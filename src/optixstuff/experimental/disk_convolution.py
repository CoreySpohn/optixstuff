"""Fixed-PSF disk convolution with optional Triton execution.

The source may have arbitrary leading candidate dimensions; the cube is shared
and remains a dynamic JAX operand. ``jax`` is the portable baseline.
``reference`` exercises the custom differentiation rules using a JAX lowering
for CPU testing. Only ``triton`` executes custom GPU kernels.

The custom operator differentiates source brightness, including higher-order
source derivatives. Differentiating or vmapping the PSF cube is deliberately
unsupported; use ``jax`` when fitting the optics themselves.
"""

import importlib
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir, xla

_contract_p = core.Primitive("optixstuff_fixed_psf_contract")
_contract_p.def_impl(partial(xla.apply_primitive, _contract_p))


def _abstract(source, matrix, *, transpose, implementation):
    if source.ndim != 2 or matrix.ndim != 2:
        raise ValueError("The fixed-PSF primitive requires two matrices.")
    inner, outer = matrix.shape[::-1] if transpose else matrix.shape
    if source.shape[1] != inner:
        raise ValueError("Source and PSF matrix contraction dimensions differ.")
    if source.dtype != matrix.dtype or source.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
    ):
        raise TypeError("The fixed-PSF primitive requires matching float32 or float64.")
    return jax.core.ShapedArray((source.shape[0], outer), source.dtype)


_contract_p.def_abstract_eval(_abstract)


def _reference(source, matrix, *, transpose):
    return jnp.matmul(
        source, matrix.T if transpose else matrix, precision=jax.lax.Precision.HIGHEST
    )


def _lower(ctx, source, matrix, *, transpose, implementation):
    if implementation != "reference":
        raise NotImplementedError(
            "The Triton disk backend requires a CUDA GPU; use backend='jax' on CPU."
        )
    return mlir.lower_fun(
        partial(_reference, transpose=transpose), multiple_results=False
    )(ctx, source, matrix)


def _lower_cuda(ctx, source, matrix, *, transpose, implementation):
    if implementation == "reference":
        return _lower(
            ctx, source, matrix, transpose=transpose, implementation=implementation
        )
    # Optional GPU dependencies are loaded only during CUDA lowering.
    try:
        kernels = importlib.import_module("optixstuff.experimental._disk_triton")
    except ImportError as exc:
        raise ImportError(
            "Install optixstuff[triton] and a CUDA-enabled JAX for the Triton backend."
        ) from exc
    return mlir.lower_fun(
        partial(kernels.contract, transpose=transpose), multiple_results=False
    )(ctx, source, matrix)


mlir.register_lowering(_contract_p, _lower)
mlir.register_lowering(_contract_p, _lower_cuda, platform="cuda")


def _jvp(primals, tangents, **params):
    source, matrix = primals
    source_dot, matrix_dot = tangents
    if not isinstance(matrix_dot, ad.Zero):
        raise NotImplementedError(
            "Triton disk convolution requires fixed PSF optics. "
            "Use backend='jax' to differentiate the cube."
        )
    result = _contract_p.bind(source, matrix, **params)
    tangent = (
        ad.Zero.from_primal_value(result)
        if isinstance(source_dot, ad.Zero)
        else _contract_p.bind(source_dot, matrix, **params)
    )
    return result, tangent


def _transpose(cotangent, source, matrix, **params):
    if ad.is_undefined_primal(matrix):
        raise NotImplementedError("The PSF cube must be fixed for the custom adjoint.")
    if not ad.is_undefined_primal(source):
        return None, None
    if isinstance(cotangent, ad.Zero):
        return ad.Zero(source.aval), None
    params = {**params, "transpose": not params["transpose"]}
    return _contract_p.bind(cotangent, matrix, **params), None


def _batch(args, dims, **params):
    source, matrix = args
    source_dim, matrix_dim = dims
    if matrix_dim is not None:
        raise NotImplementedError(
            "Triton candidates must share a fixed PSF cube; "
            "map separate optical configurations with jax.lax.map."
        )
    source = jnp.moveaxis(source, source_dim, 0)
    batch, rows, inner = source.shape
    out = _contract_p.bind(source.reshape(batch * rows, inner), matrix, **params)
    return out.reshape(batch, rows, out.shape[-1]), 0


ad.primitive_jvps[_contract_p] = _jvp
ad.primitive_transposes[_contract_p] = _transpose
batching.primitive_batchers[_contract_p] = _batch


def _fold(source):
    """Fold source quadrants with the table contract's center and edge convention."""
    ny, nx = source.shape[-2:]
    cy, cx = (ny - 1) // 2, (nx - 1) // 2
    h, w = ny - cy, nx - cx
    leading = ((0, 0),) * (source.ndim - 2)
    q1 = source[..., cy:, cx:]
    q2 = jnp.pad(source[..., cy:, :cx][..., ::-1], (*leading, (0, 0), (1, w - cx - 1)))
    q3 = jnp.pad(
        source[..., :cy, :cx][..., ::-1, ::-1],
        (*leading, (1, h - cy - 1), (1, w - cx - 1)),
    )
    q4 = jnp.pad(
        source[..., :cy, cx:][..., ::-1, :], (*leading, (1, h - cy - 1), (0, 0))
    )
    return jnp.stack((q1, q2, q3, q4), axis=-3)


def convolve_disk(source, psf_datacube, *, backend="jax"):
    """Apply a shared full or quarter PSF cube to native-grid source images.

    Args:
        source: Flux per native source pixel, shape ``(..., ny, nx)``.
        psf_datacube: PSFs indexed by source position, shape
            ``(sy, sx, detector_y, detector_x)``. The source axes must match
            ``(ny, nx)`` or ``(ny//2+1, nx//2+1)``. No normalization is applied.
        backend: ``jax`` (portable), ``triton`` (experimental CUDA), or
            ``reference`` (custom AD rules with portable JAX execution).

    Returns:
        Images shaped ``(..., detector_y, detector_x)``, in source flux units
        times the cube's transmission. Resampling and clipping belong to the
        surrounding simulation, so this operation stays linear.
    """
    if backend not in ("jax", "triton", "reference"):
        raise ValueError(f"Unknown disk convolution backend: {backend!r}")
    source, cube = jnp.asarray(source), jnp.asarray(psf_datacube)
    if source.ndim < 2 or cube.ndim != 4 or min((*source.shape, *cube.shape)) < 1:
        raise ValueError("Expected nonempty source images and a four-dimensional cube.")
    native = source.shape[-2:]
    quarter = tuple(n // 2 + 1 for n in native)
    is_quarter = cube.shape[:2] != native
    if is_quarter and cube.shape[:2] != quarter:
        raise ValueError(
            f"PSF source grid {cube.shape[:2]} must match {native} or {quarter}."
        )
    dtype = jnp.result_type(source, cube)
    source, cube = source.astype(dtype), cube.astype(dtype)
    leading = source.shape[:-2]
    detector = cube.shape[-2:]
    folded = _fold(source) if is_quarter else source
    matrix = cube.reshape(math.prod(cube.shape[:2]), math.prod(detector))
    rows = folded.reshape(-1, matrix.shape[0])
    if backend == "jax":
        result = _reference(rows, matrix, transpose=False)
    else:
        result = _contract_p.bind(rows, matrix, transpose=False, implementation=backend)
    if not is_quarter:
        return result.reshape(*leading, *detector)
    result = result.reshape(*leading, 4, *detector)
    return (
        result[..., 0, :, :]
        + result[..., 1, :, ::-1]
        + result[..., 2, ::-1, ::-1]
        + result[..., 3, ::-1, :]
    )
