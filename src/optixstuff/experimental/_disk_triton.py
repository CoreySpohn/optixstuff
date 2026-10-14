"""CUDA kernels for a fixed PSF operator and its transpose.

Split the source reduction to expose parallelism even for one candidate, then
reduce partial images in JAX. Float32 candidate batches use tiled IEEE dot
products; small batches and float64 use explicit multiply/reduce tiles. Neither
path uses TF32, float16, atomics, or an expanded quarter-symmetric PSF cube.
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton as tr
import triton.language as tl


@tr.jit
def _stream_kernel(
    Source,
    Matrix,
    Partial,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.program_id(1)
    split = tl.program_id(2)
    start = split * SPLIT_K
    k = start + tl.arange(0, BLOCK_K)
    acc = tl.full(
        (BLOCK_K, BLOCK_N),
        0,
        tl.float64 if Source.dtype.element_ty == tl.float64 else tl.float32,
    )
    for offset in range(tr.cdiv(SPLIT_K, BLOCK_K)):
        kk = k + offset * BLOCK_K
        s = tl.load(Source + b.to(tl.int64) * K + kk, kk < K, other=0)
        p = tl.load(
            Matrix
            + kk[:, None].to(tl.int64) * STRIDE_K
            + n[None, :].to(tl.int64) * STRIDE_N,
            (kk[:, None] < K) & (n[None, :] < N),
            other=0,
        )
        acc += s[:, None] * p
    total = tl.sum(acc, axis=0)
    tl.store(Partial + (split.to(tl.int64) * B + b) * N + n, total, n < N)


@tr.jit
def _batched_kernel(
    Source,
    Matrix,
    Partial,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    split = tl.program_id(2)
    k = split * SPLIT_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_B, BLOCK_N), tl.float32)
    for offset in range(tr.cdiv(SPLIT_K, BLOCK_K)):
        kk = k + offset * BLOCK_K
        s = tl.load(
            Source + b[:, None].to(tl.int64) * K + kk[None, :],
            (b[:, None] < B) & (kk[None, :] < K),
            other=0,
        )
        p = tl.load(
            Matrix
            + kk[:, None].to(tl.int64) * STRIDE_K
            + n[None, :].to(tl.int64) * STRIDE_N,
            (kk[:, None] < K) & (n[None, :] < N),
            other=0,
        )
        acc = tl.dot(s, p, acc, input_precision="ieee")
    tl.store(
        Partial + (split.to(tl.int64) * B + b[:, None]) * N + n[None, :],
        acc,
        (b[:, None] < B) & (n[None, :] < N),
    )


def contract(source, matrix, *, transpose):
    """Launch an operator or adjoint without transposing the resident PSF matrix."""
    b, k = source.shape
    n = matrix.shape[0] if transpose else matrix.shape[1]
    stride_k, stride_n = (1, matrix.shape[1]) if transpose else (matrix.shape[1], 1)
    # Cap scratch storage to approximately 64 MiB, apart from one output slice.
    max_splits = max(1, min(32, (64 * 1024**2) // (b * n * source.dtype.itemsize)))
    split_k = tr.cdiv(tr.cdiv(k, max_splits), 32) * 32
    splits = tr.cdiv(k, split_k)
    out_shape = jax.ShapeDtypeStruct((splits, b, n), source.dtype)
    common = dict(
        B=b,
        K=k,
        N=n,
        STRIDE_K=stride_k,
        STRIDE_N=stride_n,
        SPLIT_K=split_k,
        BLOCK_K=32,
        BLOCK_N=64,
    )
    if source.dtype == jnp.float32 and b >= 8:
        partials = jt.triton_call(
            source,
            matrix,
            kernel=_batched_kernel,
            out_shape=out_shape,
            grid=(tr.cdiv(n, 64), tr.cdiv(b, 16), splits),
            BLOCK_B=16,
            num_warps=4,
            **common,
        )
    else:
        partials = jt.triton_call(
            source,
            matrix,
            kernel=_stream_kernel,
            out_shape=out_shape,
            grid=(tr.cdiv(n, 64), b, splits),
            num_warps=4,
            **common,
        )
    return partials.sum(axis=0)
