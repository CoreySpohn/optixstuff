"""Measure exact disk operators; synthetic timings are not disk-fit speedups."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from optixstuff.coronagraph import _convolve_quadrants
from optixstuff.experimental.disk_convolution import convolve_disk


def apply(source, cube, backend):
    """Compare the existing mapped contraction with the explicit candidate axis."""
    if backend == "legacy":
        if source.shape[-2:] == cube.shape[:2]:
            return jax.vmap(lambda s: jnp.einsum("ij,ijxy->xy", s, cube))(source)
        return jax.vmap(_convolve_quadrants, in_axes=(0, None))(source, cube)
    return convolve_disk(source, cube, backend=backend)


def render(z, yy, xx):
    """Analytic Gaussian ring for timing only; use GraterDisk for scientific fits."""
    radius = jnp.sqrt(xx**2 + (yy / 0.7) ** 2 + 1e-8)
    return jnp.exp(z[:, 0, None, None]) * jnp.exp(
        -0.5 * ((radius - z[:, 1, None, None]) / jnp.exp(z[:, 2, None, None])) ** 2
    )


def adjoint(residual, source, cube, backend):
    """Propagate image sensitivity back to source brightness."""
    _, pullback = jax.vjp(lambda s: apply(s, cube, backend), source)
    return pullback(residual)[0]


def loss(z, cube, data, sigma, yy, xx, backend):
    """Gaussian image objective including a parameterized source render."""
    prediction = apply(render(z, yy, xx), cube, backend)
    return 0.5 * jnp.sum(((prediction - data) / sigma) ** 2)


def timed(fn, args, repeats):
    """Separate first-call compile/execution from synchronized warm latency."""
    compiled = eqx.filter_jit(fn)
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(*args))
    first = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    return result, dict(
        first_call_s=first,
        median_ms=1e3 * float(np.median(samples)),
        min_ms=1e3 * min(samples),
        p90_ms=1e3 * float(np.quantile(samples, 0.9)),
    )


def error(actual, reference):
    """Report absolute and normwise errors, avoiding relative errors at dark pixels."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return dict(
        max_abs=float(np.max(np.abs(a - b))),
        relative_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
    )


def main():
    """Write machine-readable results for CPU or CUDA, with explicit provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=65)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cube", type=Path, help="Real native PSF cube saved as .npy")
    parser.add_argument("--source-shape", type=int, nargs=2)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--backends", nargs="+", default=["legacy", "jax", "triton"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("disk-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or min(args.batches) < 1 or args.size < 1:
        parser.error("sizes, batches, and repeats must be positive")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if "triton" in args.backends and jax.default_backend() != "gpu":
        parser.error(
            "Triton requires CUDA; for CPU use --backends legacy jax reference"
        )
    rng = np.random.default_rng(0)
    dtype = np.dtype(args.dtype)
    start = time.perf_counter()
    if args.cube:
        host_cube = np.load(args.cube, allow_pickle=False).astype(dtype, copy=False)
        with args.cube.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        provenance = dict(kind="real_cube", path=str(args.cube), sha256=digest)
    else:
        n = args.size
        c = n if args.full else n // 2 + 1
        host_cube = rng.random((c, c, n, n), dtype=np.float64).astype(dtype) / n**2
        provenance = dict(
            kind="synthetic_random_psfs",
            seed=0,
            warning="Synthetic bandwidth workload; not scientific fit validation",
        )
    source_shape = tuple(args.source_shape or host_cube.shape[-2:])
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    cube = jax.block_until_ready(jax.device_put(host_cube))
    transfer_s = time.perf_counter() - start
    device = jax.devices()[0]
    versions = {}
    for package in ("jax", "jaxlib", "equinox", "jax-triton", "triton", "optixstuff"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = dict(
        device=str(device),
        device_kind=device.device_kind,
        platform=platform.platform(),
        versions=versions,
        provenance=provenance,
        cube_shape=cube.shape,
        source_shape=source_shape,
        dtype=args.dtype,
        cube_bytes=cube.nbytes,
        setup_s=setup_s,
        transfer_s=transfer_s,
        rows=[],
    )
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1, 1, source_shape[0], dtype=dtype),
        jnp.linspace(-1, 1, source_shape[1], dtype=dtype),
        indexing="ij",
    )
    value_grad = jax.value_and_grad(loss)
    for batch in args.batches:
        z = jnp.tile(jnp.array([0.0, 0.5, np.log(0.12)], dtype=dtype), (batch, 1))
        z = z.at[:, 1].add(jnp.linspace(-0.03, 0.03, batch, dtype=dtype))
        source, render_time = timed(render, (z, yy, xx), args.repeats)
        cube64, source64 = cube.astype(jnp.float64), source.astype(jnp.float64)
        reference = jax.block_until_ready(
            eqx.filter_jit(apply)(source64, cube64, "jax")
        )
        sigma = jnp.asarray(float(jnp.max(reference)) * 0.01, dtype=dtype)
        residual = jnp.asarray(rng.normal(size=reference.shape), dtype=dtype)
        data = (reference + sigma * residual).astype(dtype)
        reference_adj = jax.block_until_ready(
            eqx.filter_jit(adjoint)(
                residual.astype(jnp.float64), source64, cube64, "jax"
            )
        )
        ref_loss, ref_grad = jax.block_until_ready(
            eqx.filter_jit(value_grad)(
                z.astype(jnp.float64),
                cube64,
                data.astype(jnp.float64),
                sigma.astype(jnp.float64),
                yy.astype(jnp.float64),
                xx.astype(jnp.float64),
                "jax",
            )
        )
        del cube64, source64
        for backend in args.backends:
            prediction, forward_time = timed(
                apply, (source, cube, backend), args.repeats
            )
            sensitivity, adj_time = timed(
                adjoint, (residual, source, cube, backend), args.repeats
            )
            (value, grad), vg_time = timed(
                value_grad, (z, cube, data, sigma, yy, xx, backend), args.repeats
            )
            row = dict(
                batch=batch,
                backend=backend,
                render=render_time,
                forward=forward_time,
                adjoint=adj_time,
                value_and_grad=vg_time,
                candidates_per_s=batch * 1e3 / forward_time["median_ms"],
                image_error=error(prediction, reference),
                adjoint_error=error(sensitivity, reference_adj),
                loss_abs_error=float(abs(value - ref_loss)),
                gradient_error=error(grad, ref_grad),
            )
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"B={batch:3d} {backend:9s} forward={forward_time['median_ms']:.3f} ms "
                f"value+grad={vg_time['median_ms']:.3f} ms "
                f"image rel-L2={row['image_error']['relative_l2']:.3g}",
                flush=True,
            )
    report["memory_stats"] = device.memory_stats()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
