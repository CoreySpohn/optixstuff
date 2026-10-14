"""Real-YIP GraterDisk fit through coronagraphoto and Photomancy (FP64).

Start with --psf-size 32 or 64; the native 256-pixel quarter cube alone is
8.72 GB in FP64. Increasing --psf-size changes the optical sampling and is
not evidence of convergence to the native solution.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from coronagraphoto import disk_rate
from disk_convolution import error, timed
from hwoutils.constants import Msun2kg
from photomancy.backends import LaplaceBackend
from photomancy.disk import build_disk_logdensity
from skyscapes.disk import GraterDisk
from skyscapes.scene import FlatStar
from yippy import fetch_yip

from optixstuff import OpticalPath, YippyCoronagraph


class DiskScene(eqx.Module):
    """The fitted subset of a disk-only system; no orbit propagation is needed."""

    disk: GraterDisk
    midplane_inc_deg: jax.Array
    midplane_pa_deg: jax.Array


class DiskForward(eqx.Module):
    """Keep the instrument and its large PSF arrays visible as dynamic PyTree leaves."""

    optical_path: OpticalPath
    star: FlatStar

    def __call__(self, system):
        """Return photon rates, including rotation, sampling, and final clipping."""
        return disk_rate(
            system.disk,
            self.optical_path,
            start_time_jd=0.0,
            wavelength_nm=500.0,
            bin_width_nm=50.0,
            telescope_pa_deg=17.0,
            star=self.star,
            incl_deg=system.midplane_inc_deg,
            pa_deg=system.midplane_pa_deg,
        )


def make_scene(size):
    """Build the physical disk family independently of the optical backend."""
    disk = GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.array(0.2),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(1.5),
        rmax_AU=jnp.array(4.5),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([0.5, 0.5]),
        nx=size,
        ny=size,
        pixel_scale_arcsec=0.015,
        dist_pc=10.0,
        n_slices_los=21,
        n_scale_heights=3.0,
    )
    return DiskScene(disk, jnp.array(45.0), jnp.array(30.0))


def fit_leaves(s):
    """Fit radius and orientation; hold other morphology parameters fixed here."""
    return [s.disk.sma_AU, s.midplane_inc_deg, s.midplane_pa_deg]


def prior(s):
    """Weak Gaussian shape prior shared by all backend comparisons."""
    return -0.5 * (
        ((s.disk.sma_AU - 3) / 0.5) ** 2
        + ((s.midplane_inc_deg - 45) / 15) ** 2
        + ((s.midplane_pa_deg - 30) / 15) ** 2
    )


def score(z, model):
    """Differentiate only fitted parameters, leaving instrument arrays fixed."""
    return model(z)


def run_fit(model, z, steps):
    """Include MAP optimization and the Hessian-based covariance in fit timing."""
    return LaplaceBackend(n_steps=steps).run(model, z)


def main():
    """Compare real disk fitting and posterior summaries against the JAX reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yip", default="eac1_aavc_2d", help="Catalog name or local directory"
    )
    parser.add_argument("--psf-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backend", choices=["triton", "reference", "jax-batched"], default="triton"
    )
    parser.add_argument("--export-cube", type=Path)
    parser.add_argument("--output", type=Path, default=Path("photomancy-disk-fit.json"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if args.backend == "triton" and jax.default_backend() != "gpu":
        parser.error("CUDA is required; --backend reference tests integration on CPU")
    start = time.perf_counter()
    yip_path = args.yip if Path(args.yip).is_dir() else fetch_yip(args.yip)
    # Isolate derived caches: Yippy's performance cache key omits PSF sampling.
    with tempfile.TemporaryDirectory(prefix="disk-fit-yip-") as temporary:
        for file in Path(yip_path).iterdir():
            if file.is_file():
                (Path(temporary) / file.name).symlink_to(file.resolve())
        coro = YippyCoronagraph(
            temporary,
            ensure_psf_datacube=True,
            downsample_shape=(args.psf_size, args.psf_size),
            # Keep unused scalar-performance apertures resolved on coarse grids.
            aperture_radius_lod=3.0,
        )
    cube = coro.psf_datacube
    jax.block_until_ready(cube)
    print(f"Loaded {cube.shape}, {cube.dtype}, {cube.nbytes / 1e9:.3f} GB", flush=True)
    if args.export_cube:
        np.save(args.export_cube, np.asarray(cube), allow_pickle=False)
    path = OpticalPath.from_default_setup(
        coro,
        detector_shape=(args.image_size, args.image_size),
        pixel_scale_arcsec=0.015,
    )
    star = FlatStar(Ms_kg=Msun2kg, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    truth = make_scene(args.image_size)
    forward = DiskForward(path, star)
    noiseless = eqx.filter_jit(lambda f, s: f(s))(forward, truth)
    sigma = jnp.max(noiseless) * 0.01
    data = noiseless + sigma * jax.random.normal(jax.random.key(0), noiseless.shape)
    initial = eqx.tree_at(
        fit_leaves, truth, [jnp.array(3.1), jnp.array(47.0), jnp.array(32.0)]
    )
    report = dict(
        device=jax.devices()[0].device_kind,
        dtype=str(cube.dtype),
        yip=args.yip,
        psf_shape=coro.psf_shape,
        cube_shape=cube.shape,
        cube_bytes=cube.nbytes,
        source_pixel_scale_arcsec=0.015,
        native_pixel_scale_lod=float(coro.pixel_scale_lod),
        image_size=args.image_size,
        n_slices_los=21,
        wavelength_nm=500.0,
        bin_width_nm=50.0,
        roll_deg=17.0,
        setup_s=time.perf_counter() - start,
        rows=[],
    )
    reference = None
    for backend in dict.fromkeys(("jax", "jax-batched", args.backend)):
        fwd = eqx.tree_at(
            lambda f: f.optical_path.coronagraph.disk_convolution_backend,
            forward,
            backend,
        )
        model, z0, unravel = build_disk_logdensity(
            initial,
            data,
            fit_leaves=fit_leaves,
            noise_sigma=sigma,
            forward=fwd,
            prior=prior,
        )
        (_, gradient), vg_time = timed(
            jax.value_and_grad(score), (z0, model), args.repeats
        )
        hessian, hess_time = timed(jax.hessian(score), (z0, model), args.repeats)
        posterior, fit_time = timed(run_fit, (model, z0, args.steps), args.repeats)
        value, final_grad = jax.value_and_grad(score)(posterior.mean, model)
        fitted = eqx.combine(unravel(posterior.mean), model.static)
        row = dict(
            backend=backend,
            value_and_grad=vg_time,
            hessian=hess_time,
            fit=fit_time,
            map=posterior.mean.tolist(),
            covariance=posterior.cov.tolist(),
            parameter_names_in_physical_order=["sma_AU", "incl_deg", "pa_deg"],
            recovered=[float(x) for x in fit_leaves(fitted)],
            truth=[float(x) for x in fit_leaves(truth)],
            log_evidence=float(posterior.evidence),
            log_density=float(value),
            final_gradient_norm=float(jnp.linalg.norm(final_grad)),
            steps=args.steps,
        )
        if reference is None:
            reference = (posterior, gradient, hessian)
        else:
            ref_post, ref_grad, ref_hess = reference
            row.update(
                map_error=error(posterior.mean, ref_post.mean),
                covariance_error=error(posterior.cov, ref_post.cov),
                gradient_error=error(gradient, ref_grad),
                hessian_error=error(hessian, ref_hess),
                evidence_abs_error=float(abs(posterior.evidence - ref_post.evidence)),
            )
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{backend}: value+grad {vg_time['median_ms']:.2f} ms; "
            f"complete fit {fit_time['median_ms']:.2f} ms; "
            f"recovered {row['recovered']}; "
            f"gradient norm {row['final_gradient_norm']:.3g}",
            flush=True,
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
