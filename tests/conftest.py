"""Shared fixtures for optixstuff tests."""

import importlib.util
import inspect
import os

import numpy as np
import pytest
from astropy.io import fits
from hwoutils.grids import PixelGrid
from hypothesis import settings
from yippy import Coronagraph, EqxCoronagraph

import optixstuff as ox
from optixstuff import ScalarCoronagraph, SimplePrimary, YippyCoronagraph

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


# Small analytic YIP fixtures with independent pixel-centre coordinates.


def concrete_subclasses(base):
    classes = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            classes.append(cls)
        classes.extend(concrete_subclasses(cls))
    return classes


def synthetic_coronagraph(path, n=24, *, rules=()):
    path.mkdir(exist_ok=True)
    scale, center, sigma = 0.5, (n - 1) / 2, 0.7
    yy, xx = np.indices((n, n))
    xx, yy = (xx - center) * scale, (yy - center) * scale
    offsets = np.array(
        [(x, y) for x in (0.0, 1.0, 2.0, 3.0) for y in (0.0, 1.0, 2.0, 3.0)]
    )
    psfs = np.array(
        [
            0.6
            * (1 - 0.9 * np.exp(-(x * x + y * y) / 4))
            * scale**2
            / (2 * np.pi * sigma**2)
            * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
            for x, y in offsets
        ]
    )
    header = fits.Header(
        {
            "D": 6.0,
            "D_INSC": 4.7,
            "OBSCURED": 0.2,
            "PIXSCALE": scale,
            "LAMBDA": 500.0,
            "MINLAM": 450.0,
            "MAXLAM": 550.0,
            "XCENTER": center + 0.5,
            "YCENTER": center + 0.5,
            "N_LAM": 1,
            "N_STAR": 3,
        }
    )
    for key, unit in (
        ("D", "m"),
        ("D_INSC", "m"),
        ("LAMBDA", "nm"),
        ("MINLAM", "nm"),
        ("MAXLAM", "nm"),
    ):
        header.comments[key] = f"value in {unit}"
    for filename, data in (
        ("offax_psf", psfs),
        ("offax_psf_offset_list", offsets),
        (
            "stellar_intens",
            np.stack([np.full((n, n), 1e-8 * (1 + d * d)) for d in (0.0, 0.1, 0.2)]),
        ),
        ("stellar_intens_diam_list", np.array([0.0, 0.1, 0.2])),
        ("sky_trans", 1 - np.exp(-(xx * xx + yy * yy) / 4)),
    ):
        fits.PrimaryHDU(data, header).writeto(path / f"{filename}.fits", overwrite=True)
    host = Coronagraph(path)
    for rule in rules:
        host.compute_all_performance_curves(rule)
    host.create_psf_datacube()
    return YippyCoronagraph(
        backend=EqxCoronagraph(yippy_coro=host, ensure_psf_datacube=True)
    )


# Constant coronagraph shared by hardware-container tests.
def scalar_coronagraph():
    frame = SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=28.0).frame(
        500.0
    )
    return ScalarCoronagraph(
        grid=PixelGrid.focal(24, 0.5, frame),
        throughput_fraction=0.5,
        aperture_area_lod2=1.0,
        stellar_leakage_density_lod2=1e-10,
        outer_working_angle_lod=30.0,
    )


@pytest.fixture(scope="session")
def cached_diameter_coro(tmp_path_factory):
    """Synthetic diameter-dependent package; all generated files stay in tmp_path."""
    return synthetic_coronagraph(tmp_path_factory.mktemp("stellar_diameter_yip"))


@pytest.fixture
def simple_primary():
    """A basic 6m primary mirror with 14% obscuration."""
    return ox.SimplePrimary(
        circumscribed_diameter_m=6.0,
        collecting_area_m2=3.141592653589793 * 9 * (1 - 0.14**2),
    )


@pytest.fixture
def simple_detector():
    """A simple detector with typical HWO-like parameters."""
    return ox.IdealDetector(
        pixel_scale_arcsec=0.010,
        shape=(100, 100),
    )


@pytest.fixture
def throughput_element():
    """A constant 80% throughput optical element."""
    return ox.ConstantThroughput(throughput=0.8, name="test_filter")


class SinusoidalSpeckle(ox.LinearSpeckleField):
    """Two-mode deterministic trajectory; coefficient unit is nm of OPD."""

    offset_nm: object
    amplitude_nm: object
    angular_frequency_per_d: object

    def coefficients(self, time_d):
        import jax.numpy as jnp

        # Subtract on the host when concrete; traced absolute times need x64.
        elapsed_d = jnp.asarray(time_d - self.epoch_d)
        return self.offset_nm + self.amplitude_nm * jnp.sin(
            elapsed_d[..., None] * self.angular_frequency_per_d
        )


@pytest.fixture
def linear_speckle():
    """Small asymmetric complex field; scoped x64, never import-order precision."""
    import jax
    import jax.numpy as jnp

    with jax.enable_x64():
        primary = ox.SimplePrimary(
            circumscribed_diameter_m=6.0, collecting_area_m2=28.0
        )
        grid = PixelGrid.regular((3, 4), 0.37, "lod", frame=primary.frame(500.0))
        y, x = np.indices(grid.shape)
        yield SinusoidalSpeckle(
            e_nom=(1 + 0.1 * x) + 1j * (0.4 - 0.2 * y),
            G=np.stack([(0.3 + 0.2j) * (x + 1), (-0.2 + 0.4j) * (y + 1)]),
            input_energy=7.3,
            grid=grid,
            reference_wavelength_nm=500.0,
            offset_nm=jnp.array([0.03, -0.02]),
            amplitude_nm=jnp.array([0.05, 0.04]),
            angular_frequency_per_d=jnp.array([2.0, 3.0]),
        )
