"""Shared fixtures for optixstuff tests."""

import importlib.util
import inspect
import os

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.grids import Image, PixelGrid
from hwoutils.spectral import Band
from hypothesis import settings

import optixstuff as ox
from optixstuff import ScalarCoronagraph, SimplePrimary
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from optixstuff.performance import PerformanceCurves

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


# Small analytic optics with independent pixel-centre coordinates.


def concrete_subclasses(base):
    classes = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            classes.append(cls)
        classes.extend(concrete_subclasses(cls))
    return classes


class GaussianTableCoronagraph(ox.AbstractTableCoronagraph):
    """Gaussian source fractions and a uniform diameter-dependent stellar table.

    Native samples are amplitude*pixel_area*exp(-r^2/(2*sigma^2))/(2*pi*sigma^2),
    with sigma=0.7. Circular records use the analytic encircled fraction
    amplitude*(1-exp(-radius^2/(2*sigma^2))) and geometric area pi*radius^2.
    """

    grid: PixelGrid
    bandpass: Band
    sky_trans: Image
    psf_datacube: object
    _records: tuple
    outer_working_angle_lod: float | None = None

    def __init__(self, n=24, *, rules=()):
        self.grid = PixelGrid.focal(
            n,
            0.5,
            SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=28.0).frame(
                500.0
            ),
        )
        self.bandpass = Band(450.0, 550.0)
        x, y = self.grid.coordinates()
        self.sky_trans = Image(
            1 - jnp.exp(-(x * x + y * y) / 4),
            self.grid,
            (),
            "per_unit_area",
            "transmission",
        )
        start = n // 2
        self.psf_datacube = self._native_source_psfs(
            x[start:, start:].ravel(), y[start:, start:].ravel()
        ).data.reshape((n - start, n - start, n, n))
        axis = jnp.arange(4.0)
        amplitude = 0.6 * (1 - 0.9 * jnp.exp(-(axis**2) / 4))
        records = []
        for rule in dict.fromkeys((DEFAULT_APERTURE_RULE, *rules)):
            radius = rule.radius_lod
            tau = amplitude * (1 - jnp.exp(-(radius**2) / (2 * 0.7**2)))
            area = jnp.full_like(axis, np.pi * radius**2)
            density = jnp.full_like(axis, 1e-8 / self.grid.pixel_area)
            records.append(
                PerformanceCurves(
                    axis,
                    self.grid.frame,
                    rule,
                    tau,
                    area,
                    density * area,
                    density,
                    density * area / tau,
                    0.0,
                    min(self.grid.shape) * self.grid.step / 2,
                    throughput_variance=tau,
                    core_area_variance_lod2=area,
                    leakage_fraction_variance=density * area,
                    leakage_density_variance_lod2=density,
                )
            )
        self._records = tuple(records)

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    def _native_stellar_map(self, stellar_diam_lod):
        return Image(
            jnp.full(self.grid.shape, 1e-8) * (1 + stellar_diam_lod**2),
            self.grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
            outside="reflect",
        )

    def _native_source_psfs(self, x_lod, y_lod):
        x, y = self.grid.coordinates()
        dx, dy = jnp.atleast_1d(x_lod), jnp.atleast_1d(y_lod)
        amplitude = 0.6 * (1 - 0.9 * jnp.exp(-(dx**2 + dy**2) / 4))
        data = (
            amplitude[:, None, None]
            * self.grid.pixel_area
            / (2 * np.pi * 0.7**2)
            * jnp.exp(
                -((x - dx[:, None, None]) ** 2 + (y - dy[:, None, None]) ** 2)
                / (2 * 0.7**2)
            )
        )
        return Image(
            data,
            self.grid,
            ("source",),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
            outside="zero",
        )


def synthetic_coronagraph(n=24, *, rules=()):
    return GaussianTableCoronagraph(n, rules=rules)


# Constant coronagraph shared by hardware-container tests.
def scalar_coronagraph(diameter_m=6.0):
    frame = SimplePrimary(
        circumscribed_diameter_m=diameter_m, collecting_area_m2=28.0
    ).frame(500.0)
    return ScalarCoronagraph(
        grid=PixelGrid.focal(24, 0.5, frame),
        throughput_fraction=0.5,
        aperture_area_lod2=1.0,
        stellar_leakage_density_lod2=1e-10,
        outer_working_angle_lod=30.0,
    )


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


@pytest.fixture
def displaced_gaussian_source():
    """Gaussian source displaced by (0.08, -0.06) lod from its nominal point."""
    from hwoutils.grids import Frame, Image, PixelGrid

    def build(step):
        grid = PixelGrid.focal(int(12 / step), step, Frame(500, 6, "circumscribed"))
        x, y = grid.coordinates()

        def source(separation):
            data = (
                0.31
                * np.exp(
                    -((x - separation - 0.08) ** 2 + (y + 0.06) ** 2) / (2 * 0.6**2)
                )
                / (2 * np.pi * 0.6**2)
                * step**2
            )
            return Image(data, grid, (), "per_pixel", "stellar_flux_fraction")

        star = Image(
            np.full(grid.shape, 1e-9 * step**2),
            grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
        )
        return source, star

    return build
