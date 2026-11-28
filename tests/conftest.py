"""Shared fixtures for optixstuff tests."""

import importlib.util
import inspect
import os
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hypothesis import settings

import optixstuff as ox
from optixstuff import SimplePupil
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.performance import PerformanceCurves
from optixstuff.spectral import Band

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")
if importlib.util.find_spec("hwoutils.precision_guard") is not None:
    pytest_plugins.append("hwoutils.precision_guard")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


@pytest.fixture(scope="session")
def phoenix():
    """A rough stellar spectrum: knots [nm] and photon flux density, peak one.

    The public PHOENIX-ACES 5800 K, log g 4.5, solar-metallicity spectrum
    (Husser et al. 2013, A&A 553, A6) as tabulated by
    ``skyscapes.io.from_phoenix``, cut to 535 to 665 nm (2972 knots, linear
    between them) and divided by its maximum.
    """
    path = Path(__file__).parent / "data" / "phoenix_5800K_logg4.5_535_665nm.npz"
    with np.load(path) as data:
        return data["wavelength_nm"], data["photon_flux_density"]


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
    band_averaged: bool = eqx.field(static=True)
    sky_trans: Image
    uniform_sky_profile: ox.SampledProfile
    _records: tuple

    def __init__(
        self,
        n=24,
        *,
        rules=(),
        diameter_m=6.0,
        bandpass=Band(450.0, 550.0),  # noqa: B008
        band_averaged=False,
    ):
        self.grid = PixelGrid.focal(
            n,
            0.5,
            SimplePupil(
                circumscribed_diameter_m=diameter_m, collecting_area_m2=28.0
            ).frame(500.0),
        )
        self.bandpass = bandpass
        self.band_averaged = band_averaged
        x, y = self.grid.coordinates()
        self.sky_trans = Image(
            1 - jnp.exp(-(x * x + y * y) / 4),
            self.grid,
            (),
            Measure.intensive(),
            "transmission",
        )
        self.uniform_sky_profile = ox.SampledProfile.from_uniform_sky(self.sky_trans)
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

    def _native_stellar_map(self, stellar_diameter_lod):
        return Image(
            jnp.full(
                self.grid.shape, 1e-8, dtype=jnp.result_type(stellar_diameter_lod, 1.0)
            )
            * (1 + stellar_diameter_lod**2),
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
        )

    def _native_source_psfs(self, x_lod, y_lod):
        x, y = self.grid.coordinates(dtype=jnp.result_type(x_lod, y_lod, 1.0))
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
        image = Image(
            data,
            self.grid,
            ("source",),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
        )

        # The source domain is the disk inscribed in the native grid.
        domain = min(self.grid.shape) * self.grid.step / 2
        valid = jnp.isfinite(dx) & jnp.isfinite(dy) & (jnp.hypot(dx, dy) <= domain)
        image = replace(
            image,
            data=jnp.where(valid[:, None, None], image.data, image.data.dtype.type(0)),
        )
        return ox.SourcePSFs(image, valid)


def synthetic_coronagraph(n=24, *, rules=()):
    return GaussianTableCoronagraph(n, rules=rules)


# Hardware-container tests need a coronagraph whose frame matches the pupil.
def table_coronagraph(diameter_m=6.0):
    return GaussianTableCoronagraph(diameter_m=diameter_m)


def make_telescope(diameter_m=6.0, **kwargs):
    """Telescope with a simple circular pupil of the given diameter."""
    return ox.Telescope(
        SimplePupil(circumscribed_diameter_m=diameter_m, collecting_area_m2=28.0),
        **kwargs,
    )


def make_instrument(coronagraph, **kwargs):
    """Observatory with one configuration named "imager" around a coronagraph.

    The pupil has the coronagraph's diameter. Extra keywords go to the
    configuration; read it with ``observatory.select("imager")``.
    """
    return ox.Observatory(
        make_telescope(coronagraph.frame.diameter_m),
        (
            ox.Instrument(
                "imager",
                attenuating_elements=(
                    ox.ConstantThroughput(throughput=1.0, name="optics"),
                ),
                coronagraph=coronagraph,
                detector=ox.Detector(pixel_scale_arcsec=0.01, shape=(512, 512), qe=0.9),
                **kwargs,
            ),
        ),
    )


@pytest.fixture
def simple_pupil():
    """A basic 6m pupil with 14% obscuration."""
    return ox.SimplePupil(
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
    angular_frequency_per_s: object

    def coefficients(self, *, elapsed_s):
        import jax.numpy as jnp

        dtype = jnp.result_type(
            elapsed_s, self.offset_nm, self.amplitude_nm, self.angular_frequency_per_s
        )
        t = jnp.asarray(elapsed_s, dtype=dtype)
        return self.offset_nm + self.amplitude_nm * jnp.sin(
            t[..., None] * self.angular_frequency_per_s
        )


def make_linear_speckle(epoch_d=0.0, epoch=None):
    """Small asymmetric complex field in the active precision."""
    import jax.numpy as jnp

    pupil = ox.SimplePupil(circumscribed_diameter_m=6.0, collecting_area_m2=28.0)
    grid = PixelGrid.regular((3, 4), 0.37, "lod", frame=pupil.frame(500.0))
    y, x = np.indices(grid.shape)
    return SinusoidalSpeckle(
        ox.FocalResponse(
            (1 + 0.1 * x) + 1j * (0.4 - 0.2 * y),
            np.stack([(0.3 + 0.2j) * (x + 1), (-0.2 + 0.4j) * (y + 1)]),
            grid,
            7.3,
            500.0,
            wavelength_law="opd",
        ),
        offset_nm=jnp.array([0.03, -0.02]),
        amplitude_nm=jnp.array([0.05, 0.04]),
        angular_frequency_per_s=jnp.array([2.0, 3.0]),
        epoch=Epoch() if epoch is None else epoch,
        epoch_d=epoch_d,
    )


@pytest.fixture
def linear_speckle():
    """Small asymmetric complex field in the active precision."""
    return make_linear_speckle()


@pytest.fixture
def displaced_gaussian_source():
    """Gaussian source displaced by (0.08, -0.06) lod from its nominal point."""
    from optixstuff.grids import Frame, Image, Measure, PixelGrid

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
            return Image(data, grid, (), Measure.integral(), "stellar_flux_fraction")

        star = Image(
            np.full(grid.shape, 1e-9 * step**2),
            grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )
        return source, star

    return build
