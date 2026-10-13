"""Integration tests for optixstuff with a real coronagraph.

Loads a real coronagraph YIP via pooch/yippy and verifies that
YippyCoronagraph and OpticalPath work correctly with real data.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from yippy import fetch_yip

import optixstuff as ox

# ---------------------------------------------------------------------------
# Session-scoped fixtures (data downloaded once per test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def yip_path():
    """Download the default AAVC coronagraph YIP via pooch."""
    return fetch_yip("eac1_aavc_2d")


@pytest.fixture(scope="session")
def yippy_coronagraph(yip_path):
    """Build a YippyCoronagraph from the downloaded YIP."""
    return ox.YippyCoronagraph(yip_path)


@pytest.fixture(scope="session")
def optical_path(yippy_coronagraph):
    """Full OpticalPath with primary, coronagraph, detector, and a filter."""
    primary = ox.SimplePrimary(diameter_m=6.0, obscuration=0.14)
    detector = ox.Detector(
        pixel_scale_arcsec=0.010,
        shape=(100, 100),
        quantum_efficiency=0.9,
        dark_current_rate_e_per_s=3e-5,
        read_noise_e=0.0,
        clock_induced_charge_rate_e_per_frame=1.3e-3,
        frame_time_s=1000.0,
        read_time_s=1000.0,
        dqe=1.0,
    )
    optics_filter = ox.ConstantThroughput(throughput=0.5, name="optics")
    return ox.OpticalPath(
        primary=primary,
        coronagraph=yippy_coronagraph,
        attenuating_elements=(optics_filter,),
        detector=detector,
    )


# ---------------------------------------------------------------------------
# YippyCoronagraph tests
# ---------------------------------------------------------------------------


class TestYippyCoronagraph:
    """Verify YippyCoronagraph wraps EqxCoronagraph correctly."""

    def test_is_abstract_coronagraph(self, yippy_coronagraph):
        assert isinstance(yippy_coronagraph, ox.AbstractCoronagraph)

    def test_is_equinox_module(self, yippy_coronagraph):
        assert isinstance(yippy_coronagraph, eqx.Module)

    def test_pixel_scale_positive(self, yippy_coronagraph):
        assert yippy_coronagraph.pixel_scale_lod > 0

    def test_iwa_owa_ordering(self, yippy_coronagraph):
        assert yippy_coronagraph.IWA < yippy_coronagraph.OWA

    def test_throughput_in_range(self, yippy_coronagraph):
        t = yippy_coronagraph.throughput(5.0, 500.0)
        assert 0.0 <= float(t) <= 1.0

    def test_core_area_positive(self, yippy_coronagraph):
        a = yippy_coronagraph.core_area(5.0, 500.0)
        assert float(a) > 0

    def test_core_mean_intensity_nonnegative(self, yippy_coronagraph):
        cmi = yippy_coronagraph.core_mean_intensity(5.0, 500.0)
        assert float(cmi) >= 0

    def test_occulter_transmission_in_range(self, yippy_coronagraph):
        st = yippy_coronagraph.occulter_transmission(5.0, 500.0)
        assert 0.0 <= float(st) <= 1.0

    @pytest.mark.parametrize("separation_lod", [6.0, 10.0, 16.0])
    def test_core_mean_intensity_is_per_lod_squared(
        self, yippy_coronagraph, separation_lod
    ):
        """Core mean intensity is the star's light per (lambda/D)^2.

        The package's stellar map holds the fraction of the star's light on
        each pixel. Averaged over a thin ring of pixels at the separation and
        divided by one pixel's solid angle, it is the intensity per
        (lambda/D)^2 that the contract promises. A value reported per pixel
        would come back smaller by the pixel area in (lambda/D)^2, a factor of
        16 at 0.25 lambda/D pixels. The tolerance covers the difference between
        a one-pixel ring and the reader's radial bins, a few percent.
        """
        coro = yippy_coronagraph
        pixel_lod = coro.pixel_scale_lod
        star_map = jnp.asarray(coro.stellar_intens(0.0))
        ny, nx = star_map.shape
        # The shipped maps are point-symmetric about the array's middle,
        # between the four central pixels.
        rows, cols = jnp.indices((ny, nx))
        radius_lod = jnp.hypot(rows - (ny - 1) / 2.0, cols - (nx - 1) / 2.0) * pixel_lod
        ring = jnp.abs(radius_lod - separation_lod) < 0.5 * pixel_lod
        per_pixel = jnp.sum(jnp.where(ring, star_map, 0.0)) / jnp.sum(ring)
        expected = float(per_pixel) / pixel_lod**2

        got = float(coro.core_mean_intensity(separation_lod, 500.0))
        assert got == pytest.approx(expected, rel=0.1)


# ---------------------------------------------------------------------------
# OpticalPath integration tests
# ---------------------------------------------------------------------------


class TestOpticalPathIntegration:
    """Verify a real OpticalPath computes system throughput."""

    def test_system_throughput_value(self, optical_path):
        t = optical_path.system_throughput(500.0)
        assert t == pytest.approx(0.5)

    def test_primary_area(self, optical_path):
        expected = jnp.pi * 3.0**2 * (1.0 - 0.14**2)
        assert optical_path.primary.area_m2 == pytest.approx(float(expected), rel=1e-6)
