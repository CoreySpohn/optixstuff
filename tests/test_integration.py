"""Integration tests for optixstuff with a real coronagraph.

Loads a real coronagraph YIP via pooch/yippy and verifies that
YippyCoronagraph and OpticalPath work correctly with real data.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from yippy import fetch_yip

import optixstuff as ox

# Public YIP downloads and backend setup belong in the nightly tier.
pytestmark = pytest.mark.slow

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
    primary = ox.SimplePrimary(
        circumscribed_diameter_m=6.0,
        collecting_area_m2=3.141592653589793 * 9 * (1 - 0.14**2),
    )
    detector = ox.Detector(
        pixel_scale_arcsec=0.010,
        shape=(100, 100),
        qe=0.9,
        dark_current_e_per_pixel_s=3e-5,
        read_noise_e_rms=0.0,
        clock_induced_charge_e_per_pixel_frame=1.3e-3,
        read_time_s=1000.0,
        survival_fraction=1.0,
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
        assert yippy_coronagraph.grid.step > 0

    def test_iwa_owa_ordering(self, yippy_coronagraph):
        assert (
            yippy_coronagraph.curves().iwa_lod
            < yippy_coronagraph.curves().data_extent_lod
        )

    def test_throughput_in_range(self, yippy_coronagraph):
        t = yippy_coronagraph.throughput(5.0, 500.0)
        assert 0.0 <= float(t) <= 1.0

    def test_core_area_positive(self, yippy_coronagraph):
        a = yippy_coronagraph.core_area(5.0, 500.0)
        assert float(a) > 0

    def test_leakage_density_lod2_nonnegative(self, yippy_coronagraph):
        cmi = yippy_coronagraph.leakage_density_lod2(5.0, 500.0)
        assert float(cmi) >= 0

    def test_occulter_transmission_in_range(self, yippy_coronagraph):
        st = yippy_coronagraph.occulter_transmission(5.0, 500.0)
        assert 0.0 <= float(st) <= 1.0

    @pytest.mark.parametrize("separation_lod", [6.0, 10.0, 16.0])
    def test_leakage_density_lod2_is_per_lod_squared(
        self, yippy_coronagraph, separation_lod
    ):
        """Exact delegation to the aperture-density record, without pixel scaling."""
        coro = yippy_coronagraph
        record = coro.curves()
        # The record is an aperture mean, not the old azimuthal per-pixel profile.
        expected = record.evaluate("leakage_density_lod2", separation_lod)
        assert coro.leakage_density_lod2(separation_lod) == expected

    @pytest.mark.parametrize("separation_lod", [6.0, 10.0])
    def test_noise_floor_ayo_matches_leakage_density_lod2(
        self, yippy_coronagraph, separation_lod
    ):
        """The AYO noise floor is the core mean intensity over the ppf.

        Both are per (lambda/D)^2 on the same object, so a caller can multiply
        either by the core area. A floor left per pixel would come back low by
        the pixel area in (lambda/D)^2.
        """
        coro = yippy_coronagraph
        ppf = 30.0
        floor = float(coro.noise_floor_ayo(separation_lod, ppf))
        cmi = float(coro.leakage_density_lod2(separation_lod, 500.0))
        assert floor == pytest.approx(cmi / ppf, rel=1e-6)


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
        assert optical_path.primary.collecting_area_m2 == pytest.approx(
            float(expected), rel=1e-6
        )
