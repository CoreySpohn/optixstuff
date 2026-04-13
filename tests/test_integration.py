"""Integration tests for optixstuff with a real coronagraph.

Loads a real coronagraph YIP via pooch/yippy and verifies that
YippyCoronagraph and OpticalPath work correctly with real data.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from yippy.datasets import fetch_coronagraph

import optixstuff as ox


# ---------------------------------------------------------------------------
# Session-scoped fixtures (data downloaded once per test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def yip_path():
    """Download the default AAVC coronagraph YIP via pooch."""
    return fetch_coronagraph()


@pytest.fixture(scope="session")
def yippy_coronagraph(yip_path):
    """Build a YippyCoronagraph from the downloaded YIP."""
    return ox.YippyCoronagraph(yip_path)


@pytest.fixture(scope="session")
def optical_path(yippy_coronagraph):
    """Full OpticalPath with primary, coronagraph, detector, and a filter."""
    primary = ox.SimplePrimary(diameter_m=6.0, obscuration=0.14)
    detector = ox.Detector(
        pixel_scale=0.010,
        shape=(100, 100),
        quantum_efficiency=0.9,
        dark_current_rate=3e-5,
        read_noise_electrons=0.0,
        cic_rate=1.3e-3,
        frame_time=1000.0,
        read_time=1000.0,
        dqe=1.0,
    )
    optics_filter = ox.ConstantThroughputElement(throughput=0.5, name="optics")
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
