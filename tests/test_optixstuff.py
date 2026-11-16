"""Tests for optixstuff hardware abstractions."""

import jax.numpy as jnp
import pytest

import optixstuff as ox


class TestSimplePrimary:
    def test_diameter(self, simple_primary):
        assert simple_primary.diameter("circumscribed") == 6.0

    def test_area_no_obscuration(self):
        p = ox.SimplePrimary(
            circumscribed_diameter_m=2.0, collecting_area_m2=3.141592653589793
        )
        expected = jnp.pi * 1.0**2
        assert jnp.isclose(p.collecting_area_m2, expected)

    def test_area_with_obscuration(self, simple_primary):
        expected = jnp.pi * 3.0**2 * (1.0 - 0.14**2)
        assert jnp.isclose(simple_primary.collecting_area_m2, expected)

    def test_area_with_shape_factor(self):
        p = ox.SimplePrimary(
            circumscribed_diameter_m=6.0, collecting_area_m2=3.141592653589793 * 9 * 0.9
        )
        expected = jnp.pi * 3.0**2 * 0.9
        assert jnp.isclose(p.collecting_area_m2, expected)

    def test_is_equinox_module(self, simple_primary):
        import equinox as eqx

        assert isinstance(simple_primary, eqx.Module)


class TestConstantThroughputElement:
    def test_get_throughput(self, throughput_element):
        assert throughput_element.get_throughput(500.0) == pytest.approx(0.8)

    def test_wavelength_independent(self, throughput_element):
        t1 = throughput_element.get_throughput(400.0)
        t2 = throughput_element.get_throughput(900.0)
        assert t1 == t2

    def test_apply(self, throughput_element):
        arr = jnp.ones((10, 10))
        result = throughput_element.apply(arr, 500.0)
        assert jnp.allclose(result, 0.8 * jnp.ones((10, 10)))


class TestLinearThroughputElement:
    def test_interpolation(self):
        wls = jnp.array([400.0, 600.0, 800.0])
        tps = jnp.array([0.5, 0.9, 0.7])
        el = ox.SpectralThroughput(wavelengths_nm=wls, throughputs=tps)
        # At 600 nm should be exactly 0.9
        assert el.get_throughput(600.0) == pytest.approx(0.9, abs=1e-5)

    def test_extrapolation_returns_zero(self):
        wls = jnp.array([400.0, 600.0, 800.0])
        tps = jnp.array([0.5, 0.9, 0.7])
        el = ox.SpectralThroughput(wavelengths_nm=wls, throughputs=tps)
        # Outside range should be zero
        assert el.get_throughput(200.0) == pytest.approx(0.0, abs=1e-5)
        assert el.get_throughput(1000.0) == pytest.approx(0.0, abs=1e-5)


class TestOpticalFilter:
    def test_interpolation(self):
        wls = jnp.array([500.0, 550.0, 600.0])
        trans = jnp.array([0.0, 1.0, 0.0])
        f = ox.SpectralThroughput(wavelengths_nm=wls, throughputs=trans)
        # Peak at 550 should be 1.0
        assert f.get_throughput(550.0) == pytest.approx(1.0, abs=1e-5)


class TestOpticalPath:
    def test_system_throughput_single_element(self, simple_primary, simple_detector):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.7)

    def test_system_throughput_two_elements(self, simple_primary, simple_detector):
        el1 = ox.ConstantThroughput(throughput=0.8)
        el2 = ox.ConstantThroughput(throughput=0.9)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el1, el2),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.72)

    def test_default_n_channels_and_npix_multiplier(
        self, simple_primary, simple_detector
    ):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.n_channels == 1.0
        assert path.npix_multiplier == 1.0

    def test_custom_n_channels_and_npix_multiplier(
        self, simple_primary, simple_detector
    ):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
            n_channels=2.0,
            npix_multiplier=3.0,
        )
        assert path.n_channels == 2.0
        assert path.npix_multiplier == 3.0
