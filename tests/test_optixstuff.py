"""Tests for optixstuff hardware abstractions."""

import jax.numpy as jnp
import pytest

import optixstuff as ox


class TestSimplePrimary:
    def test_diameter(self, simple_primary):
        assert simple_primary.diameter_m == 6.0

    def test_area_no_obscuration(self):
        p = ox.SimplePrimary(diameter_m=2.0, obscuration=0.0)
        expected = jnp.pi * 1.0**2
        assert jnp.isclose(p.area_m2, expected)

    def test_area_with_obscuration(self, simple_primary):
        expected = jnp.pi * 3.0**2 * (1.0 - 0.14**2)
        assert jnp.isclose(simple_primary.area_m2, expected)

    def test_area_with_shape_factor(self):
        p = ox.SimplePrimary(diameter_m=6.0, obscuration=0.0, shape_factor=0.9)
        expected = jnp.pi * 3.0**2 * 0.9
        assert jnp.isclose(p.area_m2, expected)

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


class TestSimpleDetector:
    def test_qe(self, simple_detector):
        assert simple_detector.get_qe(500.0) == pytest.approx(0.9)

    def test_qe_wavelength_independent(self, simple_detector):
        assert simple_detector.get_qe(400.0) == simple_detector.get_qe(900.0)

    def test_dark_current_rate(self, simple_detector):
        assert simple_detector.dark_current_rate == pytest.approx(1e-4)

    def test_read_noise(self, simple_detector):
        assert simple_detector.read_noise_electrons == pytest.approx(3.0)

    def test_scalar_noise_rate(self, simple_detector):
        # dark variance: 1e-4 * 10 = 1e-3
        # cic variance: 0.02 * 10 / 1000.0 = 2e-4
        result = simple_detector.scalar_noise_rate(n_pix=10.0, t_photon=1000.0)
        expected = 1e-4 * 10.0 + 0.02 * 10.0 / 1000.0
        assert result == pytest.approx(expected)


class TestOpticalPath:
    def test_system_throughput_single_element(self, simple_primary, simple_detector):
        el = ox.ConstantThroughputElement(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,  # placeholder
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.7)

    def test_system_throughput_two_elements(self, simple_primary, simple_detector):
        el1 = ox.ConstantThroughputElement(throughput=0.8)
        el2 = ox.ConstantThroughputElement(throughput=0.9)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el1, el2),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.72)
