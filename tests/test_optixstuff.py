"""Tests for optixstuff hardware abstractions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.spectral import Transmission, spectral_breaks

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


class TestOpticalPath:
    def test_transmission_elements_compose(self, simple_primary, simple_detector):
        """Fails if held responses lose knots, attenuation or parameter gradients.

        Exact algebra: at 550 nm the linear response is (0.2+0.8)/2;
        the second element transmits 0.4. Allow 2e-15 floating-point error.
        """
        with jax.enable_x64():
            element = ox.TransmissiveElement(Transmission([500, 600], [0.2, 0.8]))
            path = ox.OpticalPath(
                primary=simple_primary,
                attenuating_elements=(element, ox.ConstantThroughput(0.4)),
                coronagraph=None,
                detector=simple_detector,
            )
            expected = (0.2 + 0.8) / 2 * 0.4
            np.testing.assert_allclose(
                eqx.filter_jit(lambda p: p.system_throughput(550.0))(path),
                expected,
                rtol=0,
                atol=2e-15,
            )
            np.testing.assert_array_equal(spectral_breaks(path), [500, 600])
            np.testing.assert_allclose(
                element.apply(jnp.array([[2.0, 3.0]]), 550),
                jnp.array([[2.0, 3.0]]) * ((0.2 + 0.8) / 2),
                rtol=0,
                atol=2e-15,
            )
            gradient = eqx.filter_grad(lambda p: p.system_throughput(550.0))(path)
            np.testing.assert_allclose(
                gradient.attenuating_elements[0].transmission.transmission,
                [0.4 / 2, 0.4 / 2],
                rtol=0,
                atol=2e-15,
            )
            assert path.system_throughput(499.0) == 0
            assert path.system_throughput(601.0) == 0

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
