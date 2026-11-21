"""Tests for optixstuff hardware abstractions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optixstuff as ox
from optixstuff.spectral import Transmission, spectral_breaks


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


class TestInstrument:
    def test_transmission_elements_compose(self, simple_primary, simple_detector):
        """Fails if held responses lose knots, attenuation or parameter gradients.

        Exact algebra: at 550 nm the linear response is (0.2+0.8)/2;
        the second element transmits 0.4. Allow 2e-15 floating-point error.
        """
        with jax.enable_x64():
            element = ox.TransmissiveElement(Transmission([500, 600], [0.2, 0.8]))
            path = ox.Instrument(
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

    def test_flux_scale_multiplies_the_effective_area(
        self, simple_primary, simple_detector
    ):
        """Fails if the scale misses the effective area or defaults to anything but one.

        Basis: exact products of the area, the 0.4 throughput and the scale,
        compared at 0 ULP (a product by exactly one changes no bit); invalid
        scales must raise.
        """
        with jax.enable_x64():
            kwargs = dict(
                primary=simple_primary,
                attenuating_elements=(ox.ConstantThroughput(0.4),),
                coronagraph=None,
                detector=simple_detector,
            )
            plain = ox.Instrument(**kwargs)
            assert float(plain.flux_scale) == 1.0
            area = simple_primary.collecting_area_m2
            np.testing.assert_array_equal(
                plain.effective_area_m2(550.0), area * plain.system_throughput(550.0)
            )
            scaled = ox.Instrument(**kwargs, flux_scale=1.25)
            np.testing.assert_allclose(
                scaled.effective_area_m2(jnp.array([500.0, 600.0])),
                1.25 * area * 0.4 * jnp.ones(2),
                rtol=1e-6,
            )
            for bad in (0.0, -1.0, np.nan, np.inf):
                with pytest.raises(ValueError, match="flux_scale"):
                    ox.Instrument(**kwargs, flux_scale=bad)
            grad = eqx.filter_grad(lambda i: i.effective_area_m2(550.0))(plain)
            np.testing.assert_allclose(grad.flux_scale, area * 0.4, rtol=1e-6)

    def test_warm_emission_is_planck_times_emissivity_times_area(
        self, simple_primary, simple_detector
    ):
        """Fails if the thermal rate drops a factor or is on by default.

        Basis: the closed-form photon radiance 2 c / lambda^4 / expm1(h c /
        (lambda k T)) written from the constants, times emissivity and the
        geometric area, in float64; rtol 1e-12 for the exp and one product
        chain. The default and a zero emissivity give exact zeros.
        """
        from hwoutils.constants import c, h, k_B

        with jax.enable_x64():
            kwargs = dict(
                primary=simple_primary,
                attenuating_elements=(ox.ConstantThroughput(0.4),),
                coronagraph=None,
                detector=simple_detector,
            )
            wavelength = np.array([500.0, 1000.0, 2000.0])
            assert not np.any(
                ox.Instrument(**kwargs).thermal_photon_rate_per_sr(wavelength)
            )
            warm = ox.WarmEmission(temperature_K=290.0, effective_emissivity=0.638)
            instrument = ox.Instrument(**kwargs, warm_emission=warm, flux_scale=1.7)
            lam_m = wavelength * 1e-9
            radiance = 2 * c / lam_m**4 / np.expm1(h * c / (lam_m * k_B * 290.0)) * 1e-9
            np.testing.assert_allclose(
                instrument.thermal_photon_rate_per_sr(wavelength),
                radiance * 0.638 * simple_primary.collecting_area_m2,
                rtol=1e-12,
            )
            off = ox.Instrument(
                **kwargs, warm_emission=ox.WarmEmission(290.0, 0.0)
            ).thermal_photon_rate_per_sr(wavelength)
            assert not np.any(off)
            for bad in ((0.0, 0.5), (np.nan, 0.5), (290.0, 1.2), (290.0, -0.1)):
                with pytest.raises(ValueError):
                    ox.WarmEmission(*bad)

    def test_system_throughput_single_element(self, simple_primary, simple_detector):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.Instrument(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.7)

    def test_system_throughput_two_elements(self, simple_primary, simple_detector):
        el1 = ox.ConstantThroughput(throughput=0.8)
        el2 = ox.ConstantThroughput(throughput=0.9)
        path = ox.Instrument(
            primary=simple_primary,
            attenuating_elements=(el1, el2),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.72)


def test_exposure_is_a_record_of_one_observation():
    """Fails if Exposure loses a field, its ecliptic defaults, or its traced roll.

    The record is a pytree whose roll is an array leaf that vmap can map, and
    the plan may be absent.
    """
    from optixstuff.spectral import Band, Quadrature

    quadrature = Quadrature.build((Band(500.0, 600.0),), order=2)
    exposure = ox.Exposure(time_d=3.0, roll_deg=15, quadrature=quadrature)
    assert exposure.plan is None
    assert exposure.roll_deg == 15.0
    assert not hasattr(exposure, "ecliptic_latitude_deg")
    assert not hasattr(exposure, "solar_longitude_deg")
    assert exposure.roll_deg in jax.tree.leaves(exposure)
    rolled = jax.vmap(
        lambda r: eqx.tree_at(lambda e: e.roll_deg, exposure, r).roll_deg * 2.0
    )(jnp.array([1.0, 2.0]))
    assert rolled.tolist() == [2.0, 4.0]
    planned = eqx.tree_at(
        lambda e: e.plan,
        exposure,
        ox.frames_for_time(60.0, 10.0),
        is_leaf=lambda x: x is None,
    )
    assert float(planned.plan.n_frames) == 6.0


def test_documented_host_surface_is_exported():
    """The performance-record builder and the grid helper are public."""
    from optixstuff.coronagraph import on_grid
    from optixstuff.performance import performance_curves

    assert "performance_curves" in ox.__all__
    assert ox.performance_curves is performance_curves
    assert callable(on_grid)
