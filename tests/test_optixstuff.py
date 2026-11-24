"""Tests for optixstuff hardware abstractions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.jax_config import float_dtype
from hwoutils.leaves import check_tree

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
    for x64 in (False, True):
        with jax.enable_x64(x64):
            rule = Quadrature.build((Band(500.0, 600.0),), order=2)
            built = ox.Exposure(time_d=3.0, roll_deg=15.0, quadrature=rule)
            assert check_tree(built) == ()
            assert built.roll_deg.dtype == float_dtype()
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


def _arcsec_per_lod(wavelength_nm, diameter_m):
    return wavelength_nm * 1e-9 / diameter_m * 180.0 / np.pi * 3600.0


def test_instrument_declares_the_detector_half_diagonal_as_its_field():
    """Fails if the declared field is not the detector half-diagonal in lambda/D.

    Truth: arithmetic from the detector geometry, written out here. A 96 by 128
    detector with 0.2 lambda/D pixels at 550 nm on a 6 m aperture has a
    half-diagonal of 0.2 * 80 = 16 lambda/D at 550 nm, and the same angle is
    16 * 550 / 450 lambda/D at the center of a 450 nm band. The result holds
    with a coronagraph (its frame) and without one (the primary). Tolerance 1e-12
    relative, basis floating-point arithmetic on a few products. The half-width
    (12.8 lambda/D at 550 nm) is 20 percent smaller, far outside it.
    """
    from conftest import table_coronagraph

    from optixstuff.spectral import Band

    diameter = 6.0
    scale = 0.2 * _arcsec_per_lod(550.0, diameter)
    detector = ox.IdealDetector(pixel_scale_arcsec=scale, shape=(96, 128))
    primary = ox.SimplePrimary(
        circumscribed_diameter_m=diameter, collecting_area_m2=28.0
    )
    elements = (ox.ConstantThroughput(throughput=1.0),)
    bare = ox.Instrument(primary, elements, None, detector)
    framed = ox.Instrument(primary, elements, table_coronagraph(diameter), detector)
    for instrument in (bare, framed):
        for center, expected in ((550.0, 16.0), (450.0, 16.0 * 550.0 / 450.0)):
            band = Band(0.9 * center, 1.1 * center)
            np.testing.assert_allclose(
                instrument.spectral_field_lod(band), expected, rtol=1e-12
            )


def test_quadrature_build_takes_the_instrument_field_when_asked():
    """Fails if the declared field misses the node count or leaks into the default.

    Truth: the same build with the declared value passed as a number, which is
    an identical layout (exact). ``field_lod=None`` reads the instrument and the
    default of zero does not, so the default layout keeps ``order`` nodes per
    piece (exact), and the declared field adds nodes to a band it slides across.
    """
    from optixstuff.spectral import Band, Quadrature

    detector = ox.IdealDetector(
        pixel_scale_arcsec=0.2 * _arcsec_per_lod(550.0, 6.0), shape=(96, 128)
    )
    primary = ox.SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=28.0)
    instrument = ox.Instrument(
        primary, (ox.ConstantThroughput(throughput=1.0),), None, detector
    )
    band = Band(495.0, 605.0)
    declared = Quadrature.build((band,), instrument, field_lod=None)
    explicit = Quadrature.build(
        (band,), instrument, field_lod=instrument.spectral_field_lod(band)
    )
    assert declared.pieces == explicit.pieces
    assert declared.nodes_nm.shape[1] > 4
    assert Quadrature.build((band,), instrument).nodes_nm.shape == (1, 4)


@pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
def test_exposure_time_is_a_strong_leaf_and_declares_no_reads(x64):
    """Exact dtype basis. Fails if a Python-float time stays a Python number,
    is weakly typed or has another dtype than the active one, or if the
    removed ``reads`` field is accepted again."""
    from optixstuff.spectral import Band, Quadrature

    with jax.enable_x64(x64):
        quadrature = Quadrature.build((Band(500.0, 600.0),), order=2)
        exposure = ox.Exposure(time_d=1.0, roll_deg=0.0, quadrature=quadrature)
        assert isinstance(exposure.time_d, jax.Array)
        assert exposure.time_d.dtype == (jnp.float64 if x64 else jnp.float32)
        assert not exposure.time_d.weak_type
        assert exposure.time_d in jax.tree.leaves(exposure)
        with pytest.raises(TypeError):
            ox.Exposure(0.0, 0.0, quadrature, reads=("star",))
