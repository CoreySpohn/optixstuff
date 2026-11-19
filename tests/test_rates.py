"""Code verification of spectral photon records and their dimensional weights."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import optixstuff as ox
from optixstuff.spectral import Band, Quadrature, Transmission


@pytest.mark.parametrize("spatial", [(), (2, 3)])
def test_weighted_integration(spatial):
    """Fails if integration misses a kink, reduces a spatial axis, or drops nm weights.

    Floating-point tolerance: 8 float64 eps for three multiply-adds.
    JIT/eager uses that same reduction-roundoff bound.
    """
    import jax

    with jax.enable_x64():
        # Asymmetric triangular filter inside the first band. The second
        # band lies entirely inside its falling branch and needs no split.
        element = ox.TransmissiveElement(Transmission([420, 453, 610], [0, 1, 0]))
        quad = Quadrature.build((Band(400, 610), Band(453, 610)), element, order=1)
        scale = np.arange(1, 1 + int(np.prod(spatial))).reshape(spatial)
        values = (
            element.get_throughput(quad.nodes_nm).reshape(
                (*quad.nodes_nm.shape, *(1 for _ in spatial))
            )
            * scale
        )
        rates = ox.Rates(quad, "photon", planet=values, thermal=2 * values)
        # Triangle area is base * height / 2. The second row is padded.
        expected = np.array([95.0, 78.5]).reshape((2, *(1 for _ in spatial))) * scale
        result = rates.integrate()
        assert result.bands == quad.bands
        compiled = eqx.filter_jit(lambda r: r.integrate())(rates)
        for name, factor in [("planet", 1), ("thermal", 2), ("star_leakage", 0)]:
            np.testing.assert_allclose(
                getattr(result, name),
                factor * expected,
                rtol=8 * np.finfo(float).eps,
                atol=0,
            )
            np.testing.assert_array_equal(
                getattr(compiled, name), getattr(result, name)
            )
        np.testing.assert_array_equal(rates.total(), 3 * values)
        np.testing.assert_allclose(
            result.total(), 3 * expected, rtol=8 * np.finfo(float).eps, atol=0
        )
        assert (rates.unit, result.unit) == ("photon/s/nm", "photon/s")


def test_zeros():
    """Fails if all-zero records lose bands or spatial pixels; exact algebra."""
    quad = Quadrature.build((Band(400, 500), Band(600, 700)), order=3)
    for spatial in [(), (2, 3)]:
        rates = ox.Rates.zeros(quad, "photon", spatial, dtype=jnp.float32)
        for name in rates.component_names():
            np.testing.assert_array_equal(
                getattr(rates, name), np.zeros((2, 3, *spatial))
            )
        assert rates.integrate().total().shape == (2, *spatial)
    with pytest.raises(ValueError, match="quadrature"):
        ox.Rates(quad, "photon", planet=jnp.ones((2, 1)))
    empty = ox.Rates(quad, "photon")
    assert empty.integrate().total().shape == (2,)


def test_aperture_fields_follow_integration():
    """Fails if integrating drops an aperture's pixel count or variance rates."""
    quad = Quadrature.build((Band(400, 500), Band(600, 700)), order=3)
    ones = jnp.ones((2, 3))
    rates = ox.Rates(
        quad,
        "photon",
        planet=ones,
        n_pix=jnp.array([4.0, 5.0]),
        variance=ox.Rates(quad, "photon", planet=ones / 2, n_pix=2.0),
    )
    result = rates.integrate()
    assert not result.spectral and result.carrier == "photon"
    np.testing.assert_array_equal(result.n_pix, [4.0, 5.0])
    np.testing.assert_array_equal(result.variance.n_pix, 2.0)
    np.testing.assert_array_equal(result.variance.planet, result.planet / 2)
    with pytest.raises(ValueError, match="already integrated"):
        result.integrate()
    with pytest.raises(ValueError, match="same unit"):
        ox.Rates(quad, "photon", variance=ox.Rates(quad, "electron"))
    with pytest.raises(ValueError, match="carrier"):
        ox.Rates(quad, "photons")
