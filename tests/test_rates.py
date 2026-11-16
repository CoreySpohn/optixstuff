"""Code verification of spectral photon records and their dimensional weights."""

from dataclasses import fields

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.spectral import Quadrature

import optixstuff as ox


@pytest.mark.parametrize("spatial", [(), (2, 3)])
def test_weighted_integration(spatial):
    """Fails if nm weights, the Q axis, or scalar absent terms are lost.

    Floating-point tolerance: 8 float64 eps for three multiply-adds.
    JIT/eager uses that same reduction-roundoff bound.
    """
    import jax

    with jax.enable_x64():
        quad = Quadrature(
            jnp.array([[400, 450, 500], [600, 700, 800]], dtype=jnp.float64),
            jnp.array([[2, 3, 5], [7, 11, 13]], dtype=jnp.float64),
        )
        values = jnp.arange(1, 1 + 6 * int(np.prod(spatial)), dtype=jnp.float64)
        values = values.reshape((2, 3, *spatial))
        rates = ox.PhotonRates(planet=values, thermal=2 * values)
        expected = np.zeros((2, *spatial))
        for w in range(2):
            for q in range(3):
                expected[w] += float(quad.weights_nm[w, q]) * np.asarray(values[w, q])
        result = rates.integrate(quad)
        compiled = eqx.filter_jit(lambda r, q: r.integrate(q))(rates, quad)
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
        assert not isinstance(result, ox.ElectronRates)


def test_zeros():
    """Fails if all-zero records lose bands or spatial pixels; exact algebra."""
    for shape in [(2, 3), (2, 3, 2, 3)]:
        rates = ox.PhotonRates.zeros(shape, dtype=jnp.float32)
        for field in fields(rates):
            np.testing.assert_array_equal(getattr(rates, field.name), np.zeros(shape))
        quad = Quadrature(jnp.ones((2, 3)), jnp.ones((2, 3)))
        assert rates.integrate(quad).total().shape == (2, *shape[2:])
    empty = ox.PhotonRates()
    assert empty.integrate(quad).total().shape == (2,)
