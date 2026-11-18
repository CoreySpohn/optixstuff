"""Code verification of spectral photon records and their dimensional weights."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.spectral import Band, Quadrature, Transmission

import optixstuff as ox


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
        rates = ox.PhotonRates(quad, planet=values, thermal=2 * values)
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
        assert not isinstance(result, ox.ElectronRates)


def test_zeros():
    """Fails if all-zero records lose bands or spatial pixels; exact algebra."""
    quad = Quadrature.build((Band(400, 500), Band(600, 700)), order=3)
    for spatial in [(), (2, 3)]:
        rates = ox.PhotonRates.zeros(quad, spatial, dtype=jnp.float32)
        for name in rates.component_names():
            np.testing.assert_array_equal(
                getattr(rates, name), np.zeros((2, 3, *spatial))
            )
        assert rates.integrate().total().shape == (2, *spatial)
    with pytest.raises(ValueError, match="quadrature"):
        ox.PhotonRates(quad, planet=jnp.ones((2, 1)))
    empty = ox.PhotonRates(quad)
    assert empty.integrate().total().shape == (2,)
