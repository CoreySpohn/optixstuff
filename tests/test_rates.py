"""Code verification of spectral photon records and their dimensional weights."""

import equinox as eqx
import jax
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
    flag = jnp.array([[True, False, True], [False, True, True]])
    rates = ox.Rates(
        quad,
        "photon",
        planet=ones,
        n_pix=jnp.array([4.0, 5.0]),
        variance=ox.Rates(quad, "photon", planet=ones / 2, n_pix=2.0),
        supported=flag,
    )
    assert ox.Rates(quad, "photon").supported is None
    result = rates.integrate()
    np.testing.assert_array_equal(result.supported, flag)
    assert result.supported.dtype == bool
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
    with pytest.raises(ValueError, match="supported"):
        ox.Rates(quad, "photon", supported=jnp.ones((3, 3), bool))


def test_static_support_is_implied_by_support_and_skips_tracers():
    """Fails if a record admits a strict flag wider than its static flag.

    ``supported`` True where ``static_supported`` is False raises, and so does
    ``supported=None`` beside a ``static_supported`` with a False entry; a
    record whose flags satisfy the implication builds, and a constructor
    mapped with ``vmap`` over both flags does not raise (tracers are not
    checked). Truth: the stated implication; exact.
    """
    bands = (Band(400, 500), Band(600, 700))
    planet = jnp.ones((2, 2, 3))
    strict = np.zeros((2, 2, 3), bool)
    static = np.ones((2, 2, 3), bool)
    strict[0, 1, 2] = True
    ok = ox.Rates(
        bands, "photon", planet=planet, supported=strict, static_supported=static
    )
    assert ok.static_supported.dtype == bool
    static[0, 1, 2] = False
    with pytest.raises(ValueError, match="static_supported is False"):
        ox.Rates(
            bands, "photon", planet=planet, supported=strict, static_supported=static
        )
    with pytest.raises(ValueError, match="supported=None"):
        ox.Rates(bands, "photon", planet=planet, static_supported=static)
    with pytest.raises(ValueError, match="static_supported must have one leading"):
        ox.Rates(
            bands, "photon", planet=planet, static_supported=np.ones((3, 2, 3), bool)
        )

    def build(a, b):
        return ox.Rates(bands, "photon", planet=planet, supported=a, static_supported=b)

    mapped = jax.vmap(build)(jnp.stack([strict, ~static]), jnp.stack([static, static]))
    assert mapped.static_supported.shape == (2, 2, 2, 3)
