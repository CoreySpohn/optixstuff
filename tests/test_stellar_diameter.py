"""Stellar diameter forwarding, point-source regressions, and JAX tracing."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_coronagraph_contract import _TableCoronagraph
from test_speckle import _MockCoro
from yippy.datasets import CATALOG

from optixstuff import YippyCoronagraph
from optixstuff.coronagraph import (
    AbstractCoronagraph,
    AbstractScalarCoronagraph,
    AbstractTableCoronagraph,
    MultiBandCoronagraph,
)


class _DiameterBackend:
    """Small backend with diameter dependence and a non-binary pixel scale."""

    pixel_scale_lod = 0.3
    IWA = 2.0
    OWA = 20.0

    def core_mean_intensity(self, separation_lod, stellar_diam_lod=0.0):
        return jnp.asarray(1e-9) / separation_lod + 1e-7 * stellar_diam_lod**2

    def noise_floor_ayo(self, separation_lod, ppf=30.0):
        return self.core_mean_intensity(separation_lod) / ppf


@pytest.fixture
def diameter_coro():
    return YippyCoronagraph(backend=_DiameterBackend())


@pytest.mark.parametrize(
    "cls", [AbstractCoronagraph, AbstractScalarCoronagraph, AbstractTableCoronagraph]
)
def test_abstract_diameter_keyword(cls):
    parameter = inspect.signature(cls.core_mean_intensity).parameters[
        "stellar_diam_lod"
    ]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 0.0


@pytest.mark.parametrize("factory", [_MockCoro, _TableCoronagraph])
def test_existing_models_ignore_diameter(factory):
    coro = factory()
    sep = jnp.asarray([3.0, 5.0, 10.0])
    baseline = coro.core_mean_intensity(sep, 550.0)
    for diameter in (0.0, 0.1):
        np.testing.assert_array_equal(
            coro.core_mean_intensity(sep, 550.0, stellar_diam_lod=diameter), baseline
        )


@pytest.mark.parametrize("x64", [False, True])
def test_yippy_point_source_regression(diameter_coro, x64):
    with jax.enable_x64(x64):
        coro = diameter_coro
        sep = jnp.asarray([3.0, 5.0, 10.0])
        # Pin the expressions used before the diameter keyword was added.
        legacy_intensity = (
            coro._backend.core_mean_intensity(sep) / coro.pixel_scale_lod**2
        )
        legacy_floor = (
            coro._backend.noise_floor_ayo(sep, 30.0) / coro.pixel_scale_lod**2
        )
        np.testing.assert_array_equal(
            coro.core_mean_intensity(sep, 550.0), legacy_intensity
        )
        np.testing.assert_array_equal(
            coro.core_mean_intensity(sep, 550.0, stellar_diam_lod=0.0), legacy_intensity
        )
        np.testing.assert_array_equal(coro.noise_floor_ayo(sep), legacy_floor)
        np.testing.assert_array_equal(
            coro.noise_floor_ayo(sep, stellar_diam_lod=0.0), legacy_floor
        )


@pytest.mark.parametrize("method", ["core_mean_intensity", "noise_floor_ayo"])
def test_yippy_diameter_forwarding(diameter_coro, method):
    coro = diameter_coro
    sep, diameter, ppf = 3.0, 0.1, 17.0
    per_pixel = coro._backend.core_mean_intensity(sep, diameter)
    if method == "core_mean_intensity":
        actual = coro.core_mean_intensity(sep, 550.0, stellar_diam_lod=diameter)
        expected = per_pixel / coro.pixel_scale_lod**2
    else:
        actual = coro.noise_floor_ayo(sep, ppf, stellar_diam_lod=diameter)
        expected = (per_pixel / ppf) / coro.pixel_scale_lod**2
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("wavelength_nm", [500.0, 850.0])
def test_multiband_forwards_diameter_and_time(wavelength_nm):
    class _DiameterScalar(_MockCoro):
        def core_mean_intensity(self, sep, wl, *, time_s=0.0, stellar_diam_lod=0.0):
            return self.pixel_scale_lod * (1.0 + stellar_diam_lod) + time_s

    blue = _DiameterScalar(pixel_scale_lod=0.25)
    red = _DiameterScalar(pixel_scale_lod=0.5)
    coro = MultiBandCoronagraph([450.0, 900.0], [blue, red])
    expected_band = blue if wavelength_nm == 500.0 else red
    np.testing.assert_array_equal(
        coro.core_mean_intensity(3.0, wavelength_nm, time_s=2.0, stellar_diam_lod=0.1),
        expected_band.core_mean_intensity(
            3.0, wavelength_nm, time_s=2.0, stellar_diam_lod=0.1
        ),
    )
    np.testing.assert_array_equal(
        coro.core_mean_intensity(3.0, wavelength_nm),
        coro.core_mean_intensity(3.0, wavelength_nm, stellar_diam_lod=0.0),
    )


def _assert_diameter_traces(coro, diameters, method):
    def evaluate(diameter):
        if method == "core_mean_intensity":
            return coro.core_mean_intensity(3.0, 550.0, stellar_diam_lod=diameter)
        return coro.noise_floor_ayo(3.0, 17.0, stellar_diam_lod=diameter)

    expected = jnp.stack([evaluate(diameter) for diameter in diameters])
    compiled = jax.jit(evaluate)
    np.testing.assert_allclose(
        jnp.stack([compiled(diameter) for diameter in diameters]), expected, rtol=1e-6
    )
    np.testing.assert_allclose(jax.vmap(evaluate)(diameters), expected, rtol=1e-6)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(diameters), expected, rtol=1e-6
    )


@pytest.mark.parametrize("method", ["core_mean_intensity", "noise_floor_ayo"])
def test_yippy_diameter_jit_and_vmap(diameter_coro, method):
    _assert_diameter_traces(diameter_coro, jnp.asarray([0.0, 0.01, 0.1]), method)


# The cached-YIP checks run nightly; the quick tier uses synthetic backends.
@pytest.mark.slow
def test_cached_yip_point_source_regression(cached_diameter_coro):
    coro = cached_diameter_coro
    sep = jnp.asarray([3.0, 5.0, 10.0])
    expected = coro._backend.core_mean_intensity(sep) / coro.pixel_scale_lod**2
    np.testing.assert_array_equal(coro.core_mean_intensity(sep, 550.0), expected)
    np.testing.assert_array_equal(
        coro.core_mean_intensity(sep, 550.0, stellar_diam_lod=0.0), expected
    )
    expected_floor = coro._backend.noise_floor_ayo(sep) / coro.pixel_scale_lod**2
    np.testing.assert_array_equal(coro.noise_floor_ayo(sep), expected_floor)
    np.testing.assert_array_equal(
        coro.noise_floor_ayo(sep, stellar_diam_lod=0.0), expected_floor
    )


@pytest.mark.parametrize(
    "separation_lod, fraction", [(3.0, 0.25), (5.0, 0.5), (7.0, 0.75)]
)
@pytest.mark.slow
def test_cached_yip_matches_backend(cached_diameter_coro, separation_lod, fraction):
    coro = cached_diameter_coro
    low, high = coro._backend._diam_knot_range
    diameter = low + fraction * (high - low)
    assert 0.0 < diameter < high
    expected = (
        coro._backend.core_mean_intensity(separation_lod, diameter)
        / coro.pixel_scale_lod**2
    )
    np.testing.assert_array_equal(
        coro.core_mean_intensity(separation_lod, 550.0, stellar_diam_lod=diameter),
        expected,
    )


@pytest.mark.slow
def test_cached_yip_extended_star_increases_leakage(cached_diameter_coro):
    coro = cached_diameter_coro
    low, high = coro._backend._diam_knot_range
    diameter = low + 0.75 * (high - low)
    assert coro.core_mean_intensity(3.0, 550.0, stellar_diam_lod=diameter) > (
        coro.core_mean_intensity(3.0, 550.0)
    )


@pytest.mark.parametrize("method", ["core_mean_intensity", "noise_floor_ayo"])
@pytest.mark.slow
def test_cached_yip_diameter_jit_and_vmap(cached_diameter_coro, method):
    low, high = cached_diameter_coro._backend._diam_knot_range
    diameters = jnp.asarray([0.0, low + 0.25 * (high - low), low + 0.75 * (high - low)])
    _assert_diameter_traces(cached_diameter_coro, diameters, method)


@pytest.mark.slow
def test_cached_yip_matches_raw_table_at_knots(cached_diameter_coro, tmp_path_factory):
    from yippy import Coronagraph

    coro = cached_diameter_coro
    (scratch,) = tmp_path_factory.getbasetemp().glob("stellar_diameter_yip[0-9]*")
    # The fixture stops at the first multi-diameter package, so it is the last
    # package copied in catalog order. Reuse that copy without fetching data.
    yip_dirs = [scratch / name for name in CATALOG if (scratch / name).is_dir()]
    reference = Coronagraph(yip_dirs[-1])
    diam_knots = np.asarray(reference.stellar_intens.diams.value)
    nonzero = np.flatnonzero(diam_knots > 0.0)
    assert len(nonzero) >= 3
    points = zip((3.0, 5.0, 10.0), nonzero[[0, len(nonzero) // 2, -1]], strict=True)
    for sep, j in points:
        diameter = float(diam_knots[j])
        actual = float(coro.core_mean_intensity(sep, 550.0, stellar_diam_lod=diameter))
        expected = reference.core_mean_intensity(sep, stellar_diam=diameter) / (
            coro.pixel_scale_lod**2
        )
        # Conftest leaves x64 disabled: JAX float32 and the NumPy float64
        # spline differ by up to 7.51e-7 here. Keep rtol=1e-6 and no atol
        # so rounding is allowed without hiding errors at tiny intensities.
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=0.0,
            err_msg=f"separation={sep}, stellar_diam_lod={diameter}",
        )
        assert np.isfinite(actual) and actual > 0.0
        if 0 < j < len(diam_knots) - 1:
            neighbours = [
                reference.core_mean_intensity(sep, stellar_diam=float(d))
                / coro.pixel_scale_lod**2
                for d in diam_knots[[j - 1, j + 1]]
            ]
            assert min(neighbours) <= actual <= max(neighbours)
