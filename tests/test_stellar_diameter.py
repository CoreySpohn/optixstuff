"""Stellar diameter forwarding, point-source regressions, and JAX tracing."""

import inspect
import shutil
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_coronagraph_contract import _TableCoronagraph
from test_speckle import _MockCoro
from yippy import EqxCoronagraph
from yippy.datasets import CATALOG, cache_dir

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


@pytest.fixture(scope="module")
def cached_diameter_coro(tmp_path_factory):
    """Use only cached public packages; keep generated caches in a temp copy."""
    root = cache_dir()
    scratch = tmp_path_factory.mktemp("stellar_diameter_yip")
    for name in CATALOG:
        candidates = (root / f"{name}.zip.unzip" / name, root / name)
        source = next((path for path in candidates if path.is_dir()), None)
        target = scratch / name
        if source is not None:
            shutil.copytree(source, target)
        elif (root / f"{name}.zip").is_file():
            with ZipFile(root / f"{name}.zip") as archive:
                archive.extractall(scratch)
        else:
            continue
        backend = EqxCoronagraph(target)
        if backend._has_2d_core_intensity:
            return YippyCoronagraph(backend=backend)
    pytest.skip(f"No cached public YIP with a stellar-diameter axis in {root}")


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


def test_cached_yip_extended_star_increases_leakage(cached_diameter_coro):
    coro = cached_diameter_coro
    low, high = coro._backend._diam_knot_range
    diameter = low + 0.75 * (high - low)
    assert coro.core_mean_intensity(3.0, 550.0, stellar_diam_lod=diameter) > (
        coro.core_mean_intensity(3.0, 550.0)
    )


@pytest.mark.parametrize("method", ["core_mean_intensity", "noise_floor_ayo"])
def test_cached_yip_diameter_jit_and_vmap(cached_diameter_coro, method):
    low, high = cached_diameter_coro._backend._diam_knot_range
    diameters = jnp.asarray([0.0, low + 0.25 * (high - low), low + 0.75 * (high - low)])
    _assert_diameter_traces(cached_diameter_coro, diameters, method)


def test_cached_yip_matches_raw_table_at_knots(cached_diameter_coro):
    coro = cached_diameter_coro
    table = coro._backend._core_mean_intensity_interp_2d
    sep_knots, diam_knots, values = table.x, table.y, table.f
    n_sep, n_diam = values.shape
    points = [(n_sep // 4, 1), (n_sep // 2, n_diam // 2), (3 * n_sep // 4, n_diam - 1)]
    for i, j in points:
        assert diam_knots[j] > 0.0
        np.testing.assert_allclose(
            coro.core_mean_intensity(
                sep_knots[i], 550.0, stellar_diam_lod=diam_knots[j]
            ),
            values[i, j] / coro.pixel_scale_lod**2,
            rtol=1e-6,
        )
