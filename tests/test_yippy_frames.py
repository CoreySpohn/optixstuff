"""Separate table frames and shared-frame compatibility in the yippy wrapper."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.transforms import resample_flux

from optixstuff import YippyCoronagraph
from optixstuff.coronagraph import AbstractTableCoronagraph


@pytest.mark.slow
def test_separate_stellar_and_sky_frames(cached_diameter_coro):
    """Off-axis pitch cannot determine stellar area or stellar/sky resampling."""
    pytest.importorskip("sinchole")
    native = cached_diameter_coro._backend
    if not hasattr(native, "stellar_pixel_scale_lod"):
        pytest.skip("Requires frame-aware yippy")
    backend = eqx.tree_at(
        lambda b: (b.pixel_scale_lod, b.stellar_pixel_scale_lod, b.sky_pixel_scale_lod),
        native,
        (0.25, 0.4, 0.3),
    )
    coro = YippyCoronagraph(backend=backend)
    sep = 3.0
    per_pixel = backend.core_mean_intensity(sep, 0.0)
    np.testing.assert_array_equal(
        coro.core_mean_intensity(sep, 500.0), per_pixel / 0.4**2
    )
    np.testing.assert_array_equal(
        coro.noise_floor_ayo(sep), (per_pixel / 30.0) / 0.4**2
    )
    shape = backend.psf_shape
    stellar = backend.stellar_intens(0.0)
    assert np.ptp(stellar) > 0
    expected = resample_flux(stellar, 0.4, 0.35, shape, 0.0)
    np.testing.assert_array_equal(
        coro.stellar_map(500.0, 0.0, pixel_scale_lod=0.35, shape=shape), expected
    )
    wrong = resample_flux(stellar, 0.25, 0.35, shape, 0.0)
    assert not np.allclose(expected, wrong, atol=0.0, rtol=1e-3)
    expected_sky = (
        resample_flux(backend.sky_trans, 0.3, 0.35, shape, 0.0) * (0.3 / 0.35) ** 2
    )
    np.testing.assert_array_equal(
        coro.background_transmission(500.0, pixel_scale_lod=0.35, shape=shape),
        expected_sky,
    )
    np.testing.assert_array_equal(coro.on_axis_psf(500.0, 1e-9, 4), stellar)
    x, y = jnp.array([3.0]), jnp.array([0.0])
    np.testing.assert_array_equal(
        coro.source_psfs(500.0, x, y, pixel_scale_lod=0.25, shape=shape),
        backend.source_psfs(500.0, x, y, pixel_scale_lod=0.25, shape=shape),
    )


@pytest.mark.slow
def test_shared_frame_bit_equality(cached_diameter_coro):
    """Delegating a shared legacy frame preserves every original output bit."""
    coro = cached_diameter_coro
    backend = coro._backend
    pitch = backend.pixel_scale_lod
    shape = backend.psf_shape
    np.testing.assert_array_equal(
        coro.core_mean_intensity(3.0, 500.0),
        backend.core_mean_intensity(3.0, 0.0) / pitch**2,
    )
    np.testing.assert_array_equal(
        coro.noise_floor_ayo(3.0),
        (backend.core_mean_intensity(3.0, 0.0) / 30.0) / pitch**2,
    )
    for method, args in (
        ("stellar_map", (500.0, 0.0)),
        ("background_transmission", (500.0,)),
        ("source_psfs", (500.0, jnp.array([3.0]), jnp.array([0.0]))),
    ):
        expected = getattr(AbstractTableCoronagraph, method)(
            coro, *args, pixel_scale_lod=pitch, shape=shape
        )
        actual = getattr(coro, method)(*args, pixel_scale_lod=pitch, shape=shape)
        np.testing.assert_array_equal(actual, expected)
