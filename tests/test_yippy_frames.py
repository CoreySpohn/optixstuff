"""Independent stellar/sky frames are respected by the shared-grid adapter."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image, PixelGrid, resample

from optixstuff import YippyCoronagraph


def test_separate_stellar_and_sky_frames(cached_diameter_coro):
    """Exact algebra: adapter and explicit resampling use each product grid."""
    backend = cached_diameter_coro._backend
    sg = PixelGrid.regular(backend.grid.shape, 0.4, "lod", frame=backend.grid.frame)
    kg = PixelGrid.regular(backend.grid.shape, 0.3, "lod", frame=backend.grid.frame)
    sky = replace(backend.sky_trans, grid=kg)
    backend = eqx.tree_at(
        lambda b: (b.stellar_grid, b.sky_grid, b.sky_trans), backend, (sg, kg, sky)
    )
    coro = YippyCoronagraph(backend=backend)
    target = PixelGrid.regular(
        (20, 22), 0.35, "lod", frame=backend.grid.frame, center=(10.2, 9.7)
    )
    stellar = backend.stellar_intens(0.1)
    expected = resample(stellar, target)
    np.testing.assert_array_equal(
        coro.stellar_map(500.0, 0.1, grid=target).data, expected.data
    )
    wrong = resample(replace(stellar, grid=backend.grid), target)
    assert not np.allclose(expected.data, wrong.data, atol=0.0, rtol=1e-3)
    np.testing.assert_array_equal(
        coro.background_transmission(500.0, grid=target).data,
        resample(sky, target).data,
    )
    # A different physical diameter and reference wavelength define another lod.
    other_frame = replace(
        backend.grid.frame, diameter_m=4.0, reference_wavelength_nm=600.0
    )
    other_grid = replace(target, frame=other_frame)
    np.testing.assert_array_equal(
        coro.stellar_map(500.0, 0.1, grid=other_grid).data,
        resample(
            replace(stellar, grid=sg.to_arcsec(500.0)), other_grid.to_arcsec(600.0)
        ).data,
    )
    # Frame conversion is explicit angular geometry; no inferred area diameter.
    angular = target.to_arcsec(500.0)
    np.testing.assert_allclose(
        coro.on_axis_psf(500.0, angular).data,
        resample(
            replace(backend.stellar_intens(0.0), grid=sg.to_arcsec(500.0)), angular
        ).data,
        rtol=8 * np.finfo(np.float32).eps,
    )


def test_shared_frame_bit_equality(cached_diameter_coro):
    """On the native grid, products are box integrals with no centre change.

    The expected value resamples the native Image onto its own grid; a frame,
    centre or reconstruction change on the consumer side breaks bit equality.
    """
    c = cached_diameter_coro
    g = c.grid
    for actual, expected in [
        (c.stellar_map(500.0, 0.1, grid=g), c._backend.stellar_intens(0.1)),
        (c.background_transmission(500.0, grid=g), c.sky_trans),
        (
            c.source_psfs(500.0, jnp.array([1.0]), jnp.array([0.0]), grid=g),
            c.source_psfs(500.0, jnp.array([1.0]), jnp.array([0.0]), grid=None),
        ),
    ]:
        assert isinstance(actual, Image)
        np.testing.assert_array_equal(actual.data, resample(expected, g).data)
