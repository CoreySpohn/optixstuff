"""Finite-diameter Images retain the backend's diameter dependence and JAX support."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import resample

from optixstuff import MultiBandCoronagraph


def test_yippy_diameter_forwarding(cached_diameter_coro):
    """Exact forwarding of native stellar Images, including finite diameters."""
    c = cached_diameter_coro
    for d in (0.0, 0.1, 0.2):
        np.testing.assert_array_equal(
            c.stellar_map(500.0, d, grid=c.grid).data,
            resample(c._backend.stellar_intens(d), c.grid).data,
        )
    assert float(c.stellar_map(500.0, 0.2, grid=c.grid).data.sum()) > float(
        c.stellar_map(500.0, 0.0, grid=c.grid).data.sum()
    )


def test_multiband_forwards_diameter_and_time(cached_diameter_coro):
    """Exact record and image dispatch; monochromatic records ignore time/wavelength."""
    c = cached_diameter_coro
    mb = MultiBandCoronagraph([450.0, 900.0], [c, c])
    for wl in (500.0, 850.0):
        np.testing.assert_array_equal(
            mb.stellar_map(wl, 0.1, grid=c.grid).data,
            c.stellar_map(wl, 0.1, grid=c.grid).data,
        )
        np.testing.assert_array_equal(
            mb.throughput(2.0, wl, time_d=3.0), c.throughput(2.0)
        )


def test_yippy_diameter_jit_and_vmap(cached_diameter_coro):
    """16 float32 eps for the log spline evaluated eagerly, JIT and vmap."""
    c = cached_diameter_coro

    def f(d):
        return c.stellar_map(500.0, d, grid=c.grid).data

    diameters = jnp.array([0.0, 0.1, 0.2])
    expected = jnp.stack([f(d) for d in diameters])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(f))(diameters),
        expected,
        rtol=16 * np.finfo(np.float32).eps,
        atol=0.0,
    )
