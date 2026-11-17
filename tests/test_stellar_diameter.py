"""Finite-diameter Images retain the backend's diameter dependence and JAX support."""

import equinox as eqx
import ineedvalidation as vv
import interpax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.grids import resample
from yippy import Circular, HalfMaximum, Matched
from yippy.performance import performance_curves

from optixstuff import MultiBandCoronagraph, ScalarCoronagraph, YippyCoronagraph


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

    np.testing.assert_array_equal(
        jax.jit(lambda s: c.leakage_fraction(s))(jnp.array(1.0)),
        c.curves().evaluate("leakage_fraction", jnp.array(1.0)),
    )
    diameters = jnp.array([0.0, 0.1, 0.2])
    expected = jnp.stack([f(d) for d in diameters])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(f))(diameters),
        expected,
        rtol=16 * np.finfo(np.float32).eps,
        atol=0.0,
    )


@pytest.fixture(
    scope="module", params=[False, True, "eac1_aavc_2d", "usort_offaxis_ovc"]
)
def finite_coro(request, cached_diameter_coro):
    """Nonuniform finite-star images, optionally in a distinct stellar frame."""
    from dataclasses import replace

    with jax.enable_x64():
        if isinstance(request.param, str):
            from yippy import EqxCoronagraph, fetch_yip

            backend = EqxCoronagraph(fetch_yip(request.param))
            return tuple(
                YippyCoronagraph(
                    backend=backend,
                    rules=(Circular(0.7), HalfMaximum(), Matched()),
                    leakage_rule=lr,
                )
                for lr in ("aperture_mean", "azimuthal_ring_mean")
            )
        backend = cached_diameter_coro._backend
        knots = jnp.array([0.0, 0.1, 0.2])
        x, y = backend.stellar_grid.coordinates()
        maps = jnp.stack(
            [
                backend.stellar_intens(0.0).data
                * jnp.exp(d * (2 + 0.2 * x + 0.1 * y**2))
                for d in knots
            ]
        )
        backend = eqx.tree_at(
            lambda b: b._stellar_ln_interp,
            backend,
            interpax.CubicSpline(knots, jnp.log(maps)),
        )
        if request.param:
            grid = replace(
                backend.stellar_grid,
                frame=replace(
                    backend.stellar_grid.frame,
                    diameter_m=4.0,
                    reference_wavelength_nm=600.0,
                ),
            )
            backend = eqx.tree_at(lambda b: b.stellar_grid, backend, grid)
        return tuple(
            YippyCoronagraph(
                backend=backend,
                rules=(Circular(0.7), HalfMaximum(), Matched()),
                leakage_rule=lr,
            )
            for lr in ("aperture_mean", "azimuthal_ring_mean")
        )


@vv.case("coronagraph-table-contract", "code-verification")
@vv.seam("yippy", "optixstuff")
@pytest.mark.parametrize("rule", [Circular(0.7), HalfMaximum(), Matched()])
@pytest.mark.parametrize("leakage_rule", ["aperture_mean", "azimuthal_ring_mean"])
def test_finite_leakage_record_reference(finite_coro, rule, leakage_rule):
    """Floating-point: 1e-9 for DFT cancellation in faint PSF tails, or 64
    stored-dtype eps for float32 leaves and log-PCHIP rounding.

    The reference remeasures the interpolated native map, at diameter knots
    and midpoints. Separations include knots, midpoints and both linear tails.
    Distinct stellar frames exercise the transpose of the spatial sampler.
    """
    with jax.enable_x64():
        c = finite_coro[leakage_rule == "azimuthal_ring_mean"]
        record = c.curves(rule, leakage_rule=leakage_rule)
        axis = record.separation_lod
        s = jnp.array(
            [
                axis[0] - 0.1,
                axis[0],
                (axis[0] + axis[1]) / 2,
                axis[len(axis) // 2],
                (axis[-2] + axis[-1]) / 2,
                axis[-1],
                axis[-1] + 0.1,
            ]
        )
        knots = c._backend._stellar_ln_interp.x
        if c.grid.shape == (256, 256):
            low, high = knots[6:8]
            diameters = jnp.array([0.0, low, (low + high) / 2])
        else:
            diameters = jnp.array([0.0, 0.05, 0.1, 0.15, 0.2])
        expected = []
        names = ("leakage_fraction", "leakage_density_lod2", "raw_contrast")
        for d in diameters:
            reference = (
                record
                if d == 0
                else performance_curves(
                    lambda s: c._backend.create_psf(s, 0.0),
                    c._backend.stellar_intens(d),
                    rule,
                    separations_lod=record.separation_lod,
                    leakage_rule=leakage_rule,
                    stellar_diameter=float(d),
                )
            )
            expected.append(jnp.stack([reference.evaluate(name, s) for name in names]))

        def query(d):
            return jnp.stack(
                [
                    getattr(c, name)(
                        s, stellar_diam_lod=d, rule=rule, leakage_rule=leakage_rule
                    )
                    for name in names
                ]
            )

        tol = max(1e-9, 64 * np.finfo(record.throughput.dtype).eps)
        np.testing.assert_allclose(
            jax.jit(jax.vmap(query))(diameters), expected, rtol=tol, atol=0
        )
        np.testing.assert_allclose(
            jnp.stack([query(d) for d in diameters]), expected, rtol=tol, atol=0
        )
        np.testing.assert_array_equal(query(0.0), expected[0])
        projected = c.leakage_density_lod2(
            c.separation_map(),
            rule=rule,
            stellar_diam_lod=0.15,
            leakage_rule=leakage_rule,
        )
        np.testing.assert_array_equal(
            c.leakage_density_map(
                [rule], stellar_diam_lod=0.15, leakage_rule=leakage_rule
            )[..., 0],
            projected,
        )
        mb = MultiBandCoronagraph([500, 800], [c, c])
        np.testing.assert_array_equal(
            mb.leakage_fraction(
                s, 800, rule=rule, leakage_rule=leakage_rule, stellar_diam_lod=0.15
            ),
            query(0.15)[0],
        )
        scalar = ScalarCoronagraph(
            grid=c.grid,
            throughput_fraction=0.3,
            aperture_area_lod2=1.0,
            stellar_leakage_density_lod2=1e-9,
        )
        for name in names:
            with pytest.raises(ValueError, match="finite stellar diameter"):
                getattr(scalar, name)(1.0, stellar_diam_lod=0.1)
