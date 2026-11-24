"""Static knot grids and the one wavelength tolerance of the band quadrature.

Two wavelengths closer than four float32 ULPs are one wavelength, in x32 and
x64 alike, at every host comparison: the break merge, the band-edge filter,
the piece-width refusal, the table-end refusal and the identity test of a
reduction. Truth sources are bitwise identities, trace counts and closed-form
trapezoid integrals of linear tables, computed here with NumPy on the cells
between knots; each tolerance is stated where it is used.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import as_float_leaf
from jax.test_util import check_grads

from optixstuff.spectral import (
    Band,
    Knots,
    Quadrature,
    Reduction,
    Resolution,
    Transmission,
    _merge_breaks,
    as_knots,
    digest_knots,
    merge_tolerance_nm,
    same_wavelength,
)

GRID = np.array([400.0, 450.1, 520.3, 700.0])


@pytest.fixture(params=[False, True], ids=["x32", "x64"])
def x64(request):
    """Run the test under each precision, scoped around construction and use."""
    with jax.enable_x64(request.param):
        yield request.param


class Linear(eqx.Module):
    """A table linear between static knots, as a source declares it."""

    knots: Knots = eqx.field(static=True, converter=as_knots)
    values: jax.Array = eqx.field(converter=as_float_leaf)

    @property
    def knot_digest(self):
        return digest_knots(self.knots, "linear")

    spectral_degree = 1

    def spectral_breaks_nm(self):
        return self.knots.nm

    def spectral_interpolant(self):
        knots = self.knots.array()
        return lambda values, wave: jnp.interp(wave, knots, values)


def trapezoid(knots, values, lo, hi):
    """Exact integral over [lo, hi] of the linear interpolant of a table."""
    inside = knots[(knots > lo) & (knots < hi)]
    edges = np.concatenate(([lo], inside, [hi]))
    y = np.interp(edges, knots, values)
    return float(np.sum(np.diff(edges) * (y[:-1] + y[1:]) / 2))


def band_integral(rule, resolution, table):
    """Integral of the table over the band through its reduction, response one."""
    reduction = resolution.rule(rule, table)
    if reduction.kind == "identity":
        values = table.spectral_interpolant()(table.values, rule.nodes_nm)
    else:
        values = table.values
    node_values = reduction.apply(values)
    return float(jnp.sum(rule.weights_nm * node_values)), reduction.kind


def test_knots_identity_and_refusals():
    """Exact digest basis (test a).

    Equal grids hash and compare equal; one changed float64 bit does not; a
    nonpositive grid, a grid with two knots within the tolerance, and a
    tracer are refused. Fails if equality compares loosely or the refusals go.
    """
    a, b = Knots(GRID), Knots(GRID.tolist())
    assert a == b and hash(a) == hash(b)
    moved = GRID.copy()
    moved[2] = np.nextafter(moved[2], np.inf)
    assert Knots(moved) != a and Knots(moved).digest != a.digest
    assert as_knots(a) is a
    with pytest.raises(ValueError, match="positive"):
        Knots([-1.0, 1.0])
    tol = float(merge_tolerance_nm(450.0))
    with pytest.raises(ValueError, match=r"closer than the wavelength tolerance"):
        Knots([400.0, 450.0, 450.0 + 0.5 * tol, 700.0])
    Knots([400.0, 450.0, 450.0 + 1.01 * tol, 700.0])
    with pytest.raises(TypeError, match="concrete"):
        jax.jit(lambda k: Knots(k).digest)(jnp.asarray(GRID))
    assert same_wavelength(450.0, 450.0 + 0.9 * tol)
    assert not same_wavelength(450.0, 450.0 + 1e-3)


def test_static_knots_retrace_only_for_a_new_grid():
    """Trace-count basis (test b).

    A table with static knots under ``eqx.filter_jit``: an equal grid in a
    new object adds no trace, a different grid adds one, and the gradient has
    no knot entry. Fails if the knots become a leaf or compare by identity.
    """
    traces = []

    @eqx.filter_jit
    def total(table):
        traces.append(1)
        return jnp.sum(table.values * table.knots.array())

    table = Linear(GRID, np.ones(4))
    total(table)
    total(Linear(GRID.copy(), 2 * np.ones(4)))
    assert len(traces) == 1
    total(Linear(GRID + 1.0, np.ones(4)))
    assert len(traces) == 2
    grad = jax.grad(lambda v: total(eqx.tree_at(lambda t: t.values, table, v)))
    assert jax.tree.leaves(table) == [table.values]
    assert grad(table.values).shape == (4,)


def test_merge_of_a_grid_with_its_float32_copy(x64):
    """Bitwise basis (test c).

    [400, 450.1, 520.3, 700] merged with its float32 copy gives four breaks,
    each bitwise one of the inputs (the smaller of each pair), in x32 and
    x64; a break 1e-3 nm from 450.1 is kept. Fails if the merge tolerance is
    1e-9 nm (six breaks, slivers of 6e-6 nm) or follows the active dtype.
    """
    copy = GRID.astype(np.float32).astype(np.float64)
    merged = _merge_breaks(np.concatenate([GRID, copy]), np.empty(0))
    np.testing.assert_array_equal(merged, np.minimum(GRID, copy))
    kept = _merge_breaks(np.concatenate([GRID, copy]), np.array([450.101]))
    assert kept.size == 5 and 450.101 in kept


def test_layout_is_independent_of_precision_and_order():
    """Exact digest basis (test d).

    The rule built on a float64 table and its float32 copy has one
    ``layout_digest`` in x32 and x64, and in either order of the models.
    Fails if the tolerance follows the active dtype or the merge keeps the
    first model's break.
    """
    first = Transmission(GRID, np.ones(4))
    second = Transmission(GRID.astype(np.float32), np.ones(4))
    digests = set()
    for flag in (False, True):
        with jax.enable_x64(flag):
            for models in ((first, second), (second, first)):
                rule = Quadrature.build((Band(390.0, 710.0),), *models)
                digests.add(rule.layout_digest)
                assert len(rule.pieces[0]) == 5
    assert len(digests) == 1


def test_linear_table_integral_matches_trapezoid(x64):
    """Closed-form basis (test e).

    A linear table integrated through its table-kind operator equals the
    trapezoid integral. Tolerance 1e-12 relative in x64 (rounding of a sum of
    order 100 terms); 5e-6 in x32 (a few float32 ULPs times the conditioning
    of the weights).
    """
    knots = np.linspace(480.0, 620.0, 71)
    values = 1.0 + 0.5 * np.sin(knots / 7.0)
    table = Linear(knots, values)
    band = Band(500.3, 597.7)
    rule = Quadrature.build((band,), order=4, breaks_nm=[530.0, 561.1])
    integral, kind = band_integral(rule, Resolution.build(rule, table), table)
    assert kind == "table"
    truth = trapezoid(knots, values, band.low_nm, band.high_nm)
    np.testing.assert_allclose(integral, truth, rtol=1e-12 if x64 else 5e-6)


def test_twin_coarse_tables_keep_the_identity_kind(x64):
    """Closed-form basis (test g).

    The float64 table and its float32 copy, both declared as breaks: each
    reduction is the identity kind in x32 and x64, and in x64 the float32
    table's integral equals its own trapezoid to 1e-11 relative (the edge
    guard bounds the error at 1e-12; measured near 1e-15). Fails if the
    identity test compares knots and piece edges exactly.
    """
    copy = GRID.astype(np.float32)
    tables = (Linear(GRID, [1.0, 2.0, 0.5, 1.5]), Linear(copy, [2.0, 1.0, 3.0, 0.5]))
    band = Band(400.0, 700.0)
    rule = Quadrature.build((band,), *tables, order=4)
    resolution = Resolution.build(rule, *tables)
    for table in tables:
        integral, kind = band_integral(rule, resolution, table)
        assert kind == "identity"
    if x64:
        truth = trapezoid(
            np.asarray(copy, float), [2.0, 1.0, 3.0, 0.5], band.low_nm, band.high_nm
        )
        np.testing.assert_allclose(integral, truth, rtol=1e-11)


def test_fine_table_near_another_grid_is_integrated_exactly():
    """Closed-form basis (test k).

    A table at resolving power 1.5e5 with fixed pseudo-random ordinates,
    declared as breaks with a second grid at 0.2 nm whose 13 inner knots sit
    1e-5 nm below one of its knots (inside the tolerance, outside the edge
    guard), on a band whose upper end is 4e-5 nm above one of its knots. The
    fine table is not the identity kind, its band integral equals the
    trapezoid sum to 1e-12 relative in x64 (rounding of a sum of 1.6e3 terms;
    measured near 1e-16), the band ends are piece edges, and its knots are
    bitwise the input after the build. Fails if every near knot counts as
    the edge (error near 2e-8), if the knots are snapped to the pieces, or
    if the band end is dropped for the nearby knot.
    """
    rng = np.random.default_rng(0)
    fine = 450.0 * (1 + 1 / 1.5e5) ** np.arange(1700)
    values = 1.0 + rng.uniform(-0.5, 0.5, fine.size)
    coarse_at = np.searchsorted(fine, 450.0 + 0.2 * np.arange(1, 14))
    coarse = np.concatenate(([449.0], fine[coarse_at] - 1e-5, [456.0]))
    band = Band(float(fine[1]), float(fine[1650] + 4e-5))
    with jax.enable_x64():
        table = Linear(fine, values)
        other = Linear(coarse, np.ones(coarse.size))
        rule = Quadrature.build((band,), table, other, order=4)
        edges = [rule.pieces[0][0][0], rule.pieces[0][-1][1]]
        assert edges == [band.low_nm, band.high_nm]
        resolution = Resolution.build(rule, table, other)
        integral, kind = band_integral(rule, resolution, table)
    assert kind != "identity"
    np.testing.assert_array_equal(table.knots.nm, fine)
    truth = trapezoid(fine, values, band.low_nm, band.high_nm)
    np.testing.assert_allclose(integral, truth, rtol=1e-12)


def test_float32_micron_table_ends_at_the_band_edge(x64):
    """Exact layout basis (tests h and i).

    A table stored in float32 microns (0.45 um widens to 449.9999881 nm) on
    a band from 420 to 450.0 nm builds with the band edges kept, one piece and
    no refusal, and is the identity kind. A band ending 1 nm past the table
    end is refused. Fails if the edge filter or the table-end test compares
    at 1e-9 nm.
    """
    microns = np.array([0.40, 0.42, 0.45], dtype=np.float32)
    table = Linear(microns.astype(np.float64) * 1e3, [1.0, 2.0, 3.0])
    assert table.knots.nm[-1] < 450.0
    rule = Quadrature.build((Band(420.0, 450.0),), table, order=4)
    assert rule.pieces == (((420.0, 450.0, 4),),)
    assert Resolution.build(rule, table).rule(rule, table).kind == "identity"
    past = Quadrature.build((Band(420.0, 451.0),), table, order=4)
    with pytest.raises(ValueError, match="leaves the table"):
        Reduction.build(past, table.knots, degree=1)


def test_transmission_has_one_leaf_and_a_true_gradient():
    """Finite-difference basis (test j).

    The wavelength grid is static auxiliary data, so the response is the one
    leaf; the gradient of a band integral with respect to it matches central
    differences (``check_grads``, x64, eps 1e-4, tolerance 1e-4). Fails if
    the grid is kept as a child.
    """
    with jax.enable_x64():
        curve = Transmission([500.0, 520.0, 545.0, 560.0], [0.2, 0.9, 0.7, 0.1])
        assert jax.tree.leaves(curve) == [curve.transmission]
        assert isinstance(curve.wavelength_nm, Knots)
        band = Band(505.0, 555.0)

        def integral(response):
            moved = jax.tree.unflatten(jax.tree.structure(curve), [response])
            return moved.integrate(band, lambda w: 1.0 + w / 500.0)

        check_grads(integral, (curve.transmission,), order=1, modes=("rev",))
        assert curve.low_nm == 500.0 and curve.high_nm == 560.0
