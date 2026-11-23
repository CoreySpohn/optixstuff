"""Node-effective values of tabulated sources: the operator, its layout and lookup.

Every expected value is independent of the operator under test: an exact
per-cell polynomial integral typed with ``numpy.polynomial``, or a
Gauss-Legendre sum on every table cell. Every response is a polynomial of
degree Q - 1 or a non-polynomial pixel, so a wrong distribution of the source
over the nodes is visible; a flat response would check only the total.
The ``phoenix`` fixture (conftest) is a public PHOENIX stellar spectrum.
"""

import re
from itertools import pairwise

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.special import j1

from optixstuff import Exposure
from optixstuff.spectral import (
    _RULE_CACHE,
    Band,
    Quadrature,
    Reduction,
    Resolution,
    SpectralTable,
    _build_arrays,
    digest_knots,
)


class Tabulated(eqx.Module):
    """A tabulated spectrum, linear or cubic between knots, as a source declares it."""

    wavelength_nm: jax.Array
    values: jax.Array
    method: str = eqx.field(static=True)
    knot_digest: str | None = eqx.field(static=True)

    def __init__(self, wavelength_nm, values, method="linear"):
        self.method = method
        self.wavelength_nm = jnp.asarray(wavelength_nm)
        self.values = jnp.asarray(values)
        traced = isinstance(self.wavelength_nm, jax.core.Tracer)
        self.knot_digest = None if traced else digest_knots(self.wavelength_nm, method)

    @property
    def spectral_degree(self):
        return 1 if self.method == "linear" else 3

    def spectral_breaks_nm(self):
        return np.asarray(self.wavelength_nm)

    def spectral_interpolant(self):
        knots, method = self.wavelength_nm, self.method
        return lambda values, wave: interpax.interp1d(wave, knots, values, method)


def gauss_cells(edges, n):
    """Gauss-Legendre points and weights with n points on every cell of edges."""
    x, w = leggauss(n)
    length = np.diff(edges)[:, None]
    return (edges[:-1, None] + length * (x + 1) / 2).ravel(), (length * w / 2).ravel()


def cubic(band):
    """A cubic response over the band (degree Q - 1 at order 4)."""

    def response(wavelength_nm):
        u = (np.asarray(wavelength_nm) - band.center_nm) / (band.width_nm / 2)
        return 1 + u - 0.7 * u**2 + 0.4 * u**3

    return response


def band_sums(rule, node_values, response):
    """Sum of weights * node values * response at the nodes, per band."""
    nodes = np.asarray(rule.nodes_nm)
    return np.sum(
        np.asarray(rule.weights_nm) * np.asarray(node_values) * response(nodes), -1
    )


def linear_integral(knots, values, lo, hi, poly_of_wavelength):
    """Exact integral over [lo, hi] of a linear table times a polynomial.

    ``poly_of_wavelength`` returns the polynomial on a cell as a function of
    ``u = (wavelength - lo)``; each cell is integrated in closed form.
    """
    inside = knots[(knots > lo) & (knots < hi)]
    edges = np.concatenate(([lo], inside, [hi]))
    total = 0.0
    for a, b in pairwise(edges):
        ya, yb = np.interp([a, b], knots, values)
        source = Polynomial([ya - (yb - ya) / (b - a) * (a - lo), (yb - ya) / (b - a)])
        integral = (source * poly_of_wavelength(a, b)).integ()
        total += integral(b - lo) - integral(a - lo)
    return total


def build(bands, *tables, **kwargs):
    rule = Quadrature.build(bands, **kwargs)
    return rule, Resolution.build(rule, *tables)


# Three layouts: one band of one piece; a band of two pieces beside a padded
# band of one; three bands of three, two and one pieces at an odd order.
LAYOUTS = {
    "one piece": ((Band(500.0, 530.0),), (), 4),
    "padded": ((Band(500.0, 530.0), Band(520.0, 560.0)), (511.3,), 4),
    "three bands, odd order": (
        (Band(500.0, 530.0), Band(531.0, 560.0), Band(562.0, 575.0)),
        (511.3, 523.9, 547.2),
        3,
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_reduction_is_exact_below_the_order_and_misses_by_the_remainder(layout):
    """Fails if a node-effective value is not the source's moment on its node.

    A random linear table (seed 0, 62 knots in 495 to 580 nm) times
    ``u**d``, ``u = (wavelength - 495) / 85``. Truth: exact per-cell
    polynomial integrals. For every degree d below the order Q the rule is
    exact (floating-point, 1e-12 relative). At degree Q it must miss by the
    analytic remainder, the integral of the table times the nodal polynomial
    of each piece (exact-algebra, 1e-12 of the integral), so a rule exact to
    a higher degree (a point sample, the Gauss bonus) fails. Padded nodes copy
    the first real node of their band exactly.
    """
    bands, breaks, order = LAYOUTS[layout]
    rng = np.random.default_rng(0)
    knots = np.sort(np.concatenate(([495.0, 580.0], rng.uniform(495, 580, 60))))
    values = rng.uniform(0.5, 2.0, knots.size)
    with jax.enable_x64():
        rule = Quadrature.build(bands, order=order, breaks_nm=breaks)
        table = Tabulated(knots, values)
        reduction = Reduction.build(
            rule, knots, degree=1, interpolant=table.spectral_interpolant()
        )
        assert reduction.kind == "table"
        node_values = np.asarray(reduction.apply(table.values))
    for w, row in enumerate(rule.pieces):
        count = sum(n for *_, n in row)
        np.testing.assert_array_equal(node_values[w, count:], node_values[w, 0])
    for degree in range(order + 1):

        def monomial(wavelength_nm, degree=degree):
            return ((np.asarray(wavelength_nm) - 495.0) / 85.0) ** degree

        estimate = band_sums(rule, node_values, monomial)
        for w, row in enumerate(rule.pieces):
            truth, remainder = 0.0, 0.0
            for lo, hi, n in row:
                start = sum(m for _, _, m in row[: row.index((lo, hi, n))])
                roots = (np.asarray(rule.nodes_nm[w, start : start + n]) - 495) / 85

                def power(a, b, lo=lo, degree=degree):
                    # u**degree as a polynomial in (wavelength - lo).
                    return Polynomial([(lo - 495.0) / 85, 1 / 85]) ** degree

                def nodal(a, b, lo=lo, roots=roots):
                    shift = Polynomial([(lo - 495.0) / 85, 1 / 85])
                    out = Polynomial([1.0])
                    for root in roots:
                        out = out * (shift - root)
                    return out

                truth += linear_integral(knots, values, lo, hi, power)
                remainder += linear_integral(knots, values, lo, hi, nodal)
            if degree < order:
                np.testing.assert_allclose(estimate[w], truth, rtol=1e-12, atol=0)
            else:
                assert abs(remainder) > 1e-7 * abs(truth)
                assert abs(estimate[w] - truth + remainder) < 1e-12 * abs(truth)


def test_fine_kind_integrates_a_product_of_two_tables():
    """Fails if the fine rule under-integrates a source of degree two between knots.

    The product of two linear tables with interleaved knots (23 and 17
    knots) over 500 to 555 nm, order 4, cubic response. Truth: 8-point Gauss
    on every merged cell (exact for degree 5). The declared degree 2 gives
    ``ceil((4 + 2) / 2) = 3`` points per cell and is exact (floating-point,
    1e-13). Mutation: two points per cell, the rule of a declared degree 0,
    must miss by more than 1e-7 (spike: -1.2e-6 on this layout).
    """
    band = Band(500.0, 555.0)
    first = np.linspace(499.0, 556.0, 23)
    second = np.linspace(499.3, 556.3, 17)

    def product(wavelength_nm):
        return np.interp(wavelength_nm, first, 1 + 0.5 * np.sin(first)) * np.interp(
            wavelength_nm, second, 2 + np.cos(0.7 * second)
        )

    edges = np.union1d([band.low_nm, band.high_nm], np.union1d(first, second))
    edges = edges[(edges >= band.low_nm) & (edges <= band.high_nm)]
    x, wx = gauss_cells(edges, 8)
    truth = np.sum(wx * product(x) * cubic(band)(x))
    knots = np.union1d(first, second)
    errors = {}
    with jax.enable_x64():
        rule = Quadrature.build((band,), order=4)
        for degree in (2, 0):
            reduction = Reduction.build(rule, knots, degree=degree)
            assert reduction.kind == "fine"
            node_values = reduction.apply(product(np.asarray(reduction.fine_nm)))
            errors[degree] = band_sums(rule, node_values, cubic(band))[0] / truth - 1
    assert abs(errors[2]) < 1e-13
    assert abs(errors[0]) > 1e-7


def test_table_kind_matches_fine_kind_and_truth_on_phoenix(phoenix):
    """Fails if transposing the interpolant differs from evaluating it on the fine rule.

    PHOENIX 5800 K over 540 to 595 nm, order 4, cubic response. The table
    kind (the interpolant transposed) must equal the fine kind (the
    interpolant evaluated at ``fine_nm``) for a linear and a cubic
    interpolant (floating-point, 1e-13; spike 2e-15), and the linear table
    must equal its exact per-cell integral (floating-point, 1e-12; spike
    1e-15). Mutation, each measured on this band before the test was written:
    a point sample at the nodes misses by +5.1e-2, the band mean at every node
    by -2.6e-2, the node rows reversed by -4.6e-2; each must miss by more than
    5e-3.
    """
    knots, values = phoenix
    band = Band(540.0, 595.0)
    response = cubic(band)
    with jax.enable_x64():
        rule = Quadrature.build((band,), order=4)
        for method in ("linear", "cubic"):
            table = Tabulated(knots, values, method)
            composed = Reduction.build(
                rule,
                knots,
                degree=table.spectral_degree,
                interpolant=table.spectral_interpolant(),
            )
            fine = Reduction.build(rule, knots, degree=table.spectral_degree)
            at_fine = table.spectral_interpolant()(
                table.values, fine.fine_nm.ravel()
            ).reshape(fine.fine_nm.shape)
            np.testing.assert_allclose(
                composed.apply(table.values), fine.apply(at_fine), rtol=1e-13
            )
            assert composed.nbytes < 0.1e6
        node_values = np.asarray(composed.apply(table.values))[0]
    center = band.center_nm

    def shifted(a, b):
        # The cubic response as a polynomial in (wavelength - low).
        u = Polynomial(
            [(band.low_nm - center) / (band.width_nm / 2), 2 / band.width_nm]
        )
        return 1 + u - 0.7 * u**2 + 0.4 * u**3

    with jax.enable_x64():
        linear = Tabulated(knots, values)
        exact = np.asarray(
            Reduction.build(
                rule, knots, degree=1, interpolant=linear.spectral_interpolant()
            ).apply(linear.values)
        )[0]
    truth = linear_integral(knots, values, band.low_nm, band.high_nm, shifted)
    weights = np.asarray(rule.weights_nm[0])
    nodes = np.asarray(rule.nodes_nm[0])

    def error(node_values):
        return np.sum(weights * node_values * response(nodes)) / truth - 1

    assert abs(error(exact)) < 1e-12
    mean = np.sum(weights * exact) / band.width_nm
    mutants = {
        "point sample": np.interp(nodes, knots, values),
        "band mean": np.full(nodes.shape, mean),
        "rows reversed": exact[::-1],
    }
    for name, mutant in mutants.items():
        assert abs(error(mutant)) > 5e-3, name
    assert node_values.shape == (4,)


def test_a_band_that_leaves_the_table_is_refused(phoenix):
    """Fails if a piece outside the table, or straddling its end, gets a value.

    Exact (the refusal). A piece wholly beyond the last knot, a piece that
    straddles it and a band starting below the first knot each raise and name
    the table end; through ``Resolution.build`` the message also names the
    table. At the boundary, a band one ULP inside the last knot builds and one
    ULP beyond it is refused, with enough digits that the two ends differ in
    the message. Mutation: a build that zero-fills beyond the table returns a
    value and fails here. The same band ended at the table end gives the exact
    per-cell integral (floating-point, 1e-12).
    """
    knots, values = phoenix
    end = knots[-1]
    with jax.enable_x64():
        table = Tabulated(knots, values)
        for band, breaks in (
            (Band(650.0, 690.0), (end,)),
            (Band(650.0, 690.0), ()),
            (Band(530.0, 560.0), ()),
        ):
            rule = Quadrature.build((band,), order=4, breaks_nm=breaks)
            named = float(knots[0] if band.low_nm < knots[0] else end)
            with pytest.raises(ValueError, match=re.escape(f"at {named!r} nm")):
                Reduction.build(rule, knots, degree=1)
            with pytest.raises(ValueError, match="Tabulated: band"):
                Resolution.build(rule, table)
        inside = float(np.nextafter(end, -np.inf))
        rule = Quadrature.build((Band(650.0, inside),), order=4)
        Reduction.build(rule, knots, degree=1)
        beyond = float(np.nextafter(end, np.inf))
        rule = Quadrature.build((Band(650.0, beyond),), order=4)
        message = f"to {beyond!r} nm leaves the table, which ends at {float(end)!r}"
        with pytest.raises(ValueError, match=re.escape(message)):
            Reduction.build(rule, knots, degree=1)
        ended = Band(650.0, end)
        rule, resolution = build((ended,), table, order=4)
        node_values = resolution.rule(rule, table).apply(table.values)
    response = cubic(ended)

    def shifted(a, b):
        u = Polynomial(
            [
                (ended.low_nm - ended.center_nm) / (ended.width_nm / 2),
                2 / ended.width_nm,
            ]
        )
        return 1 + u - 0.7 * u**2 + 0.4 * u**3

    truth = linear_integral(knots, values, ended.low_nm, ended.high_nm, shifted)
    np.testing.assert_allclose(
        band_sums(rule, node_values, response)[0], truth, rtol=1e-12
    )


def airy(separation_lod, center_nm):
    """Unocculted Airy pixel at a fixed sky angle, as a function of wavelength."""

    def response(wavelength_nm):
        x = np.pi * separation_lod * center_nm / np.asarray(wavelength_nm)
        return (2 * j1(x) / x) ** 2

    return response


def test_traced_contract_of_the_operator(phoenix):
    """Fails if jit, vmap, grad, float32 or a band slice break the operator.

    Two PHOENIX bands, 540 to 570 and 570 to 630 nm, read out to 10 lambda/D
    (orders 6 and 8, so the first row is padded), through an Airy pixel at 6
    lambda/D. The gradient with respect to one table value is the integral
    of its hat function times the per-piece Lagrange interpolant of the
    response at the nodes, typed here with ``numpy.polynomial``
    (exact-algebra, 1e-10 relative), for a knot inside a band, one on the
    shared edge and one on the outer edge. jit equals eager and vmap equals a
    loop (floating-point, 1e-14); a new table and scale reuse one trace; the
    rows of one band carried from the full rule integrate to that band's
    value, and a lookup on the band slice raises. In float32 the value holds
    to 1e-5 of the float64 value (spike: 1.1e-6 on this layout).
    """
    knots, values = phoenix
    bands = (Band(540.0, 570.0), Band(570.0, 630.0))
    response = airy(6.0, 570.0)
    with jax.enable_x64():
        table = Tabulated(knots, values)
        rule, resolution = build(bands, table, order=4, field_lod=10.0)
        assert [row[0][2] for row in rule.pieces] == [6, 8]
        weights = rule.weights_nm * response(np.asarray(rule.nodes_nm))
        traces = []

        def value(resolution, table):
            traces.append(1)
            node_values = resolution.rule(rule, table).apply(table.values)
            return jnp.sum(weights * node_values, -1)

        eager = value(resolution, table)
        jitted = eqx.filter_jit(value)
        np.testing.assert_allclose(jitted(resolution, table), eager, rtol=1e-14)
        scaled = eqx.tree_at(lambda t: t.values, table, 1.3 * table.values)
        jitted(resolution, scaled)
        assert len(traces) == 2  # one eager call and one trace
        batch = jnp.stack([table.values, 0.5 * table.values, table.values**2])
        mapped = jax.vmap(
            lambda v: value(resolution, eqx.tree_at(lambda t: t.values, table, v))
        )(batch)
        loop = [
            value(resolution, eqx.tree_at(lambda t: t.values, table, v)) for v in batch
        ]
        np.testing.assert_allclose(mapped, np.stack(loop), rtol=1e-14)
        gradient = np.asarray(
            jax.grad(
                lambda v: value(
                    resolution, eqx.tree_at(lambda t: t.values, table, v)
                ).sum()
            )(table.values)
        )
        node_values = resolution.rule(rule, table).apply(table.values)
        for i in range(len(bands)):
            row = rule.band(i)
            np.testing.assert_allclose(
                row.integrate(
                    node_values[i][None] * response(np.asarray(row.nodes_nm))
                ),
                eager[i][None],
                rtol=1e-14,
            )
            with pytest.raises(LookupError, match="band slice"):
                resolution.rule(row, table)
        f32_truth = float(eager.sum())
    for j in (
        int(np.searchsorted(knots, 555.0)),
        int(np.searchsorted(knots, 570.0)),
        int(np.searchsorted(knots, 540.0)),
    ):
        expected = 0.0
        hat = (knots[j - 1], knots[j], knots[j + 1])
        for w, row in enumerate(rule.pieces):
            start = 0
            for lo, hi, n in row:
                nodes = np.asarray(rule.nodes_nm[w, start : start + n])
                start += n
                lagrange = Polynomial.fit(nodes, response(nodes), n - 1)
                for a, b, rising in ((hat[0], hat[1], True), (hat[1], hat[2], False)):
                    a, b = max(a, lo), min(b, hi)
                    if b <= a:
                        continue
                    x, wx = gauss_cells(np.array([a, b]), n + 2)
                    shape = (
                        (x - hat[0]) / (hat[1] - hat[0])
                        if rising
                        else (hat[2] - x) / (hat[2] - hat[1])
                    )
                    expected += np.sum(wx * shape * lagrange(x))
        np.testing.assert_allclose(gradient[j], expected, rtol=1e-10)
    table32 = Tabulated(knots, values)
    rule32, resolution32 = build(bands, table32, order=4, field_lod=10.0)
    assert rule32.weights_nm.dtype == jnp.float32
    node_values = resolution32.rule(rule32, table32).apply(table32.values)
    weights32 = rule32.weights_nm * response(np.asarray(rule32.nodes_nm))
    np.testing.assert_allclose(
        float(jnp.sum(weights32 * node_values)), f32_truth, rtol=1e-5
    )


def test_resolution_finds_rules_by_digest_and_refuses_the_rest(phoenix):
    """Fails if a source can be integrated by a rule built for another table.

    Exact. A rule is found under jit by the static digests; a table that
    was not resolved on the quadrature, a band slice and a hand-made rule
    raise at trace time; a table constructed under a trace has no digest and
    its lookup names the remedy; a knot grid replaced after construction
    fails the build's digest check; two tables on one grid share one rule;
    a table whose knots are all piece edges takes the identity kind and holds
    no arrays (exact Gauss per piece: the cubic-response integral to 1e-12),
    unless a piece has fewer nodes than the source's degree.
    An exposure carries its resolution through jit and refuses one built for
    another quadrature.
    """
    knots, values = phoenix
    band = Band(560.0, 580.0)
    with jax.enable_x64():
        star = Tabulated(knots, values)
        twin = Tabulated(knots, 2 * values)
        other = Tabulated(knots + 0.001, values)
        rule, resolution = build((band,), star, twin, order=4)
        assert len(resolution.rules) == 1
        assert isinstance(star, SpectralTable)

        @eqx.filter_jit
        def read(resolution, table):
            return resolution.rule(rule, table).apply(table.values)

        np.testing.assert_array_equal(
            read(resolution, twin), 2 * read(resolution, star)
        )
        with pytest.raises(LookupError, match="no rule for Tabulated"):
            read(resolution, other)
        with pytest.raises(LookupError, match="no rule for Tabulated"):
            resolution.rule(rule, other)
        hand_made = Quadrature(rule.bands, rule.nodes_nm, rule.weights_nm)
        with pytest.raises(LookupError, match="hand-made"):
            resolution.rule(hand_made, star)

        @jax.jit
        def traced_table(wavelength_nm):
            resolution.rule(rule, Tabulated(wavelength_nm, values))

        with pytest.raises(LookupError, match="construct the table on the host"):
            traced_table(jnp.asarray(knots))
        stale = eqx.tree_at(lambda t: t.wavelength_nm, star, star.wavelength_nm + 0.01)
        with pytest.raises(ValueError, match="does not match its knots"):
            Resolution.build(rule, stale)

        inside = knots[(knots > band.low_nm) & (knots < band.high_nm)]
        exact_rule = Quadrature.build((band,), order=4, breaks_nm=inside)
        identity = Resolution.build(exact_rule, star).rule(exact_rule, star)
        assert identity.kind == "identity" and identity.nbytes == 0
        coarse = Quadrature.build((band,), order=2, breaks_nm=inside)
        assert Reduction.build(coarse, knots, degree=3).kind == "fine"
        node_values = identity.apply(
            np.interp(np.asarray(exact_rule.nodes_nm), knots, values)
        )

        exposure = Exposure(0.0, 0.0, rule, resolution=resolution, reads=("star",))
        np.testing.assert_array_equal(
            eqx.filter_jit(
                lambda e, t: e.resolution.rule(e.quadrature, t).apply(t.values)
            )(exposure, star),
            read(resolution, star),
        )
        with pytest.raises(ValueError, match="another quadrature"):
            Exposure(0.0, 0.0, exact_rule, resolution=resolution)

    def shifted(a, b):
        u = Polynomial(
            [(band.low_nm - band.center_nm) / (band.width_nm / 2), 2 / band.width_nm]
        )
        return 1 + u - 0.7 * u**2 + 0.4 * u**3

    truth = linear_integral(knots, values, band.low_nm, band.high_nm, shifted)
    np.testing.assert_allclose(
        band_sums(exact_rule, node_values, cubic(band))[0], truth, rtol=1e-12
    )


class _Declares:
    """A model that declares fixed spectral breaks."""

    def __init__(self, *breaks_nm):
        self.breaks_nm = np.asarray(breaks_nm)

    def spectral_breaks_nm(self):
        return self.breaks_nm


def _reduce(rule, knots):
    return Reduction.build(
        rule, knots, degree=1, interpolant=lambda v, w: jnp.interp(w, knots, v)
    )


def test_breaks_that_differ_by_rounding_are_one_break_and_the_reduction_is_finite():
    """Fails if breaks that differ by rounding make a piece of zero width.

    A caller break at 950.0 beside a model knot one ulp below it must give the
    pieces, nodes and weights of the build with only one of them, to rounding,
    and keep the model knot; of two model knots the smaller is kept. A break
    one ulp inside either edge of a band is dropped. Every reduction is finite.
    Truth: the build with a single break, or with none for the edge cases.
    Tolerance 1e-12 relative, basis the ulp-sized shift of one piece edge; a
    piece of 1e-13 nm gives NaN blocks.
    """
    near = float(np.nextafter(950.0, 0.0))
    band, knots = Band(900.0, 1000.0), np.linspace(890.0, 1010.0, 41)
    with jax.enable_x64():
        merged = Quadrature.build((band,), _Declares(near), breaks_nm=[950.0])
        reverse = Quadrature.build((band,), _Declares(950.0), breaks_nm=[near])
        both = Quadrature.build((band,), _Declares(950.0, near))
        for single, kept in (
            (Quadrature.build((band,), breaks_nm=[950.0]), near),
            (Quadrature.build((band,), _Declares(near)), near),
        ):
            assert len(merged.pieces[0]) == len(single.pieces[0]) == 2
            np.testing.assert_allclose(merged.nodes_nm, single.nodes_nm, rtol=1e-12)
            np.testing.assert_allclose(merged.weights_nm, single.weights_nm, rtol=1e-12)
            assert merged.pieces[0][0][1] == kept
        assert reverse.pieces[0][0][1] == 950.0
        assert both.pieces[0][0][1] == near
        edge_band = Band(900.0, 950.0)
        edges = (float(np.nextafter(950.0, 0.0)), float(np.nextafter(900.0, 1000.0)))
        trimmed = Quadrature.build((edge_band,), breaks_nm=edges)
        plain = Quadrature.build((edge_band,))
        assert trimmed.pieces == plain.pieces == (((900.0, 950.0, 4),),)
        np.testing.assert_array_equal(trimmed.nodes_nm, plain.nodes_nm)
        for rule in (merged, reverse, both, trimmed):
            assert np.isfinite(np.asarray(_reduce(rule, knots).blocks)).all()
            assert (np.asarray(rule.weights_nm)[:, :4] > 0).all()


def test_reduction_refuses_a_piece_that_is_too_narrow_or_has_equal_nodes():
    """Fails if a hand-built rule with a degenerate piece reduces to NaN blocks.

    A piece of zero width and a piece one ulp wide (nodes that coincide in
    floating point) are both refused with a message that names the cause.
    """
    band, knots = Band(900.0, 1000.0), np.linspace(890.0, 1010.0, 41)
    near = float(np.nextafter(950.0, 0.0))
    with jax.enable_x64():
        for edge, nodes in (
            (950.0, [925.0, 950.0, 975.0]),
            (near, [925.0, 950.0, 975.0]),
        ):
            rule = Quadrature(
                (band,),
                [nodes],
                [[25.0, 1e-13, 75.0]],
                pieces=[[(900.0, edge, 1), (edge, 950.0, 1), (950.0, 1000.0, 1)]],
            )
            with pytest.raises(ValueError, match="narrower than"):
                _reduce(rule, knots)
        coincident = Quadrature(
            (band,),
            [[925.0, 950.0, 950.0, 975.0]],
            [[50.0, 25.0, 25.0, 50.0]],
            pieces=[[(900.0, 950.0, 1), (950.0, 1000.0, 3)]],
        )
        with pytest.raises(ValueError, match="not distinct"):
            _reduce(coincident, knots)


def test_reduction_cache_separates_rules_with_equal_pieces_and_other_weights():
    """Fails if a rule with the pieces of another but other weights reuses its operator.

    Two quadratures with equal pieces and weights that differ by 1e-3 on one
    node are built in one process, in both orders. Each reduction must equal
    ``_build_arrays`` called directly on that rule (no cache), to rounding
    (1e-14 relative to the block scale), and the two must differ by more than
    1e-4 of it. The node-effective values divide by the weights, so the second
    build gets the first rule's operator when the cache key ignores them.
    """
    band, knots = Band(500.0, 560.0), np.linspace(490.0, 570.0, 33)
    interpolant = lambda v, w: jnp.interp(w, knots, v)  # noqa: E731
    with jax.enable_x64():
        base = Quadrature.build((band,), breaks_nm=[530.0], order=3)
        weights = np.asarray(base.weights_nm).copy()
        weights[0, 1] += 1e-3
        other = Quadrature((band,), base.nodes_nm, weights, base.pieces)
        assert other.layout_digest == base.layout_digest
        truth = {
            name: np.asarray(_build_arrays(rule, knots, 1, interpolant)[2]["blocks"])
            for name, rule in (("base", base), ("other", other))
        }
        scale = np.abs(truth["base"]).max()
        assert np.abs(truth["base"] - truth["other"]).max() > 1e-6 * scale
        for order in (("base", "other"), ("other", "base")):
            rules = {"base": base, "other": other}
            _RULE_CACHE.clear()
            for name in order:
                reduction = Reduction.build(
                    rules[name], knots, degree=1, interpolant=interpolant
                )
                np.testing.assert_allclose(
                    reduction.blocks, truth[name], rtol=0, atol=1e-14 * scale
                )


def test_resolution_refuses_a_quadrature_with_the_layout_and_other_weights(phoenix):
    """Fails if a rule built for one rule is read by a layout twin with other weights.

    A quadrature with the pieces of the built one and a weight changed by 1e-3
    is refused by ``Resolution.rule`` when both are concrete, and two such
    quadratures are refused together by ``Resolution.build``. The rule found
    for the built quadrature itself is unchanged.
    """
    knots = np.linspace(490.0, 570.0, 33)
    table = Tabulated(knots, np.linspace(1.0, 2.0, knots.size))
    with jax.enable_x64():
        base = Quadrature.build((Band(500.0, 560.0),), breaks_nm=[530.0], order=3)
        weights = np.asarray(base.weights_nm).copy()
        weights[0, 1] += 1e-3
        other = Quadrature((base.bands[0],), base.nodes_nm, weights, base.pieces)
        resolution = Resolution.build(base, table)
        assert resolution.rule(base, table).kind == "table"
        with pytest.raises(LookupError, match="other nodes or weights"):
            resolution.rule(other, table)
        with pytest.raises(ValueError, match="share a layout"):
            Resolution.build((base, other), table)
