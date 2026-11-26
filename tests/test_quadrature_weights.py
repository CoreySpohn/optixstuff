"""Band quadrature: accuracy ladders, the response-set order rule and the layout.

Every expected value is an independent Gauss-Legendre sum typed here with
NumPy (12 points per knot cell for the half-width band, 16 points per cell on
a fine grid for the Airy ladder), never a call to the rule under test.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testing import ulps
from numpy.polynomial.legendre import leggauss
from scipy.special import j1

from optixstuff.spectral import Band, Quadrature, Reduction, Transmission

LO_NM, HI_NM, CENTER_NM = 375.0, 625.0, 500.0
BAND = Band(LO_NM, HI_NM)
SECOND_RADIATION_NM_K = 1.438776877e7
# Leinert et al. (1998) Table 19 wavelengths, the knots of the local zodi model.
TABLE19_NM = 1e3 * np.array(
    [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5, 4.8, 12, 25, 60, 100, 140]
)


def planck_photon(wavelength_nm, temperature_K):
    """Blackbody photon spectral radiance up to a constant factor."""
    return wavelength_nm**-4 / np.expm1(
        SECOND_RADIATION_NM_K / (wavelength_nm * temperature_K)
    )


def thermal(wavelength_nm, temperature_K):
    return planck_photon(wavelength_nm, temperature_K) / planck_photon(
        CENTER_NM, temperature_K
    )


def contrast(wavelength_nm):
    return 0.3 * (wavelength_nm / CENTER_NM) ** -4


def legendre_row(pieces):
    """Nodes and weights of Gauss-Legendre rules on (low, high, order) pieces."""
    waves, weights = [], []
    for lo, hi, n in pieces:
        x, w = leggauss(n)
        waves.append(lo + (hi - lo) * ((x + 1) / 2))
        weights.append((hi - lo) * w / 2)
    return np.concatenate(waves), np.concatenate(weights)


class Hosts:
    """A line-blanketed 5800 K star, a coarse zodi table and a trapezoid response.

    The star is a blackbody carrying 300 seeded Gaussian absorption lines of
    depth up to 0.6 on a 0.1 nm table, which stands in for a stellar atmosphere
    grid without any download.
    """

    def __init__(self):
        rng = np.random.default_rng(0)
        self.star_nm = np.arange(350.0, 650.0, 0.1)
        self.star = planck_photon(self.star_nm, 5800.0) / planck_photon(500.0, 5800.0)
        lines = zip(
            rng.uniform(350, 650, 300),
            rng.uniform(0, 0.6, 300),
            rng.uniform(0.05, 0.3, 300),
            strict=True,
        )
        for center, depth, sigma in lines:
            near = np.abs(self.star_nm - center) < 5 * sigma
            self.star[near] *= 1 - depth * np.exp(
                -0.5 * ((self.star_nm[near] - center) / sigma) ** 2
            )
        self.zodi_nm = np.array([200.0, 300, 400, 500, 700, 900, 1000, 1200])
        self.zodi = np.array([0.1, 0.45, 0.8, 1.0, 0.95, 0.75, 0.68, 0.5])
        width = HI_NM - LO_NM
        self.response_nm = np.array(
            [LO_NM, LO_NM + 0.1 * width, HI_NM - 0.1 * width, HI_NM]
        )
        self.response = np.array([0.0, 1.0, 1.0, 0.0])
        self.models = {
            "star": Transmission(self.star_nm, self.star),
            "zodi": Transmission(self.zodi_nm, self.zodi),
            "response": Transmission(self.response_nm, self.response),
        }
        knots = np.unique(
            np.concatenate((self.star_nm, self.zodi_nm, self.response_nm))
        )
        knots = knots[(knots >= LO_NM) & (knots <= HI_NM)]
        nodes, weights = leggauss(12)
        half = np.diff(knots)[:, None] / 2
        self.wave = (knots[:-1, None] + knots[1:, None]) / 2 + half * nodes
        self.width = half * weights

    def table(self, name, wavelength_nm):
        """Linear interpolation of one host table, typed with NumPy."""
        return np.interp(
            wavelength_nm,
            getattr(self, f"{name}_nm"),
            getattr(self, name),
        )

    def integrand(self, fn, host):
        """Response times a host table (or one) times fn, at any wavelengths."""

        def value(wavelength_nm):
            factor = 1.0 if host is None else self.table(host, wavelength_nm)
            return self.table("response", wavelength_nm) * factor * fn(wavelength_nm)

        return value

    def error(self, order, value, *declared):
        """Relative error of the rule with the named tables declared as breaks."""
        rule = Quadrature.build(
            (BAND,), *(self.models[name] for name in declared), order=order
        )
        estimate = float(rule.integrate(value(np.asarray(rule.nodes_nm)))[0])
        return abs(estimate / np.sum(self.width * value(self.wave)) - 1)


@pytest.fixture(scope="module")
def hosts():
    """Analytic and tabulated spectral references."""
    return Hosts()


@pytest.mark.float64_only(
    reason="quadrature convergence reference resolves residuals below float32 roundoff"
)
def test_accuracy_ladder_on_a_half_width_band(hosts):
    """Fails if the rule loses the accuracy measured before the build.

    The integrands are sampled at the nodes, with the knots of every linear
    table they hold declared as breaks. Spike basis, relative errors at orders
    4, 6 and 8: a 600 K thermal flux 5.8e-3, 3.8e-6, 1.0e-10; the zodi table
    times a cubic decline 1.1e-9, 1.4e-14, then the floating-point floor.
    Bounds allow a factor of about three, and each thermal order must gain at
    least a factor of 30. The line-blanketed star sampled at nodes that ignore
    its knots stays above 5e-3 at every order (measured 2.2e-2, 3.4e-2,
    1.3e-2): more nodes do not converge on a point sample. With its 0.1 nm
    knots declared the same integral is exact (floating-point, 1e-12;
    measured 3e-16).
    """
    heat = [
        hosts.error(
            order, hosts.integrand(lambda w: thermal(w, 600.0), None), "response"
        )
        for order in (4, 6, 8)
    ]
    zodi = [
        hosts.error(
            order,
            hosts.integrand(lambda w: (CENTER_NM / w) ** 3, "zodi"),
            "response",
            "zodi",
        )
        for order in (4, 6, 8)
    ]
    star = hosts.integrand(contrast, "star")
    sampled = [hosts.error(order, star, "response") for order in (4, 6, 8)]
    exact = hosts.error(4, star, "response", "star")
    assert heat[0] < 2e-2 and heat[1] < 1e-5 and heat[2] < 3e-10
    assert heat[1] < heat[0] / 30 and heat[2] < heat[1] / 30
    assert zodi[0] < 3e-9 and zodi[1] < 5e-14 and zodi[2] < 1e-14
    assert min(sampled) > 5e-3
    assert exact < 1e-12


SEPARATION_BAND = Band(540.0, 660.0)
CYCLES_PER_LOD = SEPARATION_BAND.center_nm * (
    1 / SEPARATION_BAND.low_nm - 1 / SEPARATION_BAND.high_nm
)


def airy_pixel(separation_lod, wavelength_nm):
    """Unocculted Airy intensity at a fixed sky angle, (pixels, wavelengths)."""
    center = SEPARATION_BAND.center_nm
    ratio = center / np.asarray(wavelength_nm)[None, :]
    x = np.pi * np.asarray(separation_lod)[:, None] * ratio
    return (2 * j1(x) / x) ** 2 * ratio**2


def test_order_rule_resolves_a_sliding_response_on_a_smooth_source():
    """Fails if the response-set order rule under-resolves a sliding pattern.

    One 20 percent piece (540 to 660 nm) read through an Airy pixel whose
    pattern slides by 0.5, 1, 2, 4 and 6 chromatic cycles across it (pixels
    within 0.1 cycle of each), on a 5800 K blackbody photon spectrum. Truth
    is a 16-point rule on 200 cells (matches 8 points on 4000 cells to 1e-15).
    Spike basis: the first order with a worst pixel below 1e-4 is 4, 6, 8, 12
    and 16 (pinned, so the ladder itself is checked over orders 4 to 24); the
    rule's own order (6, 8, 10, 16, 22) gives at most 1e-6 (measured at most
    2.9e-8). Mutation: today's order 4 at two cycles must exceed 1e-2
    (measured 1.6e-1).
    """
    x, w = leggauss(16)
    cells = np.linspace(SEPARATION_BAND.low_nm, SEPARATION_BAND.high_nm, 201)
    length = np.diff(cells)[:, None]
    fine = (cells[:-1, None] + length * (x + 1) / 2).ravel()
    fine_weights = (length * w / 2).ravel()

    def worst(rule, separations, truth):
        nodes = np.asarray(rule.nodes_nm[0], dtype=np.float64)
        weights = np.asarray(rule.weights()[0])
        value = airy_pixel(separations, nodes) @ (weights * thermal(nodes, 5800.0))
        return np.max(np.abs(value / truth - 1))

    onsets, rule_orders = [], []
    for cycles in (0.5, 1.0, 2.0, 4.0, 6.0):
        separations = np.linspace(cycles - 0.1, cycles + 0.1, 9) / CYCLES_PER_LOD
        truth = airy_pixel(separations, fine) @ (fine_weights * thermal(fine, 5800.0))
        errors = {
            order: worst(
                Quadrature.build((SEPARATION_BAND,), order=order),
                separations,
                truth,
            )
            for order in range(4, 25)
        }
        onsets.append(min(order for order, error in errors.items() if error < 1e-4))
        rule = Quadrature.build(
            (SEPARATION_BAND,), order=4, field_lod=cycles / CYCLES_PER_LOD
        )
        rule_orders.append(rule.pieces[0][0][2])
        assert worst(rule, separations, truth) < 1e-6
        if cycles == 2.0:
            assert errors[4] > 1e-2
    assert onsets == [4, 6, 8, 12, 16]
    assert rule_orders == [6, 8, 10, 16, 22]


def test_order_rule_on_a_rough_source_through_the_reduction(phoenix):
    """Fails if the rule's order loses a rough star read through the reduction.

    The ladder above on the PHOENIX 5800 K table, linear between its 0.04 nm
    knots, reduced to node-effective values by ``Reduction`` (the fine kind:
    the table at ``fine_nm``). Truth is a 16-point rule on the table cells
    split at 200 uniform cells. The reduction is exact to degree Q - 1 in the
    response for a rough source, against 2Q - 1 for a smooth one, so it needs
    about one more node per cycle. Spike basis: the first order below 1e-4 is
    5, 7, 10, 16 and 23 at 0.5, 1, 2, 4 and 6 cycles (pinned; an even-order
    scan gives 8 at one cycle); the rule's order gives 5.0e-6, 3.6e-6,
    1.9e-5, 3.8e-5 and 1.4e-4, bounded at 5e-4 (three times the worst). The
    smooth-source bound of 1e-6 must fail at every cycle count, which keeps
    the two rows from being merged.
    """
    knots, table = phoenix
    x, w = leggauss(16)
    edges = np.union1d(
        np.linspace(SEPARATION_BAND.low_nm, SEPARATION_BAND.high_nm, 201),
        knots[(knots > SEPARATION_BAND.low_nm) & (knots < SEPARATION_BAND.high_nm)],
    )
    length = np.diff(edges)[:, None]
    fine = (edges[:-1, None] + length * (x + 1) / 2).ravel()
    fine_weights = (length * w / 2).ravel()
    reduced = {}
    for order in range(4, 25):
        rule = Quadrature.build((SEPARATION_BAND,), order=order)
        reduction = Reduction.build(rule, knots, degree=1)
        at_fine = np.interp(np.asarray(reduction.fine_nm), knots, table)
        reduced[order] = (rule, np.asarray(reduction.apply(at_fine))[0])

    def worst(order, separations, truth):
        rule, node_values = reduced[order]
        nodes, weights = np.asarray(rule.nodes_nm[0]), np.asarray(rule.weights()[0])
        value = airy_pixel(separations, nodes) @ (weights * node_values)
        return np.max(np.abs(value / truth - 1))

    onsets, at_rule = [], []
    for cycles in (0.5, 1.0, 2.0, 4.0, 6.0):
        separations = np.linspace(cycles - 0.1, cycles + 0.1, 9) / CYCLES_PER_LOD
        truth = airy_pixel(separations, fine) @ (
            fine_weights * np.interp(fine, knots, table)
        )
        errors = {order: worst(order, separations, truth) for order in reduced}
        onsets.append(min(order for order, error in errors.items() if error < 1e-4))
        rule = Quadrature.build(
            (SEPARATION_BAND,), order=4, field_lod=cycles / CYCLES_PER_LOD
        )
        at_rule.append(errors[rule.pieces[0][0][2]])
    assert onsets == [5, 7, 10, 16, 23]
    assert max(at_rule) < 5e-4
    assert min(at_rule) > 1e-6


@pytest.mark.parametrize("order", [1, 3, 4])
def test_piece_without_cycles_keeps_its_order(order, precision):
    """Fails if a piece that holds no chromatic cycles changes its layout.

    Exact (floating-point identity): with ``field_lod=0`` and with
    ``field_lod=None`` and no model that declares a field, every piece keeps
    exactly ``order`` Gauss-Legendre nodes, typed here per piece, so a rule
    that rounds an odd order up fails at orders 1 and 3. A model that declares
    a field through ``spectral_field_lod`` raises the order of a piece and the
    largest declaration wins.
    """

    class Field:
        def __init__(self, lod):
            self.lod = lod

        def spectral_breaks_nm(self):
            return np.empty(0)

        def spectral_field_lod(self, band):
            return self.lod

    bands = (Band(1.0, 6.0), Band(2.0, 3.0), Band(6.0, 7.0))
    knots = (1.0, 2.0, 4.0, 6.0)
    pieces = (
        ((1.0, 2.0, order), (2.0, 4.0, order), (4.0, 6.0, order)),
        ((2.0, 3.0, order),),
        ((6.0, 7.0, order),),
    )
    for rule in (
        Quadrature.build(bands, order=order, breaks_nm=knots),
        Quadrature.build(bands, order=order, breaks_nm=knots, field_lod=0),
        Quadrature.build(
            bands, Field(0.0), order=order, breaks_nm=knots, field_lod=None
        ),
    ):
        assert rule.pieces == pieces
        for i, row in enumerate(pieces):
            nodes, weights = legendre_row(row)
            np.testing.assert_array_equal(
                rule.nodes_nm[i, : nodes.size], nodes.astype(precision)
            )
            np.testing.assert_array_equal(
                rule.weights()[i, : nodes.size], weights.astype(precision)
            )
            np.testing.assert_array_equal(rule.weights()[i, nodes.size :], 0.0)
    declared = Quadrature.build(
        bands, Field(0.5), Field(2.0), order=order, field_lod=None
    )
    assert declared.pieces == Quadrature.build(bands, order=order, field_lod=2.0).pieces
    assert declared.pieces[1][0][2] > order


def test_layout_floor_on_eleven_twenty_percent_bands():
    """Fails if a field radius removes a node or a piece edge of today's layout.

    Eleven 20 percent bands centered from 450 to 1000 nm with the Table 19
    knots of the local zodi as breaks. Today's layout, typed here from the
    knot count, has 4 nodes per piece (76 nodes). At every field radius each
    band keeps every one of today's edges and at least today's nodes. Spike
    basis for the totals at 12, 16, 22.6 and 32 lambda/D: 172, 198, 246 and
    304 nodes, and 132 padded renders today against 330 at the half-diagonal
    of a 128 by 128 detector at 0.25 lambda/D.
    """
    bands = tuple(Band(0.9 * c, 1.1 * c) for c in np.linspace(450.0, 1000.0, 11))
    today = []
    for band in bands:
        inside = TABLE19_NM[(TABLE19_NM > band.low_nm) & (TABLE19_NM < band.high_nm)]
        today.append(np.concatenate(([band.low_nm], inside, [band.high_nm])))
    totals = {}
    for field_lod in (0.0, 12.0, 16.0, 16.0 * math.sqrt(2.0), 32.0):
        rule = Quadrature.build(bands, breaks_nm=TABLE19_NM, field_lod=field_lod)
        counts = [sum(n for *_, n in row) for row in rule.pieces]
        totals[round(field_lod, 1)] = (sum(counts), rule.nodes_nm.size)
        for row, edges, count in zip(rule.pieces, today, counts, strict=True):
            assert set(edges) <= {lo for lo, _, _ in row} | {row[-1][1]}
            assert count >= 4 * (edges.size - 1)
            if field_lod == 0:
                assert [n for *_, n in row] == [4] * (edges.size - 1)
        np.testing.assert_array_equal(np.asarray(rule.active).sum(-1), counts)
    assert totals == {
        0.0: (76, 132),
        12.0: (172, 242),
        16.0: (198, 264),
        22.6: (246, 330),
        32.0: (304, 374),
    }


def test_rows_of_unequal_order_survive_padding_slicing_and_jit():
    """Fails if padding carries weight, a slice loses its layout, or jit differs.

    Two bands read out to 10 lambda/D, one with a declared break, give rows of
    unequal order. Each band of the joint rule must equal the rule built alone
    (same host arithmetic, rtol 1e-13), NaN on padding must not reach the
    integral, jit must match eager, and the layout digest must be static
    (readable inside a traced function), equal on a rebuild and distinct for
    a slice, a different order and a hand-made rule (None).

    Floating-point basis. Allow 16 operations for weights, pairwise
    contraction in both reduction orders and normalization by the band width.
    """
    bands = (BAND, Band(400.0, 500.0))
    rule = Quadrature.build(bands, order=4, breaks_nm=(550.0,), field_lod=10.0)
    counts = [sum(n for *_, n in row) for row in rule.pieces]
    assert counts[0] != counts[1] and rule.nodes_nm.shape == (2, max(counts))
    np.testing.assert_array_equal(np.asarray(rule.active).sum(-1), counts)
    np.testing.assert_array_equal(np.asarray(rule.weights())[~rule.active], 0.0)
    values = contrast(rule.nodes_nm)
    padded = jnp.where(rule.active, values, jnp.nan)
    seen = []

    def total(q, v):
        seen.append(q.layout_digest)
        return q.integrate(v)

    joint = jax.jit(total)(rule, padded)
    assert seen == [rule.layout_digest]
    np.testing.assert_allclose(
        joint, rule.integrate(values), rtol=float(ulps(16, 1.0)), atol=0
    )
    for i, band in enumerate(bands):
        row = rule.band(i)
        assert row.pieces == (rule.pieces[i],)
        alone = Quadrature.build((band,), order=4, breaks_nm=(550.0,), field_lod=10.0)
        assert alone.pieces == row.pieces
        expected = alone.integrate(contrast(alone.nodes_nm))
        np.testing.assert_allclose(
            joint[i], expected[0], rtol=float(ulps(16, 1.0)), atol=0
        )
        assert row.layout_digest != rule.layout_digest
    constant = jnp.full(rule.nodes_nm.shape, 2.5)
    np.testing.assert_allclose(
        rule.mean(constant), 2.5, rtol=float(ulps(16, 1.0)), atol=0
    )
    rebuilt = Quadrature.build(bands, order=4, breaks_nm=(550.0,), field_lod=10.0)
    assert (rule.digest, rule.layout_digest) == (
        rebuilt.digest,
        rebuilt.layout_digest,
    )
    other = Quadrature.build(bands, order=6, breaks_nm=(550.0,), field_lod=10.0)
    assert other.layout_digest != rule.layout_digest
    assert Quadrature(bands, rule.nodes_nm, rule.weights()).layout_digest is None


def test_invalid_rules_are_refused():
    """Fails if a negative weight, a broken layout or an invalid field is accepted."""
    with pytest.raises(ValueError, match="nonnegative"):
        Quadrature((BAND,), [[400.0, 600.0]], [[1.0, -1.0]])
    with pytest.raises(ValueError, match="match nodes_nm"):
        Quadrature((BAND,), [[400.0, 600.0]], [[1.0]])
    with pytest.raises(ValueError, match="span"):
        Quadrature((BAND,), [[500.0]], [[250.0]], [((LO_NM, 600.0, 1),)])
    with pytest.raises(ValueError, match="more nodes"):
        Quadrature((BAND,), [[500.0]], [[250.0]], [((LO_NM, HI_NM, 2),)])
    for field_lod in (-1.0, np.nan, (1.0, 2.0)):
        with pytest.raises(ValueError, match="field_lod"):
            Quadrature.build((BAND,), field_lod=field_lod)
