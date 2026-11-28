"""The commanded exposure event, its gates and the spectral sampling.

Every expected value is closed-form arithmetic typed here from the stated
gates, never a call to the helper under test. Times are multiples of 2**-20 d
where a test needs exact binary64 sums.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_instrument, make_linear_speckle, table_coronagraph
from hwoutils.epoch import Epoch
from hwoutils.leaves import check_tree
from hwoutils.precision import PrecisionError
from hwoutils.provenance import StampMismatchError
from hwoutils.testing import ulps

import optixstuff as ox
from optixstuff.sampling import build_facts

EPOCH = Epoch(2460000.5, 0.0, "tdb")
# The same instant as EPOCH on a different day zero: EPOCH is 1000 d after SHIFTED.
SHIFTED = Epoch(2460000.5, -1000.0, "tdb")
BANDS = (ox.Band(480.0, 520.0), ox.Band(540.0, 600.0))
K = 3 * 2.0**-20


def event(time_d=5.0 + K, *, epoch=EPOCH, plan=None, bands=BANDS, **kwargs):
    """A two-band event on EPOCH with no scene and no observatory."""
    return ox.Exposure(
        time_d,
        kwargs.pop("pointing", ox.Pointing(0.0)),
        plan,
        epoch=epoch,
        instrument=kwargs.pop("instrument", "eac1"),
        bands=bands,
    )


def test_event_fields_are_a_leaf_and_static_metadata(precision):
    """Fails if the event holds a build product, or a static field becomes a leaf.

    Exact. ``time_d`` is a strong leaf of the active dtype; the epoch, the
    configuration name and the bands are static, absent from the leaves, and the
    event builds with no scene and no observatory. A bad name, an empty band
    tuple and a non-epoch are refused.
    """
    e = event()
    assert e.time_d.dtype == precision and not e.time_d.weak_type
    leaves = jax.tree.leaves(e)
    assert len(leaves) == 2 and e.time_d in leaves and e.pointing.roll_deg in leaves
    assert check_tree(e) == ()
    for static in (e.epoch, e.instrument, e.bands):
        assert not any(leaf is static for leaf in leaves)
    assert isinstance(e.bands, tuple)
    assert not hasattr(e, "quadrature") and not hasattr(e, "resolution")
    listed = ox.Exposure(
        1.0, ox.Pointing(0.0), epoch=EPOCH, instrument="a-b_1", bands=list(BANDS)
    )
    assert listed.bands == BANDS
    for bad in (
        dict(instrument="two words"),
        dict(instrument=""),
        dict(instrument=3),
        dict(bands=()),
        dict(bands=(500.0,)),
    ):
        with pytest.raises(ValueError, match=r"instrument name|bands"):
            event(**bad)
    with pytest.raises(TypeError, match="Epoch"):
        event(epoch=2460000.5)
    with pytest.raises(TypeError):
        ox.Exposure(1.0, ox.Pointing(0.0), instrument="eac1", bands=BANDS)


def _on_shifted(time_d, source=EPOCH, target=SHIFTED):
    return event(time_d, epoch=source).on_epoch(target)


@pytest.mark.float64_only(reason="adds the shift in binary64 on a float64 leaf")
def test_on_epoch_adds_the_shift_exactly_and_round_trips():
    """Fails if the shift is subtracted, rounded, or the instant moves.

    Closed form on exactly representable days: ``5 + 3 * 2**-20`` plus the
    1000 d between the epochs is exact in binary64, so the comparison is
    exact, the way back restores the leaf, and two events for one instant on
    shifted epochs give one Julian date. Equal epochs return the event itself.
    """
    e = event()
    assert e.on_epoch(e.epoch) is e
    moved = e.on_epoch(SHIFTED)
    assert moved.epoch is SHIFTED
    assert float(moved.time_d) == 5.0 + K + 1000.0
    assert float(moved.on_epoch(EPOCH).time_d) == float(e.time_d)
    assert moved.epoch.jd(moved.time_d) == e.epoch.jd(e.time_d)
    assert moved.bands == e.bands and moved.instrument == e.instrument
    assert moved.time_d.dtype == jnp.float64
    host = e.on_epoch(SHIFTED)
    np.testing.assert_array_equal(host.time_d, jnp.float64(5.0 + K + 1000.0))
    batched = event(jnp.array([1.0, 2.0 + K]))
    np.testing.assert_array_equal(
        batched.on_epoch(SHIFTED).time_d, np.array([1001.0, 1002.0 + K])
    )


def test_on_epoch_for_one_instant_on_a_different_split_changes_only_the_epoch(
    precision,
):
    """Fails if a split-equivalent epoch is returned stale, or the leaf is touched.

    Two epochs for one instant with different splits (``jd1 + jd2`` of
    2460000.5 + 0.25 and 2460000.75 + 0) are unequal metadata. The event comes
    back on the other epoch with a bit-identical leaf in either precision. A
    scale mismatch raises. Exact.
    """
    first, second = Epoch(2460000.5, 0.25), Epoch(2460000.75, 0.0)
    assert first != second and first.days_between(second) == 0.0
    e = event(5.0 + K, epoch=first)
    moved = e.on_epoch(second)
    assert moved is not e and moved.epoch is second
    np.testing.assert_array_equal(moved.time_d, e.time_d)
    assert moved.time_d.dtype == precision
    with pytest.raises(ValueError, match="scale"):
        e.on_epoch(Epoch(2460000.75, 0.0, "utc"))
    with pytest.raises(ValueError, match="scale"):
        e.on_epoch(Epoch(3.0, 0.0, "utc"))


@pytest.mark.float32_only(reason="a float32 leaf cannot carry a nonzero shift")
def test_on_epoch_refuses_a_narrow_leaf_in_every_context():
    """Fails if a float32 leaf is shifted in float32, eagerly or inside a transform.

    Exact. A nonzero shift of a float32 time raises ``PrecisionError`` eagerly,
    at trace time under jit, under vmap and inside grad, and the message gives
    the remedy; the remedy (restating the host date on the target epoch)
    builds the event directly.
    """
    e = event()
    with pytest.raises(PrecisionError, match="original host date"):
        e.on_epoch(SHIFTED)
    with pytest.raises(PrecisionError):
        eqx.filter_jit(lambda x: x.on_epoch(SHIFTED).time_d)(e)
    with pytest.raises(PrecisionError):
        jax.vmap(lambda t: event(t).on_epoch(SHIFTED).time_d)(jnp.arange(3.0))
    with pytest.raises(PrecisionError):
        jax.grad(lambda t: event(t).on_epoch(SHIFTED).time_d)(1.0)
    jd = 2460005.5 + K
    restated = event(SHIFTED.days(jd=jd), epoch=SHIFTED)
    assert restated.epoch.jd(np.float64(restated.time_d)) == pytest.approx(
        jd, abs=2.0e-4
    )
    assert float(restated.time_d) == np.float32(SHIFTED.days(jd=jd))


@pytest.mark.float64_only(reason="a float64 tracer is the only traced shift allowed")
def test_on_epoch_adds_on_the_device_for_a_float64_tracer():
    """Fails if a float64 tracer is refused, or its time gradient is not one.

    Closed form: the shifted time is ``t + 1000``, with derivative one. Exact.
    """
    out = jax.jit(lambda t: event(t).on_epoch(SHIFTED).time_d)(jnp.float64(5.0))
    assert out == 1005.0 and out.dtype == jnp.float64
    assert (
        jax.grad(lambda t: event(t).on_epoch(SHIFTED).time_d)(jnp.float64(5.0)) == 1.0
    )


def test_on_epoch_refuses_a_scale_mismatch(precision):
    """Fails if a mismatched time scale is converted silently. Exact."""
    with pytest.raises(ValueError, match="scale"):
        event().on_epoch(Epoch(2460000.5, 0.0, "utc"))
    with pytest.raises(ValueError, match="scale"):
        jax.jit(lambda t: event(t).on_epoch(Epoch(2460000.5, 1.0, "tt")).time_d)(1.0)


# Gates of two 10 s frames and a 4 s read that does not overlap: [0, 10], [14, 24].
PLAN = ox.FramePlan(10.0, 2)


def test_gates_with_a_read_between_frames(precision):
    """Fails if the span omits the last frame or the midpoint is half the occupied time.

    Closed form from the stated gates ``[0, 10]`` and ``[14, 24]``: span 24 s and
    midpoint 12 s, not 14 s (half of the 28 s occupied time). One frame gives
    half the frame time. ``ulps(2, expected)`` for the algebra.
    """
    for read in (4.0, jnp.array(4.0)):
        span = PLAN.collection_span_s(read_time_s=read, overlaps_read=False)
        mid = PLAN.collection_midpoint_s(read_time_s=read, overlaps_read=False)
        np.testing.assert_allclose(span, 24.0, rtol=float(ulps(2, 1.0)), atol=0)
        np.testing.assert_allclose(mid, 12.0, rtol=float(ulps(2, 1.0)), atol=0)
        assert mid.dtype == precision
    detector = ox.Detector(shape=(2, 2), pixel_scale_arcsec=0.01, read_time_s=4.0)
    assert float(PLAN.occupied_time_s(detector)) == 28.0
    single = ox.FramePlan(10.0, 1)
    np.testing.assert_allclose(
        single.collection_midpoint_s(read_time_s=4.0, overlaps_read=False), 5.0
    )


def test_gates_when_the_read_overlaps_collection():
    """Fails if an overlapped read still separates the gates.

    Closed form: gates ``[0, 10]`` and ``[10, 20]``, span 20 s, midpoint 10 s,
    whatever the read time. ``ulps(2, expected)``.
    """
    for read in (4.0, 0.0):
        span = PLAN.collection_span_s(read_time_s=read, overlaps_read=True)
        mid = PLAN.collection_midpoint_s(read_time_s=read, overlaps_read=True)
        np.testing.assert_allclose(span, 20.0, rtol=float(ulps(2, 1.0)), atol=0)
        np.testing.assert_allclose(mid, 10.0, rtol=float(ulps(2, 1.0)), atol=0)


def test_no_collection_has_no_midpoint(precision):
    """Fails if zero frames or a NaN count return a number, or an instant a NaN.

    Exact. Zero frames and a NaN count give NaN span and midpoint for both gate
    cases; ``plan=None`` is an instant with a zero offset of the dtype of
    ``time_d``.
    """
    for count in (0, np.nan):
        plan = ox.FramePlan(10.0, count)
        for overlaps in (False, True):
            for helper in (plan.collection_span_s, plan.collection_midpoint_s):
                assert np.isnan(helper(read_time_s=4.0, overlaps_read=overlaps))
        assert np.isnan(
            event(plan=plan).collection_midpoint_s(read_time_s=4.0, overlaps_read=False)
        )
    instant = event().collection_midpoint_s(read_time_s=4.0, overlaps_read=False)
    assert instant == 0.0 and instant.dtype == precision and instant.shape == ()


def test_per_band_plans_give_one_midpoint_per_band():
    """Fails if bands share a midpoint, a plan of another band count is accepted,
    or integer counts lose their value.

    Closed form per band: frame times 10 and 20 s with counts 2 and 3 and a 4 s
    read give spans ``1 * 14 + 10 = 24`` and ``2 * 24 + 20 = 68`` s. A count of
    ``2**24 + 1`` stays an exact integer in the plan and is within half a float32
    ulp of ``n / 2`` at the midpoint. ``ulps(2, expected)``.
    """
    plan = ox.FramePlan([10.0, 20.0], [2, 3])
    assert jnp.issubdtype(plan.n_frames.dtype, jnp.integer)
    mid = event(plan=plan).collection_midpoint_s(read_time_s=4.0, overlaps_read=False)
    np.testing.assert_allclose(mid, [12.0, 34.0], rtol=float(ulps(2, 1.0)), atol=0)
    with pytest.raises(ValueError, match="bands"):
        event(plan=ox.FramePlan([10.0, 20.0, 30.0], 2))
    with pytest.raises(ValueError, match="bands"):
        event(plan=ox.FramePlan(10.0, [1, 2, 3]))
    big = ox.FramePlan(1.0, 2**24 + 1)
    assert int(big.n_frames) == 2**24 + 1
    got = big.collection_midpoint_s(read_time_s=0.0, overlaps_read=True)
    np.testing.assert_allclose(got, (2**24 + 1) / 2, rtol=float(ulps(2, 1.0)), atol=0)


def test_midpoint_under_transforms_when_the_read_does_not_overlap():
    """Fails if the midpoint breaks under jit, vmap or grad, or its slopes are wrong.

    Closed form ``((n - 1) * (t + r) + t) / 2``: the derivative is ``n / 2`` in the
    frame time and ``(n - 1) / 2`` in the read time. ``ulps(2, expected)``.
    """
    n = 5

    def mid(t, r):
        plan = ox.FramePlan(t, n)
        return plan.collection_midpoint_s(read_time_s=r, overlaps_read=False)

    tol = dict(rtol=float(ulps(2, 1.0)), atol=0)
    np.testing.assert_allclose(jax.jit(mid)(10.0, 4.0), (4 * 14 + 10) / 2, **tol)
    times = jnp.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        jax.vmap(mid, in_axes=(0, None))(times, 4.0),
        (4 * (times + 4) + times) / 2,
        **tol,
    )
    np.testing.assert_allclose(jax.grad(mid, argnums=0)(10.0, 4.0), n / 2, **tol)
    np.testing.assert_allclose(jax.grad(mid, argnums=1)(10.0, 4.0), (n - 1) / 2, **tol)


def test_midpoint_under_transforms_when_the_read_overlaps():
    """Fails if an overlapped midpoint depends on the read or loses its frame slope.

    Closed form ``n * t / 2``: the derivative is ``n / 2`` in the frame time
    and zero in the read time. ``ulps(2, expected)``.
    """
    n = 5

    def mid(t, r):
        plan = ox.FramePlan(t, n)
        return plan.collection_midpoint_s(read_time_s=r, overlaps_read=True)

    tol = dict(rtol=float(ulps(2, 1.0)), atol=0)
    np.testing.assert_allclose(jax.jit(mid)(10.0, 4.0), n * 10.0 / 2, **tol)
    times = jnp.array([10.0, 20.0])
    np.testing.assert_allclose(
        jax.vmap(mid, in_axes=(0, None))(times, 4.0), n * times / 2, **tol
    )
    np.testing.assert_allclose(jax.grad(mid, argnums=0)(10.0, 4.0), n / 2, **tol)
    assert jax.grad(mid, argnums=1)(10.0, 4.0) == 0.0


def test_events_stack_only_with_equal_static_fields():
    """Fails if a static field is a leaf or events of another layout share a structure.

    Pytree structure, exact. Two events with equal static fields stack with
    ``jax.tree.map(jnp.stack)`` and map under vmap; a different band tuple,
    configuration name or epoch is another tree structure.
    """
    first, second = event(1.0), event(2.0, pointing=ox.Pointing(30.0))
    stacked = jax.tree.map(lambda *x: jnp.stack(x), first, second)
    assert stacked.time_d.shape == (2,) and stacked.pointing.roll_deg.shape == (2,)
    np.testing.assert_array_equal(
        jax.vmap(lambda e: e.time_d + e.pointing.roll_deg)(stacked), [1.0, 32.0]
    )
    structure = jax.tree.structure(first)
    assert structure == jax.tree.structure(second)
    for other in (
        event(1.0, bands=BANDS[::-1]),
        event(1.0, bands=BANDS[:1]),
        event(1.0, instrument="other"),
        event(1.0, epoch=SHIFTED),
    ):
        assert jax.tree.structure(other) != structure
    with pytest.raises(ValueError):
        jax.tree.map(lambda *x: x, first, event(1.0, instrument="other"))


def test_pointing_offsets_are_structural_and_the_sign_anchor_holds():
    """Fails if a None offset equals a zero offset in structure, or the roll sign flips.

    Structure and sky-to-detector arithmetic, as released: at roll 90 degrees a
    one pixel offset along x moves a source at the target by one pixel along
    +x; a None and a zero offset agree on the detector offsets, exact.
    """
    bare = event(pointing=ox.Pointing(30.0, None))
    zero = event(pointing=ox.Pointing(30.0, jnp.zeros(2)))
    assert jax.tree.structure(bare) != jax.tree.structure(zero)
    east, north = jnp.array([0.1, -0.2]), jnp.array([0.3, 0.05])
    for a, b in zip(
        bare.pointing.detector_offsets_arcsec(east, north),
        zero.pointing.detector_offsets_arcsec(east, north),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    one_pixel_mas = 0.05 * 1000.0
    rolled = event(pointing=ox.Pointing(90.0, (one_pixel_mas, 0.0)))
    x, y = jax.jit(lambda e: e.pointing.detector_offsets_arcsec(0.0, 0.0))(rolled)
    np.testing.assert_allclose([x, y], [0.05, 0.0], rtol=float(ulps(4, 1.0)), atol=0)


def _tracer_rule(row):
    return ox.Quadrature(BANDS, row[None, None] * jnp.ones((2, 3)), jnp.ones((2, 3)))


def test_sampling_holds_a_resolution_for_its_own_quadrature():
    """Fails if the layout guard is removed or a hand-built rule is refused.

    A resolution built for another layout raises at construction; a hand-built
    rule with the empty resolution constructs, also from tracers inside jit and
    scan. Exact.
    """
    band = ox.Band(480.0, 528.0)
    rule = ox.Quadrature.build((band,), order=3)
    other = ox.Quadrature.build((band,), order=4)
    with pytest.raises(ValueError, match="another quadrature"):
        ox.SpectralSampling(other, ox.Resolution.build(rule))
    assert ox.SpectralSampling(rule, ox.Resolution.build(rule)).bands == (band,)
    hand = ox.SpectralSampling(ox.Quadrature((band,), rule.nodes_nm, rule.weights_nm))
    assert hand.built_at == () and hand.reads == () and hand.resolution.keys == ()
    assert jax.jit(lambda r: ox.SpectralSampling(_tracer_rule(r)).quadrature.nodes_nm)(
        2.0
    ).shape == (2, 3)
    scanned = jax.lax.scan(
        lambda c, r: (
            c,
            ox.SpectralSampling(_tracer_rule(r)).quadrature.integrate(jnp.ones((2, 3))),
        ),
        0.0,
        jnp.array([1.0, 2.0]),
    )[1]
    np.testing.assert_array_equal(scanned, np.full((2, 2), 3.0))
    for bad in ("3", "9scene.x", "bad owner.x", 3):
        with pytest.raises(ValueError, match="owner-qualified"):
            ox.SpectralSampling(hand.quadrature, reads=(bad,))
    ok = ox.SpectralSampling(
        hand.quadrature, reads=["scene.star", "observatory.telescope"]
    )
    assert ok.reads == ("scene.star", "observatory.telescope")
    with pytest.raises(ValueError, match="sorted"):
        ox.SpectralSampling(hand.quadrature, built_at=(("b", "x"), ("a", "y")))


class _Table:
    """Static digest facts of a tabulated source."""

    def __init__(self, digest, degree=1):
        self.knot_digest, self.spectral_degree = digest, degree


def _view(pupil="p1", child_pupils=None, breaks=(540.0,), lod=9.0):
    """Duck-typed selected view with the reads the checks use."""
    if child_pupils is None:
        coronagraph = SimpleNamespace(pupil_id="c1")
    else:
        coronagraph = SimpleNamespace(
            coronagraphs=tuple(SimpleNamespace(pupil_id=c) for c in child_pupils)
        )
    return SimpleNamespace(
        pupil=SimpleNamespace(pupil_id=pupil),
        coronagraph=coronagraph,
        spectral_breaks_nm=lambda: np.asarray(breaks),
        spectral_field_lod=lambda band: lod,
    )


def _scene(*digests):
    return SimpleNamespace(spectral_tables=lambda: tuple(_Table(d) for d in digests))


def _built(view=None, scene=None, **kwargs):
    rule = ox.Quadrature.build(BANDS, order=3)
    stamps = build_facts(BANDS, instrument="eac1", selected=view, scene=scene, **kwargs)
    return ox.SpectralSampling(rule, built_at=stamps), rule


def test_check_event_compares_ordered_bands_and_the_configuration():
    """Fails if bands compare as a set, a known instrument mismatch passes, or
    a hand-built sampling is refused.

    Exact. Equal ordered bands pass with nothing unverified; reordered bands of
    unequal node count raise; a known different ``instrument`` stamp raises
    ``StampMismatchError``; a hand-built sampling is admitted with
    ``("instrument",)`` unverified.
    """
    sampling, _ = _built()
    assert sampling.check_event(event()) == ()
    with pytest.raises(StampMismatchError, match="instrument"):
        sampling.check_event(event(instrument="other"))
    with pytest.raises(ValueError, match="bands"):
        sampling.check_event(event(bands=BANDS[:1]))
    uneven = ox.Quadrature.build(BANDS, order=3, breaks_nm=(550.0,), field_lod=10.0)
    assert len({int(row.sum()) for row in np.asarray(uneven.active)}) == 2
    with pytest.raises(ValueError, match="bands"):
        ox.SpectralSampling(uneven).check_event(event(bands=BANDS[::-1]))
    hand = ox.SpectralSampling(ox.Quadrature.build(BANDS, order=3))
    assert hand.check_event(event()) == ("instrument",)
    assert hand.check_event(event(), selected=_view(), scene=_scene("k")) == tuple(
        sorted(
            (
                "instrument",
                "telescope_pupil_id",
                "coronagraph_pupil_id",
                "breaks_nm",
                "field_lod",
                "scene_layout",
            )
        )
    )


def test_check_event_recomputes_the_hardware_and_scene_facts():
    """Fails if a hardware or scene change is compared by name only.

    Exact. A different pupil identifier, a same-name swap with another filter
    break, another derived field radius and a changed scene each raise naming
    the label; unknown on either side is unverified; with no scene, the
    recorded scene layout is unverified rather than compared.
    """
    view, scene = _view(), _scene("k1", "k2")
    sampling, _ = _built(view, scene)
    assert sampling.check_event(event(), selected=view, scene=scene) == ()
    assert sampling.check_event(event(), selected=view) == ("scene_layout",)
    assert sampling.check_event(event(), scene=scene) == (
        "breaks_nm",
        "coronagraph_pupil_id",
        "field_lod",
        "telescope_pupil_id",
    )
    for changed, label in (
        (_view(pupil="p2"), "telescope_pupil_id"),
        (_view(breaks=(545.0,)), "breaks_nm"),
        (_view(lod=8.0), "field_lod"),
    ):
        with pytest.raises(StampMismatchError, match=label):
            sampling.check_event(event(), selected=changed, scene=scene)
    with pytest.raises(StampMismatchError, match="scene_layout"):
        sampling.check_event(event(), selected=view, scene=_scene("k1", "other"))
    unknown_pupil = _view(pupil=None)
    assert "telescope_pupil_id" in sampling.check_event(
        event(), selected=unknown_pupil, scene=scene
    )
    blind, _ = _built(_view(pupil=None), scene)
    assert "telescope_pupil_id" in blind.check_event(
        event(), selected=view, scene=scene
    )


def test_check_event_stamps_an_explicit_field_override_and_reads_a_multiband_view():
    """Fails if an explicit override is compared to the derived radius, or a
    multiband coronagraph shares one identifier.

    Exact. An explicit ``field_lod`` is stamped with its source and is not
    compared to the view (it is returned unverified); each child of a multiband
    coronagraph is stamped under its own label and compared.
    """
    view = _view(lod=9.0)
    sampling, _ = _built(view, field_lod=(2.5, 3.5))
    recorded = dict(sampling.built_at)
    assert recorded["field_lod_source"] == "str:explicit"
    assert (
        recorded["field_lod"]
        != dict(build_facts(BANDS, instrument="eac1", selected=view))["field_lod"]
    )
    assert "field_lod" in sampling.check_event(event(), selected=_view(lod=1.0))
    multi = _view(child_pupils=("a", "b"))
    sampling, _ = _built(multi)
    assert dict(sampling.built_at).keys() >= {
        "child0.coronagraph_pupil_id",
        "child1.coronagraph_pupil_id",
    }
    assert sampling.check_event(event(), selected=multi) == ()
    with pytest.raises(StampMismatchError, match=r"child1\.coronagraph_pupil_id"):
        sampling.check_event(event(), selected=_view(child_pupils=("a", "z")))


def test_check_event_accepts_a_sampling_built_from_the_selected_view():
    """Fails if a rule built from a configuration is refused by its own view.

    ``Quadrature.build(exposure.bands, observatory.select(exposure.instrument))``
    wrapped in a sampling with the view's facts passes ``check_event`` with the
    breaks and field radius verified, and a view of another telescope pupil
    raises, exact.
    """
    observatory = make_instrument(table_coronagraph())
    exposure = event(instrument="imager")
    selected = observatory.select(exposure.instrument)
    sampling = ox.SpectralSampling(
        ox.Quadrature.build(exposure.bands, selected),
        built_at=build_facts(exposure.bands, instrument="imager", selected=selected),
    )
    unverified = sampling.check_event(exposure, selected=selected)
    assert "breaks_nm" not in unverified and "field_lod" not in unverified
    assert "instrument" not in unverified


def test_sampling_nodes_are_live_leaves():
    """Fails if the nodes are static or stopped for gradients.

    Closed form: for ``sum(integrate(a * nodes + b))`` the derivative with
    respect to each node is ``a * weight``, zero on padded nodes, and the
    weights' derivative is the integrand. ``ulps(4, expected)``.
    """
    rule = ox.Quadrature.build(BANDS, order=3, breaks_nm=(550.0,), field_lod=10.0)
    assert not bool(np.all(np.asarray(rule.active)))
    sampling = ox.SpectralSampling(rule)
    a, b = 0.37, 1.5

    def total(s):
        q = s.quadrature
        return jnp.sum(q.integrate(a * q.nodes_nm + b))

    grad = jax.grad(total)(sampling).quadrature
    np.testing.assert_allclose(
        grad.nodes_nm, a * rule.weights_nm, rtol=float(ulps(4, 1.0)), atol=0
    )
    assert np.all(np.asarray(grad.nodes_nm)[~np.asarray(rule.active)] == 0.0)
    assert len(jax.tree.leaves(sampling)) == 2


def test_hardware_clock_reads_the_event_on_the_field_epoch(precision):
    """Fails if a field reads a scene day, or two events for one instant disagree.

    Closed form on exactly representable days: events for one instant stated on
    the field epoch and on an epoch ten days later give equal elapsed seconds
    in binary64, and a float32 event on the shifted epoch raises. The scene
    translation is a different number. A field built without an epoch, or with
    a non-epoch, raises ``TypeError``.
    """
    field_epoch = Epoch(2460000.5, 0.0)
    later = Epoch(2460000.5, 10.0)
    scene_epoch = Epoch(2460000.5, 3.0)
    field = make_linear_speckle(epoch_d=0.125, epoch=field_epoch)
    time_d = 1.0 + K
    on_field = event(time_d, epoch=field_epoch)
    on_later = event(time_d - 10.0, epoch=later)
    if precision == np.float64:
        first = field.elapsed_s(on_field.on_epoch(field.epoch).time_d)
        second = field.elapsed_s(on_later.on_epoch(field.epoch).time_d)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, (time_d - 0.125) * 86400.0)
        scene_days = on_field.on_epoch(scene_epoch).time_d
        assert field.elapsed_s(scene_days) != first
    else:
        with pytest.raises(PrecisionError):
            on_later.on_epoch(field.epoch)
        np.testing.assert_array_equal(
            on_field.on_epoch(field.epoch).time_d, on_field.time_d
        )
    fields = {f.name: getattr(field, f.name) for f in dataclasses.fields(field)}
    fields.pop("epoch")
    with pytest.raises(TypeError):
        type(field)(**fields)
    with pytest.raises(TypeError, match="Epoch"):
        type(field)(**fields, epoch=2460000.5)
    assert not any(leaf is field.epoch for leaf in jax.tree.leaves(field))
