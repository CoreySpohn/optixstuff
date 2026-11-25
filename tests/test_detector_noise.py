"""Code verification of frame accounting, electron moments, and draws."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optixstuff as ox
from optixstuff.spectral import Band

BANDS = (Band(450, 500), Band(500, 550))


def test_frames_for_time(precision):
    """Fails if tolerance rounds down, subframes vanish, or JIT casts to integer.

    NumPy products in the working dtype give exact counts from the first
    covering candidate. The discrete count derivative is exactly zero.
    """
    dtype = precision
    integer = np.array(3.0, dtype=dtype)
    ratios = np.array(
        [
            0.0,
            0.1,
            1.0,
            3.0,
            np.nextafter(integer, -np.inf, dtype=dtype),
            np.nextafter(integer, np.inf, dtype=dtype),
        ],
        dtype=np.dtype(dtype),
    )
    plan = eqx.filter_jit(ox.frames_for_time)(
        jnp.array(ratios), jnp.array(1.0, dtype=dtype)
    )
    np.testing.assert_array_equal(plan.n_frames, [0, 1, 1, 3, 3, 4])
    assert jnp.issubdtype(plan.n_frames.dtype, jnp.floating)
    rng = np.random.default_rng(0)
    frame = rng.uniform(0.01, 100.0, 2000)
    duration = rng.uniform(0.0, 1000.0, 2000) * frame
    duration[:1000] = (
        rng.integers(1, 1000, 1000)
        * frame[:1000]
        * (1 + rng.uniform(-2e-12, 2e-12, 1000))
    )
    frame, duration = frame.astype(dtype), duration.astype(dtype)
    ceiling = np.ceil(duration / frame)
    candidates = ceiling[:, None] + np.array([-1.0, 0.0, 1.0], dtype=dtype)
    covers = (candidates >= 1) & (candidates * frame[:, None] >= duration[:, None])
    expected = np.min(np.where(covers, candidates, np.inf), axis=1)
    np.testing.assert_array_equal(
        ox.frames_for_time(duration, frame).n_frames, expected
    )
    derivative = jax.vmap(jax.grad(lambda t: ox.frames_for_time(t, 1.0).n_frames))
    np.testing.assert_array_equal(derivative(jnp.array(ratios)), 0.0)

    with pytest.raises(ValueError, match=r"scalar.*\(W,\)"):
        ox.frames_for_time(jnp.ones((2, 1)), frame)
    with pytest.raises(ValueError, match=r"scalar.*\(W,\)"):
        ox.frames_for_time(jnp.ones(1), jnp.ones(2))


@pytest.mark.parametrize(
    "make_plan", [ox.frames_for_time, eqx.filter_jit(ox.frames_for_time)]
)
def test_frames_for_invalid_and_zero_time(make_plan):
    """Exact frame-plan rules catch invalid requests reported as zero frames."""
    times = jnp.array([jnp.nan, jnp.inf, -jnp.inf, -1.0, 0.0])
    plan = make_plan(times, jnp.array(1.0))
    expected = [np.nan, np.nan, np.nan, np.nan, 0.0]
    np.testing.assert_array_equal(plan.n_frames, expected)
    np.testing.assert_array_equal(plan.live_time_s, expected)


@pytest.mark.parametrize(
    ("dtype", "time_s", "frame_time_s", "expected"),
    [
        pytest.param(
            jnp.float32,
            655360.25,
            1.0,
            655361,
            marks=pytest.mark.float32_only(
                reason="float32 frame count near a coarse spacing"
            ),
        ),
        pytest.param(
            jnp.float64,
            0.3,
            0.1,
            3,
            marks=pytest.mark.float64_only(
                reason="float64 products land on the frame-count boundary"
            ),
        ),
        pytest.param(
            jnp.float64,
            0.7,
            0.1,
            7,
            marks=pytest.mark.float64_only(
                reason="float64 products land on the frame-count boundary"
            ),
        ),
    ],
)
def test_frames_cover_the_request_in_the_working_dtype(
    dtype, time_s, frame_time_s, expected
):
    """Hand values give exact counts and catch tolerant or division-only rounding."""
    request = jnp.asarray(time_s, dtype=dtype)
    plan = ox.frames_for_time(request, jnp.asarray(frame_time_s, dtype=dtype))
    assert plan.n_frames == expected
    assert plan.live_time_s >= request
    assert (plan.n_frames - 1) * plan.frame_time_s < request


def test_frames_cover_a_request_past_the_float32_integer_range():
    """Exact float32 products catch a count step that cannot advance past 2**24.

    3963.617431640625 s at 0.00021 s needs more than 2**24 frames, where adding
    one frame rounds back to the same float32 count; the plan must still cover
    the request and its next lower representable count must not.
    """
    request = np.float32(3963.617431640625)
    frame = np.float32(0.00021)
    for make_plan in (ox.frames_for_time, eqx.filter_jit(ox.frames_for_time)):
        plan = make_plan(jnp.float32(request), jnp.float32(frame))
        count = np.float32(plan.n_frames)
        assert count > 2**24
        assert count * frame >= request
        assert np.nextafter(count, np.float32(-np.inf)) * frame < request


def test_frame_plan_validates_concrete_values_only():
    """Exact value checks catch invalid concrete fields and validation of tracers."""
    for frame_time_s, n_frames, field in (
        (-1, 1, "frame_time_s"),
        (0, 1, "frame_time_s"),
        (np.inf, 1, "frame_time_s"),
        (1 + 1j, 1, "frame_time_s"),
        (1, 2.5, "n_frames"),
        (1, -1, "n_frames"),
        (1, np.inf, "n_frames"),
        (1, 1 + 1j, "n_frames"),
    ):
        with pytest.raises(ValueError, match=field):
            ox.FramePlan(frame_time_s, n_frames)
    assert np.isnan(ox.FramePlan(1, np.nan).n_frames)
    plan = jax.jit(ox.FramePlan)(-1.0, 2.5)
    assert plan.frame_time_s == -1
    assert plan.n_frames == 2.5


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
def test_frame_occupancy(detector_cls):
    """Fails if read overlap changes live time or is ignored; exact algebra."""
    plan = ox.FramePlan(3.0, 4.0)
    for overlap, expected in [(False, 20.0), (True, 12.0)]:
        det = detector_cls(
            pixel_scale_arcsec=0.01,
            shape=(2, 3),
            read_time_s=2.0,
            overlaps_read=overlap,
        )
        assert plan.live_time_s == 12.0
        assert plan.occupied_time_s(det) == expected


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
def test_moment_contract(detector_cls):
    """Fails if detector terms or band reduction are misplaced; 16 float32 eps.

    The full detector has every term nonzero; the ideal has source only.
    """
    kwargs = (
        {}
        if detector_cls is ox.IdealDetector
        else dict(
            dark_current_e_per_pixel_s=0.2,
            clock_induced_charge_e_per_pixel_frame=0.3,
            read_noise_e_rms=0.4,
            excess_noise_factor_sq=2.0,
        )
    )
    det = detector_cls(pixel_scale_arcsec=0.01, shape=(2, 3), **kwargs)
    rates = ox.Rates(
        BANDS,
        "electron",
        planet=jnp.arange(1.0, 13.0).reshape((2, 2, 3)),
        thermal=jnp.full((2, 2, 3), 2.0),
        n_pix=5.0,
    )
    plan = ox.FramePlan(3.0, 4.0)
    n_pix = 5.0
    source = np.arange(1.0, 13.0).reshape((2, 2, 3)) + 2.0
    dark, cic, rn, f2 = (
        (0.0, 0.0, 0.0, 1.0)
        if detector_cls is ox.IdealDetector
        else (0.2, 0.3, 0.4, 2.0)
    )
    expected_mean = (source + n_pix * dark) * 12.0 + n_pix * cic * 4.0
    expected_var = f2 * expected_mean + n_pix * rn**2 * 4.0
    eager = det.mean_and_variance(rates, plan)
    old_mean = (
        rates.total() + n_pix * det.dark_current_e_per_pixel_s
    ) * plan.live_time_s + (
        n_pix * det.clock_induced_charge_e_per_pixel_frame * plan.n_frames
    )
    old_var = det.excess_noise_factor_sq * old_mean + (
        n_pix * det.read_noise_e_rms**2 * plan.n_frames
    )
    np.testing.assert_array_equal(eager[0], old_mean)
    np.testing.assert_array_equal(eager[1], old_var)
    weighted = dict(planet=rates.planet / 2, thermal=rates.thermal)
    unweighted = dict(planet=rates.planet, thermal=rates.thermal)
    for source_override, pixel_override in [
        (weighted, None),
        (None, 3.5),
        (weighted, 3.5),
    ]:
        variance_source = source if source_override is None else (source - 2) / 2 + 2
        pixels = n_pix if pixel_override is None else pixel_override
        variance_rates = ox.Rates(
            BANDS, "electron", n_pix=pixel_override, **(source_override or unweighted)
        )
        mean, variance = det.mean_and_variance(
            eqx.tree_at(
                lambda r: r.variance, rates, variance_rates, is_leaf=lambda x: x is None
            ),
            plan,
            sum_bands=True,
        )
        poisson = (variance_source + pixels * dark) * 12 + pixels * cic * 4
        expected = f2 * poisson + pixels * rn**2 * 4
        np.testing.assert_array_equal(mean, jnp.sum(old_mean, axis=0))
        np.testing.assert_allclose(
            variance, expected.sum(axis=0), rtol=16 * np.finfo(np.float32).eps
        )
    compiled = eqx.filter_jit(lambda d, r, p: d.mean_and_variance(r, p))(
        det, rates, plan
    )
    for actual in [eager, compiled]:
        for got, expected in zip(actual, [expected_mean, expected_var], strict=True):
            np.testing.assert_allclose(
                got, expected, rtol=16 * np.finfo(np.float32).eps
            )
    reduced = det.mean_and_variance(rates, plan, sum_bands=True)
    for got, expected in zip(reduced, [expected_mean, expected_var], strict=True):
        np.testing.assert_allclose(
            got, expected.sum(axis=0), rtol=16 * np.finfo(np.float32).eps
        )


def test_noise_terms_independently():
    """Fails if a noise term changes another budget term.

    Floating-point: 16 float32 eps relative, plus 2e-6 absolute for
    cancellation when subtracting count budgets of order 100.
    """
    det = ox.Detector(pixel_scale_arcsec=0.01, shape=(1, 1))
    rates = ox.Rates(BANDS, "electron", planet=jnp.array([2.0, 3.0]), n_pix=5.0)
    plan = ox.FramePlan(3.0, 4.0)
    baseline = det.mean_and_variance(rates, plan)
    variants = [
        ("dark_current_e_per_pixel_s", 0.2, 12.0, 12.0),
        ("clock_induced_charge_e_per_pixel_frame", 0.3, 6.0, 6.0),
        ("read_noise_e_rms", 0.4, 0.0, 3.2),
        ("excess_noise_factor_sq", 2.0, 0.0, np.array([24.0, 36.0])),
    ]
    for name, value, delta_mean, delta_var in variants:
        changed = eqx.tree_at(
            lambda d, name=name: getattr(d, name), det, jnp.array(value)
        )
        got = changed.mean_and_variance(rates, plan)
        for observed, original, expected in zip(
            got, baseline, [delta_mean, delta_var], strict=True
        ):
            np.testing.assert_allclose(
                observed - original,
                expected,
                rtol=16 * np.finfo(np.float32).eps,
                atol=2e-6,
            )


@eqx.filter_jit
def _samples(det, rates, plan, keys):
    return jax.vmap(lambda key: det.draw(key, rates, plan))(keys)


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
@pytest.mark.parametrize("f2", [1.0, 2.0])
def test_draw_moments(detector_cls, f2):
    """Fails if source/dark/CIC gain or read variance disagrees with moments.

    N=2000, seed=0, one pixel in two bands at low/high source mean (0.2, 80).
    Six SE: mean SE=sqrt(var/N); variance SE=sqrt(2 var^2/(N-1)+k4/N).
    Compound-Poisson k4=lambda*F2*(2 F2-1)*(3 F2-2); Gaussian k4=0.
    This includes low-mean excess kurtosis; near-Gaussian two-sided false
    rejection is about 2e-9 per check, <4e-8 across 16 moment checks.
    IdealDetector cases explicitly verify Poisson mean=variance, no floor.
    """
    kwargs = (
        {}
        if detector_cls is ox.IdealDetector
        else dict(
            dark_current_e_per_pixel_s=0.1,
            clock_induced_charge_e_per_pixel_frame=0.2,
            read_noise_e_rms=0.3,
            excess_noise_factor_sq=f2,
        )
    )
    det = detector_cls(pixel_scale_arcsec=0.01, shape=(1, 1), **kwargs)
    rates = ox.Rates(BANDS, "electron", planet=jnp.array([0.2, 80.0]).reshape(2, 1, 1))
    plan = ox.FramePlan(0.5, 2.0)
    n = 2000
    keys = jax.random.split(jax.random.key(0), n)
    samples = np.asarray(_samples(det, rates, plan, keys))
    mean, var = map(np.asarray, det.mean_and_variance(rates, plan))
    if detector_cls is ox.IdealDetector:
        np.testing.assert_array_equal(mean, rates.planet)
        np.testing.assert_array_equal(var, rates.planet)
        f2 = 1.0
        lam = mean
    else:
        lam = np.asarray(rates.planet) + 0.1 + 0.2 * 2
    k4 = lam * f2 * (2 * f2 - 1) * (3 * f2 - 2)
    assert np.all(np.abs(samples.mean(axis=0) - mean) < 6 * np.sqrt(var / n))
    assert np.all(
        np.abs(samples.var(axis=0, ddof=1) - var)
        < 6 * np.sqrt(2 * var**2 / (n - 1) + k4 / n)
    )


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
def test_draw_key_contract(detector_cls):
    """Fails for key reuse/nondeterminism or batch-dependent sampling; bitwise."""
    kwargs = (
        {}
        if detector_cls is ox.IdealDetector
        else dict(
            dark_current_e_per_pixel_s=0.1,
            clock_induced_charge_e_per_pixel_frame=0.2,
            read_noise_e_rms=0.3,
            excess_noise_factor_sq=2.0,
        )
    )
    det = detector_cls(pixel_scale_arcsec=0.01, shape=(1, 1), **kwargs)
    rates = ox.Rates(BANDS, "electron", planet=jnp.array([1.0, 20.0]).reshape(2, 1, 1))
    plan = ox.FramePlan(1.0, 2.0)
    keys = jax.random.split(jax.random.key(0), 3)
    draw = eqx.filter_jit(lambda d, r, p, k: d.draw(k, r, p))
    loop = jnp.stack([draw(det, rates, plan, k) for k in keys])
    np.testing.assert_array_equal(loop, _samples(det, rates, plan, keys))
    np.testing.assert_array_equal(
        draw(det, rates, plan, keys[0]), draw(det, rates, plan, keys[0])
    )
    zero = det.draw(keys[0], rates, ox.FramePlan(1.0, 0.0))
    np.testing.assert_array_equal(zero, 0.0)


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
def test_frame_plan_band_axis(detector_cls):
    """Fails when an image's x axis takes the durations or frames for each band.

    Exact multiplication uses integer-valued rates and durations. The
    zero-frame band must also produce zero in a draw with nonzero CIC/read.
    """
    shape = (2, 3, 2)
    rates = ox.Rates(BANDS, "electron", planet=jnp.arange(1.0, 13.0).reshape(shape))
    det = detector_cls(pixel_scale_arcsec=0.01, shape=shape[1:])
    duration = jnp.array([2.0, 7.0])
    plan = ox.FramePlan(duration, 1.0)
    mean, var = det.mean_and_variance(rates, plan)
    expected = np.stack([2.0 * rates.planet[0], 7.0 * rates.planet[1]])
    np.testing.assert_array_equal(mean, expected)
    np.testing.assert_array_equal(var, expected)

    if detector_cls is ox.Detector:
        det = eqx.tree_at(
            lambda d: (d.clock_induced_charge_e_per_pixel_frame, d.read_noise_e_rms),
            det,
            (jnp.asarray(1.0), jnp.asarray(1.0)),
        )
    draw = det.draw(jax.random.key(0), rates, ox.FramePlan(duration, jnp.array([0, 1])))
    np.testing.assert_array_equal(draw[0], 0)
    assert jnp.any(draw[1] != 0)

    with pytest.raises(ValueError, match=r"scalar.*\(W,\)"):
        ox.FramePlan(jnp.ones((2, 1)), 1.0)
    with pytest.raises(ValueError, match=r"scalar.*\(W,\)"):
        ox.FramePlan(duration, jnp.ones(1))
    empty_rates = ox.Rates(BANDS, "electron")
    empty_mean, _ = det.mean_and_variance(empty_rates, plan)
    np.testing.assert_array_equal(
        empty_mean, jnp.full(2, det.clock_induced_charge_e_per_pixel_frame)
    )
    with pytest.raises(ValueError, match="same bands"):
        ox.Rates(BANDS, "electron", variance=ox.Rates(BANDS[::-1], "electron"))
    with pytest.raises(ValueError, match="electron/s"):
        det.mean_and_variance(ox.Rates(BANDS, "photon"), plan)
    with pytest.raises(ValueError, match="per-pixel"):
        det.draw(jax.random.key(0), ox.Rates(BANDS, "electron", n_pix=2.0), plan)
    mismatched = ox.FramePlan(jnp.ones(1), 1.0)
    with pytest.raises(ValueError, match="leading"):
        det.mean_and_variance(rates, mismatched)
    with pytest.raises(ValueError, match="leading"):
        det.draw(jax.random.key(0), rates, mismatched)


def test_detector_noise_fields_are_strong_leaves_of_the_active_dtype(precision):
    """Python numbers given to the detector are stored as strong float arrays.

    Truth source: definitional (the leaf rule: every float leaf is a strong array of
    the active dtype). The six noise fields and the scalar quantum efficiency are set
    from Python floats under each precision. Tolerance basis: exact. Fails if any
    field keeps a weak type or a dtype other than the active one.
    """
    from hwoutils.leaves import check_tree

    det = ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(1, 1),
        qe=0.9,
        survival_fraction=0.95,
        dark_current_e_per_pixel_s=1e-4,
        clock_induced_charge_e_per_pixel_frame=1e-3,
        read_noise_e_rms=0.0,
        excess_noise_factor_sq=1.0,
        read_time_s=0.1,
    )
    names = (
        "_qe",
        "survival_fraction",
        "dark_current_e_per_pixel_s",
        "clock_induced_charge_e_per_pixel_frame",
        "read_noise_e_rms",
        "excess_noise_factor_sq",
        "read_time_s",
    )
    for name in names:
        leaf = getattr(det, name)
        assert isinstance(leaf, jax.Array), name
        assert leaf.dtype == precision and not leaf.weak_type, name
    assert check_tree(det) == ()
