"""Aperture S/N and exposure-time algebra through detector statistics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optixstuff as ox
from optixstuff import Detector, FramePlan, Rates
from optixstuff.spectral import Band, Quadrature


@pytest.fixture(autouse=True, scope="module")
def _float64():
    """Hold the algebra to float64 rounding, scoped to this module."""
    with jax.enable_x64():
        yield


def _detector():
    """Chromatic yield with every detector noise term switched on."""
    return Detector(
        shape=(8, 8),
        pixel_scale_arcsec=0.01,
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=3e-4,
        clock_induced_charge_e_per_pixel_frame=0.005,
        read_noise_e_rms=0.2,
    )


_BAND = (Band(500.0, 600.0),)


def _rates(bands=_BAND):
    """Hand-set aperture photon rates with distinct mean and variance moments."""
    q = Quadrature.build(bands, order=4)
    shape = q.nodes_nm.shape
    slope = 1 + 0.002 * (q.nodes_nm - 550.0)

    def moment(scale, n_pix):
        return dict(
            planet=scale * 2e-4 * slope,
            star_leakage=scale * jnp.full(shape, 1e-4),
            local_zodi=scale * jnp.full(shape, 3e-4),
            exozodi=scale * 2e-4 / slope,
            n_pix=jnp.full(len(bands), n_pix),
        )

    return Rates(
        q,
        "photon",
        **moment(1.0, 17.1),
        variance=Rates(q, "photon", **moment(0.8, 14.6)),
    )


def test_snr_is_detector_statistics_plus_residual_squares():
    """Fails if S/N departs from detector variance or residuals lose time squared.

    Basis: exact algebra against ``detector.mean_and_variance`` at a whole
    number of frames, with a 1e-12 relative float64 rounding allowance.
    """
    rates, detector = _rates(), _detector()
    e = detector.electrons(rates)
    plan = FramePlan(100.0, 36)
    _, base = detector.mean_and_variance(e, plan)
    for f, g in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.2)):
        actual = ox.snr(
            rates,
            detector,
            3600.0,
            frame_time_s=100.0,
            post_processing=ox.PostProcessing(f, g),
        )
        expected = base + (f * e.star_leakage * 3600) ** 2 + (g * e.exozodi * 3600) ** 2
        np.testing.assert_allclose(
            (e.planet * 3600 / actual) ** 2, expected, rtol=1e-12, atol=0
        )
    kwargs = dict(frame_time_s=100.0)
    np.testing.assert_allclose(
        ox.snr(rates, detector, 3600.0, **kwargs),
        e.planet * 3600 / jnp.sqrt(base),
        rtol=1e-12,
    )
    # Electron rates are read as they are; zero time gives zero, not NaN.
    np.testing.assert_allclose(
        ox.snr(e, detector, 3600.0, **kwargs),
        ox.snr(rates, detector, 3600.0, **kwargs),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(ox.snr(rates, detector, 0.0, **kwargs), [0.0])
    for bad in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ox.PostProcessing(bad, 0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ox.PostProcessing(0, bad)


def test_exposure_time_inverts_snr_and_flags_unreachable_targets():
    """Fails if the solved time misses the target or an unreachable one looks valid.

    Basis: exact inverse algebra, 1e-12 relative float64 rounding allowance;
    the derivative is compared with a central difference of step 1e-4, whose
    O(h^2) truncation allows 1e-7 relative.
    """
    rates, detector = _rates(), _detector()
    pp = ox.PostProcessing(0.1, 0.2)
    kwargs = dict(frame_time_s=100.0, post_processing=pp)
    times = jnp.array([100.0, 723.4, 3600.0, 1e5])
    targets = jax.vmap(lambda t: ox.snr(rates, detector, t, **kwargs)[0])(times)

    def solve(target):
        return ox.exposure_time(rates, detector, target, **kwargs)

    solved = jax.jit(jax.vmap(solve))(targets)
    np.testing.assert_allclose(solved.time_s[:, 0], times, rtol=1e-12)
    assert solved.valid.all()

    e = detector.electrons(rates)
    ceiling = e.planet / (0.1 * e.star_leakage)
    assert ox.snr(rates, detector, 1e30, **kwargs)[0] < ceiling[0]
    for target in (ceiling * 1.001, ceiling, jnp.sqrt(ceiling**2 * 1.0)):
        beyond = solve(target)
        assert not beyond.valid.any()
        np.testing.assert_array_equal(beyond.time_s, [0.0])
    nothing = solve(0.0)
    assert nothing.valid.all()
    np.testing.assert_array_equal(nothing.time_s, [0.0])

    step = 1e-4
    slope = jax.grad(lambda t: solve(t).time_s[0])(5.0)
    difference = (solve(5.0 + step).time_s[0] - solve(5.0 - step).time_s[0]) / (
        2 * step
    )
    np.testing.assert_allclose(slope, difference, rtol=1e-7)
    # An unreachable target has a finite (zero) derivative, not NaN.
    assert jax.grad(lambda t: solve(t).time_s[0])(ceiling[0] * 2) == 0.0


def test_unknown_rates_stay_unknown():
    """Fails if a NaN rate reaches S/N or a time as a valid-looking number.

    Basis: exact; NaN must propagate to S/N and clear ``valid``.
    """
    rates, detector = _rates(), _detector()
    nan = jnp.full(rates.quadrature.nodes_nm.shape, jnp.nan)
    # The signal comes from the mean record and the noise from its variance.
    for where in (lambda r: r.planet, lambda r: r.variance.local_zodi):
        unknown = eqx.tree_at(where, rates, nan)
        assert jnp.isnan(ox.snr(unknown, detector, 3600.0, frame_time_s=100.0)).all()
        solved = ox.exposure_time(unknown, detector, 5.0, frame_time_s=100.0)
        assert not solved.valid.any()
        np.testing.assert_array_equal(solved.time_s, [0.0])


def test_per_band_targets_and_frame_times():
    """Fails if (W,) targets, times or frame times mix bands.

    Basis: each band of a two-band call equals its own one-band call to a
    1e-12 relative rounding allowance.
    """
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    rates, detector = _rates(bands), _detector()
    pp = ox.PostProcessing(0.05, 0.1)
    target, frame, time = jnp.array([4.0, 6.0]), jnp.array([50.0, 200.0]), 5000.0
    both = ox.exposure_time(
        rates, detector, target, frame_time_s=frame, post_processing=pp
    )
    snrs = ox.snr(rates, detector, time, frame_time_s=frame, post_processing=pp)
    assert both.time_s.shape == snrs.shape == (2,)
    for i, band in enumerate(bands):
        one = _rates((band,))
        single = ox.exposure_time(
            one, detector, target[i], frame_time_s=frame[i], post_processing=pp
        )
        np.testing.assert_allclose(both.time_s[i], single.time_s[0], rtol=1e-12)
        np.testing.assert_allclose(
            snrs[i],
            ox.snr(one, detector, time, frame_time_s=frame[i], post_processing=pp)[0],
            rtol=1e-12,
        )
    with pytest.raises(ValueError, match="scalar or shape"):
        ox.snr(rates, detector, jnp.ones((2, 2)), frame_time_s=100.0)


def test_background_variance_factor_one_is_the_unscaled_result():
    """Fails if a factor of one changes any number.

    Basis: exact equality (0 ULP) with the hand-built variance, source shot
    noise plus detector terms plus the squared floor, at factor one.
    """
    rates, detector = _rates(), _detector()
    e = detector.electrons(rates)
    _, one_frame = detector.mean_and_variance(e, FramePlan(100.0, 1))
    pp = ox.PostProcessing(0.1, 0.2)
    assert float(pp.background_variance_factor) == 1.0
    floor = (0.1 * e.star_leakage) ** 2 + (0.2 * e.exozodi) ** 2
    variance = (one_frame / 100.0) * 3600.0 + floor * 3600.0**2
    np.testing.assert_array_equal(
        ox.snr(rates, detector, 3600.0, frame_time_s=100.0, post_processing=pp),
        e.planet * 3600.0 / jnp.sqrt(variance),
    )
    explicit = ox.PostProcessing(0.1, 0.2, 1.0)
    np.testing.assert_array_equal(
        ox.exposure_time(
            rates, detector, 5.0, frame_time_s=100.0, post_processing=pp
        ).time_s,
        ox.exposure_time(
            rates, detector, 5.0, frame_time_s=100.0, post_processing=explicit
        ).time_s,
    )


def test_background_variance_factor_scales_everything_but_the_planet():
    """Fails if the factor touches the planet shot noise or misses a background.

    Basis: exact algebra, 1e-12 relative float64 rounding allowance, against
    the AYO equation ``t = snr**2 (p + m b') / (p**2 - snr**2 f2)`` with ``b'``
    the detector variance rate minus the planet's own shot variance rate.
    """
    rates, detector = _rates(), _detector()
    e = detector.electrons(rates)
    _, one_frame = detector.mean_and_variance(e, FramePlan(100.0, 1))
    background = one_frame / 100.0 - e.variance.planet * detector.excess_noise_factor_sq
    f, g, target = 0.05, 0.1, 6.0
    floor = (f * e.star_leakage) ** 2 + (g * e.exozodi) ** 2
    for factor in (1.5, 2.0, 4.0):
        solved = ox.exposure_time(
            rates,
            detector,
            target,
            frame_time_s=100.0,
            post_processing=ox.PostProcessing(f, g, factor),
        )
        expected = (
            target**2
            * (
                e.variance.planet * detector.excess_noise_factor_sq
                + factor * background
            )
            / (e.planet**2 - target**2 * floor)
        )
        np.testing.assert_allclose(solved.time_s, expected, rtol=1e-12)
    for bad in (0.99, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="background_variance_factor"):
            ox.PostProcessing(0, 0, bad)


def test_ayo_science_time_from_dumped_rates():
    """Fails if a factor of two stops reproducing the community time.

    Basis: electron rates, detector and frame time of a published toy case
    (500 nm, S/N 7, one second photon-counting frames). The AYO equation on
    the same rates gives 424413.4 s; the dump prints seven significant
    figures, so the calculator is held to 1e-6 relative of the stored
    424413.5 s. Without the factor the time is 215336.1 s.
    """
    planet, nf = 7.829059e-03, 5.396355e-11
    leak, zodi, exozodi, dark_cic = 1.798785e-09, 1.935725e-02, 1.091779e-01, 1.33e-01
    snr_target = 7.0
    ayo = (
        snr_target**2
        * (planet + 2 * (leak + zodi + exozodi + dark_cic))
        / (planet**2 - (snr_target * nf) ** 2)
    )
    np.testing.assert_allclose(ayo, 424413.4, rtol=1e-6)
    detector = Detector(
        pixel_scale_arcsec=0.0065,
        shape=(8, 8),
        dark_current_e_per_pixel_s=3e-5,
        clock_induced_charge_e_per_pixel_frame=1.3e-3,
    )
    n_pix = dark_cic / (3e-5 + 1.3e-3 / 1.0)
    rates = Rates(
        (Band(450.0, 550.0),),
        "electron",
        planet=jnp.array([planet]),
        star_leakage=jnp.array([leak]),
        local_zodi=jnp.array([zodi]),
        exozodi=jnp.array([exozodi]),
        n_pix=jnp.array([n_pix]),
    )
    floor = ox.PostProcessing(nf / leak, 0.0, 2.0)
    time = ox.exposure_time(
        rates, detector, snr_target, frame_time_s=1.0, post_processing=floor
    )
    np.testing.assert_allclose(time.time_s, [424413.5], rtol=1e-6)
    np.testing.assert_allclose(time.time_s, [ayo], rtol=1e-6)
    once = ox.exposure_time(
        rates,
        detector,
        snr_target,
        frame_time_s=1.0,
        post_processing=ox.PostProcessing(nf / leak, 0.0),
    )
    np.testing.assert_allclose(once.time_s, [215336.1], rtol=1e-6)


def test_trailing_batch_axes_broadcast_against_the_band_axis():
    """Fails if rate fields with a trailing batch axis break the calculation.

    Basis: each batch member equals its own unbatched call to a 1e-12 relative
    rounding allowance. Per-band pixel counts, frame times and targets must
    align with the band axis, not the last axis, of ``(W, K)`` electron rates.
    """
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    detector = _detector()
    pp = ox.PostProcessing(0.05, 0.1, 2.0)
    scales = jnp.array([0.5, 1.0, 3.0])
    base = _rates(bands)

    def scaled(record, scale):
        """Scale the backgrounds; a batch axis is added where scale has one."""
        scale = jnp.asarray(scale)
        names = ("planet", "star_leakage", "local_zodi", "exozodi")
        fields = {}
        for name in names:
            value = getattr(record, name)
            factor = 1.0 if name == "planet" else scale
            fields[name] = (
                value[..., None] * jnp.ones_like(scale) * factor
                if scale.ndim
                else value * factor
            )
        return Rates(record.quadrature, "photon", n_pix=record.n_pix, **fields)

    def with_variance(scale):
        return eqx.tree_at(
            lambda r: r.variance,
            scaled(base, scale),
            scaled(base.variance, scale),
            is_leaf=lambda x: x is None,
        )

    frame, target = jnp.array([50.0, 200.0]), jnp.array([4.0, 6.0])
    batched = with_variance(scales)
    got = ox.exposure_time(
        batched, detector, target, frame_time_s=frame, post_processing=pp
    )
    got_snr = ox.snr(batched, detector, 5000.0, frame_time_s=frame, post_processing=pp)
    assert got.time_s.shape == got_snr.shape == (2, 3)
    for k, scale in enumerate(scales):
        one = with_variance(scale)
        single = ox.exposure_time(
            one, detector, target, frame_time_s=frame, post_processing=pp
        )
        np.testing.assert_allclose(got.time_s[:, k], single.time_s, rtol=1e-12)
        np.testing.assert_allclose(
            got_snr[:, k],
            ox.snr(one, detector, 5000.0, frame_time_s=frame, post_processing=pp),
            rtol=1e-12,
        )
