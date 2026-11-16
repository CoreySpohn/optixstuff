"""Code verification of frame accounting, electron moments, and draws."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optixstuff as ox


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_frames_for_time(dtype):
    """Fails at rounded multiples, positive subframes, or integer casts in JIT.

    Exact counts against an independently written tolerance rule; float64
    random cases use seed 0. The discrete count derivative is exactly zero.
    """
    with jax.enable_x64():
        integer = np.array(3.0, dtype=np.dtype(dtype))
        ratios = np.array(
            [
                0.0,
                0.1,
                1.0,
                3.0,
                np.nextafter(integer, -np.inf),
                np.nextafter(integer, np.inf),
            ],
            dtype=np.dtype(dtype),
        )
        plan = eqx.filter_jit(ox.frames_for_time)(
            jnp.array(ratios), jnp.array(1.0, dtype=dtype)
        )
        np.testing.assert_array_equal(plan.n_frames, [0, 1, 1, 3, 3, 3])
        assert jnp.issubdtype(plan.n_frames.dtype, jnp.floating)
        rng = np.random.default_rng(0)
        frame = rng.uniform(0.01, 100.0, 2000)
        duration = rng.uniform(0.0, 1000.0, 2000) * frame
        duration[:1000] = (
            rng.integers(1, 1000, 1000)
            * frame[:1000]
            * (1 + rng.uniform(-2e-12, 2e-12, 1000))
        )
        x = duration / frame
        k = np.rint(x)
        close = (k >= 1) & (
            np.abs(x - k) / np.maximum(k, 1) <= max(1e-12, 4 * np.finfo(float).eps)
        )
        expected = np.where(
            duration == 0, 0.0, np.maximum(1.0, np.ceil(np.where(close, k, x)))
        )
        np.testing.assert_array_equal(
            ox.frames_for_time(duration, frame).n_frames, expected
        )
        derivative = jax.vmap(jax.grad(lambda t: ox.frames_for_time(t, 1.0).n_frames))
        np.testing.assert_array_equal(derivative(jnp.array(ratios)), 0.0)


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
    rates = ox.ElectronRates(
        planet=jnp.arange(1.0, 13.0).reshape((2, 2, 3)),
        thermal=jnp.full((2, 2, 3), 2.0),
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
    eager = det.mean_and_variance(rates, plan, n_pix=n_pix)
    compiled = eqx.filter_jit(lambda d, r, p: d.mean_and_variance(r, p, n_pix=5.0))(
        det, rates, plan
    )
    for actual in [eager, compiled]:
        for got, expected in zip(actual, [expected_mean, expected_var], strict=True):
            np.testing.assert_allclose(
                got, expected, rtol=16 * np.finfo(np.float32).eps
            )
    reduced = det.mean_and_variance(rates, plan, n_pix=n_pix, sum_bands=True)
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
    rates = ox.ElectronRates(planet=jnp.array([2.0, 3.0]))
    plan = ox.FramePlan(3.0, 4.0)
    baseline = det.mean_and_variance(rates, plan, n_pix=5.0)
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
        got = changed.mean_and_variance(rates, plan, n_pix=5.0)
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
    rates = ox.ElectronRates(planet=jnp.array([0.2, 80.0]).reshape(2, 1, 1))
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
    rates = ox.ElectronRates(planet=jnp.array([1.0, 20.0]).reshape(2, 1, 1))
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
