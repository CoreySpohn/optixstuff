"""Code verification of the detector's pixel log density."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import optixstuff as ox
from optixstuff.spectral import Band

BANDS = (Band(450, 500),)
PLAN = ox.FramePlan(3.0, 4.0)


def _rates(values):
    return ox.Rates(BANDS, "electron", planet=jnp.asarray(values))


def _noisy():
    return ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(2, 3),
        dark_current_e_per_pixel_s=0.2,
        clock_induced_charge_e_per_pixel_frame=0.3,
        read_noise_e_rms=1.5,
        excess_noise_factor_sq=1.4,
    )


def test_gaussian_form_matches_photomancy_gaussian_loglike():
    """Fails if the noisy form drifts from the normalized per-pixel Gaussian.

    Reference: the photomancy ``gaussian_loglike`` expression
    ``sum(-0.5 z^2 - log sigma - 0.5 log 2 pi)``, ``sigma = sqrt(variance)``,
    rebuilt here with scipy, compared in float64 to 1e-12 relative.
    """
    with jax.enable_x64():
        det = _noisy()
        rates = _rates(np.linspace(1.0, 9.0, 6).reshape(1, 2, 3))
        mean, var = det.mean_and_variance(rates, PLAN)
        observed = np.asarray(mean) + np.array([[[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]]])
        expected = stats.norm.logpdf(observed, np.asarray(mean), np.sqrt(var)).sum()
        np.testing.assert_allclose(
            det.log_prob(observed, rates, PLAN).sum(), expected, rtol=1e-12
        )


def test_poisson_form_matches_scipy():
    """Fails if the photon-counting form is not the Poisson pmf, constants included."""
    with jax.enable_x64():
        det = ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(2, 3))
        rates = _rates(np.linspace(0.05, 0.6, 6).reshape(1, 2, 3))
        mean, _ = det.mean_and_variance(rates, PLAN)
        observed = np.array([[[0, 1, 0], [2, 0, 1]]])
        expected = stats.poisson.logpmf(observed, np.asarray(mean)).sum()
        np.testing.assert_allclose(
            det.log_prob(observed, rates, PLAN).sum(), expected, rtol=1e-12
        )


@pytest.mark.parametrize("dark", [0.0, 0.1])
def test_poisson_form_normalizes_at_low_counts(dark):
    """Fails if the pmf over observed counts of a 2-pixel frame does not sum to one."""
    with jax.enable_x64():
        det = ox.Detector(
            pixel_scale_arcsec=0.01,
            shape=(1, 2),
            dark_current_e_per_pixel_s=dark,
            photon_counting=True,
        )
        rates = _rates([[[0.1, 0.4]]])
        counts = np.arange(0, 40)
        grid = np.stack(np.meshgrid(counts, counts, indexing="ij"), -1).reshape(-1, 2)
        observed = grid.reshape(-1, 1, 1, 2)
        total = jnp.exp(det.log_prob(observed, rates, PLAN).sum(-1)).sum()
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)


def test_gaussian_form_normalizes():
    """Fails if the noisy form does not integrate to one over a pixel."""
    with jax.enable_x64():
        det = _noisy()
        rates = _rates(np.full((1, 1, 1), 3.0))
        mean, var = (float(x.squeeze()) for x in det.mean_and_variance(rates, PLAN))
        y = np.linspace(
            float(mean) - 12 * np.sqrt(var), float(mean) + 12 * np.sqrt(var), 20001
        )
        dens = np.exp(np.asarray(det.log_prob(y.reshape(-1, 1, 1, 1), rates, PLAN)))[
            :, 0, 0, 0
        ]
        np.testing.assert_allclose(np.trapezoid(dens, y), 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    "make", [_noisy, lambda: ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(2, 3))]
)
def test_gradient_finite_including_zero_rate(make):
    """Fails if a gradient is NaN or infinite, also at zero rate with zero counts."""
    det = make()
    observed = jnp.array([[[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]]])

    def f(planet):
        return det.log_prob(observed, _rates(planet), PLAN).sum()

    grad = jax.grad(f)(jnp.array([[[0.0, 0.5, 1.0], [0.0, 0.2, 2.0]]]))
    assert np.all(np.isfinite(grad))


def test_elementwise_shape_vmap_and_jit():
    """Fails if the result is reduced, or vmap, jit and a loop disagree."""
    det = _noisy()
    planet = jnp.stack([jnp.full((1, 2, 3), 2.0), jnp.full((1, 2, 3), 5.0)])
    observed = jnp.full((2, 1, 2, 3), 4.0)
    mapped = jax.vmap(lambda o, p: det.log_prob(o, _rates(p), PLAN))(observed, planet)
    loop = jnp.stack(
        [det.log_prob(observed[i], _rates(planet[i]), PLAN) for i in range(2)]
    )
    assert mapped.shape == (2, 1, 2, 3)
    np.testing.assert_allclose(mapped, loop, rtol=1e-6)
    batched = det.log_prob(observed, _rates(planet[0]), PLAN)
    assert batched.shape == (2, 1, 2, 3)
    jitted = eqx.filter_jit(det.log_prob)(observed[0], _rates(planet[0]), PLAN)
    np.testing.assert_allclose(jitted, loop[0], rtol=1e-6)


def test_zero_variance_gaussian_edges():
    """Fails if zero variance is not 0 at the mean and -inf elsewhere."""
    det = ox.Detector(pixel_scale_arcsec=0.01, shape=(1, 2))
    rates = _rates([[[0.0, 0.0]]])
    out = det.log_prob(jnp.array([[[0.0, 5.0]]]), rates, PLAN)
    assert out[0, 0, 0] == 0.0
    assert out[0, 0, 1] == -jnp.inf


def test_impossible_observation_is_minus_inf():
    """Fails if excess noise above one, zero rate and 5 electrons is finite."""
    det = ox.Detector(pixel_scale_arcsec=0.01, shape=(1, 1), excess_noise_factor_sq=1.5)
    out = det.log_prob(jnp.array([[[5.0]]]), _rates([[[0.0]]]), PLAN)
    assert out[0, 0, 0] == -jnp.inf


def test_poisson_negative_mean_is_outside_the_law():
    """Fails if a negative Poisson mean scores a zero count as finite.

    The pmf of zero counts is exp(-mean), so a negative mean would score above
    one (log above zero) and reward a model that drives a pixel negative.
    """
    det = ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(1, 3))
    rates = _rates([[[-0.5, 0.0, 0.5]]])
    out = det.log_prob(jnp.zeros((1, 1, 3)), rates, PLAN)
    mean = 0.5 * PLAN.frame_time_s * PLAN.n_frames
    assert out[0, 0, 0] == -jnp.inf
    assert out[0, 0, 1] == 0.0
    np.testing.assert_allclose(out[0, 0, 2], -mean, rtol=1e-6)

    def score(r):
        return det.log_prob(jnp.zeros((1, 1, 1)), _rates(r), PLAN).sum()

    grad = jax.grad(score)(jnp.array([[[-0.5]]]))
    assert jnp.all(jnp.isfinite(grad))


def test_photon_counting_contract():
    """Fails if a counting detector may carry read noise or excess noise."""
    with pytest.raises(ValueError):
        ox.Detector(
            pixel_scale_arcsec=0.01,
            shape=(1, 1),
            read_noise_e_rms=1.0,
            photon_counting=True,
        )
    with pytest.raises(ValueError):
        ox.Detector(
            pixel_scale_arcsec=0.01,
            shape=(1, 1),
            excess_noise_factor_sq=1.2,
            photon_counting=True,
        )
    assert ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(1, 1)).photon_counting


def test_poisson_form_matches_draw_frequencies():
    """Fails if exp(log_prob) differs from the empirical frequencies of draw."""
    det = ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(1, 1))
    rates = _rates([[[0.4]]])
    keys = jax.random.split(jax.random.key(0), 40000)
    draws = np.asarray(jax.vmap(lambda k: det.draw(k, rates, PLAN))(keys)).ravel()
    for k in range(5):
        freq = np.mean(draws == k)
        p = float(
            jnp.exp(det.log_prob(jnp.array([[[float(k)]]]), rates, PLAN))[0, 0, 0]
        )
        assert abs(freq - p) < 5 * np.sqrt(p * (1 - p) / draws.size) + 1e-4


def test_gaussian_form_matches_draw_moments():
    """Fails if the density's first two moments differ from those of draw."""
    det = _noisy()
    rates = _rates([[[6.0]]])
    keys = jax.random.split(jax.random.key(1), 40000)
    draws = np.asarray(jax.vmap(lambda k: det.draw(k, rates, PLAN))(keys)).ravel()
    mean, var = (float(x.squeeze()) for x in det.mean_and_variance(rates, PLAN))
    se = np.sqrt(var / draws.size)
    assert abs(draws.mean() - mean) < 5 * se
    assert abs(draws.var() / var - 1) < 0.05


# Fisher root contract, over every concrete detector law.


class _ConstantVariance(ox.Detector):
    """Gaussian detector with a variance that ignores the rates."""

    def mean_and_variance(self, rates, plan, *, sum_bands=False):
        mean, _ = super().mean_and_variance(rates, plan, sum_bands=sum_bands)
        return mean, jnp.full_like(mean, 2.5)


class _GaussianDraw(ox.Detector):
    """Overrides ``draw`` only; the assumed law is still the inherited one."""

    def draw(self, key, rates, plan):
        mean, variance = self.mean_and_variance(rates, plan)
        return mean + jnp.sqrt(variance) * jax.random.normal(key, mean.shape)


def _kw(**extra):
    return dict(pixel_scale_arcsec=0.01, shape=(2, 3), **extra)


DETECTORS = {
    "ideal": lambda: ox.IdealDetector(**_kw()),
    "poisson_dark_cic": lambda: ox.Detector(
        **_kw(
            dark_current_e_per_pixel_s=0.2,
            clock_induced_charge_e_per_pixel_frame=0.3,
            photon_counting=True,
        )
    ),
    "gaussian_noisy": _noisy,
    "gaussian_no_read": lambda: ox.Detector(**_kw(excess_noise_factor_sq=1.4)),
    "gaussian_constant_variance": lambda: _ConstantVariance(
        **_kw(read_noise_e_rms=1.0)
    ),
    "gaussian_draw_override": lambda: _GaussianDraw(**_kw(read_noise_e_rms=1.0)),
    # Low counts (mean 0.6 to 5.4 per pixel): the variance part of the Gaussian
    # root is 19 percent of the information here, 1 percent at the means above.
    "poisson_low_counts": lambda: ox.IdealDetector(**_kw()),
    "gaussian_low_counts": lambda: ox.Detector(
        **_kw(excess_noise_factor_sq=1.4, read_noise_e_rms=0.3)
    ),
}
LOW = {"poisson_low_counts", "gaussian_low_counts"}
TEMPLATE = np.linspace(1.0, 9.0, 6).reshape(1, 2, 3)


def test_every_concrete_detector_is_in_the_contract():
    """Fails if a detector class is added without a Fisher-root contract case."""
    covered = {type(make()) for make in DETECTORS.values()}
    from conftest import concrete_subclasses

    assert set(concrete_subclasses(ox.AbstractDetector)) <= covered


def _template(name):
    return TEMPLATE * (0.05 if name in LOW else 1.0)


def _root(det, theta, name):
    return det.fisher_root(_rates(theta * _template(name)), PLAN)


@pytest.mark.parametrize("name", DETECTORS)
def test_fisher_root_matches_score_variance(name):
    """Fails if J^T J of the root is not the variance of the score.

    Truth from primitives: ``N = 400000`` observed frames are drawn from the law
    ``log_prob`` assumes (Poisson of the mean for a counting detector, a normal
    with the mean and variance of ``mean_and_variance`` otherwise; never from
    ``draw``). The score is ``d log_prob / d theta`` for ``rates = theta *
    template`` at ``theta = 1`` (the template is scaled by 0.05 in the
    low-count cases), summed over the six independent pixels. The
    sample variance of the score must equal ``J^T J`` within 5 standard errors,
    the error of a sample variance from its own fourth moment
    ``sqrt((m4 - s^4) / N)``; a mean of N draws has no bias to allow for.
    """
    n_draws = 400_000
    with jax.enable_x64():
        det = DETECTORS[name]()
        theta = jnp.asarray(1.0)
        rng = np.random.default_rng(7)
        mean, variance = (
            np.asarray(x) for x in det.mean_and_variance(_rates(_template(name)), PLAN)
        )
        if det.photon_counting:
            y = rng.poisson(mean, size=(n_draws, *mean.shape)).astype(float)
        else:
            y = rng.normal(mean, np.sqrt(variance), size=(n_draws, *mean.shape))

        def logp(t):
            return det.log_prob(y, _rates(t * _template(name)), PLAN)

        score = np.asarray(jax.jvp(logp, (theta,), (jnp.ones(()),))[1]).sum((1, 2, 3))
        centred = score - score.mean()
        var_hat = centred.var(ddof=1)
        se = np.sqrt((np.mean(centred**4) - var_hat**2) / n_draws)
        jac = jax.jacfwd(lambda t: _root(det, t, name))(theta)
        fisher = float(jnp.sum(jac**2))
        assert abs(var_hat - fisher) < 5 * se, (var_hat, fisher, se)
        assert se / fisher < 0.02


@pytest.mark.parametrize("name", DETECTORS)
def test_fisher_root_zero_outside_the_law_and_finite(name):
    """Fails if the root is NaN or nonzero where the law has no support.

    The support is a positive mean for a counting detector and a positive
    variance for the Gaussian law (the walls of ``log_prob``); with read noise
    the Gaussian law is defined at a negative mean and its root stays finite.
    """
    det = DETECTORS[name]()
    planet = np.array([[[-4.0, -1.0, 0.0], [0.0, 2.0, 5.0]]])
    rates = _rates(planet)
    mean, variance = det.mean_and_variance(rates, PLAN)
    root = np.asarray(det.fisher_root(rates, PLAN))
    assert np.all(np.isfinite(root))
    outside = np.asarray((mean <= 0) if det.photon_counting else (variance <= 0))
    # A constant variance has no wall: the Gaussian law is defined everywhere.
    assert outside.any() == (name != "gaussian_constant_variance")
    assert np.all(root[..., outside] == 0.0)
    assert np.any(root[..., ~outside] != 0.0)
    grad = jax.grad(lambda p: det.fisher_root(_rates(p), PLAN).sum())(
        jnp.asarray(planet)
    )
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize("name", DETECTORS)
def test_fisher_root_jit_equals_eager(name):
    """Fails if jit changes the root or its Jacobian beyond float32 rounding."""
    det = DETECTORS[name]()
    planet = jnp.asarray(TEMPLATE)

    def root(p):
        return det.fisher_root(_rates(p), PLAN)

    np.testing.assert_allclose(eqx.filter_jit(root)(planet), root(planet), rtol=1e-6)
    np.testing.assert_allclose(
        eqx.filter_jit(jax.jacfwd(root))(planet),
        jax.jacfwd(root)(planet),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fisher_root_of_an_unknown_law_raises():
    """Fails if a subclass with its own log_prob silently inherits a root."""

    class Robust(ox.Detector):
        def log_prob(self, observed, rates, plan):
            return -jnp.abs(observed - self.mean_and_variance(rates, plan)[0])

    with pytest.raises(NotImplementedError, match="Robust"):
        Robust(**_kw()).fisher_root(_rates(TEMPLATE), PLAN)
