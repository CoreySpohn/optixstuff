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


# The rate floor that draw and log_prob share: 1e-4 of the frame peak.
FLOOR_DETECTORS = {
    "poisson": lambda: ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(1, 2)),
    "gaussian": lambda: ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(1, 2),
        dark_current_e_per_pixel_s=0.2,
        read_noise_e_rms=1.5,
        excess_noise_factor_sq=1.4,
    ),
}


def _frame(negative, peak=1.0):
    """A frame of two pixels: a positive peak and one negative component."""
    return ox.Rates(
        BANDS,
        "electron",
        planet=jnp.asarray([[[peak, 0.0]]]),
        star_leakage=jnp.asarray([[[0.0, negative]]]),
    )


@pytest.mark.parametrize("name", FLOOR_DETECTORS)
def test_log_prob_and_draw_share_the_rate_floor(name):
    """Fails if the likelihood and the draws treat a negative rate differently.

    Exact. A total rate below zero by half the tolerance (1e-4 of the frame
    peak) is a rate of zero for both: ``log_prob`` equals its value at a rate
    of zero and the draw succeeds. Twice the tolerance scores ``-inf``, with a
    finite gradient, and the draw raises. A Poisson mean below zero must
    never score a zero count above one (log above zero), which would reward a
    model that drives a pixel negative. At a rate of exactly zero the floor
    passes the full gradient (float64, 1e-6 against a rate of 1e-12), so a
    fit started at zero flux is not slowed by half.
    """
    det = FLOOR_DETECTORS[name]()
    observed = jnp.asarray([[[3.0, 0.0]]])
    zero = det.log_prob(observed, _frame(0.0), PLAN)
    inside = det.log_prob(observed, _frame(-0.5e-4), PLAN)
    np.testing.assert_array_equal(inside, zero)
    det.draw(jax.random.key(0), _frame(-0.5e-4), PLAN).block_until_ready()
    beyond = det.log_prob(observed, _frame(-2e-4), PLAN)
    assert beyond[0, 0, 1] == -jnp.inf and beyond[0, 0, 0] == zero[0, 0, 0]
    with pytest.raises(eqx.EquinoxRuntimeError, match="frame peak"):
        det.draw(jax.random.key(0), _frame(-2e-4), PLAN).block_until_ready()

    def score(r):
        return det.log_prob(observed, _frame(r), PLAN)[0, 0, 1]

    assert jnp.isfinite(jax.grad(score)(jnp.asarray(-2e-4)))
    with jax.enable_x64():
        at_zero, above = (jax.grad(score)(jnp.asarray(r)) for r in (0.0, 1e-12))
    np.testing.assert_allclose(at_zero, above, rtol=1e-6)


def _saturated_edge(hostile):
    """Electron rates of a band whose first node-effective value is negative.

    The node values and weights of a saturated absorption edge (the first
    node at -0.076 of the band mean), read through a smooth response in the
    first pixel and, when ``hostile``, a response on the negative node only
    in the second.
    """
    band = Band(500.0, 555.0)
    nodes = (
        band.low_nm + band.width_nm * (np.polynomial.legendre.leggauss(4)[0] + 1) / 2
    )
    rule = ox.Quadrature((band,), [nodes], [[9.6, 17.9, 17.9, 9.6]])
    values = jnp.asarray([-0.076, 0.663, 1.385, 1.227])
    smooth = jnp.asarray([1.0, 0.9, 0.8, 0.7])
    second = jnp.asarray([1.0, 0.0, 0.0, 0.0]) if hostile else smooth
    response = jnp.stack([smooth, second], -1)  # (Q, pixels)
    photons = ox.Rates(
        rule, "photon", star_leakage=(values[:, None] * response)[None, :, None]
    )
    return FLOOR_DETECTORS["poisson"]().electrons(photons)


@pytest.mark.parametrize("x64", [False, True])
def test_draw_floors_round_off_and_raises_beyond_it(x64):
    """Fails if a draw samples a negative rate or refuses integration error.

    Exact, in float32 and float64. A saturated edge read through a smooth
    response passes; a response that sees only the negative node raises,
    under ``filter_jit``, ``vmap`` and ``grad``. A total at half the
    tolerance is drawn as exactly zero by both detectors, and twice it
    raises. A pixel where a ringing planet is the only light, at -2e-7 of the
    frame peak (the reconstruction ringing measured on the synthetic YIP
    fixture of the consumer suite), passes. Mutation: a tolerance relative to
    the pixel's own light raises on that pixel.
    """
    key = jax.random.key(0)
    ideal = FLOOR_DETECTORS["poisson"]()
    with jax.enable_x64(x64):
        assert _frame(0.0).planet.dtype == (jnp.float64 if x64 else jnp.float32)
        ideal.draw(key, _saturated_edge(False), PLAN).block_until_ready()
        hostile = _saturated_edge(True)
        for draw in (
            eqx.filter_jit(ideal.draw),
            lambda k, r, p: jax.vmap(lambda k: ideal.draw(k, r, p))(
                jax.random.split(k, 2)
            ),
        ):
            with pytest.raises(eqx.EquinoxRuntimeError, match="frame peak"):
                draw(key, hostile, PLAN).block_until_ready()

        def fit(scale):
            rates = eqx.tree_at(
                lambda r: r.star_leakage, hostile, hostile.star_leakage * scale
            )
            frame = jax.lax.stop_gradient(ideal.draw(key, rates, PLAN))
            return ideal.log_prob(frame.astype(float), rates, PLAN).sum()

        with pytest.raises(eqx.EquinoxRuntimeError, match="frame peak"):
            jax.grad(fit)(jnp.asarray(1.0)).block_until_ready()
        ringing = ox.Rates(
            BANDS,
            "electron",
            planet=jnp.asarray([[[0.0, -2e-7]]]),
            star_leakage=jnp.asarray([[[1.0, 0.0]]]),
        )
        ideal.draw(key, ringing, PLAN).block_until_ready()
        for det in (ideal, ox.Detector(pixel_scale_arcsec=0.01, shape=(1, 2))):
            drawn = det.draw(key, _frame(-0.5e-4), PLAN)
            assert drawn[0, 0, 1] == 0
            with pytest.raises(eqx.EquinoxRuntimeError, match="frame peak"):
                det.draw(key, _frame(-2e-4), PLAN).block_until_ready()


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
