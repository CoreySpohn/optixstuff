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
            det.log_prob(observed, rates, PLAN), expected, rtol=1e-12
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
            det.log_prob(observed, rates, PLAN), expected, rtol=1e-12
        )


@pytest.mark.parametrize("dark", [0.0, 0.1])
def test_poisson_form_normalizes_at_low_counts(dark):
    """Fails if the pmf over observed counts of a 2-pixel frame does not sum to one."""
    with jax.enable_x64():
        det = ox.Detector(
            pixel_scale_arcsec=0.01, shape=(1, 2), dark_current_e_per_pixel_s=dark
        )
        rates = _rates([[[0.1, 0.4]]])
        counts = np.arange(0, 40)
        grid = np.stack(np.meshgrid(counts, counts, indexing="ij"), -1).reshape(-1, 2)
        observed = grid.reshape(-1, 1, 1, 2)
        total = jnp.exp(det.log_prob(observed, rates, PLAN)).sum()
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
        dens = np.exp(np.asarray(det.log_prob(y.reshape(-1, 1, 1, 1), rates, PLAN)))
        np.testing.assert_allclose(np.trapezoid(dens, y), 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    "make", [_noisy, lambda: ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(2, 3))]
)
def test_gradient_finite_including_zero_rate(make):
    """Fails if a gradient is NaN or infinite, also at zero rate with zero counts."""
    det = make()
    observed = jnp.array([[[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]]])

    def f(planet):
        return det.log_prob(observed, _rates(planet), PLAN)

    grad = jax.grad(f)(jnp.array([[[0.0, 0.5, 1.0], [0.0, 0.2, 2.0]]]))
    assert np.all(np.isfinite(grad))


def test_batch_prefix_and_vmap():
    """Fails if leading batch axes are summed or vmap over rates disagrees."""
    det = _noisy()
    planet = jnp.stack([jnp.full((1, 2, 3), 2.0), jnp.full((1, 2, 3), 5.0)])
    observed = jnp.full((2, 1, 2, 3), 4.0)
    mapped = jax.vmap(lambda o, p: det.log_prob(o, _rates(p), PLAN))(observed, planet)
    loop = jnp.stack(
        [det.log_prob(observed[i], _rates(planet[i]), PLAN) for i in range(2)]
    )
    np.testing.assert_allclose(mapped, loop, rtol=1e-6)
    batched = det.log_prob(observed, _rates(planet[0]), PLAN)
    assert batched.shape == (2,)
    np.testing.assert_allclose(batched, loop[0] * jnp.ones(2), rtol=1e-6)
    jitted = eqx.filter_jit(det.log_prob)(observed[0], _rates(planet[0]), PLAN)
    np.testing.assert_allclose(jitted, loop[0], rtol=1e-6)


def test_mask_drops_pixels():
    """Fails if masked pixels contribute."""
    det = _noisy()
    rates = _rates(np.full((1, 2, 3), 2.0))
    observed = np.full((1, 2, 3), 3.0)
    mask = np.array([[[True, True, False], [False, False, False]]])
    full = det.log_prob(observed, rates, PLAN)
    part = det.log_prob(observed, rates, PLAN, mask=mask)
    np.testing.assert_allclose(part, full * 2 / 6, rtol=1e-6)
