"""Deterministic per-pixel noise variance on detectors."""

import jax
import jax.numpy as jnp

from optixstuff import Detector


def test_noise_variance_matches_formula():
    """N = QE*rate*t + dark*t + CIC*n_frames + read^2*n_frames, per pixel."""
    det = Detector(
        pixel_scale_arcsec=0.01,
        shape=(4, 4),
        quantum_efficiency=0.8,
        dark_current_rate_e_per_s=0.01,
        read_noise_e=3.0,
        clock_induced_charge_rate_e_per_frame=0.1,
        frame_time_s=100.0,
    )
    rate = jnp.full((4, 4), 5.0)
    t = 1000.0
    n_frames = jnp.ceil(t / 100.0)  # 10
    expected = 0.8 * 5.0 * t + 0.01 * t + 0.1 * n_frames + 3.0**2 * n_frames
    N = det.noise_variance(rate, t)
    assert N.shape == (4, 4)
    assert jnp.allclose(N, expected)


def test_noise_variance_matches_readout_montecarlo():
    """Deterministic variance equals the empirical variance of stochastic readout."""
    det = Detector(
        pixel_scale_arcsec=0.01,
        shape=(6, 6),
        quantum_efficiency=0.9,
        dark_current_rate_e_per_s=0.05,
        read_noise_e=2.0,
        clock_induced_charge_rate_e_per_frame=0.02,
        frame_time_s=50.0,
    )
    rate = jnp.full((6, 6), 3.0)
    t = 500.0
    keys = jax.random.split(jax.random.PRNGKey(0), 6000)
    draws = jax.vmap(lambda k: det.readout(rate, t, k))(keys)  # (6000, 6, 6)
    emp_var = draws.var(axis=0)
    pred = det.noise_variance(rate, t)
    assert jnp.allclose(emp_var, pred, rtol=0.1)
