"""Tests for the IFS disperser descriptors."""

import jax.numpy as jnp

from optixstuff.disperser import AbstractDisperser, LensletDisperser


def _lenslet():
    # crispy WFIRST660-like: npixperdlam=2, R=50 -> linear coeff 100
    return LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        grid_kind="square",
        n_lenslets=8,
        psflet_kind="gaussian",
        detector_shape=(256, 256),
    )


def test_is_abstract_disperser():
    assert isinstance(_lenslet(), AbstractDisperser)


def test_spectral_sampling_returns_config():
    assert float(_lenslet().spectral_sampling()) == 2.0


def test_spectral_resolution_at_ref():
    # local dispersion derivative at u=0 is coeff 100; R = 100 / 2.0 = 50
    d = _lenslet()
    assert jnp.allclose(d.spectral_resolution(660.0), 50.0)


def test_throughput_default_unity():
    assert jnp.allclose(_lenslet().throughput(660.0), 1.0)


def test_abstract_cannot_instantiate():
    import pytest

    with pytest.raises(TypeError):
        AbstractDisperser()
