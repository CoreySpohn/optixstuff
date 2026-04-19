"""Tests for optixstuff.Exposure."""

import jax
import jax.numpy as jnp
from optixstuff import Exposure


def test_exposure_scalar_construction():
    exp = Exposure(
        start_time_jd=jnp.asarray(60000.0),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(100.0),
        position_angle_deg=jnp.asarray(0.0),
    )
    assert exp.exposure_time_s == 3600.0
    assert exp.central_wavelength_nm == 500.0


def test_exposure_vector_construction():
    wls = jnp.asarray([400.0, 500.0, 600.0])
    exp = Exposure(
        start_time_jd=jnp.asarray(60000.0),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=wls,
        bin_width_nm=jnp.asarray(100.0),
        position_angle_deg=jnp.asarray(0.0),
    )
    assert exp.central_wavelength_nm.shape == (3,)


def test_exposure_in_axes_helper():
    spec = Exposure.in_axes(central_wavelength_nm=0, bin_width_nm=0)
    assert spec.central_wavelength_nm == 0
    assert spec.bin_width_nm == 0
    assert spec.start_time_jd is None
    assert spec.exposure_time_s is None
    assert spec.position_angle_deg is None


def test_exposure_is_pytree():
    exp = Exposure(
        start_time_jd=jnp.asarray(60000.0),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(100.0),
        position_angle_deg=jnp.asarray(0.0),
    )
    leaves, treedef = jax.tree_util.tree_flatten(exp)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.exposure_time_s == 3600.0
