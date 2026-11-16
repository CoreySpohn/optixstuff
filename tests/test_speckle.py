"""Tests for AbstractSpeckleField and the OpticalPath.speckle field."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import scalar_coronagraph
from hwoutils.constants import d2s, s2d

import optixstuff as ox


class _MockSpeckle(ox.AbstractSpeckleField):
    """Incoherent-halo-style speckle field with a mild time modulation."""

    pixel_scale_lod: float = 0.25
    epoch_d: float = 0.0

    def realize(self, *, wavelength_nm, time_d=0.0):
        scale = 1.0 + 0.1 * jnp.cos(
            jnp.asarray(time_d - self.epoch_d, dtype=float) * d2s
        )
        return jnp.full((8, 8), 1e-11) * scale


class TestInterface:
    """The realize contract and abstractness."""

    def test_realize_shape(self):
        m = _MockSpeckle().realize(wavelength_nm=500.0, time_d=0.0)
        assert m.shape == (8, 8)

    def test_realize_is_keyword_only(self):
        with pytest.raises(TypeError):
            _MockSpeckle().realize(500.0)

    def test_realize_time_varying(self):
        sp = _MockSpeckle()
        a = sp.realize(wavelength_nm=500.0, time_d=0.0)
        b = sp.realize(wavelength_nm=500.0, time_d=float(jnp.pi) * s2d)
        assert float(jnp.max(jnp.abs(a - b))) > 0.0

    def test_realize_defaults_time_zero(self):
        sp = _MockSpeckle()
        assert jnp.allclose(
            sp.realize(wavelength_nm=500.0),
            sp.realize(wavelength_nm=500.0, time_d=0.0),
        )

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ox.AbstractSpeckleField()

    def test_is_a_pytree(self):
        leaves = eqx.filter(_MockSpeckle(), eqx.is_array)
        assert leaves is not None


class TestOpticalPathField:
    """OpticalPath carries an optional speckle field."""

    def _path(self, simple_primary, simple_detector, **kw):
        return ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(),
            coronagraph=scalar_coronagraph(),
            detector=simple_detector,
            **kw,
        )

    def test_defaults_to_none(self, simple_primary, simple_detector):
        path = self._path(simple_primary, simple_detector)
        assert path.speckle is None

    def test_repr_omits_speckle_when_none(self, simple_primary, simple_detector):
        path = self._path(simple_primary, simple_detector)
        assert "speckle:" not in repr(path)

    def test_accepts_speckle(self, simple_primary, simple_detector):
        sp = _MockSpeckle()
        path = self._path(simple_primary, simple_detector, speckle=sp)
        assert isinstance(path.speckle, ox.AbstractSpeckleField)
        assert path.speckle is sp

    def test_repr_shows_speckle_when_set(self, simple_primary, simple_detector):
        path = self._path(simple_primary, simple_detector, speckle=_MockSpeckle())
        assert any(line.startswith("  speckle:") for line in repr(path).split("\n"))


class TestFromDefaultSetup:
    """The convenience factory threads the speckle field through."""

    def test_default_is_none(self):
        path = ox.OpticalPath.from_default_setup(scalar_coronagraph())
        assert path.speckle is None

    def test_accepts_speckle(self):
        sp = _MockSpeckle()
        path = ox.OpticalPath.from_default_setup(scalar_coronagraph(), speckle=sp)
        assert path.speckle is sp


class TestRealizeAverage:
    """Exposure averaging over the contract, provided once for every field."""

    def test_matches_the_midpoint_rule_by_hand(self):
        sp = _MockSpeckle()
        got = sp.realize_average(wavelength_nm=500.0, exposure_s=4.0, n_sub=4)
        expected = (
            sum(
                sp.realize(wavelength_nm=500.0, time_d=t * s2d)
                for t in (0.5, 1.5, 2.5, 3.5)
            )
            / 4.0
        )
        assert jnp.allclose(got, expected, rtol=8 * np.finfo(np.float32).eps, atol=0.0)

    def test_one_substep_is_the_exposure_midpoint(self):
        sp = _MockSpeckle()
        got = sp.realize_average(wavelength_nm=500.0, exposure_s=6.0, n_sub=1)
        expected = sp.realize(wavelength_nm=500.0, time_d=3.0 * s2d)
        assert jnp.allclose(got, expected, rtol=8 * np.finfo(np.float32).eps, atol=0.0)

    def test_honors_the_start_time(self):
        sp = _MockSpeckle()
        shifted = sp.realize_average(
            wavelength_nm=500.0, exposure_s=2.0, start_time_d=10.0 * s2d, n_sub=4
        )
        base = sp.realize_average(wavelength_nm=500.0, exposure_s=2.0, n_sub=4)
        assert float(jnp.max(jnp.abs(shifted - base))) > 0.0

    def test_a_static_field_is_unchanged_by_averaging(self):
        class _Static(ox.AbstractSpeckleField):
            pixel_scale_lod: float = 0.25
            epoch_d: float = 0.0

            def realize(self, *, wavelength_nm, time_d=0.0):
                return jnp.full((4, 4), 3e-11)

        averaged = _Static().realize_average(wavelength_nm=500.0, exposure_s=1e4)
        assert jnp.allclose(
            averaged, 3e-11, rtol=8 * np.finfo(np.float32).eps, atol=0.0
        )

    def test_converges_as_n_sub_grows(self):
        """Second-order midpoint rule: quadrupling the sub-steps cuts the
        residual against a fine reference by about 16x."""
        sp = _MockSpeckle()
        kw = dict(wavelength_nm=500.0, exposure_s=8.0)
        reference = sp.realize_average(**kw, n_sub=4096)
        coarse = float(jnp.max(jnp.abs(sp.realize_average(**kw, n_sub=4) - reference)))
        fine = float(jnp.max(jnp.abs(sp.realize_average(**kw, n_sub=16) - reference)))
        assert coarse / fine > 8.0

    def test_is_jit_and_grad_safe(self):
        sp = _MockSpeckle()

        def total(exposure_s):
            return jnp.sum(
                sp.realize_average(wavelength_nm=500.0, exposure_s=exposure_s, n_sub=8)
            )

        assert jnp.isfinite(eqx.filter_jit(total)(4.0))
        assert jnp.isfinite(jax.grad(total)(4.0))

    def test_rejects_a_bad_substep_count(self):
        with pytest.raises(ValueError, match="n_sub"):
            _MockSpeckle().realize_average(wavelength_nm=500.0, exposure_s=1.0, n_sub=0)


def test_day_clock_matches_parent_formula():
    """Parent cos(seconds) model: 16 eps, or 1e-12 after epoch subtraction.

    At epoch=.25, day subtraction loses at most eps*.25*86400 seconds;
    the .1*cos phase response bounds its relative effect below 1e-12.
    """
    with jax.enable_x64():
        t = 2.3 / 86400.0
        assert d2s == 86400.0
        np.testing.assert_allclose(
            _MockSpeckle().realize(wavelength_nm=500.0, time_d=t),
            1e-11 * (1 + 0.1 * np.cos(86400.0 * t)),
            rtol=16 * np.finfo(float).eps,
            atol=0.0,
        )
        sp = _MockSpeckle(epoch_d=0.25)
        np.testing.assert_allclose(
            sp.realize(wavelength_nm=500.0, time_d=0.25 + t),
            1e-11 * (1 + 0.1 * np.cos(86400.0 * t)),
            rtol=1e-12,
            atol=0.0,
        )
        midpoint_s = (np.arange(8) + 0.5) * 4.0 / 8
        # Parent averaging evaluated the same cos field at elapsed seconds.
        expected_average = np.mean(1e-11 * (1 + 0.1 * np.cos(86400.0 * t + midpoint_s)))
        np.testing.assert_allclose(
            sp.realize_average(
                wavelength_nm=500.0, start_time_d=0.25 + t, exposure_s=4.0, n_sub=8
            ),
            expected_average,
            rtol=1e-12,
            atol=0.0,
        )
