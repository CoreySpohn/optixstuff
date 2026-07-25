"""Tests for AbstractSpeckleField and the OpticalPath.speckle field."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import optixstuff as ox
from optixstuff.coronagraph import AbstractScalarCoronagraph


class _MockCoro(AbstractScalarCoronagraph):
    """Minimal scalar-only coronagraph for OpticalPath construction."""

    pixel_scale_lod: float = 0.25
    IWA: float = 2.0
    OWA: float = 30.0

    def throughput(self, sep, wl, *, time_s=0.0):
        return 0.5

    def core_area(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(self, sep, wl, *, time_s=0.0):
        return 1e-10

    def occulter_transmission(self, sep, wl, *, time_s=0.0):
        return 1.0


class _MockSpeckle(ox.AbstractSpeckleField):
    """Incoherent-halo-style speckle field with a mild time modulation."""

    pixel_scale_lod: float = 0.25
    epoch_jd: float = 2451545.0  # J2000

    def realize(self, *, wavelength_nm, time_s=0.0):
        scale = 1.0 + 0.1 * jnp.cos(jnp.asarray(time_s, dtype=float))
        return jnp.full((8, 8), 1e-11) * scale


class TestInterface:
    """The realize contract and abstractness."""

    def test_realize_shape(self):
        m = _MockSpeckle().realize(wavelength_nm=500.0, time_s=0.0)
        assert m.shape == (8, 8)

    def test_realize_is_keyword_only(self):
        with pytest.raises(TypeError):
            _MockSpeckle().realize(500.0)

    def test_realize_time_varying(self):
        sp = _MockSpeckle()
        a = sp.realize(wavelength_nm=500.0, time_s=0.0)
        b = sp.realize(wavelength_nm=500.0, time_s=float(jnp.pi))
        assert float(jnp.max(jnp.abs(a - b))) > 0.0

    def test_realize_defaults_time_zero(self):
        sp = _MockSpeckle()
        assert jnp.allclose(
            sp.realize(wavelength_nm=500.0),
            sp.realize(wavelength_nm=500.0, time_s=0.0),
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
            coronagraph=_MockCoro(),
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
        path = ox.OpticalPath.from_default_setup(_MockCoro())
        assert path.speckle is None

    def test_accepts_speckle(self):
        sp = _MockSpeckle()
        path = ox.OpticalPath.from_default_setup(_MockCoro(), speckle=sp)
        assert path.speckle is sp


class TestRealizeAverage:
    """Exposure averaging over the contract, provided once for every field."""

    def test_matches_the_midpoint_rule_by_hand(self):
        sp = _MockSpeckle()
        got = sp.realize_average(wavelength_nm=500.0, exposure_s=4.0, n_sub=4)
        expected = (
            sum(sp.realize(wavelength_nm=500.0, time_s=t) for t in (0.5, 1.5, 2.5, 3.5))
            / 4.0
        )
        assert jnp.allclose(got, expected, rtol=1e-12)

    def test_one_substep_is_the_exposure_midpoint(self):
        sp = _MockSpeckle()
        got = sp.realize_average(wavelength_nm=500.0, exposure_s=6.0, n_sub=1)
        expected = sp.realize(wavelength_nm=500.0, time_s=3.0)
        assert jnp.allclose(got, expected, rtol=1e-12)

    def test_honors_the_start_time(self):
        sp = _MockSpeckle()
        shifted = sp.realize_average(
            wavelength_nm=500.0, exposure_s=2.0, start_time_s=10.0, n_sub=4
        )
        base = sp.realize_average(wavelength_nm=500.0, exposure_s=2.0, n_sub=4)
        assert float(jnp.max(jnp.abs(shifted - base))) > 0.0

    def test_a_static_field_is_unchanged_by_averaging(self):
        class _Static(ox.AbstractSpeckleField):
            pixel_scale_lod: float = 0.25
            epoch_jd: float = 2451545.0

            def realize(self, *, wavelength_nm, time_s=0.0):
                return jnp.full((4, 4), 3e-11)

        averaged = _Static().realize_average(wavelength_nm=500.0, exposure_s=1e4)
        assert jnp.allclose(averaged, 3e-11, rtol=1e-12)

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
