"""Tests for AbstractSpeckleField and the OpticalPath.speckle field."""

from __future__ import annotations

import equinox as eqx
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
