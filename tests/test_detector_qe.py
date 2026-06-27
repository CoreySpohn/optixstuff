"""Spectral quantum efficiency on Detector.get_qe.

Detector accepts an optional (wavelengths_nm, qe) curve; get_qe interpolates
it (linear, zero outside the measured range), reusing the SpectralThroughput
interpolation pattern. Without a curve, get_qe returns the scalar QE, and
IdealDetector stays constant-QE by design.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

import optixstuff as ox

WL = jnp.array([400.0, 600.0, 800.0])
QE = jnp.array([0.2, 0.8, 0.4])


def _spectral_detector():
    return ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(32, 32),
        qe_curve=(WL, QE),
    )


def _scalar_detector():
    return ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(32, 32),
        quantum_efficiency=0.7,
    )


class TestSpectralQE:
    def test_interpolates_at_sample_points(self):
        det = _spectral_detector()
        assert float(det.get_qe(400.0)) == pytest.approx(0.2)
        assert float(det.get_qe(600.0)) == pytest.approx(0.8)
        assert float(det.get_qe(800.0)) == pytest.approx(0.4)

    def test_linear_between_samples(self):
        det = _spectral_detector()
        assert float(det.get_qe(500.0)) == pytest.approx(0.5)  # midway 0.2->0.8
        assert float(det.get_qe(700.0)) == pytest.approx(0.6)  # midway 0.8->0.4

    def test_zero_outside_measured_range(self):
        det = _spectral_detector()
        assert float(det.get_qe(300.0)) == pytest.approx(0.0)
        assert float(det.get_qe(900.0)) == pytest.approx(0.0)

    def test_vectorized_over_wavelength(self):
        det = _spectral_detector()
        out = det.get_qe(jnp.array([400.0, 500.0, 600.0]))
        assert jnp.allclose(out, jnp.array([0.2, 0.5, 0.8]))

    def test_jittable(self):
        # idiomatic: detector is a traced PyTree argument (as in a jitted ETC),
        # not a static closure -- it holds interpax arrays, like SpectralThroughput.
        det = _spectral_detector()
        qe_at = eqx.filter_jit(lambda d, wl: d.get_qe(wl))
        assert float(qe_at(det, 500.0)) == pytest.approx(0.5)


class TestScalarQEUnchanged:
    def test_scalar_detector_ignores_wavelength(self):
        det = _scalar_detector()
        assert float(det.get_qe(450.0)) == pytest.approx(0.7)
        assert float(det.get_qe(900.0)) == pytest.approx(0.7)

    def test_ideal_detector_stays_constant(self):
        det = ox.IdealDetector(
            pixel_scale_arcsec=0.01, shape=(8, 8), quantum_efficiency=0.6
        )
        assert float(det.get_qe(400.0)) == pytest.approx(0.6)
        assert float(det.get_qe(1200.0)) == pytest.approx(0.6)
