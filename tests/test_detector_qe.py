"""Code verification of the photon-to-electron boundary."""

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.spectral import Quadrature

import optixstuff as ox


@pytest.mark.parametrize("spatial", [(), (2, 3)])
def test_electrons_from_primitives(spatial):
    """Fails if survival is omitted or QE is applied after integration.

    Floating-point: 16 float64 eps for interpolation and three weighted
    products. Negative controls isolate survival and spectral ordering.
    """
    with jax.enable_x64():
        quad = Quadrature(
            jnp.array([[400, 450, 500], [600, 700, 800]], dtype=jnp.float64),
            jnp.array([[2, 3, 5], [7, 11, 13]], dtype=jnp.float64),
        )
        det = ox.Detector(
            pixel_scale_arcsec=0.01,
            shape=(2, 3),
            survival_fraction=0.9,
            qe_curve=([400, 600, 800], [0.2, 0.8, 0.4]),
        )
        base = np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
        image = base.reshape((2, 3, *(1 for _ in spatial))) * np.ones(spatial)
        rates = ox.PhotonRates(
            **{f.name: (i + 1) * image for i, f in enumerate(fields(ox.PhotonRates))}
        )
        qe = [[0.2, 0.35, 0.5], [0.8, 0.6, 0.4]]
        expected = np.zeros((2, *spatial))
        for w in range(2):
            for q in range(3):
                expected[w] += (
                    float(quad.weights_nm[w, q]) * 0.9 * qe[w][q] * image[w, q]
                )
        actual = det.electrons(rates, quad)
        compiled = eqx.filter_jit(lambda d, r, q: d.electrons(r, q))(det, rates, quad)
        tol = 16 * np.finfo(float).eps
        for i, field in enumerate(fields(rates)):
            np.testing.assert_allclose(
                getattr(actual, field.name), (i + 1) * expected, rtol=tol, atol=0
            )
            np.testing.assert_allclose(
                getattr(compiled, field.name), (i + 1) * expected, rtol=tol, atol=0
            )
        unit_survival = eqx.tree_at(lambda d: d.survival_fraction, det, jnp.array(1.0))
        np.testing.assert_allclose(
            actual.planet / unit_survival.electrons(rates, quad).planet,
            0.9,
            rtol=tol,
            atol=0,
        )
        integrated = np.asarray(rates.integrate(quad).planet)
        mean_qe = np.sum(np.asarray(quad.weights_nm) * qe, axis=1) / np.sum(
            np.asarray(quad.weights_nm), axis=1
        )
        wrong_order = 0.9 * integrated * mean_qe.reshape((2, *(1 for _ in spatial)))
        assert np.all(np.abs(np.asarray(actual.planet) / wrong_order - 1) > 0.02)


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
def test_yield_contract(detector_cls):
    """Fails if constant QE is scalar or yield is applied twice; 8 float32 eps."""
    det = detector_cls(pixel_scale_arcsec=0.01, shape=(2, 3), survival_fraction=0.9)
    quad = Quadrature(jnp.array([[450.0, 550.0, 650.0]]), jnp.array([[2.0, 3.0, 5.0]]))
    rates = ox.PhotonRates(planet=jnp.ones((1, 3, 2, 3)))
    assert det.quantum_efficiency(quad.nodes_nm).shape == (1, 3)
    result = det.electrons(rates, quad)
    expected = jnp.sum(quad.weights_nm * det.quantum_efficiency(quad.nodes_nm)) * 0.9
    np.testing.assert_allclose(
        result.total(), expected, rtol=8 * np.finfo(np.float32).eps
    )
    mean, _ = det.mean_and_variance(result, ox.FramePlan(1.0, 1.0), n_pix=0.0)
    np.testing.assert_allclose(mean, result.total(), rtol=8 * np.finfo(np.float32).eps)
    assert result.star_leakage.shape == (1, 2, 3)


def test_linear_qe():
    """Fails if interpolation or zero extrapolation changes; 8 float32 eps."""
    det = ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(1, 1),
        qe_curve=([400, 600, 800], [0.2, 0.8, 0.4]),
    )
    wavelengths = jnp.array([[300.0, 400.0, 500.0], [600.0, 700.0, 900.0]])
    expected = [[0.0, 0.2, 0.5], [0.8, 0.6, 0.0]]
    for result in [
        det.quantum_efficiency(wavelengths),
        eqx.filter_jit(lambda d, w: d.quantum_efficiency(w))(det, wavelengths),
    ]:
        np.testing.assert_allclose(
            result, expected, rtol=8 * np.finfo(np.float32).eps, atol=0
        )
    constant = ox.Detector(pixel_scale_arcsec=0.01, shape=(1, 1), qe=0.7)
    np.testing.assert_allclose(
        constant.quantum_efficiency(wavelengths),
        0.7,
        rtol=8 * np.finfo(np.float32).eps,
        atol=0,
    )
    # Quadrature breakpoints: the table points, none for a constant QE (exact).
    np.testing.assert_array_equal(det.quadrature_knots_nm(), [400.0, 600.0, 800.0])
    assert constant.quadrature_knots_nm().shape == (0,)


@pytest.mark.parametrize("detector_cls", ox.AbstractDetector.__subclasses__())
def test_concrete_validation(detector_cls):
    """Fails if concrete out-of-range survival or QE is accepted; exact rejection."""
    for bad in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="survival_fraction"):
            detector_cls(pixel_scale_arcsec=0.01, shape=(1, 1), survival_fraction=bad)
    if detector_cls is ox.Detector:
        for kwargs in [
            {"qe": -0.1},
            {"qe": 1.1},
            {"qe_curve": ([400, 500], [0.5, 1.1])},
        ]:
            with pytest.raises(ValueError, match="QE"):
                detector_cls(pixel_scale_arcsec=0.01, shape=(1, 1), **kwargs)
