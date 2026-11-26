"""Code verification of detector rate units and sampling metadata."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.precision import PrecisionError

from optixstuff import Band, Detector, FramePlan, Quadrature, Rates
from optixstuff.detector import _floored


@vv.case("detector-readout", "code-verification")
def test_statistics_require_band_integration(precision):
    """Truth is the band-integrated electron input contract and zero-count law.

    Unit rejection, finite score after integration, and the zero-variance
    score at zero counts are exact predicates.
    """
    quad = Quadrature.build((Band(500, 501),), order=2)
    rates = Rates(quad, "electron", planet=jnp.ones((1, 2)))
    detector = Detector(pixel_scale_arcsec=0.01, shape=(1, 1))
    plan = FramePlan(1.0, 1.0)
    for evaluate in (
        lambda r: detector.log_prob(0, r, plan),
        lambda r: detector.draw(jax.random.key(0), r, plan),
        lambda r: detector.mean_and_variance(r, plan),
        lambda r: detector.fisher_root(r, plan),
    ):
        with pytest.raises(ValueError, match="band-integrat"):
            evaluate(rates)
    # Unit errors take precedence even when a replaced leaf would fail flooring.
    malformed = eqx.tree_at(lambda r: r.thermal, rates, jnp.ones(3))
    with pytest.raises(ValueError, match="band-integrat"):
        detector.log_prob(0, malformed, plan)
    if precision == np.float32:
        with pytest.raises(PrecisionError, match="log density"):
            detector.log_prob(1, rates.integrate(), plan)
        return
    score = detector.log_prob(1, rates.integrate(), plan)
    assert score.shape == (1,) and np.isfinite(score).all()
    zero = Rates(quad.bands, "electron")
    np.testing.assert_array_equal(detector.log_prob(0, zero, plan), [0.0])


@vv.case("detector-readout", "code-verification")
@pytest.mark.parametrize("carrier", ["electron", "photon"])
def test_floor_preserves_sampling(carrier):
    """Truth is the input carrier and quadrature of both aperture moments.

    Metadata, quadrature identity, and PyTree structure must agree exactly.
    """
    quad = Quadrature.build((Band(500, 550), Band(550, 600)), order=2)
    rates = Rates(quad, carrier, variance=Rates(quad, carrier))
    floored, _ = _floored(rates)
    assert jax.tree.structure(floored) == jax.tree.structure(rates)
    for moment in (floored, floored.variance):
        assert moment.carrier == carrier and moment.spectral
        assert moment.quadrature is quad
