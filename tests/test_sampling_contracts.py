"""Grid sampling, transmission and trapezoid-weight contracts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from optixstuff.grids import Image, Measure, PixelGrid, resample
from optixstuff.spectral import Band, Transmission, spectral_breaks, trapezoid_weights


@pytest.fixture(autouse=True, scope="module")
def _clear_caches():
    yield
    jax.clear_caches()


@pytest.mark.float64_only(reason="float64 roundoff in the reference calculation")
def test_density_evaluates_centers_without_area_or_box_factors():
    """Fails if point samples are deposited, box-averaged or area-scaled.

    Local linear/Keys kernels reproduce an interior affine density exactly;
    nearest chooses floor(p+1/2). DFT modes have exact trigonometric
    reconstructions. Absolute tolerance 2e-13 allows float64 kernel and
    matrix summation at density <10. Each kernel needs a distinct stencil.
    """
    source = PixelGrid.regular((8, 8), 0.5, "arcsec")
    x, y = source.coordinates()
    image = Image(3 + 2 * x - y, source, (), Measure.density("grid"), "ph/s")
    for step in (0.25, 0.75):
        target = PixelGrid.regular((2, 3), step, "arcsec")
        tx, ty = target.coordinates()
        px, py = source.unit_to_pixel(tx, ty)
        nearest_x, nearest_y = source.pixel_to_unit(
            np.floor(px + 0.5), np.floor(py + 0.5)
        )
        for kernel in ("nearest", "linear", "cubic"):
            expected = (
                3 + 2 * nearest_x - nearest_y
                if kernel == "nearest"
                else 3 + 2 * tx - ty
            )
            result = resample(image, target, reconstruction=kernel)
            np.testing.assert_allclose(result.data, expected, rtol=0, atol=2e-13)
            assert result.measure == Measure.density("grid")
        sy, sx = np.indices(source.shape)
        data = 4 + np.cos(2 * np.pi * sx / 8) * np.cos(2 * np.pi * sy / 8)
        expected = 4 + np.cos(2 * np.pi * px / 8) * np.cos(2 * np.pi * py / 8)
        sampled = replace(image, data=data, reconstruction="band_limited")
        result = resample(sampled, target, pad_factor=1)
        np.testing.assert_allclose(result.data, expected, rtol=0, atol=2e-13)


@pytest.mark.float64_only(reason="finite-difference gradient reference")
def test_transmission_support_moments_band_integrals_and_transforms():
    """Fails if interpolation clamps, knots go missing or band edges are ignored.

    Triangle (400,0),(500,1),(700,0) has area 150 nm and centroid 1600/3.
    Over [450,600], area=225/2 and first moment=58750 from integrating
    (w-400)/100 and (700-w)/200 on either side of 500. Order two is exact
    for a linear source times response. rtol=2e-14 is float64 roundoff;
    An added source kink at 550 contributes integral_0^50 t*(3/4-t/200) dt.
    Derivative finite differences allow 1e-7, including table parameters.
    No prior owner table test exists.
    """
    curve = Transmission([400, 500, 700], [0, 1, 0])
    wavelengths = jnp.array([399, 400, 450, 600, 700, 701.0])
    np.testing.assert_allclose(
        jax.jit(lambda t, w: t(w))(curve, wavelengths),
        [0, 0, 0.5, 0.5, 0, 0],
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_array_equal(spectral_breaks(curve), [400, 500, 700])
    np.testing.assert_allclose(curve.mean_nm, 1600 / 3, rtol=2e-14)
    np.testing.assert_allclose(
        curve.integrate(Band(300, 800), order=1), 150, rtol=2e-14
    )
    band = Band(450, 600)
    np.testing.assert_allclose(curve.integrate(band, order=1), 225 / 2, rtol=2e-14)
    np.testing.assert_allclose(
        curve.integrate(band, lambda w: w, order=2),
        58750,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        curve.integrate(
            band,
            lambda w: jnp.array([1.0, jnp.maximum(w - 550, 0)]),
            order=2,
            breaks_nm=jnp.array([550.0]),
        ),
        [225 / 2, (3 / 4) * 50**2 / 2 - 50**3 / 600],
        rtol=2e-14,
    )
    hat = Transmission.from_band(Band(500, 600))
    np.testing.assert_array_equal(hat(jnp.array([499, 500, 600, 601])), [0, 1, 1, 0])
    assert hat.low_nm == 500 and hat.high_nm == 600
    np.testing.assert_allclose(hat.mean_nm, 550, rtol=2e-14)
    np.testing.assert_allclose(hat.integrate(Band(510, 570)), 60, rtol=2e-14)
    assert hat.integrate(Band(610, 700)) == 0
    opaque = Transmission([400, 700], [0, 0])
    assert opaque.integrate(Band(400, 700)) == 0
    assert np.isnan(opaque.mean_nm)
    for wave, response in (
        ([500], [1]),
        ([500, 400], [1, 1]),
        ([400, 400], [1, 1]),
        ([0, 500], [1, 1]),
        ([400, np.nan], [1, 1]),
        ([400, 500], [-1, 1]),
        ([400, 500], [1, np.inf]),
        ([400, 500], [1]),
    ):
        with pytest.raises(ValueError):
            Transmission(wave, response)
    check_grads(
        lambda t, w: t(w),
        (curve, jnp.array(550.0)),
        order=1,
        modes=("fwd", "rev"),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.float64_only(reason="float64 roundoff in the reference calculation")
def test_nonuniform_trapezoid_weights_batched_and_degenerate():
    """Fails if nonuniform PSD line widths or endpoint half-cells are wrong.

    The integral of 2*x+3 is x**2+3*x on each increasing grid. This exact
    algebra allows 2e-15 relative float64 summation error. Singleton span is
    exactly zero; validation assertions require exact failures.
    """
    x = jnp.array([[0, 0.25, 1, 3], [1, 2, 4, 8.0]])
    weights = jax.jit(trapezoid_weights)(x)
    expected = np.asarray(x[:, -1] ** 2 + 3 * x[:, -1] - x[:, 0] ** 2 - 3 * x[:, 0])
    np.testing.assert_allclose(jnp.sum(weights * (2 * x + 3), -1), expected, rtol=2e-15)
    np.testing.assert_array_equal(weights[0], [0.125, 0.5, 1.375, 1.0])
    np.testing.assert_array_equal(trapezoid_weights([2.0]), [0.0])
    for bad in ([], [1, 1], [2, 1], [0, np.inf]):
        with pytest.raises(ValueError):
            trapezoid_weights(bad)
