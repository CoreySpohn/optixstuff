"""Aperture quadrature and immutable, single-axis photometric records.

All inputs are source-flux Images and lambda/D coordinates; no data reader is
required. Host construction prepares geometry and interpolation coefficients.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from types import MappingProxyType
from typing import ClassVar

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import root_scalar

from optixstuff.grids import (
    Frame,
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
)
from optixstuff.radial import radial_profile

from .apertures import ApertureRule, Circular, HalfMaximum

CORE_OVERSAMPLE = 10
PEAK_SEARCH_RADIUS_LOD = 1.0
CORE_WINDOW_LOD = 3.0
PROFILE_STEP_PIX = 0.25
RING_RULE = "aperture_ring_mean"
LEAKAGE_RULES = ("aperture_mean", RING_RULE, "azimuthal_ring_mean")
# Quantities that carry the leakage of the star and so depend on its diameter.
DIAMETER_QUANTITIES = (
    "leakage_fraction",
    "leakage_fraction_variance",
    "leakage_density_lod2",
    "leakage_density_variance_lod2",
    "raw_contrast",
)
# A record at one diameter serves a star within this relative tolerance of it
# (plus the absolute floor): the leakage excess grows as the diameter squared.
_SINGLE_DIAMETER_RTOL = 1e-3
_SINGLE_DIAMETER_ATOL = 1e-6
# Slack, as a fraction of the largest tabulated diameter, on the table's range.
_DIAMETER_RANGE_RTOL = 1e-6
_QUINTIC_PAD = 48


def core_window_half_width_pix(pixel_scale_lod: float) -> int:
    """Half-width in pixels of the core-area window for a pixel scale.

    Args:
        pixel_scale_lod: Pixel scale in lambda/D per pixel.

    Returns:
        Half-width in pixels covering ``CORE_WINDOW_LOD`` lambda/D, rounded up.
    """
    return int(np.ceil(CORE_WINDOW_LOD / pixel_scale_lod))


def _band_limited_samples(spectrum, start, n_pix, oversample):
    """Inverse-DFT basis onto the subpixel centers of ``n_pix`` pixels.

    Row ``j`` evaluates the image's trigonometric interpolant at pixel
    coordinate ``start + (j + 0.5) / oversample - 0.5``, the center of
    subpixel ``j`` counted from pixel ``start``.
    """
    size = spectrum.shape[0]
    coords = start + (jnp.arange(n_pix * oversample) + 0.5) / oversample - 0.5
    freqs = jnp.fft.fftfreq(size)
    return jnp.exp(2j * jnp.pi * coords[:, None] * freqs[None, :]) / size


@functools.partial(
    jax.jit, static_argnames=("oversample", "half_width_pix", "return_gradient")
)
def core_subpixel_window(
    psf,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    nominal_center=None,
    search_radius_pix=None,
    return_gradient=False,
):
    """The PSF's band-limited interpolant on subpixels about the source peak.

    The window is ``2 * half_width_pix + 1`` pixels on a side (clipped to the
    image) about the brightest pixel, and each pixel is split into
    ``oversample`` x ``oversample`` subpixels. The values are the PSF's
    trigonometric interpolant, a matrix DFT of the whole image evaluated at
    the subpixel centers, in the PSF's units (flux per pixel at that point).
    This is the image :func:`truncation_core` thresholds.

    Args:
        psf: 2-D PSF image.
        oversample: Subpixels per pixel side.
        half_width_pix: Half-width in pixels of the window.
        nominal_center: Optional source (row, column) in original pixels.
        search_radius_pix: Peak-search radius about ``nominal_center``.
        return_gradient: Also return the two spectral derivatives, in
            intensity per original pixel coordinate.

    Returns:
        ``(fine, start_y, start_x)``: the subpixel image, and the row and
        column of the window's first pixel in ``psf``. With
        ``return_gradient=True``, append ``grad_y, grad_x``.
    """
    center = _peak_index(psf, nominal_center, search_radius_pix)
    return _subpixel_window_at(psf, center, oversample, half_width_pix, return_gradient)


def _subpixel_window_at(psf, center, oversample, half_width_pix, return_gradient=False):
    """Evaluate a window about a supplied center without searching for a peak."""
    ny, nx = psf.shape
    n_y = min(2 * half_width_pix + 1, ny)
    n_x = min(2 * half_width_pix + 1, nx)
    row, col = jnp.rint(jnp.asarray(center)).astype(int)

    start_y = jnp.clip(row - half_width_pix, 0, ny - n_y)
    start_x = jnp.clip(col - half_width_pix, 0, nx - n_x)
    spectrum = jnp.fft.fft2(psf)
    basis_y = _band_limited_samples(spectrum, start_y, n_y, oversample)
    basis_x = _band_limited_samples(spectrum.T, start_x, n_x, oversample)
    fine = jnp.real(basis_y @ spectrum @ basis_x.T)
    if return_gradient:
        grad_y = jnp.real(
            (basis_y * (2j * jnp.pi * jnp.fft.fftfreq(ny))) @ spectrum @ basis_x.T
        )
        grad_x = jnp.real(
            basis_y @ spectrum @ (basis_x * (2j * jnp.pi * jnp.fft.fftfreq(nx))).T
        )
        return fine, start_y, start_x, grad_y, grad_x
    return fine, start_y, start_x


def _peak_index(image, nominal_center=None, search_radius_pix=None, valid=None):
    """Locate the brightest sample within the nominal source's search disk."""
    if nominal_center is None:
        return jnp.unravel_index(jnp.argmax(image), image.shape)
    rows = jnp.arange(image.shape[0])[:, None] - nominal_center[0]
    cols = jnp.arange(image.shape[1])[None, :] - nominal_center[1]
    inside = rows**2 + cols**2 <= search_radius_pix**2
    # Native offsets beyond the image retain the global, in-image core.
    valid = _nominal_in_image(nominal_center, image.shape) if valid is None else valid
    candidates = jnp.where(inside | ~valid, image, -jnp.inf)
    return jnp.unravel_index(jnp.argmax(candidates), image.shape)


def _nominal_in_image(nominal_center, shape):
    """Keep source validity in original image coordinates."""
    return (
        (nominal_center[0] >= 0)
        & (nominal_center[0] < shape[0])
        & (nominal_center[1] >= 0)
        & (nominal_center[1] < shape[1])
    )


def _window_peak(
    fine, start_y, start_x, oversample, nominal_center, search_radius_pix, image_shape
):
    """Peak sample in a window, restricted in global pixel coordinates."""
    center = (
        None
        if nominal_center is None
        else (
            (nominal_center[0] - start_y + 0.5) * oversample - 0.5,
            (nominal_center[1] - start_x + 0.5) * oversample - 0.5,
        )
    )
    row, col = _peak_index(
        fine,
        center,
        search_radius_pix * oversample,
        valid=None
        if nominal_center is None
        else _nominal_in_image(nominal_center, image_shape),
    )
    return jnp.array(
        [
            start_y + (row + 0.5) / oversample - 0.5,
            start_x + (col + 0.5) / oversample - 0.5,
        ]
    )


def _refine_peak(psf, center, oversample, nominal_center=None, search_radius_pix=None):
    """Refine an interior maximum using derivatives of the same DFT basis.

    Interior Newton steps require a negative-definite Hessian and stay
    within one subpixel of the initial sample. Near the search-disk boundary,
    also optimize along its circumference; flat maxima stay put.
    """
    spectrum = jnp.fft.fft2(psf)
    fy = 2j * jnp.pi * jnp.fft.fftfreq(psf.shape[0])
    fx = 2j * jnp.pi * jnp.fft.fftfreq(psf.shape[1])

    def evaluate(point):
        modes = (
            jnp.exp(fy * point[0])[:, None]
            * spectrum
            * jnp.exp(fx * point[1])[None, :]
            / psf.size
        )
        value = jnp.real(jnp.sum(modes))
        gy = jnp.real(jnp.sum(modes * fy[:, None]))
        gx = jnp.real(jnp.sum(modes * fx[None, :]))
        yy = jnp.real(jnp.sum(modes * fy[:, None] ** 2))
        xx = jnp.real(jnp.sum(modes * fx[None, :] ** 2))
        yx = jnp.real(jnp.sum(modes * fy[:, None] * fx[None, :]))
        det = yy * xx - yx**2
        safe_det = jnp.where(det > 0, det, 1.0)
        step = jnp.array([xx * gy - yx * gx, yy * gx - yx * gy]) / safe_det
        return (
            value,
            step,
            (yy < 0) & (det > 0),
            jnp.array([gy, gx]),
            jnp.array([[yy, yx], [yx, xx]]),
        )

    def refine(_, point):
        _, step, maximum, _, _ = evaluate(point)
        candidate = point - step
        accept = maximum & jnp.all(jnp.abs(candidate - center) <= 1 / oversample)
        return jnp.where(accept, candidate, point)

    point = jax.lax.fori_loop(0, 3, refine, center)
    if nominal_center is not None:
        nominal = jnp.asarray(nominal_center)
        in_image = jnp.all(nominal >= 0) & jnp.all(nominal < jnp.array(psf.shape))
        distance = jnp.linalg.norm(center - nominal)
        inside = jnp.linalg.norm(point - nominal) <= search_radius_pix
        point = jnp.where(in_image & ~inside, center, point)

        def boundary_peak(point):
            def on_circle(angle):
                return nominal + search_radius_pix * jnp.array(
                    [jnp.sin(angle), jnp.cos(angle)]
                )

            def refine_angle(_, angle):
                candidate = on_circle(angle)
                _, _, _, gradient, hessian = evaluate(candidate)
                radial = candidate - nominal
                tangent = jnp.array([radial[1], -radial[0]])
                first = gradient @ tangent
                second = tangent @ hessian @ tangent - gradient @ radial
                step = first / jnp.where(second < 0, second, 1.0)
                return angle - jnp.where(second < 0, jnp.clip(step, -0.2, 0.2), 0.0)

            delta = center - nominal
            angle = jnp.arctan2(delta[0], delta[1])
            angle = jax.lax.fori_loop(0, 5, refine_angle, angle)
            candidate = on_circle(angle)
            valid = jnp.all(candidate >= -0.5) & jnp.all(
                candidate <= jnp.array(psf.shape) - 0.5
            )
            better = evaluate(candidate)[0] > evaluate(point)[0]
            return jnp.where(valid & better, candidate, point)

        point = jax.lax.cond(
            in_image & (distance >= search_radius_pix - 2 / oversample),
            boundary_peak,
            lambda p: p,
            point,
        )
    return point, evaluate(point)[0]


def _circle_square_weights(rows, cols, radius):
    """Exact disk overlap of unit squares centered at rows and cols.

    Integrate sqrt(radius**2 - x**2) analytically in each quadrant,
    then difference its rectangle primitive at the four square corners.
    The angle is atan2(z, w) with w = sqrt(radius**2 - z**2) supplied by the
    caller where it is known exactly: arcsin(z / radius) loses about
    sqrt(eps) relative near z = radius, where its slope diverges.
    """

    def primitive(x, y):
        a = jnp.minimum(jnp.abs(x), radius)
        b = jnp.minimum(jnp.abs(y), radius)
        w_a = jnp.sqrt(jnp.maximum(radius**2 - a**2, 0.0))
        chord = jnp.sqrt(jnp.maximum(radius**2 - b**2, 0.0))
        clipped = chord < a
        split = jnp.where(clipped, chord, a)
        # At split = chord the complementary leg is b exactly.
        w_split = jnp.where(clipped, b, w_a)

        def integral(z, w):
            return 0.5 * (z * w + radius**2 * jnp.arctan2(z, w))

        return (
            jnp.sign(x)
            * jnp.sign(y)
            * (b * split + integral(a, w_a) - integral(split, w_split))
        )

    fraction = (
        primitive(cols + 0.5, rows + 0.5)
        - primitive(cols - 0.5, rows + 0.5)
        - primitive(cols + 0.5, rows - 0.5)
        + primitive(cols - 0.5, rows - 0.5)
    )
    inner = (jnp.abs(rows) + 0.5) ** 2 + (jnp.abs(cols) + 0.5) ** 2 <= radius**2
    outer = (
        jnp.maximum(jnp.abs(rows) - 0.5, 0) ** 2
        + jnp.maximum(jnp.abs(cols) - 0.5, 0) ** 2
        >= radius**2
    )
    return jnp.where(inner, 1.0, jnp.where(outer, 0.0, jnp.clip(fraction, 0, 1)))


def _measure_circular_aperture(
    psf,
    radius_pix,
    pixel_scale_lod,
    oversample,
    half_width_pix,
    star_psf=None,
    nominal_center=None,
    star_frame=None,
):
    """Host measurement boundary using float64 to resolve faint stellar leakage."""
    with jax.enable_x64():
        values = circular_aperture(
            jnp.asarray(psf, dtype=jnp.float64),
            radius_pix,
            pixel_scale_lod,
            oversample=oversample,
            half_width_pix=half_width_pix,
            nominal_center=nominal_center,
            star_frame=star_frame,
            star_psf=None
            if star_psf is None
            else jnp.asarray(star_psf, dtype=jnp.float64),
        )
        return tuple(float(value) for value in values)


@functools.partial(
    jax.jit, static_argnames=("oversample", "half_width_pix", "return_samples")
)
def circular_aperture(
    psf,
    radius_pix,
    pixel_scale_lod,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    star_psf=None,
    nominal_center=None,
    fixed_center=None,
    star_frame=None,
    return_samples=False,
):
    """Flux and exact area of a circle on the PSF's trigonometric interpolant.

    Each subpixel carries its exact area fraction inside ``radius_pix``.
    Flux is the sum of these weights times the interpolated center intensity,
    divided by ``oversample**2``; area is their sum times the subpixel solid
    angle. The circle is centered on the refined interpolated peak. An optional
    ``nominal_center=(row, column)`` restricts the peak search to a disk of
    radius :data:`PEAK_SEARCH_RADIUS_LOD` in lambda/D about the source.
    Set ``fixed_center=(row, column)`` for photometry at a known centroid
    on any image, without peak search. It takes precedence over
    ``nominal_center``. No flux renormalization or clipping is applied.

    The window must contain the circle plus a margin for locating the peak;
    it is clipped to the image as in :func:`core_subpixel_window`. When
    ``star_psf`` is supplied, its interpolant is measured at the same global
    coordinates through the same weights as the planet. An optional
    ``star_frame=(star_cy,star_cx,off_cy,off_cx,pitch_ratio)`` maps detector
    coordinates to a distinct stellar frame and converts pixel areas.
    Sampling at 0.5 lambda/D per pixel or finer is required for an unaliased
    reconstruction; finite-image periodic boundaries can still affect edges.
    This kernel uses the active JAX precision and supports ``jit`` and
    ``vmap``. Performance curves evaluate it in float64 to avoid cancellation
    in faint stellar sums, without changing the global precision setting.

    Returns:
        ``(planet_flux, area, star_flux)``; ``star_flux`` is zero when omitted.

    Args:
        psf: 2-D PSF image, flux per pixel.
        radius_pix: Circle radius in pixels.
        pixel_scale_lod: Pixel scale in lambda/D per pixel.
        oversample: Subpixels per pixel side.
        half_width_pix: Half-width in pixels of the measurement window.
        star_psf: Optional stellar image measured through the same weights.
        nominal_center: Optional source (row, column) in pixels restricting the
            peak search.
        fixed_center: Optional (row, column) in pixels of the circle center, with
            no peak search.
        star_frame: Optional ``(star_cy, star_cx, off_cy, off_cx, pitch_ratio)``
            mapping detector coordinates to the stellar frame.
        return_samples: Return the subpixel image, circle weights and window
            start instead of the sums, as ``(fine, circle, start_y, start_x)``.
    """
    search_radius = PEAK_SEARCH_RADIUS_LOD / pixel_scale_lod
    if fixed_center is None:
        fine, start_y, start_x = core_subpixel_window(
            psf, oversample, half_width_pix, nominal_center, search_radius
        )
        center = _window_peak(
            fine, start_y, start_x, oversample, nominal_center, search_radius, psf.shape
        )
        refined, _ = _refine_peak(
            psf, center, oversample, nominal_center, search_radius
        )
    else:
        refined = jnp.asarray(fixed_center)
        fine, start_y, start_x = _subpixel_window_at(
            psf, refined, oversample, half_width_pix
        )
    # The refined peak can lie across the search disk from the native peak.
    # Recenter the integration window so its edge cannot clip an interior disk.
    fine, start_y, start_x = _subpixel_window_at(
        psf, refined, oversample, half_width_pix
    )
    rows = (
        start_y
        + (jnp.arange(fine.shape[0])[:, None] + 0.5) / oversample
        - 0.5
        - refined[0]
    ) * oversample
    cols = (
        start_x
        + (jnp.arange(fine.shape[1])[None, :] + 0.5) / oversample
        - 0.5
        - refined[1]
    ) * oversample
    circle = _circle_square_weights(rows, cols, radius_pix * oversample)
    if return_samples:
        return fine, circle, start_y, start_x
    flux = jnp.sum(circle * fine) / oversample**2
    area = jnp.sum(circle) * (pixel_scale_lod / oversample) ** 2
    star_flux = 0.0
    if star_psf is not None:
        spectrum = jnp.fft.fft2(star_psf)
        if star_frame is None:
            basis_y = _band_limited_samples(
                spectrum, start_y, fine.shape[0] // oversample, oversample
            )
            basis_x = _band_limited_samples(
                spectrum.T, start_x, fine.shape[1] // oversample, oversample
            )
            area_ratio = 1.0
        else:
            star_cy, star_cx, off_cy, off_cx, pitch_ratio = star_frame
            row_coords = (
                star_cy
                + (
                    start_y
                    + (jnp.arange(fine.shape[0]) + 0.5) / oversample
                    - 0.5
                    - off_cy
                )
                * pitch_ratio
            )
            col_coords = (
                star_cx
                + (
                    start_x
                    + (jnp.arange(fine.shape[1]) + 0.5) / oversample
                    - 0.5
                    - off_cx
                )
                * pitch_ratio
            )
            basis_y = (
                jnp.exp(
                    2j
                    * jnp.pi
                    * row_coords[:, None]
                    * jnp.fft.fftfreq(star_psf.shape[0])
                )
                / star_psf.shape[0]
            )
            basis_x = (
                jnp.exp(
                    2j
                    * jnp.pi
                    * col_coords[:, None]
                    * jnp.fft.fftfreq(star_psf.shape[1])
                )
                / star_psf.shape[1]
            )
            area_ratio = pitch_ratio**2
        star_fine = jnp.real(basis_y @ spectrum @ basis_x.T) * area_ratio
        star_flux = jnp.sum(circle * star_fine) / oversample**2
    return flux, area, star_flux


def _level_set_weights(fine, grad_y, grad_x, threshold, oversample):
    """Fraction of each subpixel above the tangent-plane threshold."""
    # Exact area under the local tangent line in a square. The sum of two
    # independent uniforms has a trapezoidal CDF; this also handles an
    # axis-aligned normal without dividing by its vanishing component.
    a = jnp.maximum(jnp.abs(grad_y), jnp.abs(grad_x)) / oversample
    b = jnp.minimum(jnp.abs(grad_y), jnp.abs(grad_x)) / oversample
    z = jnp.abs(fine - threshold)
    safe_a = jnp.where(a > 0, a, 1.0)
    safe_b = jnp.where(b > 0, b, 1.0)
    tail = jnp.where(
        z <= (a - b) / 2,
        0.5 - z / safe_a,
        jnp.maximum((a + b) / 2 - z, 0) ** 2 / (2 * safe_a * safe_b),
    )
    core = jnp.where(fine >= threshold, 1 - tail, tail)
    return jnp.where(a > 0, core, (fine > threshold).astype(fine.dtype))


def _global_truncation(psf, threshold, pixel_scale_lod, oversample):
    """Integrate the whole interpolant in strips with bounded working memory."""
    ny, nx = psf.shape
    spectrum = jnp.fft.fft2(psf)
    fx = jnp.fft.fftfreq(nx)
    # Zero-pad Fourier modes onto the same subpixel centers as the matrix
    # DFT window. Keep the signed Nyquist mode, then take the real part only
    # after both axes have been evaluated, as in core_subpixel_window.
    modes = jnp.mod(jnp.rint(fx * nx).astype(int), nx * oversample)
    phase = jnp.exp(2j * jnp.pi * fx * (0.5 / oversample - 0.5))
    padded = jnp.zeros((ny, nx * oversample), dtype=spectrum.dtype)
    along_x = (
        jnp.fft.ifft(padded.at[:, modes].set(spectrum * phase), axis=1) * oversample
    )
    dx = (
        jnp.fft.ifft(
            padded.at[:, modes].set(spectrum * phase * (2j * jnp.pi * fx)), axis=1
        )
        * oversample
    )
    strip_height = min(4, ny)

    def integrate(index, state):
        total, highest, count = state
        start = index * strip_height
        basis = _band_limited_samples(spectrum, start, strip_height, oversample)
        fine = jnp.real(basis @ along_x)
        gy = jnp.real((basis * (2j * jnp.pi * jnp.fft.fftfreq(ny))) @ along_x)
        gx = jnp.real(basis @ dx)
        weights = _level_set_weights(fine, gy, gx, threshold, oversample)
        inside = start + jnp.arange(strip_height * oversample) / oversample < ny
        weights = jnp.where(inside[:, None], weights, 0.0)
        maximum = jnp.max(jnp.where(inside[:, None], fine, -jnp.inf))
        ties = jnp.sum((fine == maximum) & inside[:, None], dtype=jnp.int32)
        count = jnp.where(
            maximum > highest, ties, jnp.where(maximum == highest, count + ties, count)
        )
        return (
            total + jnp.array([jnp.sum(weights * fine), jnp.sum(weights)]),
            jnp.maximum(highest, maximum),
            count,
        )

    sums, highest, count = jax.lax.fori_loop(
        0,
        (ny + strip_height - 1) // strip_height,
        integrate,
        (
            jnp.zeros(2, dtype=along_x.real.dtype),
            jnp.array(-jnp.inf, dtype=along_x.real.dtype),
            jnp.array(0, dtype=jnp.int32),
        ),
    )
    sums = jnp.where(highest > threshold, sums, jnp.array([highest * count, count]))
    return sums[0] / oversample**2, sums[1] * (pixel_scale_lod / oversample) ** 2


@functools.partial(jax.jit, static_argnames=("oversample", "half_width_pix"))
def _truncation_samples(
    psf, fraction, pixel_scale_lod, oversample, half_width_pix, nominal_center
):
    search_radius = PEAK_SEARCH_RADIUS_LOD / pixel_scale_lod
    fine, start_y, start_x, grad_y, grad_x = core_subpixel_window(
        psf,
        oversample,
        half_width_pix,
        nominal_center,
        search_radius,
        return_gradient=True,
    )
    center = _window_peak(
        fine, start_y, start_x, oversample, nominal_center, search_radius, psf.shape
    )
    _, peak = _refine_peak(psf, center, oversample, nominal_center, search_radius)
    threshold = fraction * peak
    core = _level_set_weights(fine, grad_y, grad_x, threshold, oversample)
    core = jnp.where(
        jnp.any(fine > threshold), core, (fine == jnp.max(fine)).astype(fine.dtype)
    )
    return fine, core, start_y, start_x, threshold


@functools.partial(jax.jit, static_argnames=("oversample", "half_width_pix"))
def truncation_core(
    psf,
    fraction,
    pixel_scale_lod,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    nominal_center=None,
):
    """Throughput and area of the region where a PSF exceeds a fraction of its peak.

    A subpixel's weight is the exact square fraction on the bright side of
    the tangent to the level set at its center. Both the intensity and its
    gradient come from the same trigonometric interpolant (derivatives of
    the DFT basis). Area and throughput use the same weights. This local
    linear boundary approximation is second order in subpixel width.

    The threshold is ``fraction`` times the refined interpolated peak.
    If no subpixel exceeds it, the original ``fine == fine.max()`` safeguard
    selects all tied peak samples across the measurement domain: the local
    window for an in-image source, otherwise the whole image.

    With an in-image ``nominal_center``, the peak search stays within
    :data:`PEAK_SEARCH_RADIUS_LOD` of that position and only the local core
    window contributes. This excludes unrelated leakage elsewhere inside the
    IWA. Without a nominal position, or for a native offset outside the image,
    retain the global peak and integrate the whole image in strips with the
    same interpolant and fractional weights, preserving in-image power.

    An intensity PSF sampled at 0.5 lambda/D per pixel or finer supports
    band-limited reconstruction. The default ten subpixels per side resolve
    the tested Airy boundaries to 0.1%; finite-image interpolation and more
    complicated level sets still require convergence checks.

    The shapes are fixed by the static arguments, so the function is
    ``jit``-compiled once and can be ``vmap``-ped over a stack of PSFs.

    Args:
        psf: 2-D PSF image.
        fraction: Fraction of the peak that bounds the core.
        pixel_scale_lod: Pixel scale in lambda/D per pixel.
        oversample: Subpixels per pixel side.
        half_width_pix: Half-width in pixels of the subpixel window, clipped
            to the image.
        nominal_center: Optional source (row, column) in original pixels.

    Returns:
        ``(throughput, area)``: the core throughput in the PSF's flux units
        and the core area in (lambda/D)^2.
    """
    fine, core, _start_y, _start_x, threshold = _truncation_samples(
        psf,
        fraction,
        pixel_scale_lod,
        oversample,
        half_width_pix,
        nominal_center,
    )
    local = (
        jnp.sum(core * fine) / oversample**2,
        jnp.sum(core) * (pixel_scale_lod / oversample) ** 2,
    )
    in_image = (
        False
        if nominal_center is None
        else _nominal_in_image(nominal_center, psf.shape)
    )
    return jax.lax.cond(
        in_image,
        lambda: local,
        lambda: _global_truncation(psf, threshold, pixel_scale_lod, oversample),
    )


def profile_radii(shape: tuple[int, int], center: tuple[float, float]) -> np.ndarray:
    """Radii, in pixels, at which the core mean intensity profile is tabulated.

    The grid starts on the axis and steps by :data:`PROFILE_STEP_PIX` out to
    the farthest pixel center of an image of ``shape`` measured from
    ``center``, so the whole map is covered.

    Args:
        shape: ``(ny, nx)`` of the stellar intensity map.
        center: ``(row, column)`` of the optical axis as pixel-center indices.

    Returns:
        1-D array of radii in pixels.
    """
    ny, nx = shape
    cy, cx = center
    r_max = max(np.hypot(y - cy, x - cx) for y in (0, ny - 1) for x in (0, nx - 1))
    return np.arange(int(np.floor(r_max / PROFILE_STEP_PIX)) + 1) * PROFILE_STEP_PIX


@eqx.filter_jit
def _quintic_coefficients(image):
    """Prefilter nearest-extended samples into cardinal quintic coefficients.

    Uses the causal/anticausal inverse B-spline filter of P. Thevenaz,
    T. Blu and M. Unser, "Interpolation Revisited", IEEE Transactions on
    Medical Imaging 19(7), 739-758 (2000), doi:10.1109/42.875199.
    The two stable poles factor the degree-5 interpolation system; each
    scan solves one triangular recurrence for every line simultaneously.

    A 48-sample nearest-edge halo makes the finite initialization error
    smaller than float64 roundoff at the image (largest pole magnitude
    0.431, so its tail is below 3e-18). Keep the halo for stencil reads:
    extending coefficients directly would not extend the original samples.
    """
    poles = (-0.4305753470999738, -0.04309628820326465)
    coefficients = jnp.pad(image, _QUINTIC_PAD, mode="edge")
    for axis in (0, 1):
        lines = jnp.moveaxis(coefficients, axis, 0)
        for pole in poles:
            lines = lines * ((1 - pole) * (1 - 1 / pole))

            def causal(previous, sample, pole=pole):
                value = sample + pole * previous
                return value, value

            first = lines[0] / (1 - pole)
            _, forward = jax.lax.scan(causal, first, lines[1:])
            forward = jnp.concatenate([first[None], forward])

            def anticausal(following, sample, pole=pole):
                value = pole * (following - sample)
                return value, value

            last = -pole * forward[-1] / (1 - pole)
            _, backward = jax.lax.scan(anticausal, last, forward[:-1], reverse=True)
            lines = jnp.concatenate([backward, last[None]])
        coefficients = jnp.moveaxis(lines, 0, axis)
    return coefficients


def _quintic_weights(distance):
    """Evaluate the centered degree-5 B-spline on its six-sample support."""
    t = jnp.abs(distance)
    inner = (66 + t**2 * (-60 + t**2 * (30 - 10 * t))) / 120
    middle = ((3 - t) ** 5 - 6 * (2 - t) ** 5) / 120
    outer = jnp.maximum(3 - t, 0) ** 5 / 120
    return jnp.where(t < 1, inner, jnp.where(t < 2, middle, outer))


def _quintic_sample(coefficients, coordinates):
    """Read a prefiltered image with a separable six-by-six stencil."""
    rows, cols = coordinates
    offsets = jnp.arange(-2, 4)
    iy = jnp.floor(rows).astype(jnp.int32)[..., None] + offsets
    ix = jnp.floor(cols).astype(jnp.int32)[..., None] + offsets
    wy = _quintic_weights(rows[..., None] - iy)
    wx = _quintic_weights(cols[..., None] - ix)
    samples = coefficients[
        jnp.clip(iy + _QUINTIC_PAD, 0, coefficients.shape[0] - 1)[..., :, None],
        jnp.clip(ix + _QUINTIC_PAD, 0, coefficients.shape[1] - 1)[..., None, :],
    ]
    return jnp.sum(jnp.sum(samples * wx[..., None, :], axis=-1) * wy, axis=-1)


def azimuthal_mean_profile(
    image, center, radii_pix, n_theta=None, interpolation="keys"
):
    """Azimuthal mean of an image on circles about ``center``.

    Each circle is sampled at ``n_theta`` equally spaced positions along its
    in-image angular arcs with the selected spatial interpolant
    and the samples that fall inside the image are averaged. This is the mean
    intensity at exactly that radius. Averaging the pixels whose centers fall
    in an annulus instead mixes a range of radii, which biases the profile by
    several percent wherever it is curved on the scale of the annulus width.

    Args:
        image: 2-D array.
        center: ``(row, column)`` of the circle centers as pixel-center
            indices.
        radii_pix: 1-D array of circle radii in pixels.
        n_theta: Position angles per circle. Defaults to twice the larger
            image dimension, more than the highest azimuthal harmonic a map
            sampled at or above Nyquist carries at its corner radius.
        interpolation: ``"keys"`` (default, unchanged Keys cubic) or
            ``"quintic"`` (cardinal degree-5 B-spline). Quintic coefficients
            are prefiltered once per image per call, shared by all radii.
            Both extend samples beyond the edge with the nearest pixel and
            average only in-image arcs; tangencies use endpoint values.
            This string and ``n_theta`` must be static under JIT (for example,
            use ``eqx.filter_jit``). Quintic follows P. Thevenaz, T. Blu and
            M. Unser, "Interpolation Revisited", IEEE Transactions on Medical
            Imaging 19(7), 739-758 (2000), doi:10.1109/42.875199; the prefilter
            and evaluation use only JAX and differentiate through the image.

    Returns:
        1-D JAX array, the mean of each circle's in-image samples (0 for a
        circle with none).
    """
    image = jnp.asarray(image, dtype=jnp.result_type(image, 1.0))
    if interpolation not in ("keys", "quintic"):
        raise ValueError("interpolation must be 'keys' or 'quintic'")
    if interpolation == "quintic":
        coefficients = _quintic_coefficients(image)
    ny, nx = image.shape
    if n_theta is None:
        n_theta = 2 * max(ny, nx)
    radii = jnp.asarray(radii_pix, dtype=image.dtype)[:, None]
    safe_radii = jnp.where(radii > 0, radii, 1.0)
    # Circle/edge intersections split the circumference into wholly inside
    # or outside arcs. Sampling arc length cannot miss a short corner arc.
    y_edges = (jnp.array([0, ny - 1]) - center[0]) / safe_radii
    x_edges = (jnp.array([0, nx - 1]) - center[1]) / safe_radii
    if interpolation == "keys":
        y_angles = jnp.arcsin(jnp.clip(y_edges, -1, 1))
        x_angles = jnp.arccos(jnp.clip(x_edges, -1, 1))
    else:
        # Clipped inverse trig has 0 * inf derivatives outside [-1, 1].
        # Sanitize the unused branch before evaluation. Preserve the Keys
        # path exactly; quintic has finite coordinate gradients off tangency.
        y_crosses = jnp.abs(y_edges) < 1
        x_crosses = jnp.abs(x_edges) < 1
        y_angles = jnp.where(
            y_crosses,
            jnp.arcsin(jnp.where(y_crosses, y_edges, 0.0)),
            jnp.sign(y_edges) * (jnp.pi / 2),
        )
        x_angles = jnp.where(
            x_crosses,
            jnp.arccos(jnp.where(x_crosses, x_edges, 0.0)),
            jnp.where(x_edges < 0, jnp.pi, 0.0),
        )
    crossings = jnp.mod(
        jnp.concatenate([y_angles, jnp.pi - y_angles, x_angles, -x_angles], axis=1),
        2 * jnp.pi,
    )
    ends = jnp.broadcast_to(jnp.array([0.0, 2 * jnp.pi]), (radii.shape[0], 2))
    edges = jnp.sort(jnp.concatenate([ends, crossings], axis=1), axis=1)

    def coordinates(theta):
        return center[0] + radii * jnp.sin(theta), center[1] + radii * jnp.cos(theta)

    def in_image(rows, cols, tolerance=0.0):
        return (
            (rows >= -tolerance)
            & (rows <= ny - 1 + tolerance)
            & (cols >= -tolerance)
            & (cols <= nx - 1 + tolerance)
        )

    mid = (edges[:, :-1] + edges[:, 1:]) / 2
    lengths = jnp.where(in_image(*coordinates(mid)), jnp.diff(edges, axis=1), 0.0)
    cumulative = jnp.concatenate(
        [jnp.zeros_like(radii), jnp.cumsum(lengths, axis=1)], axis=1
    )
    total_length = cumulative[:, -1:]
    positions = total_length * ((jnp.arange(n_theta) + 0.5) / n_theta)
    index = (
        jax.vmap(lambda c, t: jnp.searchsorted(c, t, side="right"))(
            cumulative, positions
        )
        - 1
    )
    index = jnp.clip(index, 0, lengths.shape[1] - 1)
    theta = (
        jnp.take_along_axis(edges, index, axis=1)
        + positions
        - jnp.take_along_axis(cumulative, index, axis=1)
    )
    if interpolation == "keys":
        values = map_coordinates(image, coordinates(theta), order=3, mode="nearest")
    else:
        values = _quintic_sample(coefficients, coordinates(theta))
    # A tangent circle has zero arc length but can touch one or more corners.
    rows, cols = coordinates(edges)
    inside = in_image(rows, cols, 8 * jnp.finfo(image.dtype).eps * max(ny, nx))
    if interpolation == "keys":
        endpoints = map_coordinates(image, [rows, cols], order=3, mode="nearest")
    else:
        endpoints = _quintic_sample(coefficients, (rows, cols))
    count = jnp.sum(inside, axis=1)
    tangent_mean = jnp.sum(jnp.where(inside, endpoints, 0.0), axis=1) / jnp.maximum(
        count, 1
    )
    return jnp.where(total_length[:, 0] > 0, jnp.mean(values, axis=1), tangent_mean)


def _validate_photometry_image(image):
    if not isinstance(image, Image) or image.data.ndim != 2:
        raise ValueError("photometry requires a two-dimensional Image")
    if image.grid.unit != "lod" or image.measure == Measure.intensive():
        raise ValueError("photometry requires flux fractions on a lod grid")
    # The retained spectral kernels require positive, axis-aligned square pixels.
    _ = image.grid.step


@jax.jit
def _connected_core(weights, seed_index):
    """Eight-connected positive-weight component containing the source-local peak."""
    support = weights > 0
    seed = jnp.zeros_like(support).at[seed_index[0], seed_index[1]].set(True)

    def grow(state):
        old, _ = state
        padded = jnp.pad(old, 1)
        neighbors = jnp.zeros_like(old)
        for row in range(3):
            for col in range(3):
                neighbors = (
                    neighbors
                    | padded[row : row + old.shape[0], col : col + old.shape[1]]
                )
        new = support & neighbors
        return new, jnp.any(new != old)

    component, _ = jax.lax.while_loop(lambda state: state[1], grow, (seed, True))
    return jnp.where(component, weights, 0)


def _rule_samples(rule, psf, source_position, oversample=CORE_OVERSAMPLE):
    """Shared quadrature: per-native-pixel samples, weights, window origin, factor."""
    _validate_photometry_image(psf)
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    grid = psf.grid
    pixel = grid.unit_to_pixel(*source_position)
    nominal = (pixel.y, pixel.x)
    data = jnp.asarray(psf.integrated().data, dtype=jnp.float64)
    if isinstance(rule, Circular):
        half_width = (
            int(np.ceil((rule.radius_lod + PEAK_SEARCH_RADIUS_LOD) / grid.step)) + 1
        )
        fine, weights, sy, sx = circular_aperture(
            data,
            rule.radius_lod / grid.step,
            grid.step,
            oversample=oversample,
            half_width_pix=half_width,
            nominal_center=nominal,
            fixed_center=None
            if rule.centre == "peak" and bool(_nominal_in_image(nominal, grid.shape))
            else nominal,
            return_samples=True,
        )
    elif isinstance(rule, HalfMaximum):
        half_width = core_window_half_width_pix(grid.step)
        if not bool(_nominal_in_image(nominal, grid.shape)):
            half_width = max(grid.shape)
        fine, weights, sy, sx, _ = _truncation_samples(
            data,
            rule.fraction,
            grid.step,
            oversample,
            half_width,
            nominal,
        )
        center = _window_peak(
            fine,
            sy,
            sx,
            oversample,
            nominal,
            PEAK_SEARCH_RADIUS_LOD / grid.step,
            grid.shape,
        )
        seed = jnp.rint((center - jnp.array([sy, sx]) + 0.5) * oversample - 0.5)
        weights = _connected_core(weights, seed.astype(jnp.int32))
        weights = jnp.where(jnp.max(data) > 0, weights, 0)
    else:
        raise TypeError("rule must be Circular or HalfMaximum")
    return fine, weights, sy, sx, oversample


def _rule_weights(rule, psf, source_position, oversample):
    """Aggregate the common subpixel coverage to native fractional pixel weights."""
    with jax.enable_x64():
        _, weights, sy, sx, factor = _rule_samples(
            rule, psf, source_position, oversample
        )
        ny, nx = weights.shape[0] // factor, weights.shape[1] // factor
        # A fully covered subpixel has unit area in subpixel coordinates.
        # The collected area per native cell gives its fractional coverage.
        subpixels = PixelGrid.regular(weights.shape, 1, "pixel")
        pixels = PixelGrid.regular((ny, nx), factor, "pixel")
        coverage = Image(weights, subpixels, (), Measure.integral(), "aperture_area")
        collected = integrate_footprints(coverage, footprint_overlap(subpixels, pixels))
        native = collected.density("grid").data
        return jax.lax.dynamic_update_slice(jnp.zeros(psf.grid.shape), native, (sy, sx))


def _stellar_samples(data, sg, planet_grid, sy, sx, shape, factor):
    """Band-limited stellar per-pixel fractions in the planet's pixel units."""
    data = jnp.asarray(data, dtype=jnp.float64)
    if sg == planet_grid:
        if factor == 1:
            return jax.lax.dynamic_slice(data, (sy, sx), shape)
        spectrum = jnp.fft.fft2(data)
        by = _band_limited_samples(spectrum, sy, shape[0] // factor, factor)
        bx = _band_limited_samples(spectrum.T, sx, shape[1] // factor, factor)
        return jnp.real(by @ spectrum @ bx.T)
    sf, pf = sg.frame, planet_grid.frame
    ratio = (
        planet_grid.step
        * pf.lod_to_arcsec(pf.reference_wavelength_nm)
        / (sg.step * sf.lod_to_arcsec(sf.reference_wavelength_nm))
    )
    rows = (
        sg.center_y
        + (sy + (jnp.arange(shape[0]) + 0.5) / factor - 0.5 - planet_grid.center_y)
        * ratio
    )
    cols = (
        sg.center_x
        + (sx + (jnp.arange(shape[1]) + 0.5) / factor - 0.5 - planet_grid.center_x)
        * ratio
    )
    spectrum = jnp.fft.fft2(data)
    by = (
        jnp.exp(2j * jnp.pi * rows[:, None] * jnp.fft.fftfreq(sg.shape[0]))
        / sg.shape[0]
    )
    bx = (
        jnp.exp(2j * jnp.pi * cols[:, None] * jnp.fft.fftfreq(sg.shape[1]))
        / sg.shape[1]
    )
    inside = (
        (rows[:, None] >= -0.5)
        & (rows[:, None] < sg.shape[0] - 0.5)
        & (cols[None, :] >= -0.5)
        & (cols[None, :] < sg.shape[1] - 0.5)
    )
    return jnp.where(inside, jnp.real(by @ spectrum @ bx.T) * ratio**2, 0)


def _separation_axis(separation_lod):
    if isinstance(separation_lod, (jax.Array, jax.core.Tracer)):
        x = jnp.asarray(separation_lod, dtype=jnp.result_type(separation_lod, 1.0))
    else:
        x = np.array(separation_lod, dtype=float, copy=True)
        x.flags.writeable = False
    if x.ndim != 1 or x.size == 0:
        raise ValueError("separation_lod must be a nonempty vector")
    if not isinstance(x, jax.core.Tracer):
        axis = np.asarray(x)
        if not np.all(np.isfinite(axis)) or axis[0] < 0 or np.any(np.diff(axis) <= 0):
            raise ValueError(
                "separation_lod must contain increasing nonnegative finite values"
            )
    return x


def _profile_fit(x, values, interpolation):
    """Prepare a PCHIP and the end-segment slopes in the selected space."""
    if len(x) == 1:
        return None
    traced = isinstance(values, (jax.Array, jax.core.Tracer)) or isinstance(
        x, (jax.Array, jax.core.Tracer)
    )
    if traced:
        y = jnp.asarray(values)
        if interpolation == "log":
            # The host's 1e-300 floor underflows in float32.
            y = jnp.where(
                y > 0, jnp.maximum(jnp.log10(jnp.where(y > 0, y, 1)), -300), -300
            )
        coefficients = interpax.PchipInterpolator(jnp.asarray(x), y, check=False).c
        stack = jnp.stack
    else:
        y = np.log10(np.maximum(values, 1e-300)) if interpolation == "log" else values
        coefficients = PchipInterpolator(x, y).c
        stack = np.stack
    # Preserve native knot spacings before casting traced ordinates to their dtype.
    tails = stack(
        (
            y[0],
            y[-1],
            (y[1] - y[0]) / (x[1] - x[0]),
            (y[-1] - y[-2]) / (x[-1] - x[-2]),
        )
    )
    return coefficients, tails


def _evaluate_profile(x, values, fit, interpolation, separation_lod, *, exact_knots):
    xp = (
        jnp
        if isinstance(separation_lod, (jax.Array, jax.core.Tracer))
        or isinstance(values, jax.core.Tracer)
        or isinstance(x, jax.core.Tracer)
        or (fit is not None and isinstance(fit[0], jax.core.Tracer))
        else np
    )
    q, x, values = xp.asarray(separation_lod), xp.asarray(x), xp.asarray(values)
    if fit is None:
        return xp.full_like(q, values[0], dtype=values.dtype)
    c, t = map(xp.asarray, fit)
    k = xp.clip(xp.searchsorted(x, q, side="right") - 1, 0, len(x) - 1)
    i = xp.minimum(k, len(x) - 2)

    def polynomial(dx):
        return ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]

    value = xp.where(
        q < x[0],
        t[0] + (q - x[0]) * t[2],
        xp.where(q > x[-1], t[1] + (q - x[-1]) * t[3], polynomial(q - x[i])),
    )
    if interpolation == "log":
        value = xp.power(10.0, value)
        if not exact_knots and xp is jnp:
            correction = xp.where(q == x[k], values[k] - value, 0)
            return value + jax.lax.stop_gradient(correction)
    elif not exact_knots:
        # Aperture records historically return the polynomial at a linear endpoint.
        return value
    if xp is np:
        return np.where(q == x[k], values[k], value)
    # The residual cancels the fit's sample tangent at a knot and retains
    # its radial derivative. The measured sample supplies its own tangent.
    anchor = polynomial(x[k] - x[i])
    if interpolation == "log":
        anchor = xp.power(10.0, anchor)
    at_knot = values[k] + (value - anchor)
    return jnp.where(q == x[k], at_knot, value)


class SampledProfile(eqx.Module):
    """One radial quantity with shape-preserving interpolation in linear or log space.

    Separations are in the supplied lambda/D frame. The data span the first to the
    last sample (:meth:`supported`); ``evaluate`` continues the two end segments
    linearly in the interpolation space beyond them, an interpolant and not data,
    so gate on :meth:`supported` before treating a read as a measurement. A single
    sample is constant. Sample values can depend on a traced parameter such as
    stellar diameter.

    Attributes:
        separation_lod: Increasing sample separations in lambda/D.
        values: Sample values, one per separation, in ``quantity`` units.
        quantity: Name of the sampled quantity.
        frame: Frame defining lambda/D.
        interpolation: ``"linear"`` or ``"log"`` interpolation space.
    """

    separation_lod: np.ndarray | jax.Array
    values: np.ndarray | jax.Array
    quantity: str = eqx.field(static=True)
    frame: Frame = eqx.field(static=True)
    interpolation: str = eqx.field(static=True)
    _fit: tuple | None

    def __init__(
        self, separation_lod, values, *, quantity, frame, interpolation, _fit=None
    ):
        """Fit the supplied samples in their declared quantity and frame.

        Args:
            separation_lod: Increasing sample separations in lambda/D.
            values: Sample values, one per separation.
            quantity: Name of the sampled quantity.
            frame: Frame defining lambda/D.
            interpolation: ``"linear"`` or ``"log"`` interpolation space.
            _fit: Precomputed fit carried through ``dataclasses.replace``; leave None.
        """
        if interpolation not in ("linear", "log"):
            raise ValueError("interpolation must be 'linear' or 'log'")
        self.quantity, self.frame, self.interpolation = quantity, frame, interpolation
        self.separation_lod = _separation_axis(separation_lod)
        if isinstance(values, (jax.Array, jax.core.Tracer)):
            self.values = jnp.asarray(values, dtype=jnp.result_type(values, 1.0))
        else:
            self.values = np.array(values, dtype=float, copy=True)
            self.values.flags.writeable = False
        if self.values.shape != self.separation_lod.shape:
            raise ValueError(f"{quantity} must have shape {self.separation_lod.shape}")
        # dataclasses.replace passes the previous fit; refit the current samples.
        self._fit = _profile_fit(self.separation_lod, self.values, interpolation)

    @classmethod
    def from_uniform_sky(cls, sky):
        """Prepare the annular mean of a dimensionless native uniform-sky map.

        Args:
            sky: Dimensionless uniform-sky Image on a native lod grid.

        Returns:
            Profile of the annular mean against separation in lambda/D.
        """
        radii, values, counts = radial_profile(sky)
        populated = np.asarray(counts) > 0
        return cls(
            np.asarray(radii)[populated],
            np.asarray(values)[populated],
            quantity="uniform_sky_response",
            frame=sky.grid.frame,
            interpolation="linear",
        )

    def evaluate(self, separation_lod):
        """Evaluate on an arbitrary scalar or array of separations in this frame.

        Args:
            separation_lod: Separations in lambda/D, scalar or array.

        Returns:
            Interpolated values, shaped like ``separation_lod``.
        """
        return _evaluate_profile(
            self.separation_lod,
            self.values,
            self._fit,
            self.interpolation,
            separation_lod,
            exact_knots=True,
        )

    def supported(self, separation_lod):
        """Report where the profile holds data: from its first to its last sample.

        Beyond that span ``evaluate`` only continues the end segments.

        Args:
            separation_lod: Separations in lambda/D, scalar or array.

        Returns:
            Boolean array shaped like ``separation_lod``, True inside the samples.
        """
        return _within_samples(self.separation_lod, separation_lod)[1]


def _within_samples(x, separation_lod):
    xp = jnp if isinstance(separation_lod, (jax.Array, jax.core.Tracer)) else np
    q = xp.asarray(separation_lod)
    return q, (q >= x[0]) & (q <= x[-1])


def ring_azimuth_samples(shape, center):
    """Default position angles per ring: ``ceil(2 pi r_max)``.

    A map carries harmonics up to half its circumference in pixels at its corner
    radius, so this many equally spaced angles average every ring without
    aliasing (1131 for a 256 by 256 map).

    Args:
        shape: Map shape (ny, nx).
        center: Optical axis (row, column) in pixels.

    Returns:
        Number of equally spaced position angles per ring.
    """
    return int(np.ceil(2 * np.pi * np.max(profile_radii(shape, center))))


def ring_symmetrized(image, grid=None, n_theta=None):
    """Return ``image`` with every pixel replaced by the mean of its ring.

    The angle average of an aperture sum at distance s from the axis equals the
    aperture sum at +x of the azimuthally averaged map, so weighting this map
    with ordinary aperture pixel weights at +x gives the ring-averaged aperture
    leakage with the same circle and pixel weights.

    The ring mean at radius r is the mean of the map at ``n_theta`` equally
    spaced angles (in-image arcs only), evaluated at the radii of
    :func:`profile_radii` and linear in r between them. ``n_theta`` defaults to
    :func:`ring_azimuth_samples`.

    ``grid`` is the pixel grid of the result, for example the PSF grid when it
    differs from the image's: each pixel takes the ring mean at its own angular
    radius, in flux fraction per pixel of ``grid``, and zero beyond the image
    footprint. The default is the image's own grid.

    Args:
        image: Stellar Image to symmetrize.
        grid: Pixel grid of the result; the image's own grid when None.
        n_theta: Number of position angles per ring; ``ring_azimuth_samples`` when
            None.

    Returns:
        Image on ``grid`` with each pixel set to the mean of its ring, in flux
        fraction per pixel of ``grid``.
    """
    image = image.integrated()
    sg = image.grid
    grid = sg if grid is None else grid
    center = (sg.center_y, sg.center_x)
    radii = profile_radii(sg.shape, center)
    if n_theta is None:
        n_theta = ring_azimuth_samples(sg.shape, center)
    sf, pf = sg.frame, grid.frame
    # Image pixels per pixel of the target grid, along a radius.
    ratio = (grid.step * pf.lod_to_arcsec(pf.reference_wavelength_nm)) / (
        sg.step * sf.lod_to_arcsec(sf.reference_wavelength_nm)
    )
    with jax.enable_x64():
        data = jnp.asarray(image.data, dtype=jnp.float64)
        profile = azimuthal_mean_profile(data, center, jnp.asarray(radii), n_theta)
        rows, cols = jnp.indices(grid.shape)
        stellar_radius = jnp.hypot(rows - grid.center_y, cols - grid.center_x) * ratio
        symmetric = jnp.interp(stellar_radius, jnp.asarray(radii), profile)
        symmetric = jnp.where(stellar_radius <= radii[-1], symmetric, 0.0) * ratio**2
    return replace(image, data=symmetric, grid=grid)


def _require(ok, value, message):
    """Return ``value`` when ``ok`` holds everywhere, else raise ``message``.

    Concrete inputs raise ``ValueError``; traced inputs raise at run time
    through ``equinox.error_if``, so the check survives ``jit`` and ``vmap``.
    """
    if isinstance(ok, jax.core.Tracer) or isinstance(value, jax.core.Tracer):
        return eqx.error_if(jnp.asarray(value), ~jnp.all(ok), message)
    if not np.all(np.asarray(ok)):
        raise ValueError(message)
    return value


def _diameter_abscissa(diameters):
    """Spline abscissa of a diameter axis: a zero diameter is nudged to float32 eps.

    The nudge matches the one the stellar map interpolation applies, so the
    leakage at a diameter between knots is read with the rule the image route
    reads the map with.
    """
    x = np.array(diameters, dtype=float)
    x[x == 0] = np.finfo(np.float32).eps
    return x


def _diameter_spline_basis(diameters):
    """Cubic-spline basis of a diameter axis: coefficients for each unit sample.

    Returns an array of shape ``(K - 1, 4, K)``: the polynomial coefficients,
    highest power first, of the not-a-knot spline through the ``k``-th unit
    sample, on each of the ``K - 1`` intervals. A spline is linear in its
    samples, so the weights of a read at one diameter are these polynomials
    evaluated there, and a read is a weighted sum of the K samples.
    """
    x = _diameter_abscissa(diameters)
    if len(x) == 1:
        return None
    return np.ascontiguousarray(CubicSpline(x, np.eye(len(x))).c.transpose(1, 0, 2))


def _diameter_weights(diameters, basis, diameter):
    """Spline weights of the K tabulated diameters at ``diameter``, shape (..., K)."""
    traced = (jax.Array, jax.core.Tracer)
    xp = jnp if isinstance(diameter, traced) or isinstance(diameters, traced) else np
    # Under JAX the arrays take the default float dtype, so float32 mode does not warn.
    dtype = jnp.result_type(float) if xp is jnp else float
    if isinstance(diameters, (np.ndarray, *traced)):
        diameters = diameters.astype(dtype)
    if isinstance(basis, (np.ndarray, *traced)):
        basis = basis.astype(dtype)
    x = xp.asarray(diameters)
    x = xp.where(x == 0, np.finfo(np.float32).eps, x)
    d = xp.asarray(diameter)
    i = xp.clip(xp.searchsorted(x, d, side="right") - 1, 0, len(x) - 2)
    dx = (d - x[i])[..., None]
    c = xp.asarray(basis)[i]
    return ((c[..., 0, :] * dx + c[..., 1, :]) * dx + c[..., 2, :]) * dx + c[..., 3, :]


class PerformanceCurves(eqx.Module):
    """Frozen photometry on one separation axis; all fractions are absolute.

    Attributes:
        separation_lod: Nonnegative increasing sample offsets in lambda/D, shape
            (S,).
        frame: Wavelength and circumscribed/inscribed diameter defining
            lambda/D.
        rule: The static, hashable aperture rule.
        throughput: Weighted fraction of unocculted planet flux, tau, shape (S,).
        throughput_variance: Planet flux fraction integrated with squared
            weights, shape (S,).
        core_area_lod2: Pixel area times the sum of weights, A, in (lambda/D)^2,
            shape (S,).
        core_area_variance_lod2: Pixel area times the sum of squared weights, in
            (lambda/D)^2, shape (S,).
        leakage_fraction: Stellar per-pixel fractions integrated with the
            weights, shape (K, S): one profile per stellar diameter.
        leakage_fraction_variance: Stellar fractions integrated with squared
            weights, shape (K, S).
        leakage_density_lod2: ``leakage_fraction / core_area_lod2`` in
            1 / (lambda/D)^2 for aperture_mean and aperture_ring_mean; the radial
            azimuthal point-sample mean per unit area for azimuthal_ring_mean.
            These differ in general. Shape (K, S).
        leakage_density_variance_lod2: Variance leakage per variance area for the
            aperture rules, or the same azimuthal density for
            azimuthal_ring_mean. Shape (K, S).
        raw_contrast: ``leakage_density_lod2 * A / tau``, dimensionless, with 1e20
            when tau <= 0. Shape (K, S).
        iwa_lod: First half-maximum-throughput crossing of PCHIP, in lambda/D; if
            no crossing exists, the first positive-throughput separation (NaN for
            an all-zero curve).
        data_extent_lod: Minimum array half-width in lambda/D, not an outer
            working angle.
        leakage_rule: The rule that produced the leakage fields (see below).
        stellar_diameters_lod: The K stellar diameters, in lambda/D of ``frame``,
            increasing, that the leakage fields are tabulated at, or None for one
            profile of unknown diameter. Between the K diameters the leakage
            is interpolated; the accuracy there is limited by the spacing of the
            table (see the architecture page), not by the read. A record of
            None is read for any star without a check.
        uniform_sky: Annular mean of the coronagraph's dimensionless uniform-sky
            transmission map against separation in lambda/D of ``frame``.
            Consumers read it for extended sources such as local zodi and
            exozodi. None when the record carries no profile.
        interpolation: Immutable quantity-to-space mapping. PCHIP within the
            sampled interval and linear end-segment tails in that same space.
            The tails only continue the interpolant; the record holds data where
            :meth:`supported` is True. Log interpolation uses a positive floor
            for zeros. A read at a tabulated diameter returns that row's
            separation profile, including exact zeros at its measured knots.

    Throughput and core area are functions of separation. The leakage fields
    and ``raw_contrast`` are functions of separation and of the stellar
    diameter: ``evaluate`` reads them at a diameter, interpolating the K
    profiles with a cubic spline of the logarithm in diameter, the rule the
    stellar map itself is interpolated with. A record of K > 1 diameters
    requires the diameter at every leakage read and raises outside the
    tabulated range; reading it at zero diameter is an explicit choice, not a
    default. A record of one profile serves a star within ``1e-3`` of the
    profile's diameter (relative, with an absolute floor of ``1e-6``) and
    raises for any other; a profile of unknown diameter (None) serves any star
    and the caller vouches for it.

    At one diameter every field is a function of separation only. For a stellar
    map that is not rotationally symmetric (a 2D product) the leakage is therefore
    a reduction over position angle, not the leakage at a planet's own position
    angle. The rules, by ``leakage_rule``:

    - ``aperture_mean``: the exact-coverage aperture of ``rule`` read at (s, 0),
      the +x axis. It is the same number at every angle for a rotationally
      symmetric map; for a 2D map it is a single cut.
    - ``aperture_ring_mean``: the same aperture averaged over position angle,
      (1 / 2 pi) times the integral over phi of the aperture sum centred at
      (s cos phi, s sin phi). It is computed as the aperture weights summed over
      the azimuthally symmetrized stellar map (:func:`ring_symmetrized`), which
      is the same quantity because the angle average of an aperture is
      rotationally symmetric.
    - ``azimuthal_ring_mean`` (deprecated): the point-sample ring mean times the
      aperture area; it ignores the aperture extent and agrees with
      ``aperture_ring_mean`` only as the aperture shrinks to a point.

    Circular and HalfMaximum describe an idealized aperture in the limit of
    infinitely fine detector pixels; detector pixelization is not modelled here.
    Their indicator function equals its square, so all variance moments equal
    the corresponding means.

    Host NumPy inputs retain float64. JAX inputs use the same stored polynomial
    coefficients and evaluator, and support jit/vmap/grad. An Equinox consumer
    can convert the array leaves to its active JAX dtype once at construction.
    """

    separation_lod: np.ndarray
    frame: Frame = eqx.field(static=True)
    rule: ApertureRule = eqx.field(static=True)
    throughput: np.ndarray
    throughput_variance: np.ndarray
    core_area_lod2: np.ndarray
    core_area_variance_lod2: np.ndarray
    leakage_fraction: np.ndarray
    leakage_fraction_variance: np.ndarray
    leakage_density_lod2: np.ndarray
    leakage_density_variance_lod2: np.ndarray
    raw_contrast: np.ndarray
    iwa_lod: float
    data_extent_lod: float = eqx.field(static=True)
    leakage_rule: str = eqx.field(static=True)
    stellar_diameters_lod: np.ndarray | None
    _diameter_basis: np.ndarray | None
    _fits: dict
    uniform_sky: SampledProfile | None = None

    interpolation: ClassVar = MappingProxyType(
        {
            "throughput": "linear",
            "throughput_variance": "linear",
            "core_area_lod2": "linear",
            "core_area_variance_lod2": "linear",
            "leakage_fraction": "log",
            "leakage_fraction_variance": "log",
            "leakage_density_lod2": "log",
            "leakage_density_variance_lod2": "log",
            "raw_contrast": "log",
        }
    )

    def __init__(
        self,
        separation_lod,
        frame,
        rule,
        throughput,
        core_area_lod2,
        leakage_fraction,
        leakage_density_lod2,
        raw_contrast,
        iwa_lod,
        data_extent_lod,
        leakage_rule="aperture_mean",
        stellar_diameters_lod=None,
        *,
        throughput_variance,
        core_area_variance_lod2,
        leakage_fraction_variance,
        leakage_density_variance_lod2,
        uniform_sky: SampledProfile | None = None,
        _diameter_basis=None,
        _fits=None,
    ):
        """Validate measured arrays and prepare one set of PCHIP coefficients.

        Args:
            separation_lod: Nonnegative increasing separations in lambda/D, shape (S,).
            frame: Frame defining lambda/D.
            rule: Aperture rule, Circular or HalfMaximum.
            throughput: Planet flux fraction in the aperture, shape (S,).
            core_area_lod2: Aperture area in lambda/D squared, shape (S,).
            leakage_fraction: Stellar flux fraction in the aperture, shape (K, S),
                or (S,) for one profile.
            leakage_density_lod2: Leakage per unit area in 1 / (lambda/D)^2, shape
                (K, S) or (S,).
            raw_contrast: Dimensionless raw contrast, shape (K, S) or (S,).
            iwa_lod: Inner working angle in lambda/D.
            data_extent_lod: Minimum array half-width in lambda/D.
            leakage_rule: One of ``LEAKAGE_RULES``.
            stellar_diameters_lod: The diameters in lambda/D of ``frame`` the leakage
                rows are tabulated at: increasing, nonnegative, one per row. A
                scalar is one profile of that diameter; None is one profile of
                unknown diameter.
            throughput_variance: Planet flux fraction with squared weights, shape (S,).
            core_area_variance_lod2: Area with squared weights in lambda/D squared,
                shape (S,).
            leakage_fraction_variance: Stellar flux fraction with squared weights,
                shape (K, S) or (S,).
            leakage_density_variance_lod2: Variance leakage per variance area, shape
                (K, S) or (S,).
            uniform_sky: Optional annular mean sky transmission in ``frame``.
            _diameter_basis: Carried through ``dataclasses.replace``; leave None.
            _fits: Precomputed fits carried through ``dataclasses.replace``; leave
                None.
        """
        if not isinstance(rule, (Circular, HalfMaximum)):
            raise TypeError("rule must be Circular or HalfMaximum")
        if leakage_rule not in LEAKAGE_RULES:
            raise ValueError(f"unknown leakage_rule; expected one of {LEAKAGE_RULES}")
        if uniform_sky is not None:
            if not isinstance(uniform_sky, SampledProfile):
                raise TypeError("uniform_sky must be a SampledProfile")
            if uniform_sky.frame != frame:
                raise ValueError("uniform_sky must be in the record's lambda/D frame")
        self.uniform_sky = uniform_sky
        x = _separation_axis(np.asarray(separation_lod))
        self.separation_lod = x
        self.frame, self.rule = frame, rule
        self.iwa_lod, self.data_extent_lod = float(iwa_lod), float(data_extent_lod)
        self.leakage_rule = leakage_rule
        if stellar_diameters_lod is None:
            diameters = None
        else:
            traced = isinstance(stellar_diameters_lod, jax.core.Tracer)
            diameters = (
                jnp.atleast_1d(stellar_diameters_lod)
                if traced
                else np.atleast_1d(np.array(stellar_diameters_lod, dtype=float))
            )
            if diameters.ndim != 1:
                raise ValueError("stellar_diameters_lod must be a scalar or a vector")
            if not traced and (
                not np.all(np.isfinite(diameters))
                or diameters[0] < 0
                or np.any(np.diff(diameters) <= 0)
            ):
                raise ValueError(
                    "stellar_diameters_lod must be increasing, finite and nonnegative"
                )
        n_diameters = 1 if diameters is None else len(diameters)
        self.stellar_diameters_lod = diameters
        self._diameter_basis = (
            None
            if diameters is None or n_diameters == 1
            else _diameter_spline_basis(np.asarray(diameters, dtype=float))
        )
        values = dict(
            throughput=throughput,
            throughput_variance=throughput_variance,
            core_area_lod2=core_area_lod2,
            core_area_variance_lod2=core_area_variance_lod2,
            leakage_fraction=leakage_fraction,
            leakage_fraction_variance=leakage_fraction_variance,
            leakage_density_lod2=leakage_density_lod2,
            leakage_density_variance_lod2=leakage_density_variance_lod2,
            raw_contrast=raw_contrast,
        )
        # Replacements can change any measured array, invalidating its former fit.
        self._fits = {}
        for name, value in values.items():
            a = np.array(value, dtype=float, copy=True)
            by_diameter = name in DIAMETER_QUANTITIES
            if by_diameter and a.ndim == 1:
                a = a[None]
            shape = (n_diameters, *x.shape) if by_diameter else x.shape
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be finite and have shape {shape}")
            a.flags.writeable = False
            setattr(self, name, a)
            self._fits[name] = (
                tuple(_profile_fit(x, row, self.interpolation[name]) for row in a)
                if by_diameter
                else _profile_fit(x, a, self.interpolation[name])
            )

    def evaluate(self, name, separation_lod, stellar_diameter_lod=None):
        """Evaluate one named quantity using its declared interpolation space.

        Args:
            name: Quantity name, a key of ``interpolation``.
            separation_lod: Separations in lambda/D, scalar or array.
            stellar_diameter_lod: Stellar diameter in lambda/D of this record's
                frame, scalar or broadcastable with ``separation_lod``; may be
                traced. Required for the leakage quantities of a record
                tabulated at several diameters, optional for one profile, and
                not accepted for throughput and core area. A diameter outside
                the table (or, for one profile, away from its diameter) raises.

        Returns:
            Interpolated values, shaped like ``separation_lod`` (broadcast with
            the diameter). Outside the samples these are end-segment
            continuations, not data: read them only where :meth:`supported` is
            True.
        """
        if name not in self.interpolation:
            raise ValueError(f"quantity {name!r} is not present in this record")
        if name in DIAMETER_QUANTITIES:
            return self._evaluate_by_diameter(
                name, separation_lod, stellar_diameter_lod
            )
        if stellar_diameter_lod is not None:
            raise ValueError(f"{name} does not depend on the stellar diameter")
        return _evaluate_profile(
            self.separation_lod,
            getattr(self, name),
            self._fits[name],
            self.interpolation[name],
            separation_lod,
            exact_knots=False,
        )

    def _evaluate_by_diameter(self, name, separation_lod, stellar_diameter_lod):
        """Read a leakage quantity at a separation and a stellar diameter."""
        rows = getattr(self, name)
        fits = self._fits[name]
        diameters = self.stellar_diameters_lod
        if isinstance(diameters, (jax.Array, jax.core.Tracer)):
            # A record built under x64 keeps float64 leaves; read the active dtype.
            diameters = diameters.astype(jnp.result_type(float))
        n_rows = rows.shape[0]

        def profile(k, q):
            return _evaluate_profile(
                self.separation_lod,
                rows[k],
                fits[k],
                self.interpolation[name],
                q,
                exact_knots=False,
            )

        if n_rows == 1:
            value = profile(0, separation_lod)
            if stellar_diameter_lod is None or diameters is None:
                return value
            # The checked value is the result, so the check survives jit.
            return _require(
                abs(stellar_diameter_lod - diameters[0])
                <= _SINGLE_DIAMETER_RTOL * diameters[0] + _SINGLE_DIAMETER_ATOL,
                value,
                "this record holds one profile, at one stellar diameter; a star "
                "of a different diameter needs a record tabulated over diameter",
            )
        if stellar_diameter_lod is None:
            raise ValueError(
                f"{name} of a record tabulated over diameter needs stellar_diameter_lod"
            )
        traced = (jax.Array, jax.core.Tracer)
        use_jax = (
            isinstance(separation_lod, traced)
            or isinstance(stellar_diameter_lod, traced)
            or isinstance(diameters, jax.core.Tracer)
            or isinstance(rows, jax.core.Tracer)
        )
        xp = jnp if use_jax else np
        q, d = xp.broadcast_arrays(
            xp.asarray(separation_lod), xp.asarray(stellar_diameter_lod)
        )
        slack = _DIAMETER_RANGE_RTOL * diameters[-1]
        d = _require(
            (d >= diameters[0] - slack) & (d <= diameters[-1] + slack),
            d,
            "the stellar diameter is outside the tabulated range of the record",
        )
        per_row = xp.stack([profile(k, q) for k in range(n_rows)], axis=-1)
        knots = xp.asarray(diameters)
        k = xp.minimum(xp.searchsorted(knots, d), n_rows - 1)
        at_knot = d == knots[k]
        measured = xp.take_along_axis(per_row, k[..., None], axis=-1)[..., 0]
        weights = _diameter_weights(diameters, self._diameter_basis, d)
        tiny = np.finfo(per_row.dtype).tiny
        logs = xp.log(xp.maximum(per_row, tiny))
        value = xp.exp(xp.sum(weights * logs, axis=-1))
        interpolated = xp.where(xp.all(per_row <= 0, axis=-1), 0.0, value)
        if use_jax:
            # The measured value at a knot, with the interpolant's derivatives.
            measured = jax.lax.stop_gradient(measured) + (
                interpolated - jax.lax.stop_gradient(interpolated)
            )
        return xp.where(at_knot, measured, interpolated)

    def supported(self, separation_lod):
        """Report where the record holds data for this record's aperture rule.

        True where the separation lies from the first to the last sample and the
        aperture (separation plus ``rule.radius_lod``) fits inside
        ``data_extent_lod``; a ``HalfMaximum`` rule has no fixed radius, so only
        the separation is tested against the extent. The array extent alone is
        not enough: the samples can end inside it, and past the last sample
        ``evaluate`` only continues the end segment.

        Args:
            separation_lod: Separations in lambda/D, scalar or array.

        Returns:
            Boolean array shaped like ``separation_lod``.
        """
        q, inside = _within_samples(self.separation_lod, separation_lod)
        radius = getattr(self.rule, "radius_lod", 0.0)
        return inside & (q + radius <= self.data_extent_lod)


def _iwa(separation, throughput):
    positive = throughput > 0
    if not np.any(positive):
        return np.nan
    if len(separation) == 1:
        return float(separation[0])
    half = np.max(throughput) / 2
    f = PchipInterpolator(separation, throughput)
    for left, right, yl, yr in zip(
        separation[:-1], separation[1:], throughput[:-1], throughput[1:], strict=True
    ):
        if yl == half:
            return float(left)
        if (yl - half) * (yr - half) < 0:
            return root_scalar(lambda s: f(s) - half, bracket=(left, right)).root
    return float(separation[np.argmax(positive)])


def performance_curves(
    offaxis,
    stellar,
    rule,
    *,
    separations_lod=None,
    source_positions_lod=None,
    leakage_rule="aperture_mean",
    stellar_diameter_lod=None,
    stellar_diameters_lod=None,
    uniform_sky: SampledProfile | Image | None = None,
) -> PerformanceCurves:
    """Tabulate one aperture rule on one axis.

    Minimal protocol: ``offaxis(separation_lod: float) -> Image`` on the +x
    axis. Supply ``separations_lod`` or an object attribute of that name.
    The callable is evaluated on the explicit separation axis. Explicit
    separations synthesize PSFs at (s, 0). The returned images must share one
    positive axis-aligned square lod grid, carrying per-pixel source fractions.
    Stellar is a stellar-flux Image, potentially on a distinct grid
    and frame; its samples are mapped through the frames' angular scales.
    Geometric rules integrate both images on the same subpixel region.
    Stellar samples outside the stellar footprint are zero.
    Circular's ``centre`` chooses the supplied source position (default) or
    its refined local peak; every field, including IWA, uses that same rule.

    ``stellar_diameter_lod`` records the diameter of the one stellar map supplied;
    it does not broaden or reevaluate that map, and the record then serves only
    a star of that diameter. To tabulate the leakage over stellar diameter, give
    ``stellar`` a leading diameter axis, one map per entry of
    ``stellar_diameters_lod``; the record then holds one leakage profile per
    diameter and reads any diameter between them. Geometric rules resolve
    coverage at CORE_OVERSAMPLE subpixels per side.

    ``leakage_rule`` is ``aperture_mean`` (the +x aperture), ``aperture_ring_mean``
    (that aperture averaged over position angle, summed as native exact-coverage
    pixel weights over the ring-symmetrized stellar map on the PSF grid) or the
    deprecated ``azimuthal_ring_mean``; see :class:`PerformanceCurves`.

    Args:
        offaxis: Function of a separation in lambda/D returning the off-axis source
            Image on the +x axis.
        stellar: Stellar-flux Image, possibly on a distinct grid and frame; with
            ``stellar_diameters_lod`` its data are ``(K, ny, nx)``, one map per
            diameter.
        rule: Aperture rule, Circular or HalfMaximum.
        separations_lod: Increasing nonnegative separations in lambda/D, required.
        source_positions_lod: Optional (x, y) source position in lambda/D for each
            separation; ``(s, 0)`` by default.
        leakage_rule: ``"aperture_mean"``, ``"aperture_ring_mean"`` or the
            deprecated ``"azimuthal_ring_mean"``.
        stellar_diameter_lod: Optional diameter in lambda/D of the one stellar
            map supplied, recorded for the check on later reads.
        stellar_diameters_lod: Increasing diameters in lambda/D of the stellar
            Image's frame, one per map of a diameter-axis ``stellar``; excludes
            ``stellar_diameter_lod``.
        uniform_sky: Optional dimensionless uniform-sky Image or annular mean
            profile in the off-axis Image's lambda/D frame.

    Returns:
        The photometric record on the separation axis.
    """
    if leakage_rule == "azimuthal_ring_mean":
        warnings.warn(
            "leakage_rule 'azimuthal_ring_mean' is deprecated: it multiplies a "
            "point-sample ring mean by the aperture area and ignores the aperture "
            "extent; use 'aperture_ring_mean'",
            DeprecationWarning,
            stacklevel=2,
        )
    return _performance_curves(
        offaxis,
        stellar,
        rule,
        separations_lod=separations_lod,
        source_positions_lod=source_positions_lod,
        leakage_rule=leakage_rule,
        stellar_diameter_lod=stellar_diameter_lod,
        stellar_diameters_lod=stellar_diameters_lod,
        uniform_sky=uniform_sky,
    )


def _performance_curves(
    offaxis,
    stellar,
    rule,
    *,
    separations_lod=None,
    source_positions_lod=None,
    leakage_rule="aperture_mean",
    stellar_diameter_lod=None,
    stellar_diameters_lod=None,
    uniform_sky: SampledProfile | Image | None = None,
    oversample=CORE_OVERSAMPLE,
):

    if not isinstance(rule, (Circular, HalfMaximum)):
        raise TypeError("rule must be Circular or HalfMaximum")
    if stellar_diameters_lod is None:
        stellars = [stellar]
    else:
        if stellar_diameter_lod is not None:
            raise ValueError(
                "give stellar_diameter_lod or stellar_diameters_lod, not both"
            )
        diameters = np.asarray(stellar_diameters_lod, dtype=float)
        if (
            diameters.ndim != 1
            or stellar.data.ndim != 3
            or stellar.data.shape[0] != len(diameters)
        ):
            raise ValueError(
                "stellar must hold one map per entry of stellar_diameters_lod, "
                "shape (K, ny, nx)"
            )
        stellars = [
            replace(stellar, data=stellar.data[k], leading_axes=())
            for k in range(len(diameters))
        ]
    for image in stellars:
        _validate_photometry_image(image)
    stellars = [image.integrated() for image in stellars]
    if leakage_rule not in LEAKAGE_RULES:
        raise ValueError(f"unknown leakage_rule; expected one of {LEAKAGE_RULES}")
    if separations_lod is None:
        raise ValueError("separations_lod is required for a callable source")
    separations = np.asarray(separations_lod, dtype=float)
    if (
        separations.ndim != 1
        or len(separations) < 1
        or np.any(np.diff(separations) <= 0)
        or np.any(separations < 0)
        or not np.all(np.isfinite(separations))
    ):
        raise ValueError(
            "separations_lod must contain increasing nonnegative finite values"
        )
    images = [offaxis(float(s)) for s in separations]
    locations = (
        [(float(s), 0.0) for s in separations]
        if source_positions_lod is None
        else source_positions_lod
    )
    if not images:
        raise ValueError("no off-axis source positions lie inside the image")
    grid = images[0].grid
    sf, pf = stellars[0].grid.frame, grid.frame
    # Stellar lambda/D of the supplied maps per lambda/D of the planet's frame.
    angular_ratio = pf.lod_to_arcsec(pf.reference_wavelength_nm) / sf.lod_to_arcsec(
        sf.reference_wavelength_nm
    )
    tau, area = [], []
    leakage = [[] for _ in stellars]
    with jax.enable_x64():
        symmetric = (
            [ring_symmetrized(image, grid).data for image in stellars]
            if leakage_rule == RING_RULE
            else None
        )
        for psf, position in zip(images, locations, strict=True):
            if psf.grid != grid:
                raise ValueError("off-axis Images must share one grid")
            fine, weights, sy, sx, factor = _rule_samples(
                rule, psf, position, oversample
            )
            tau.append(float(jnp.sum(fine * weights) / factor**2))
            area.append(float(jnp.sum(weights)) * grid.pixel_area / factor**2)
            if symmetric is None:
                for row, image in zip(leakage, stellars, strict=True):
                    star = _stellar_samples(
                        image.data, image.grid, grid, sy, sx, weights.shape, factor
                    )
                    row.append(float(jnp.sum(star * weights) / factor**2))
            else:
                # Native exact-coverage weights on the symmetrized map; the
                # band-limited samples ripple on its ring edges.
                native = _rule_weights(rule, psf, position, oversample)
                for row, ring in zip(leakage, symmetric, strict=True):
                    row.append(float(jnp.sum(native * ring)))
        tau, area, leakage = np.asarray(tau), np.asarray(area), np.asarray(leakage)
        density = np.divide(leakage, area, out=np.zeros_like(leakage), where=area > 0)
        if leakage_rule == "azimuthal_ring_mean":
            density = np.stack(
                [
                    np.asarray(
                        azimuthal_mean_profile(
                            jnp.asarray(image.data, dtype=jnp.float64),
                            (image.grid.center_y, image.grid.center_x),
                            jnp.asarray(separations * angular_ratio / image.grid.step),
                        )
                    )
                    / image.grid.pixel_area
                    * angular_ratio**2
                    for image in stellars
                ]
            )
    contrast = np.divide(
        density * area,
        tau,
        out=np.full_like(density, 1e20),
        where=np.broadcast_to(tau > 0, density.shape),
    )
    if stellar_diameters_lod is None:
        diameters = (
            None
            if stellar_diameter_lod is None
            else np.asarray([stellar_diameter_lod], dtype=float) / angular_ratio
        )
    else:
        # The record's axis is in lambda/D of its own (planet) frame.
        diameters = np.asarray(stellar_diameters_lod, dtype=float) / angular_ratio
    if isinstance(uniform_sky, Image):
        uniform_sky = SampledProfile.from_uniform_sky(uniform_sky)
    return PerformanceCurves(
        separations,
        grid.frame,
        rule,
        tau,
        area,
        leakage,
        density,
        contrast,
        _iwa(separations, tau),
        min(grid.shape) * grid.step / 2,
        leakage_rule,
        diameters,
        throughput_variance=tau,
        core_area_variance_lod2=area,
        leakage_fraction_variance=leakage,
        leakage_density_variance_lod2=density,
        uniform_sky=uniform_sky,
    )
