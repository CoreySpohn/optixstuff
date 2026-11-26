"""Aperture S/N and continuous exposure time through detector statistics.

Both functions read one aperture's :class:`optixstuff.Rates`: a record read
from a coronagraph table or an aperture sum of simulated images, or the
analytic rates of a scalar calculator. With planet electrons/s ``p``, the
detector variance of one frame divided by its live seconds ``b``, and the
squared residual rate ``f2``, the count variance after ``t`` live seconds is
``b t + f2 t**2``, so ``S/N = p t / sqrt(b t + f2 t**2)`` and
``t = snr**2 b / (p**2 - snr**2 f2)``.

A record carries what its producer fills: ``jaxedith.rates`` fills ``thermal``
from ``Instrument.warm_emission``, and the stray-light companions are zero.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.detector import AbstractDetector
from optixstuff.exposure import FramePlan
from optixstuff.rates import Rates


class PostProcessing(eqx.Module):
    """Assumptions of the PSF subtraction that follows the observation.

    ``speckle_residual_fraction`` and ``exozodi_residual_fraction`` are in
    [0, 1]; zero means perfect subtraction. The residual speckle and exozodi
    rates are these fractions of the mean aperture rates. Their squares add
    to the count variance as ``(fraction * rate * time)**2``; photon and
    electron rates are unchanged.

    ``background_variance_factor`` ``m >= 1`` multiplies the variance rate of
    every source except the planet's own shot noise (star leakage, zodi,
    exozodi, thermal, companions, dark current, CIC and read noise), the
    variance the subtraction of a noisy background adds. It is required and
    keyword-only: the caller states whether a reference image is subtracted.
    Two is differential imaging, where the reference image carries the same
    background variance as the science image (the convention of Stark et al.
    2014, AYO and pyEDITH); one is no reference subtraction, each background
    counted once.

    Attributes:
        speckle_residual_fraction: Fraction of the mean star-leakage rate left
            after subtraction, in [0, 1].
        exozodi_residual_fraction: Fraction of the mean exozodi rate left after
            subtraction, in [0, 1].
        background_variance_factor: Multiplier ``m >= 1`` on the variance rate of
            every source except the planet's own shot noise. Keyword-only and
            required.
    """

    speckle_residual_fraction: Array = eqx.field(default=0.0, converter=jnp.asarray)
    exozodi_residual_fraction: Array = eqx.field(default=0.0, converter=jnp.asarray)
    background_variance_factor: Array = eqx.field(kw_only=True, converter=jnp.asarray)

    def __check_init__(self):
        """Reject concrete nonfinite or out-of-range values."""
        for value in (self.speckle_residual_fraction, self.exozodi_residual_fraction):
            if not isinstance(value, jax.core.Tracer) and not bool(
                jnp.all(jnp.isfinite(value) & (value >= 0) & (value <= 1))
            ):
                raise ValueError("residual fractions must be finite and in [0, 1]")
        value = self.background_variance_factor
        if not isinstance(value, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(value) & (value >= 1))
        ):
            raise ValueError("background_variance_factor must be finite and >= 1")


class ExposureTime(eqx.Module):
    """Continuous live seconds reaching a target S/N, one value per band.

    ``valid`` is False where no time reaches the target: the planet rate is at
    or below the residual floor, the rates are not finite, or the read is not
    ``Rates.supported``. ``time_s`` is
    NaN there, so an unreachable target cannot be summed or scheduled as a
    zero-cost one; mask on ``valid`` first. Round to whole
    frames with ``optixstuff.frames_for_time(time_s, frame_time_s)``.

    Attributes:
        time_s: Continuous live seconds reaching the target S/N, shape (W,), NaN
            where unreachable.
        valid: Boolean, shape (W,), True where ``time_s`` is a reachable time.
    """

    time_s: Array
    valid: Array


def _per_band(value, ndim):
    """Align a scalar or (W,) value with the band axis of ndim-dimensional rates."""
    value = jnp.asarray(value)
    if value.ndim > 1:
        raise ValueError("per-band inputs must be scalar or shape (W,)")
    return value.reshape(value.shape + (1,) * (ndim - 1)) if value.ndim else value


def _like(value, shape):
    """Broadcast a field with fewer trailing axes than ``shape``, band axis first."""
    value = jnp.asarray(value)
    return jnp.broadcast_to(
        value.reshape(value.shape + (1,) * (len(shape) - value.ndim)), shape
    )


def _supported(rates, shape):
    """Boolean support of the read broadcast to shape, or None when all True."""
    return None if rates.supported is None else _like(rates.supported, shape)


def _statistics(rates, detector, frame_time_s, post_processing):
    """Return planet electrons/s, variance per live second and squared floor."""
    electrons = detector.electrons(rates) if rates.carrier == "photon" else rates
    frame_time_s = jnp.asarray(frame_time_s)
    # An undefined cadence (NaN, infinite or not positive) gives NaN statistics,
    # so the target reads as unreachable instead of raising in the plan.
    usable = jnp.isfinite(frame_time_s) & (frame_time_s > 0)
    one = frame_time_s.dtype.type(1)
    plan = FramePlan(jnp.where(usable, frame_time_s, one), one)
    _, one_frame = detector.mean_and_variance(electrons, plan)
    b = jnp.where(
        _per_band(usable, one_frame.ndim),
        one_frame / _per_band(frame_time_s, one_frame.ndim),
        one_frame.dtype.type(jnp.nan),
    )
    floor_sq = jnp.zeros_like(b)
    speckle = _per_band(post_processing.speckle_residual_fraction, b.ndim)
    exozodi = _per_band(post_processing.exozodi_residual_fraction, b.ndim)
    floor_sq = (
        floor_sq
        + (speckle * _like(electrons.star_leakage, b.shape)) ** 2
        + (exozodi * _like(electrons.exozodi, b.shape)) ** 2
    )
    shot = electrons if electrons.variance is None else electrons.variance
    planet_variance = detector.excess_noise_factor_sq * _like(shot.planet, b.shape)
    # m = 1 adds exactly zero, leaving the detector variance unchanged.
    factor = _per_band(post_processing.background_variance_factor, b.ndim)
    b = b + (factor - 1.0) * (b - planet_variance)
    return _like(electrons.planet, b.shape), b, floor_sq


def snr(
    rates: Rates,
    detector: AbstractDetector,
    time_s: ArrayLike,
    *,
    frame_time_s: ArrayLike,
    post_processing: PostProcessing,
) -> Array:
    """Return S/N per band, shape (W,), after ``time_s`` live seconds.

    ``rates`` is one aperture in photons/s/nm, to which the detector applies
    its yield here, or in electrons/s. Signal is planet electrons times live
    time. The detector supplies all independent count variance: source shot
    noise from the squared-weight ``rates.variance``, dark current, clock
    induced charge and read noise over the aperture's pixels, with CIC and
    read noise accruing once per ``frame_time_s``. ``post_processing`` adds
    the squared residual speckle and exozodi counts and scales the background
    variance; it is required, as no subtraction is not the same as perfect
    subtraction.

    ``time_s`` and ``frame_time_s`` are scalar or shape (W,). The result
    equals ``detector.mean_and_variance`` at every whole number of frames and
    interpolates between them. Zero time gives zero. NaN rates, as read
    outside an aperture record, give NaN, and so does a read with
    ``rates.supported`` False, whatever finite rates it holds. Rate fields
    may carry trailing batch axes after the band axis.

    Args:
        rates: One aperture's Rates, in photon/s/nm (the detector applies its
            yield) or electron/s.
        detector: Detector supplying yield and count variance.
        time_s: Live seconds, scalar or shape (W,).
        frame_time_s: Seconds per frame, scalar or shape (W,); CIC and read noise
            accrue once per frame.
        post_processing: Residual fractions and background variance factor of the
            subtraction.

    Returns:
        S/N per band, shape (W,), plus any trailing batch axes of the rates.
    """
    p, b, floor_sq = _statistics(rates, detector, frame_time_s, post_processing)
    time_s = _per_band(time_s, p.ndim)
    variance = b * time_s + floor_sq * time_s**2
    # Only an exact zero is replaced; a NaN variance propagates.
    result = p * time_s / jnp.sqrt(jnp.where(variance == 0, 1, variance))
    supported = _supported(rates, result.shape)
    return (
        result
        if supported is None
        else jnp.where(supported, result, result.dtype.type(jnp.nan))
    )


def exposure_time(
    rates: Rates,
    detector: AbstractDetector,
    target_snr: ArrayLike,
    *,
    frame_time_s: ArrayLike,
    post_processing: PostProcessing,
) -> ExposureTime:
    """Return the continuous live time reaching each band's target S/N.

    The inverse of :func:`snr`: ``t = target_snr**2 b / (p**2 - target_snr**2
    f2)``. Read noise and CIC are exact in ``b`` because they accrue per frame
    of ``frame_time_s``. ``target_snr`` and ``frame_time_s`` are scalar or
    shape (W,).

    A positive target with ``p <= target_snr sqrt(f2)`` cannot be reached at
    any time: ``valid`` is False and ``time_s`` is NaN. The same holds for
    rates that are not finite and for a read with ``rates.supported`` False. A
    nonpositive target needs zero time, except on an unsupported read.

    Args:
        rates: One aperture's Rates, in photon/s/nm (the detector applies its
            yield) or electron/s.
        detector: Detector supplying yield and count variance.
        target_snr: Target S/N, scalar or shape (W,).
        frame_time_s: Seconds per frame, scalar or shape (W,); CIC and read noise
            accrue once per frame.
        post_processing: Residual fractions and background variance factor of the
            subtraction.

    Returns:
        Live seconds per band and the mask of bands where the target is reachable.
    """
    p, b, floor_sq = _statistics(rates, detector, frame_time_s, post_processing)
    target_snr = _per_band(target_snr, p.ndim)
    denominator = p**2 - target_snr**2 * floor_sq
    # Feasibility compares rates, not a difference of squares, so a target
    # exactly at the residual ceiling is unreachable despite rounding.
    reachable = (
        (p > target_snr * jnp.sqrt(floor_sq)) & (denominator > 0) & jnp.isfinite(b)
    )
    time_s = (
        target_snr**2
        * jnp.where(reachable, b, b.dtype.type(0))
        / jnp.where(reachable, denominator, denominator.dtype.type(1))
    )
    trivial = (target_snr <= 0) & jnp.isfinite(p) & jnp.isfinite(b)
    valid = reachable | trivial
    zero, nan = time_s.dtype.type(0), time_s.dtype.type(jnp.nan)
    time_s = jnp.where(trivial, zero, jnp.where(reachable, time_s, nan))
    supported = _supported(rates, time_s.shape)
    if supported is not None:
        valid = valid & supported
        time_s = jnp.where(supported, time_s, nan)
    return ExposureTime(time_s, valid)
