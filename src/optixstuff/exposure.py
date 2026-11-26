"""Observation pointing and whole-frame detector integration plans."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import mas_to_arcsec
from hwoutils.leaves import as_float_leaf, as_leaf, is_traced

from optixstuff.grids import SkyOrientation
from optixstuff.spectral import Quadrature, Resolution


def _band_shape(first, second):
    """Validate two scalar or per-band shapes and return their band shape."""
    if max(len(first), len(second)) > 1 or (first and second and first != second):
        raise ValueError(
            "FramePlan inputs must be scalar or (W,) with a shared W; "
            f"got shapes {first} and {second}"
        )
    return first or second


def _timing_dtype(*values):
    """Select the floating arithmetic dtype without promoting on counts."""
    return jnp.result_type(
        *(
            value
            for value in values
            if jnp.issubdtype(jnp.result_type(value), jnp.floating)
        ),
        1.0,
    )


def _timing_arraylike(value):
    """Normalize sequences while retaining checked host integer conversion."""
    if isinstance(value, (tuple, list)):
        traced = any(is_traced(item) for item in jax.tree.leaves(value))
        return jnp.asarray(value) if traced else np.asarray(value)
    return value


class FramePlan(eqx.Module):
    """Whole-frame integration plan with array-valued times and counts.

    ``frame_time_s`` is finite and strictly positive. ``n_frames`` is a
    nonnegative whole number, or NaN for an unreachable request. Integer
    counts retain their exact value until used in floating arithmetic.
    Each field is scalar or shape (W,), with one value per band on
    axis 0 of the rates. Use vmap for other batching.

    Attributes:
        frame_time_s: Finite positive seconds per frame, scalar or shape (W,).
        n_frames: Integer or floating count or NaN, scalar or shape (W,).
    """

    frame_time_s: jnp.ndarray
    n_frames: jnp.ndarray

    def __init__(self, frame_time_s, n_frames):
        """Store frame times and preserve integer counts.

        Args:
            frame_time_s: Finite positive seconds per frame, scalar or shape (W,).
            n_frames: Nonnegative whole count or NaN, scalar or shape (W,).
        """
        frame_time_s, n_frames = (
            _timing_arraylike(x) for x in (frame_time_s, n_frames)
        )
        dtype = _timing_dtype(frame_time_s, n_frames)
        self.frame_time_s = jnp.asarray(
            frame_time_s, dtype=None if jnp.iscomplexobj(frame_time_s) else dtype
        )
        self.n_frames = (
            as_leaf(n_frames)
            if jnp.issubdtype(jnp.result_type(n_frames), jnp.integer)
            else jnp.asarray(
                n_frames, dtype=None if jnp.iscomplexobj(n_frames) else dtype
            )
        )

    def __check_init__(self):
        """Validate field shapes and concrete values before the plan is used."""
        _band_shape(self.frame_time_s.shape, self.n_frames.shape)
        if not isinstance(self.frame_time_s, jax.core.Tracer):
            frame = np.asarray(self.frame_time_s)
            if np.iscomplexobj(frame) or not np.all(np.isfinite(frame) & (frame > 0)):
                raise ValueError("frame_time_s must be finite and strictly positive")
        if not isinstance(self.n_frames, jax.core.Tracer):
            count = np.asarray(self.n_frames)
            if np.iscomplexobj(count) or not np.all(
                np.isnan(count)
                | (np.isfinite(count) & (count >= 0) & (count == np.floor(count)))
            ):
                raise ValueError("n_frames must be a nonnegative whole number or NaN")

    def _for_bands(self, rate_bands, ndim, *operands):
        """Return live time and frame counts aligned to the rates' band axis."""
        bands = _band_shape(self.frame_time_s.shape, self.n_frames.shape)
        dtype = _timing_dtype(self.frame_time_s, self.n_frames, *operands)
        counts = self.n_frames.astype(dtype)
        live_time = counts * self.frame_time_s.astype(dtype)
        if not bands:
            return live_time, counts
        if bands != (len(rate_bands),):
            raise ValueError(
                "FramePlan fields must be scalar or (W,) matching the rates' "
                f"leading band axis; got plan {bands} for {len(rate_bands)} bands"
            )
        band_index = (slice(None),) + (None,) * (ndim - 1)
        return live_time[band_index], counts[band_index] if counts.ndim else counts

    @property
    def live_time_s(self) -> jnp.ndarray:
        """Total photon-collecting time.

        Returns:
            Seconds of collection, including the entire final frame.
        """
        dtype = _timing_dtype(self.frame_time_s, self.n_frames)
        return self.n_frames.astype(dtype) * self.frame_time_s

    def occupied_time_s(self, detector) -> jnp.ndarray:
        """Observatory seconds, adding read time when collection cannot overlap.

        Args:
            detector: Detector whose ``read_time_s`` and ``overlaps_read`` set the read
                overhead.

        Returns:
            Observatory seconds per band: frames times frame time plus read time when
            reads cannot overlap collection.
        """
        read_time_s = 0.0 if detector.overlaps_read else detector.read_time_s
        dtype = _timing_dtype(self.frame_time_s, self.n_frames, read_time_s)
        return self.n_frames.astype(dtype) * (self.frame_time_s + read_time_s)


def frames_for_time(time_s, frame_time_s) -> FramePlan:
    """Round requested collecting seconds up to a whole-frame plan.

    For positive finite requests, the frame count is at least one and its
    product with the frame duration covers the request in the working dtype.
    Zero seconds takes zero frames. Non-finite or negative requests give NaN
    counts and live times. Counts remain floating under JIT with zero derivative.

    Args:
        time_s: Requested collecting seconds, scalar or shape (W,).
        frame_time_s: Finite positive seconds per frame, scalar or shape (W,).

    Returns:
        Plan with the smallest whole number of frames covering ``time_s``.
    """
    time_s, frame_time_s = jnp.asarray(time_s), jnp.asarray(frame_time_s)
    _band_shape(time_s.shape, frame_time_s.shape)
    ratio = time_s / frame_time_s
    time_s, frame_time_s = time_s.astype(ratio.dtype), frame_time_s.astype(ratio.dtype)
    count = jnp.maximum(jnp.ceil(ratio), 1.0)
    # Division and multiplication round separately in this dtype.
    # Past the integer range of the dtype the neighbouring counts are its spacing away.
    infinity = count.dtype.type(np.inf)
    following = jnp.maximum(count + 1.0, jnp.nextafter(count, infinity))
    count = jnp.where(count * frame_time_s < time_s, following, count)
    previous = jnp.minimum(count - 1.0, jnp.nextafter(count, -infinity))
    count = jnp.where(
        (previous >= 1) & (previous * frame_time_s >= time_s), previous, count
    )
    count = jnp.where(time_s == 0, count.dtype.type(0), count)
    valid = jnp.isfinite(time_s) & (time_s >= 0)
    return FramePlan(
        frame_time_s,
        jax.lax.stop_gradient(jnp.where(valid, count, count.dtype.type(np.nan))),
    )


class Pointing(eqx.Module):
    """Tangent-plane attitude for an exposure.

    Roll is the position angle of detector +y, measured north through east.
    The offset is the target's position relative to the coronagraph optical
    axis at `PixelGrid.center_x, center_y`. Its components are sky-projected
    milliarcseconds along detector +x (columns) and +y (rows). A positive x
    offset moves every source toward higher columns at any roll.

    Source coordinates are east and north of the target, in arcseconds.
    Rotate about the target with `SkyOrientation`, then add the detector-frame
    offset in arcseconds. At parity -1 the rotation gives
    `x = -east * cos(roll) + north * sin(roll)` and
    `y = east * sin(roll) + north * cos(roll)`. The boresight is the target
    minus the inverse-rotated offset.

    The pointing offset carries the exact degeneracy with entrance-pupil
    tip/tilt for all sources. Keep that degree of freedom in `Pointing` when
    fitting. Noll tilt with RMS wavefront error `a` gives `4 * a / D` radians,
    where `D` is the pupil diameter in the same length units as `a`. Post-mask
    tip/tilt is a rigid image shift degenerate with the calibrated detector
    zero point, carried by `PixelGrid.center_x, center_y`.

    Attributes:
        roll_deg: Position angle of detector +y in degrees, a traced float leaf.
        offset_mas: Detector (x, y) offset as a traced float leaf of shape
            (2,), or a structural None for on-axis pointing.
    """

    roll_deg: jax.Array = eqx.field(converter=as_float_leaf)
    offset_mas: jax.Array | None = eqx.field(
        default=None,
        converter=lambda offset: None if offset is None else as_float_leaf(offset),
    )

    def __check_init__(self):
        """Require a single detector offset pair."""
        if self.offset_mas is not None and self.offset_mas.shape != (2,):
            raise ValueError("offset_mas must have shape (2,)")

    def detector_offsets_arcsec(self, east_arcsec, north_arcsec, parity=-1):
        """Map target-relative sky coordinates to optical-axis-relative offsets.

        Args:
            east_arcsec: East coordinate from the target, in arcseconds.
            north_arcsec: North coordinate from the target, in arcseconds.
            parity: Detector handedness, -1 or +1.

        Returns:
            Detector (x, y) offsets from the optical axis, in arcseconds.

        Raises:
            ValueError: An invalid parity or non-finite concrete roll was supplied.
        """
        x, y = SkyOrientation(self.roll_deg, parity).sky_to_detector(
            east_arcsec, north_arcsec
        )
        if self.offset_mas is not None:
            dx, dy = mas_to_arcsec(self.offset_mas)
            return x + dx, y + dy
        return x, y


class Exposure(eqx.Module):
    """Everything about one observation except the scene and the instrument.

    The record holds when and where the telescope points, how each
    band is integrated over wavelength, and (for a drawn frame) how long the
    detector collects. Build it on the host with ``coronagraphoto.build_exposure``.
    The pointing values, observation time and quadrature nodes and weights
    are array leaves, so an exposure can be an argument of JIT, vmap and grad.

    ``resolution`` holds the operators that turn each tabulated source of the
    scene into node-effective values on ``quadrature``; it is empty for a scene
    with no table. The build reads the knot grids of those tables, and
    ``reads`` names them, so a fit refuses to free a leaf the operators were
    built from.

    Attributes:
        time_d: Observation time in days from the scene epoch, a float leaf of
            the active dtype. The local-zodi
            sightline is derived from it, the scene's epoch and the target's
            sky position, not stored here.
        pointing: Roll and optional detector-frame offset for this observation.
        quadrature: Per-band wavelength nodes and weights.
        plan: Whole-frame integration plan; None when only expected rates
            are wanted.
        resolution: Operators of the scene's tabulated sources on
            ``quadrature``; empty when the scene has none.
    """

    time_d: jnp.ndarray | float = eqx.field(converter=as_float_leaf)
    pointing: Pointing
    quadrature: Quadrature
    plan: FramePlan | None = None
    resolution: Resolution = eqx.field(default_factory=Resolution)

    def __check_init__(self):
        """Require a resolution built for this quadrature."""
        built = self.resolution.quadratures
        if built and built[0].layout_digest != self.quadrature.layout_digest:
            raise ValueError("the resolution was built for another quadrature")
