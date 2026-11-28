"""Observation pointing and whole-frame detector integration plans."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import mas_to_arcsec
from hwoutils.epoch import Epoch
from hwoutils.leaves import as_float_leaf, as_leaf, is_traced
from hwoutils.precision import PrecisionError, require_float64

from optixstuff.grids import SkyOrientation
from optixstuff.optical_elements import check_name
from optixstuff.spectral import Band


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

    def collection_span_s(self, *, read_time_s, overlaps_read: bool) -> jnp.ndarray:
        """Seconds from the first gate opening to the last gate closing.

        Frame ``k`` collects on ``[k * (frame_time_s + r), k * (frame_time_s + r)
        + frame_time_s]``, with ``r`` zero when reads overlap collection and
        ``read_time_s`` otherwise. The caller states the gates: a simulation may
        pass its detector's values, and archived data passes the recorded
        values. A fitted read time is never the source for archived data.

        Args:
            read_time_s: Seconds of read between frames, scalar or shape (W,).
            overlaps_read: Whether collection continues during the read.

        Returns:
            Span in seconds, scalar or shape (W,). NaN for zero frames or a NaN
            count, which have no collection.
        """
        dtype = _timing_dtype(self.frame_time_s, self.n_frames, read_time_s)
        counts = self.n_frames.astype(dtype)
        frame = self.frame_time_s.astype(dtype)
        stride = frame if overlaps_read else frame + jnp.asarray(read_time_s, dtype)
        span = (counts - 1) * stride + frame
        return jnp.where(counts >= 1, span, jnp.asarray(jnp.nan, dtype))

    def collection_midpoint_s(self, *, read_time_s, overlaps_read: bool) -> jnp.ndarray:
        """Seconds from the first gate opening to the middle of collection.

        The gates are symmetric about the midpoint of the span, which is not
        half of ``occupied_time_s``: the last read follows the last gate.

        Args:
            read_time_s: Seconds of read between frames, scalar or shape (W,).
            overlaps_read: Whether collection continues during the read.

        Returns:
            Midpoint offset in seconds, scalar or shape (W,); NaN when there is
            no collection. Add it to an elapsed-seconds value, never to a time
            in days.
        """
        return (
            self.collection_span_s(read_time_s=read_time_s, overlaps_read=overlaps_read)
            / 2
        )


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
    """The commanded observation event: when, where and with which configuration.

    The event holds no scene, no observatory and no build product, so a
    scheduler can build it alone. The pointing values and the start time are
    array leaves, so an event can be an argument of JIT, vmap and grad. The
    epoch, configuration name and commanded bands are static: events stack
    only when all three are equal. The wavelength rule and the operators that
    serve it live in ``SpectralSampling``, passed beside the event.

    ``time_d`` is the start of photon collection, in days from ``epoch``. It is
    never a midpoint and never days from a scene epoch. A consumer that needs
    scene days calls ``exposure.on_epoch(scene.epoch).time_d`` on the host; a
    hardware clock takes ``exposure.on_epoch(clock_epoch).time_d``.

    Attributes:
        time_d: Start of collection in days from ``epoch``, a float leaf of the
            active dtype.
        pointing: Commanded roll and optional detector-frame offset.
        plan: Whole-frame integration plan; None when only an instant rate is
            wanted.
        epoch: Day zero of ``time_d``; static host metadata.
        instrument: Name of the instrument configuration; static.
        bands: Ordered commanded science bands; static and the only band axis
            of an image.
    """

    time_d: jnp.ndarray | float = eqx.field(converter=as_float_leaf)
    pointing: Pointing
    plan: FramePlan | None = None
    epoch: Epoch = eqx.field(static=True, kw_only=True)
    instrument: str = eqx.field(static=True, kw_only=True)
    bands: tuple[Band, ...] = eqx.field(static=True, converter=tuple, kw_only=True)

    def __check_init__(self):
        """Require typed static fields and a plan that fits the bands."""
        if not isinstance(self.epoch, Epoch):
            raise TypeError(f"epoch must be an Epoch, got {type(self.epoch).__name__}")
        check_name(self.instrument, "instrument")
        if not self.bands or not all(isinstance(band, Band) for band in self.bands):
            raise ValueError("bands must be a nonempty tuple of Band")
        if self.plan is not None and jnp.ndim(self.time_d) == 0:
            for leaf in (self.plan.frame_time_s, self.plan.n_frames):
                if not isinstance(leaf, jax.core.Tracer) and leaf.shape not in (
                    (),
                    (len(self.bands),),
                ):
                    raise ValueError(
                        f"plan has shape {leaf.shape} but the event has "
                        f"{len(self.bands)} bands"
                    )

    def on_epoch(self, epoch: Epoch) -> "Exposure":
        """State the same instant on another day zero.

        Equal epochs (the same instant on the same scale, see ``Epoch``) return
        the event itself. Otherwise the shift is added in binary64 on the host,
        or on the device for a float64 tracer, or the call is refused.

        Args:
            epoch: The target epoch, on the same time scale.

        Returns:
            The event with ``epoch`` and ``time_d`` stated on the target epoch.

        Raises:
            ValueError: If the time scales differ.
            PrecisionError: If a nonzero shift would be added to a float32
                ``time_d``. Build the event on the target epoch from the
                original host date instead.
        """
        if epoch == self.epoch:
            return self
        shift = self.epoch.days_between(epoch)
        time_d = self.time_d
        if time_d.dtype != np.float64:
            raise PrecisionError(
                f"time_d of dtype {time_d.dtype} may have lost its offset "
                f"of {shift!r} d between the two epochs. Build the event on "
                "the target epoch from the original host date."
            )
        if isinstance(time_d, jax.core.Tracer):
            require_float64("Traced epoch shift (or build the event on the target)")
            time_d = time_d + shift
        else:
            time_d = np.asarray(time_d, dtype=np.float64) + np.float64(shift)
        return Exposure(
            time_d,
            self.pointing,
            self.plan,
            epoch=epoch,
            instrument=self.instrument,
            bands=self.bands,
        )

    def collection_midpoint_s(self, *, read_time_s, overlaps_read: bool) -> jnp.ndarray:
        """Seconds from ``time_d`` to the middle of collection.

        Args:
            read_time_s: Seconds of read between frames, from the stated gates.
            overlaps_read: Whether collection continues during the read.

        Returns:
            Zero in the dtype of ``time_d`` when there is no plan (an instant),
            else ``plan.collection_midpoint_s``. An offset in seconds: add it to
            an elapsed-seconds value, never to ``time_d`` in days.
        """
        if self.plan is None:
            return jnp.zeros((), dtype=self.time_d.dtype)
        return self.plan.collection_midpoint_s(
            read_time_s=read_time_s, overlaps_read=overlaps_read
        )
