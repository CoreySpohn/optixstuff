"""Whole-frame detector integration plans."""

import equinox as eqx
import jax.numpy as jnp
from hwoutils.leaves import as_float_leaf

from optixstuff.spectral import Quadrature, Resolution


def _band_shape(first, second):
    """Validate two scalar or per-band shapes and return their band shape."""
    if max(len(first), len(second)) > 1 or (first and second and first != second):
        raise ValueError(
            "FramePlan inputs must be scalar or (W,) with a shared W; "
            f"got shapes {first} and {second}"
        )
    return first or second


class FramePlan(eqx.Module):
    """Whole-frame integration plan with array-valued times and counts.

    ``frame_time_s`` is strictly positive; ``n_frames`` is a nonnegative
    integral count stored as a float. Each field is scalar or shape (W,),
    with one value per band on axis 0 of the rates. Use vmap for other
    batching.

    Attributes:
        frame_time_s: Seconds per frame, strictly positive, scalar or shape (W,).
        n_frames: Whole number of frames, stored as a float, scalar or shape (W,).
    """

    frame_time_s: jnp.ndarray = eqx.field(converter=lambda x: jnp.asarray(x) * 1.0)
    n_frames: jnp.ndarray = eqx.field(converter=lambda x: jnp.asarray(x) * 1.0)

    def __check_init__(self):
        """Validate field shapes before the plan is used."""
        _band_shape(self.frame_time_s.shape, self.n_frames.shape)

    def _for_bands(self, rate_bands, ndim):
        """Return live time and frame counts aligned to the rates' band axis."""
        bands = _band_shape(self.frame_time_s.shape, self.n_frames.shape)
        if not bands:
            return self.live_time_s, self.n_frames
        if bands != (len(rate_bands),):
            raise ValueError(
                "FramePlan fields must be scalar or (W,) matching the rates' "
                f"leading band axis; got plan {bands} for {len(rate_bands)} bands"
            )
        band_index = (slice(None),) + (None,) * (ndim - 1)
        counts = self.n_frames
        return self.live_time_s[band_index], counts[
            band_index
        ] if counts.ndim else counts

    @property
    def live_time_s(self) -> jnp.ndarray:
        """Photon-collecting seconds, including the entire final frame."""
        return self.n_frames * self.frame_time_s

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
        return self.n_frames * (self.frame_time_s + read_time_s)


def frames_for_time(time_s, frame_time_s) -> FramePlan:
    """Round requested collecting seconds up to a whole-frame plan.

    Inputs are scalar or (W,), with ``time_s >= 0`` and ``frame_time_s > 0``.
    Ratios within relative tolerance ``max(1e-12, 4 * eps)`` of a positive integer
    are exact multiples. Every positive sub-frame duration takes one frame;
    zero duration takes zero frames. Counts remain floating under JIT,
    with zero derivative, including at boundaries (JAX rounding convention).

    Args:
        time_s: Requested collecting seconds, scalar or shape (W,), at least zero.
        frame_time_s: Seconds per frame, scalar or shape (W,), positive.

    Returns:
        Plan with the smallest whole number of frames covering ``time_s``.
    """
    time_s, frame_time_s = jnp.asarray(time_s), jnp.asarray(frame_time_s)
    _band_shape(time_s.shape, frame_time_s.shape)
    ratio = time_s / frame_time_s
    nearest = jnp.round(ratio)
    rtol = max(1e-12, 4 * jnp.finfo(ratio.dtype).eps)
    exact = (nearest > 0) & (jnp.abs(ratio - nearest) <= rtol * nearest)
    count = jnp.maximum(jnp.ceil(jnp.where(exact, nearest, ratio)), 1.0)
    return FramePlan(frame_time_s, jnp.where(time_s > 0, count, 0.0))


class Exposure(eqx.Module):
    """Everything about one observation except the scene and the instrument.

    The record holds when the telescope observes and at what roll, how each
    band is integrated over wavelength, and (for a drawn frame) how long the
    detector collects. Build it on the host with ``coronagraphoto.build_exposure``.
    The roll, the observation time and the quadrature nodes and weights are array
    leaves, so an exposure can be an argument of JIT, vmap and grad.

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
        roll_deg: Position angle of detector +y, north through east, in
            degrees; a traced leaf.
        quadrature: Per-band wavelength nodes and weights.
        plan: Whole-frame integration plan; None when only expected rates
            are wanted.
        resolution: Operators of the scene's tabulated sources on
            ``quadrature``; empty when the scene has none.
    """

    time_d: jnp.ndarray | float = eqx.field(converter=as_float_leaf)
    roll_deg: jnp.ndarray = eqx.field(converter=lambda x: jnp.asarray(x) * 1.0)
    quadrature: Quadrature
    plan: FramePlan | None = None
    resolution: Resolution = eqx.field(default_factory=Resolution)

    def __check_init__(self):
        """Require a resolution built for this quadrature."""
        built = self.resolution.quadratures
        if built and built[0].layout_digest != self.quadrature.layout_digest:
            raise ValueError("the resolution was built for another quadrature")
