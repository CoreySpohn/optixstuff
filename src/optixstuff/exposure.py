"""Whole-frame detector integration plans."""

import equinox as eqx
import jax.numpy as jnp


class FramePlan(eqx.Module):
    """Whole-frame integration plan with array-valued times and counts.

    ``frame_time_s`` is strictly positive; ``n_frames`` is a nonnegative
    integral count stored as a float. Batched plans broadcast with rates
    using ordinary array broadcasting.
    """

    frame_time_s: jnp.ndarray = eqx.field(converter=lambda x: jnp.asarray(x) * 1.0)
    n_frames: jnp.ndarray = eqx.field(converter=lambda x: jnp.asarray(x) * 1.0)

    @property
    def live_time_s(self) -> jnp.ndarray:
        """Photon-collecting seconds, including the entire final frame."""
        return self.n_frames * self.frame_time_s

    def occupied_time_s(self, detector) -> jnp.ndarray:
        """Observatory seconds, adding read time when collection cannot overlap."""
        read_time_s = 0.0 if detector.overlaps_read else detector.read_time_s
        return self.n_frames * (self.frame_time_s + read_time_s)


def frames_for_time(time_s, frame_time_s) -> FramePlan:
    """Round requested collecting seconds up to a whole-frame plan.

    Inputs require ``time_s >= 0`` and ``frame_time_s > 0``. Ratios within
    relative tolerance ``max(1e-12, 4 * eps)`` of a positive integer are
    exact multiples. Every positive sub-frame duration takes one frame;
    zero duration takes zero frames. Counts remain floating under JIT,
    with zero derivative, including at boundaries (JAX rounding convention).
    """
    time_s = jnp.asarray(time_s)
    ratio = time_s / frame_time_s
    nearest = jnp.round(ratio)
    rtol = max(1e-12, 4 * jnp.finfo(ratio.dtype).eps)
    exact = (nearest > 0) & (jnp.abs(ratio - nearest) <= rtol * nearest)
    count = jnp.maximum(jnp.ceil(jnp.where(exact, nearest, ratio)), 1.0)
    return FramePlan(frame_time_s, jnp.where(time_s > 0, count, 0.0))
