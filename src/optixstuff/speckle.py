"""Speckle field abstractions.

A speckle field is the stochastic, time-varying residual-starlight pattern
left by wavefront errors that wavefront control does not null. It sits *on
top of* the deterministic coronagraphic leakage floor (the YIP
``stellar_intens`` map already applied in ``coronagraphoto.star_rate``), so
it is an instrument effect, not an astrophysical scene source -- it lives on
:class:`optixstuff.OpticalPath`, next to the coronagraph, rather than in the
skyscapes scene.

The standard linear speckle model is ``I(t) = |E_nom + G eps(t)|^2``, where
``E_nom`` is the static coherent residual field (``|E_nom|^2`` is the
deterministic floor), ``G`` maps wavefront-error modes to the focal-plane
field, and ``eps(t)`` are the drifting mode coefficients. Concrete fields
differ only in how they source ``G`` and ``eps`` (analytic, replayed
intensity cubes, fitted reduced-order, or learned); they share the
:class:`AbstractSpeckleField` contract below.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from jax.typing import ArrayLike
from jaxtyping import Array


class AbstractSpeckleField(eqx.Module):
    """Abstract interface for time-varying speckle fields.

    Implementations can be backed by an analytic generator (physicaloptix),
    replayed designer/testbed intensity cubes, a fitted reduced-order model,
    or a learned generator. All produce a coronagraph-plane contrast map
    through :meth:`realize`.

    All wavelength arguments are in nanometres; ``time_s`` is seconds since
    mission start. The returned map is on the field's native coronagraph
    plane at :attr:`pixel_scale_lod`; the caller resamples it to the
    detector grid (as for the coronagraph ``stellar_intens`` map).
    """

    pixel_scale_lod: AbstractVar[float]
    """Native pixel scale in lambda/D per pixel."""

    epoch_jd: AbstractVar[float]
    """Julian Date that maps to ``time_s = 0`` -- the realization's clock
    origin. Consumers (e.g. ``coronagraphoto.speckle_rate``) convert an
    observation's absolute JD to the elapsed seconds :meth:`realize` expects
    via ``(start_time_jd - epoch_jd)``; anchoring the clock here keeps the
    physical interface in elapsed seconds and avoids feeding large absolute
    JDs into the temporal synthesis."""

    @abc.abstractmethod
    def realize(
        self,
        *,
        wavelength_nm: ArrayLike,
        time_s: ArrayLike = 0.0,
    ) -> Array:
        """Speckle contrast *delta* at a given time and wavelength.

        Returns the wavefront-error-induced excess over the deterministic
        coronagraphic floor -- i.e. ``I(t) - |E_nom|^2``, expanded as
        ``2 Re(E_nom* . G eps(t)) + |G eps(t)|^2``, in contrast units
        (fraction of the host-star flux per pixel). It must not include the
        ``|E_nom|^2`` floor itself: ``star_rate`` already applies that via
        the YIP ``stellar_intens`` map, so re-emitting it here would double
        count. An incoherent-halo implementation returns only the strictly
        positive ``|G eps(t)|^2`` term (no speckle pinning); a coherent
        implementation adds the cross term, which carries the bright-tail
        pinning and requires the complex ``E_nom``.

        Evolution is driven by ``time_s``, not a per-call PRNG key: the
        realization is fixed at construction so the map is deterministic and
        differentiable, and temporal correlation survives across an exposure
        / roll sequence.

        Args:
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            2D float contrast-delta array on the native coronagraph plane.
        """
        ...

    def realize_average(
        self,
        *,
        wavelength_nm: ArrayLike,
        exposure_s: ArrayLike,
        start_time_s: ArrayLike = 0.0,
        n_sub: int = 8,
    ) -> Array:
        """Speckle contrast delta averaged over an exposure.

        :meth:`realize` is instantaneous, but a detector integrates. Averaging
        matters because the map is QUADRATIC in the drifting coefficients:
        the mean of the realized maps is not the map of the mean coefficients,
        so an instantaneous sample is a biased stand-in for an exposure
        whenever the field decorrelates on the exposure timescale. This
        evaluates :meth:`realize` at ``n_sub`` sub-exposure midpoints and
        averages, which is exact in the limit of many sub-steps and
        second-order accurate at finite ``n_sub``.

        How many sub-steps are enough, and how much the average suppresses
        the speckle variance relative to a snapshot, are both set by how many
        independent realizations the exposure spans. A generator that knows
        its own temporal statistics can say so in closed form (see
        ``physicaloptix.SpeckleProcess.exposure_neff``); when the exposure is
        short against the decorrelation time, one sub-step is already right
        and this reduces to :meth:`realize` at the exposure midpoint.

        Args:
            wavelength_nm: Wavelength in nanometres.
            exposure_s: Exposure length in seconds.
            start_time_s: Exposure start, in seconds since ``epoch_jd``.
            n_sub: Number of sub-exposure samples. Must be at least 1.

        Returns:
            2D float contrast-delta array, averaged over the exposure.

        Raises:
            ValueError: If ``n_sub`` is not a positive integer.
        """
        if int(n_sub) < 1 or int(n_sub) != n_sub:
            raise ValueError(f"n_sub must be a positive integer, got {n_sub}")
        n_sub = int(n_sub)
        start = jnp.asarray(start_time_s, dtype=float)
        step = jnp.asarray(exposure_s, dtype=float) / n_sub

        def sample(index):
            return self.realize(
                wavelength_nm=wavelength_nm, time_s=start + (index + 0.5) * step
            )

        # Accumulate rather than stacking: one map is held at a time, so a
        # large native grid does not cost n_sub copies.
        def accumulate(index, total):
            return total + sample(index)

        return jax.lax.fori_loop(1, n_sub, accumulate, sample(0)) / n_sub
