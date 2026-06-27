"""Speckle field abstractions.

A speckle field is the stochastic, time-varying residual-starlight pattern
left by wavefront errors that wavefront control does not null. It sits *on
top of* the deterministic coronagraphic leakage floor (the YIP
``stellar_intens`` map already applied in ``coronagraphoto.star_rate``), so
it is an instrument effect, not an astrophysical scene source -- it lives on
:class:`optixstuff.OpticalPath`, next to the coronagraph, rather than in the
skyscapes scene.

The unifying generator (EFOR / Pogorelyuk / Manojkumar; see the
coronagraphoto-rasti speckle design) is ``I(t) = |E_nom + G eps(t)|^2``,
where ``E_nom`` is the static coherent residual field (``|E_nom|^2`` is the
deterministic floor), ``G`` maps wavefront-error modes to the focal-plane
field, and ``eps(t)`` are the drifting mode coefficients. Concrete fields
differ only in how they source ``G`` and ``eps`` (analytic, replayed
library cubes, fitted reduced-order, or learned); they share the
:class:`AbstractSpeckleField` contract below.
"""

import abc

import equinox as eqx
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
