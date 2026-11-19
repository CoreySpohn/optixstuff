"""Named source contributions, as photons or electrons, per pixel or per aperture."""

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.spectral import Band, Quadrature

_COMPONENTS = (
    "planet",
    "star_leakage",
    "local_zodi",
    "exozodi",
    "thermal",
    "stellar_companions",
)
_CARRIERS = ("photon", "electron")


class Rates(eqx.Module):
    """Rates of six additive sources, with their carrier and sampling tagged.

    Three things tell records apart, and all three are read off the record:

    - ``carrier`` (static) is ``"photon"`` before detector yield and
      ``"electron"`` after it.
    - ``quadrature`` is set for spectral integrands, per nm at its nodes, with
      fields shaped ``(W, Q, *spatial)``. It is None once the bands are
      integrated, and the fields are then shaped ``(W, *spatial)``.
    - ``n_pix`` is None for per-pixel values, where ``spatial`` is ``(ny, nx)``
      or empty for a single pixel. For a weighted aperture it is the sum of the
      weights in detector pixels, scalar or one value per band, and each field
      is the weighted sum over the aperture, usually with empty ``spatial``.

    ``unit`` spells out the result: ``photon/s/nm``, ``photon/s``,
    ``electron/s/nm`` or ``electron/s``, per pixel or per aperture.

    ``variance``, when set, is a second record of the same kind summed with
    the squared weights; its ``n_pix`` is the sum of squared weights. Poisson
    variance of a weighted aperture follows it instead of the mean record.

    An absent source may be scalar zero. ``bands`` (static) names the leading
    axis and equals ``quadrature.bands`` for spectral records.
    """

    planet: Array
    star_leakage: Array
    local_zodi: Array
    exozodi: Array
    thermal: Array
    stellar_companions: Array
    bands: tuple[Band, ...] = eqx.field(static=True)
    carrier: Literal["photon", "electron"] = eqx.field(static=True)
    quadrature: Quadrature | None
    n_pix: Array | None
    variance: "Rates | None"

    def __init__(
        self,
        sampling: Quadrature | Sequence[Band],
        carrier: Literal["photon", "electron"],
        *,
        planet: ArrayLike = 0.0,
        star_leakage: ArrayLike = 0.0,
        local_zodi: ArrayLike = 0.0,
        exozodi: ArrayLike = 0.0,
        thermal: ArrayLike = 0.0,
        stellar_companions: ArrayLike = 0.0,
        n_pix: ArrayLike | None = None,
        variance: "Rates | None" = None,
    ):
        """Store rates sampled on a Quadrature (per nm) or integrated over bands.

        Pass the Quadrature for spectral integrands or the bands for
        band-integrated rates; the carrier is always named.
        """
        if isinstance(sampling, Quadrature):
            self.quadrature = sampling
            self.bands = sampling.bands
        else:
            self.quadrature = None
            self.bands = tuple(sampling)
        self.carrier = carrier
        self.planet = jnp.asarray(planet)
        self.star_leakage = jnp.asarray(star_leakage)
        self.local_zodi = jnp.asarray(local_zodi)
        self.exozodi = jnp.asarray(exozodi)
        self.thermal = jnp.asarray(thermal)
        self.stellar_companions = jnp.asarray(stellar_companions)
        self.n_pix = None if n_pix is None else jnp.asarray(n_pix)
        self.variance = variance

    def __check_init__(self):
        """Require a known carrier, matching leading axes and a like variance."""
        if self.carrier not in _CARRIERS:
            raise ValueError(f"carrier must be one of {_CARRIERS}")
        if self.quadrature is not None and self.quadrature.bands != self.bands:
            raise ValueError("bands must be the quadrature's bands")
        leading = self._leading_shape
        for name in _COMPONENTS:
            value = getattr(self, name)
            if value.ndim and value.shape[: len(leading)] != leading:
                raise ValueError(
                    f"{name} must match the quadrature's (W, Q) shape"
                    if self.spectral
                    else f"{name} must have one leading entry per band"
                )
        other = self.variance
        if other is not None:
            if other.bands != self.bands:
                raise ValueError("mean and variance rates must have the same bands")
            if other.carrier != self.carrier or other.spectral != self.spectral:
                raise ValueError("mean and variance rates must have the same unit")
            if other.variance is not None:
                raise ValueError("variance rates cannot carry their own variance")

    @staticmethod
    def component_names() -> tuple[str, ...]:
        """Names of additive source terms, in their declared order."""
        return _COMPONENTS

    @property
    def spectral(self) -> bool:
        """Whether fields are per-nm integrands at quadrature nodes."""
        return self.quadrature is not None

    @property
    def unit(self) -> str:
        """Unit of every field: carrier per second, per nm while spectral."""
        return f"{self.carrier}/s/nm" if self.spectral else f"{self.carrier}/s"

    @property
    def _leading_shape(self) -> tuple[int, ...]:
        if self.quadrature is None:
            return (len(self.bands),)
        return self.quadrature.nodes_nm.shape

    def total(self) -> Array:
        """Sum the six sources, keeping the leading axes if all are scalar."""
        total = sum(getattr(self, name) for name in _COMPONENTS)
        return jnp.broadcast_to(total, total.shape or self._leading_shape)

    def integrate(self) -> "Rates":
        """Integrate spectral fields over each band, to shape ``(W, *spatial)``.

        Dimensional weights are in nm, so photon/s/nm becomes photon/s.
        Scalar absent terms broadcast to the shape shared by the sources.
        The carrier and ``n_pix`` are kept; ``variance`` is integrated too.
        """
        quadrature = self.quadrature
        if quadrature is None:
            raise ValueError("rates are already integrated over their bands")
        shape = self.total().shape
        return Rates(
            self.bands,
            self.carrier,
            n_pix=self.n_pix,
            variance=None if self.variance is None else self.variance.integrate(),
            **{
                name: quadrature.integrate(jnp.broadcast_to(getattr(self, name), shape))
                for name in _COMPONENTS
            },
        )

    @classmethod
    def zeros(
        cls,
        sampling: Quadrature | Sequence[Band],
        carrier: Literal["photon", "electron"],
        spatial_shape: tuple[int, ...] = (),
        *,
        dtype=None,
    ) -> "Rates":
        """Create six zero fields with full shape, leading axes then spatial."""
        empty = cls(sampling, carrier)
        zero = jnp.zeros((*empty._leading_shape, *spatial_shape), dtype=dtype)
        return cls(sampling, carrier, **dict.fromkeys(_COMPONENTS, zero))
