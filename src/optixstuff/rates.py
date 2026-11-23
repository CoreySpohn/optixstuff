"""Named source contributions, as photons or electrons, per pixel or per aperture."""

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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

    ``supported``, when set, is a boolean beside the values with shape
    ``(W, *spatial)``: False marks a read the coronagraph data do not cover.
    The values there are finite zeros, not NaN, so sums and gradients stay
    intact; ``optixstuff.snr`` and ``optixstuff.exposure_time`` turn the flag
    into NaN. None means every read is supported. ``integrate`` and the
    detector's ``electrons`` carry the flag through unchanged.

    ``static_supported``, when set, is the same kind of flag for every product
    except a speckle field: True where the static coronagraph products cover
    the read, whether or not the drift model does. ``supported`` is the strict
    one, so ``supported`` True implies ``static_supported`` True, and a read
    that lacks only the speckle field is ``static_supported & ~supported``.
    None means "not distinguished; read ``supported``". Terminal statistics
    read only ``supported``. The constructor checks the implication on
    concrete arrays (traced arrays are not checked) and refuses
    ``supported=None`` beside a ``static_supported`` with a False entry.

    An absent source may be scalar zero. ``bands`` (static) names the leading
    axis and equals ``quadrature.bands`` for spectral records.

    Attributes:
        planet: Planet rate, photon or electron per second (per nm while spectral).
        star_leakage: Residual starlight rate, in the same unit.
        local_zodi: Local zodiacal light rate, in the same unit.
        exozodi: Exozodiacal light rate, in the same unit.
        thermal: Warm-optics thermal emission rate, in the same unit.
        stellar_companions: Rate of other stars in the field, in the same unit.
        bands: Wavelength bands naming the leading axis.
        carrier: ``"photon"`` before detector yield, ``"electron"`` after it.
        quadrature: Wavelength nodes and weights of spectral records, or None once
            the bands are integrated.
        n_pix: Sum of aperture weights in detector pixels, or None for per-pixel
            values.
        variance: Record summed with squared weights, or None.
        supported: Boolean flag, shape ``(W, *spatial)``, False where the
            coronagraph data do not cover the read, or None when all reads are
            supported.
        static_supported: Boolean flag of the same shape for every product but
            the speckle field, or None when the record does not distinguish.
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
    supported: Array | None
    static_supported: Array | None

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
        supported: ArrayLike | None = None,
        static_supported: ArrayLike | None = None,
    ):
        """Store rates sampled on a Quadrature (per nm) or integrated over bands.

        Pass the Quadrature for spectral integrands or the bands for
        band-integrated rates; the carrier is always named.

        Args:
            sampling: Quadrature for spectral integrands (per nm at its nodes) or the
                bands of band-integrated rates.
            carrier: ``"photon"`` or ``"electron"``.
            planet: Planet rate in the record's unit; scalar zero when absent.
            star_leakage: Residual starlight rate; scalar zero when absent.
            local_zodi: Local zodiacal light rate; scalar zero when absent.
            exozodi: Exozodiacal light rate; scalar zero when absent.
            thermal: Warm-optics thermal emission rate; scalar zero when absent.
            stellar_companions: Rate of other stars in the field; scalar zero when
                absent.
            n_pix: Sum of aperture weights in detector pixels, scalar or one value per
                band; None for per-pixel values.
            variance: Rates of the same kind summed with squared weights; its ``n_pix``
                is the sum of squared weights.
            supported: Boolean flag, False where the coronagraph data do not cover the
                read; None means all reads are supported.
            static_supported: Boolean flag, False where the products other than the
                speckle field do not cover the read; None means not distinguished.
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
        self.supported = None if supported is None else jnp.asarray(supported, bool)
        self.static_supported = (
            None if static_supported is None else jnp.asarray(static_supported, bool)
        )

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
        for name in ("supported", "static_supported"):
            flag = getattr(self, name)
            if flag is not None and flag.ndim and flag.shape[:1] != leading[:1]:
                raise ValueError(f"{name} must have one leading entry per band")
        self._check_flags()
        other = self.variance
        if other is not None:
            if other.bands != self.bands:
                raise ValueError("mean and variance rates must have the same bands")
            if other.carrier != self.carrier or other.spectral != self.spectral:
                raise ValueError("mean and variance rates must have the same unit")
            if other.variance is not None:
                raise ValueError("variance rates cannot carry their own variance")

    def _check_flags(self):
        """Require ``supported`` to imply ``static_supported`` on concrete flags."""
        strict, static = self.supported, self.static_supported
        if static is None or isinstance(static, jax.core.Tracer):
            return
        if strict is None:
            if not np.all(np.asarray(static)):
                raise ValueError(
                    "supported=None beside a static_supported with a False entry"
                )
            return
        if isinstance(strict, jax.core.Tracer):
            return
        if np.any(np.asarray(strict) & ~np.asarray(static)):
            raise ValueError("supported is True where static_supported is False")

    @staticmethod
    def component_names() -> tuple[str, ...]:
        """Names of additive source terms, in their declared order.

        Returns:
            The six source names.
        """
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
        """Sum the six sources, keeping the leading axes if all are scalar.

        Returns:
            Sum over the six sources, in the record's unit.
        """
        total = sum(getattr(self, name) for name in _COMPONENTS)
        return jnp.broadcast_to(total, total.shape or self._leading_shape)

    def integrate(self) -> "Rates":
        """Integrate spectral fields over each band, to shape ``(W, *spatial)``.

        Dimensional weights are in nm, so photon/s/nm becomes photon/s.
        Scalar absent terms broadcast to the shape shared by the sources.
        The carrier, ``n_pix`` and both support flags are kept; ``variance`` is
        integrated too.

        Returns:
            Band-integrated Rates of the same carrier.
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
            supported=self.supported,
            static_supported=self.static_supported,
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
        """Create six zero fields with full shape, leading axes then spatial.

        Args:
            sampling: Quadrature or bands, as for the constructor.
            carrier: ``"photon"`` or ``"electron"``.
            spatial_shape: Trailing spatial shape; empty for a single pixel or aperture.
            dtype: Dtype of the zero fields; the JAX default when None.

        Returns:
            Rates with six zero fields of shape ``(*leading, *spatial_shape)``.
        """
        empty = cls(sampling, carrier)
        zero = jnp.zeros((*empty._leading_shape, *spatial_shape), dtype=dtype)
        return cls(sampling, carrier, **dict.fromkeys(_COMPONENTS, zero))


RATE_FLOOR_RTOL = 1e-4
"""Largest negative rate, relative to the frame peak, that is drawn as zero."""


def floor_rate(rate: ArrayLike, scale: ArrayLike) -> tuple[Array, Array]:
    """Clip a rate at zero and flag a value too negative to be integration error.

    A band rate is a sum of node values, and a node value may be negative
    without harm: a reconstruction or a source reduction rings, and the total
    of a well-resolved band stays within its integration error of the truth. A
    negative total within ``RATE_FLOOR_RTOL`` of ``scale`` is that error and is
    returned as zero. A more negative total means the band rule under-resolves
    the scene; it is flagged, and also returned as zero.

    ``scale`` is the largest summed absolute component rate of the frame (for a
    scalar read, that read's own sum), so the tolerance is the per-pixel
    accuracy the stack claims, 1e-4 of the frame peak. The detector draws raise
    on a flagged rate and ``log_prob`` scores it as ``-inf``.

    Args:
        rate: Total rate, any shape.
        scale: Frame peak of the summed absolute component rates, broadcast
            against ``rate``.

    Returns:
        The rate clipped at zero, and a boolean that is True where the rate is
        below ``-RATE_FLOOR_RTOL * scale``.
    """
    rate = jnp.asarray(rate)
    bad = rate < -RATE_FLOOR_RTOL * jnp.asarray(scale)
    # A where, not a maximum: the gradient at a rate of exactly zero is one.
    return jnp.where(rate < 0, jnp.zeros_like(rate), rate), bad
