"""Named source contributions before and after spectral integration."""

from dataclasses import fields

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from optixstuff.spectral import Band, Quadrature


class _Rates(eqx.Module):
    planet: Array = eqx.field(kw_only=True, default=0.0, converter=jnp.asarray)
    star_leakage: Array = eqx.field(kw_only=True, default=0.0, converter=jnp.asarray)
    local_zodi: Array = eqx.field(kw_only=True, default=0.0, converter=jnp.asarray)
    exozodi: Array = eqx.field(kw_only=True, default=0.0, converter=jnp.asarray)
    thermal: Array = eqx.field(kw_only=True, default=0.0, converter=jnp.asarray)
    stellar_companions: Array = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )

    @staticmethod
    def component_names() -> tuple[str, ...]:
        """Names of additive source terms, in their declared order."""
        return tuple(field.name for field in fields(_Rates))

    def total(self) -> Array:
        """Sum the six source contributions with ordinary array broadcasting."""
        return sum(getattr(self, name) for name in self.component_names())


class _BandRates(_Rates):
    bands: tuple[Band, ...] = eqx.field(static=True, converter=tuple)

    def __check_init__(self):
        """Require one leading entry per band on each non-scalar component."""
        for name in self.component_names():
            value = getattr(self, name)
            if value.ndim and value.shape[0] != len(self.bands):
                raise ValueError(f"{name} must have one leading entry per band")

    def total(self) -> Array:
        """Preserve the band axis even when every contribution is scalar."""
        total = super().total()
        return jnp.broadcast_to(total, total.shape or (len(self.bands),))


class IntegratedPhotonRates(_BandRates):
    """Band-integrated photons/s, each field shaped ``(W, *spatial)``.

    ``spatial`` is empty for an aperture or ``(ny, nx)`` for an image.
    This record contains no detector yield or noise quantities.
    """


class PhotonRates(_Rates):
    """Spectral integrands in photons/s/nm at quadrature nodes.

    Each field has shape ``(W, Q, *spatial)``, where ``spatial`` is empty
    for an aperture or ``(ny, nx)`` for an image. An absent contribution
    may instead be scalar zero. No detector quantity belongs in this type.
    """

    quadrature: Quadrature

    def __check_init__(self):
        """Require one sample for every node of each non-scalar component."""
        for name in self.component_names():
            value = getattr(self, name)
            if value.ndim and value.shape[:2] != self.quadrature.nodes_nm.shape:
                raise ValueError(f"{name} must match the quadrature's (W, Q) shape")

    def integrate(self) -> IntegratedPhotonRates:
        """Integrate each field to photons/s of shape ``(W, *spatial)``.

        Dimensional weights are in nm. Scalar absent terms broadcast to
        the shape shared by the source fields; all-scalar zeros yield ``(W,)``.
        """
        quadrature = self.quadrature
        shape = self.total().shape or quadrature.nodes_nm.shape
        return IntegratedPhotonRates(
            quadrature.bands,
            **{
                name: quadrature.integrate(jnp.broadcast_to(getattr(self, name), shape))
                for name in self.component_names()
            },
        )

    @classmethod
    def zeros(
        cls, quadrature: Quadrature, spatial_shape: tuple[int, ...] = (), *, dtype=None
    ) -> "PhotonRates":
        """Create six zero integrands with full shape (W, Q, *spatial_shape)."""
        shape = (*quadrature.nodes_nm.shape, *spatial_shape)
        return cls(
            quadrature,
            **{name: jnp.zeros(shape, dtype=dtype) for name in cls.component_names()},
        )


class ElectronRates(_BandRates):
    """Detected electrons/s, each field shaped ``(W, *spatial)``.

    ``spatial`` is empty for an aperture or ``(ny, nx)`` for an image.
    An absent contribution may be scalar zero. Yield has already been applied.
    """
