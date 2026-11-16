"""Named source contributions before and after spectral integration."""

from dataclasses import fields

import equinox as eqx
import jax.numpy as jnp
from hwoutils.spectral import Quadrature
from jaxtyping import Array


class _Rates(eqx.Module):
    planet: Array = eqx.field(default=0.0, converter=jnp.asarray)
    star_leakage: Array = eqx.field(default=0.0, converter=jnp.asarray)
    local_zodi: Array = eqx.field(default=0.0, converter=jnp.asarray)
    exozodi: Array = eqx.field(default=0.0, converter=jnp.asarray)
    thermal: Array = eqx.field(default=0.0, converter=jnp.asarray)
    stellar_companions: Array = eqx.field(default=0.0, converter=jnp.asarray)

    def total(self) -> Array:
        """Sum the six source contributions with ordinary array broadcasting."""
        return sum(getattr(self, field.name) for field in fields(self))


class IntegratedPhotonRates(_Rates):
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

    def integrate(self, quadrature: Quadrature) -> IntegratedPhotonRates:
        """Integrate each field to photons/s of shape ``(W, *spatial)``.

        Dimensional weights are in nm. Scalar absent terms broadcast to
        the shape shared by the source fields; all-scalar zeros yield ``(W,)``.
        """
        shape = self.total().shape or quadrature.nodes_nm.shape
        return IntegratedPhotonRates(
            **{
                field.name: quadrature.integrate(
                    jnp.broadcast_to(getattr(self, field.name), shape)
                )
                for field in fields(self)
            }
        )

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, dtype=None) -> "PhotonRates":
        """Create six zero integrands with full shape ``(W, Q, *spatial)``."""
        return cls(
            **{field.name: jnp.zeros(shape, dtype=dtype) for field in fields(cls)}
        )


class ElectronRates(_Rates):
    """Detected electrons/s, each field shaped ``(W, *spatial)``.

    ``spatial`` is empty for an aperture or ``(ny, nx)`` for an image.
    An absent contribution may be scalar zero. Yield has already been applied.
    """
