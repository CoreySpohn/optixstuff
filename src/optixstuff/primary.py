"""Primary mirror abstractions."""

from __future__ import annotations

import abc

import equinox as eqx
from equinox import AbstractVar
from jaxtyping import Float


class AbstractPrimary(eqx.Module):
    """Abstract interface for a primary aperture.

    Any concrete implementation must provide the diameter and collecting
    area of the primary mirror as scalar values in SI units. These are
    consumed by exposure time calculators and simulation tools alike.
    """

    @property
    @abc.abstractmethod
    def diameter_m(self) -> float:
        """Primary mirror diameter in metres."""
        ...

    @property
    @abc.abstractmethod
    def area_m2(self) -> float:
        """Effective collecting area in square metres."""
        ...


class SimplePrimary(AbstractPrimary):
    """A simple circular primary mirror with a central obscuration.

    Args:
        diameter_m: Primary mirror diameter in metres.
        obscuration: Linear obscuration fraction (0 = no obscuration).
        shape_factor: Fraction of unobscured area that is collecting
            (accounts for struts, segment gaps, etc.). Default 1.0.
    """

    _diameter_m: float
    obscuration: float
    shape_factor: float

    def __init__(
        self,
        diameter_m: float,
        obscuration: float = 0.0,
        shape_factor: float = 1.0,
    ) -> None:
        self._diameter_m = diameter_m
        self.obscuration = obscuration
        self.shape_factor = shape_factor

    @property
    def diameter_m(self) -> float:
        """Primary mirror diameter in metres."""
        return self._diameter_m

    @property
    def area_m2(self) -> float:
        """Effective collecting area in square metres."""
        import jax.numpy as jnp

        r = self._diameter_m / 2.0
        gross_area = jnp.pi * r**2
        return gross_area * (1.0 - self.obscuration**2) * self.shape_factor
