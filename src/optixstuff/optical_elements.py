"""Optical element abstractions (throughput, filters, field stops)."""

from __future__ import annotations

import abc

import equinox as eqx
from jaxtyping import Array


class AbstractOpticalElement(eqx.Module):
    """Abstract interface for an optical element in the beam path.

    Elements reduce photon flux via wavelength-dependent throughput.
    The ETC calls get_throughput() for scalar efficiency calculations.
    The simulator calls apply() to attenuate 2D photon arrays.

    Both methods are abstract: use AbstractUniformElement for elements
    with spatially uniform throughput, which provides a default apply().
    """

    @abc.abstractmethod
    def get_throughput(self, wavelength_nm: float) -> float:
        """Fractional throughput at a given wavelength.

        Args:
            wavelength_nm: Wavelength in nanometres.

        Returns:
            Scalar throughput in [0, 1].
        """
        ...

    @abc.abstractmethod
    def apply(self, arr: Array, wavelength_nm: float) -> Array:
        """Apply this element to a 2D photon array.

        Args:
            arr: Input photon rate array [ph/s/pixel].
            wavelength_nm: Wavelength in nanometres.

        Returns:
            Attenuated photon rate array, same shape as arr.
        """
        ...


class AbstractUniformElement(AbstractOpticalElement):
    """Base for elements with spatially uniform throughput.

    Provides a default apply() that multiplies the array by the scalar
    throughput. Override apply() for elements with spatially varying
    transmission (e.g., field-dependent filter transmission maps).
    """

    def apply(self, arr: Array, wavelength_nm: float) -> Array:
        """Apply uniform throughput to a photon array."""
        return arr * self.get_throughput(wavelength_nm)


class ConstantThroughputElement(AbstractUniformElement):
    """An optical element with wavelength-independent throughput.

    Useful for modeling simple attenuators, beamsplitters, or as a
    placeholder during instrument design studies.

    Args:
        throughput: Constant fractional throughput in [0, 1].
        name: Optional descriptive label (e.g. "beamsplitter").
    """

    throughput: float
    name: str = "element"

    def get_throughput(self, wavelength_nm: float) -> float:
        """Return constant throughput, ignoring wavelength."""
        return self.throughput
