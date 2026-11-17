"""Optical element abstractions (throughput, filters, field stops)."""

import abc
from typing import final

import equinox as eqx
from hwoutils.spectral import Transmission
from jax.typing import ArrayLike
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
    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Fractional throughput at a given wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Scalar throughput in [0, 1].
        """
        ...

    @abc.abstractmethod
    def apply(self, arr: Array, wavelength_nm: ArrayLike) -> Array:
        """Apply this element to a 2D photon array.

        Args:
            arr: Input photon rate array [ph/s/pixel].
            wavelength_nm: Wavelength in nanometers.

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

    def apply(self, arr: Array, wavelength_nm: ArrayLike) -> Array:
        """Apply uniform throughput to a photon array."""
        return arr * self.get_throughput(wavelength_nm)


@final
class ConstantThroughput(AbstractUniformElement):
    """An optical element with wavelength-independent throughput.

    Useful for modeling simple attenuators, beamsplitters, or as a
    placeholder during instrument design studies.
    """

    throughput: float
    name: str = eqx.field(default="element", static=True)

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant throughput, ignoring wavelength."""
        return self.throughput

    def __repr__(self) -> str:
        """One-line summary of throughput value."""
        return (
            f"ConstantThroughput(name={self.name!r}, throughput={self.throughput:.3g})"
        )


@final
class TransmissiveElement(AbstractUniformElement):
    """Spatially uniform optical element holding a spectral Transmission.

    The response table, interpolation and support belong to Transmission.
    This element supplies the optical-path apply/get_throughput interface;
    an OpticalPath multiplies the responses of its individual elements.
    Use ConstantThroughput for a wavelength-independent scalar efficiency.
    """

    transmission: Transmission

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Evaluate the held response at wavelengths in nanometers."""
        return self.transmission(wavelength_nm)
