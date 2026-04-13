"""Optical element abstractions (throughput, filters, field stops)."""


import abc
from typing import final

import equinox as eqx
import interpax
import jax.numpy as jnp
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
            wavelength_nm: Wavelength in nanometres.

        Returns:
            Scalar throughput in [0, 1].
        """
        ...

    @abc.abstractmethod
    def apply(self, arr: Array, wavelength_nm: ArrayLike) -> Array:
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

    def apply(self, arr: Array, wavelength_nm: ArrayLike) -> Array:
        """Apply uniform throughput to a photon array."""
        return arr * self.get_throughput(wavelength_nm)


@final
class ConstantThroughputElement(AbstractUniformElement):
    """An optical element with wavelength-independent throughput.

    Useful for modeling simple attenuators, beamsplitters, or as a
    placeholder during instrument design studies.
    """

    throughput: float
    name: str = eqx.field(default="element", static=True)

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant throughput, ignoring wavelength."""
        return self.throughput


@final
class LinearThroughputElement(AbstractUniformElement):
    """An optical element with linearly interpolated wavelength-dependent throughput.

    Throughput is specified at a set of wavelengths and linearly
    interpolated between them. Extrapolation returns zero outside the
    defined wavelength range.
    """

    wavelengths_nm: Array
    throughputs: Array
    interp: interpax.Interpolator1D

    def __init__(self, wavelengths_nm: Array, throughputs: Array) -> None:
        """Create a throughput element from sampled wavelength/throughput pairs."""
        self.wavelengths_nm = wavelengths_nm
        self.throughputs = throughputs
        self.interp = interpax.Interpolator1D(
            wavelengths_nm, throughputs, method="linear", extrap=jnp.array([0.0, 0.0])
        )

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Interpolate throughput at the requested wavelength."""
        return self.interp(wavelength_nm)


@final
class OpticalFilter(AbstractUniformElement):
    """A bandpass filter with linearly interpolated transmittance.

    Structurally identical to LinearThroughputElement but semantically
    distinct -- represents a spectral bandpass selection rather than a
    reflective coating or attenuator.
    """

    wavelengths_nm: Array
    transmittances: Array
    interp: interpax.Interpolator1D

    def __init__(self, wavelengths_nm: Array, transmittances: Array) -> None:
        """Create an optical filter from sampled wavelength/transmittance pairs."""
        self.wavelengths_nm = wavelengths_nm
        self.transmittances = transmittances
        self.interp = interpax.Interpolator1D(
            wavelengths_nm,
            transmittances,
            method="linear",
            extrap=jnp.array([0.0, 0.0]),
        )

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Interpolate filter transmittance at the requested wavelength."""
        return self.interp(wavelength_nm)
