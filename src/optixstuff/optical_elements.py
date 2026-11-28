"""Optical element abstractions (throughput, filters, field stops)."""

import abc
import re
from collections import Counter
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.spectral import Transmission

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


def check_name(name, what):
    """Require a name that can serve as a bracket key in a model path.

    Args:
        name: The candidate name.
        what: What is being named, for the error message.

    Raises:
        ValueError: If ``name`` is not a string of letters, digits, underscores
            and hyphens that starts with a letter or underscore.
    """
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(
            f"{what} name {name!r} must start with a letter or underscore and "
            "hold only letters, digits, underscores and hyphens"
        )


def check_element_names(elements):
    """Check the explicit names of one train of elements.

    An unnamed element (``name`` is None) is addressed by position and takes no
    part in the check. Explicit names must be valid keys and unique in the train.

    Args:
        elements: The tuple of optical elements of one train.

    Raises:
        ValueError: If an explicit name is invalid or repeated.
    """
    names = [element.name for element in elements if element.name is not None]
    for name in names:
        check_name(name, "attenuating element")
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"attenuating_elements repeat the element names {repeated}")


class AbstractOpticalElement(eqx.Module):
    """Abstract interface for an optical element in the beam path.

    Elements reduce photon flux via wavelength-dependent throughput.
    The ETC calls get_throughput() for scalar efficiency calculations.
    The simulator calls apply() to attenuate 2D photon arrays.

    Both methods are abstract: use AbstractUniformElement for elements
    with spatially uniform throughput, which provides a default apply().

    Attributes:
        name: Optional key that addresses the element in a train (a model
            path such as ``attenuating_elements[mirror]``); None, the default,
            leaves the element addressed by position.
    """

    name: str | None = eqx.field(default=None, static=True, kw_only=True)

    @abc.abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the host wavelength breaks [nm] for this model.

        Returns:
            Break wavelengths in nm as a 1D array; empty when the model declares none.
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
        """Apply uniform throughput to a photon array.

        Args:
            arr: Input photon rate array [ph/s/pixel].
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Attenuated photon rate array, same shape as ``arr``.
        """
        return arr * self.get_throughput(wavelength_nm)


@final
class ConstantThroughput(AbstractUniformElement):
    """An optical element with wavelength-independent throughput.

    Useful for modeling simple attenuators, beamsplitters, or as a
    placeholder during instrument design studies.

    Attributes:
        throughput: Fractional throughput in [0, 1], independent of wavelength.
    """

    throughput: float

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant throughput, ignoring wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers; ignored.

        Returns:
            The constant throughput.
        """
        return self.throughput

    def spectral_breaks_nm(self) -> np.ndarray:
        """The wavelength dependence is smooth throughout the band.

        Returns:
            An empty array.
        """
        return np.empty(0)

    def __repr__(self) -> str:
        """One-line summary of throughput value.

        Returns:
            One-line summary string.
        """
        return (
            f"ConstantThroughput(name={self.name!r}, throughput={self.throughput:.3g})"
        )


@final
class TransmissiveElement(AbstractUniformElement):
    """Spatially uniform optical element holding a spectral Transmission.

    The response table, interpolation and support belong to Transmission.
    This element supplies the optical-path apply/get_throughput interface;
    an Instrument multiplies the responses of its individual elements.
    Use ConstantThroughput for a wavelength-independent scalar efficiency.

    Attributes:
        transmission: Spectral Transmission giving the throughput.
    """

    transmission: Transmission

    def spectral_breaks_nm(self) -> np.ndarray:
        """Delegate to the held transmission table.

        Returns:
            Break wavelengths in nm of the held Transmission.
        """
        return self.transmission.spectral_breaks_nm()

    def get_throughput(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Evaluate the held response at wavelengths in nanometers.

        Args:
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Throughput from the held Transmission, shaped like ``wavelength_nm``.
        """
        return self.transmission(wavelength_nm)
