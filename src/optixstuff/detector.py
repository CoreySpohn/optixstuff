"""Detector abstractions."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class AbstractDetector(eqx.Module):
    """Abstract interface for a focal-plane detector.

    Provides both scalar noise rates (for ETC use) and wavelength-dependent
    quantum efficiency (for simulation and IFS use).
    """

    @abc.abstractmethod
    def get_qe(self, wavelength_nm: float) -> float:
        """Quantum efficiency at a given wavelength.

        Args:
            wavelength_nm: Wavelength in nanometres.

        Returns:
            QE as a fraction in [0, 1].
        """
        ...

    @property
    @abc.abstractmethod
    def dark_current_rate(self) -> float:
        """Dark current rate in electrons/pixel/second."""
        ...

    @property
    @abc.abstractmethod
    def read_noise_electrons(self) -> float:
        """Read noise in electrons RMS per pixel per read."""
        ...

    @abc.abstractmethod
    def scalar_noise_rate(self, n_pix: float, t_photon: float) -> float:
        """Total scalar noise variance rate for the ETC.

        Returns the combined noise variance per unit time (electrons^2/s)
        for a photometric aperture of n_pix pixels.

        Args:
            n_pix: Number of pixels in the photometric aperture.
            t_photon: Photon counting integration time in seconds.
                Required for clock-induced charge (CIC) calculation.

        Returns:
            Noise variance rate in electrons^2/second.
        """
        ...


class SimpleDetector(AbstractDetector):
    """A simple detector with constant QE and standard noise sources.

    Suitable for broadband imager studies where wavelength-dependent
    QE variation is not important.

    Args:
        qe: Quantum efficiency (constant across wavelengths).
        dark_current_electrons_per_s: Dark current in e-/pix/s.
        read_noise_electrons: Read noise in e- RMS per read.
        cic_electrons: Clock-induced charge in e-/pix/frame.
            Default 0.0 (not applicable for non-EMCCDs).
    """

    _qe: float
    _dark_current_rate: float
    _read_noise_electrons: float
    cic_electrons: float

    def __init__(
        self,
        qe: float,
        dark_current_electrons_per_s: float,
        read_noise_electrons: float,
        cic_electrons: float = 0.0,
    ) -> None:
        self._qe = qe
        self._dark_current_rate = dark_current_electrons_per_s
        self._read_noise_electrons = read_noise_electrons
        self.cic_electrons = cic_electrons

    def get_qe(self, wavelength_nm: float) -> float:
        """Return constant QE, ignoring wavelength."""
        return self._qe

    @property
    def dark_current_rate(self) -> float:
        """Dark current rate in electrons/pixel/second."""
        return self._dark_current_rate

    @property
    def read_noise_electrons(self) -> float:
        """Read noise in electrons RMS per pixel per read."""
        return self._read_noise_electrons

    def scalar_noise_rate(self, n_pix: float, t_photon: float) -> float:
        """Total noise variance rate for a photometric aperture.

        Combines dark current and CIC contributions. Read noise is
        not included here as it is per-read rather than per-second;
        callers should add (read_noise^2 * n_reads) / t_exp separately.

        Args:
            n_pix: Number of pixels in the aperture.
            t_photon: Photon counting time in seconds (for CIC).

        Returns:
            Noise variance rate in electrons^2/second.
        """
        dark_variance_rate = self._dark_current_rate * n_pix
        cic_variance_rate = self.cic_electrons * n_pix / t_photon
        return dark_variance_rate + cic_variance_rate
