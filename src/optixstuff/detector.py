"""Detector abstractions and concrete implementations."""


import abc
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from jax.typing import ArrayLike
from jaxtyping import Array


class AbstractDetector(eqx.Module):
    """Abstract interface for a focal-plane detector.

    Provides both scalar noise rates (for ETC use) and stochastic noise
    realization (for image simulation).  All concrete implementations
    must define the hardware parameters listed as ``AbstractVar`` fields.
    """

    pixel_scale: AbstractVar[float]
    """Detector plate scale in arcsec/pixel."""

    quantum_efficiency: AbstractVar[float]
    """Baseline quantum efficiency as a fraction in [0, 1]."""

    dark_current_rate: AbstractVar[float]
    """Dark current rate in electrons/pixel/second."""

    read_noise_electrons: AbstractVar[float]
    """Read noise in electrons RMS per pixel per read."""

    cic_rate: AbstractVar[float]
    """Clock-induced charge in electrons/pixel/frame."""

    frame_time: AbstractVar[float]
    """Integration time per frame/read in seconds."""

    read_time: AbstractVar[float]
    """Time per read cycle in seconds (for RN^2/t_read in ETC)."""

    dqe: AbstractVar[float]
    """QE degradation factor (multiplicative correction over mission life)."""

    shape: AbstractVar[tuple[int, int]]
    """Detector dimensions (ny, nx) in pixels."""

    @abc.abstractmethod
    def get_qe(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Quantum efficiency at a given wavelength.

        Args:
            wavelength_nm: Wavelength in nanometres.

        Returns:
            QE as a fraction in [0, 1].
        """
        ...

    @abc.abstractmethod
    def scalar_noise_rate(self, n_pix: ArrayLike, t_photon: ArrayLike) -> ArrayLike:
        """Total scalar noise variance rate for the ETC.

        Returns the combined noise variance per unit time (electrons^2/s)
        for a photometric aperture of n_pix pixels.

        Args:
            n_pix: Number of pixels in the photometric aperture.
            t_photon: Photon counting integration time in seconds.

        Returns:
            Noise variance rate in electrons^2/second.
        """
        ...

    @abc.abstractmethod
    def add_noise(
        self,
        image_rate: Array,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Apply stochastic noise realization to a photon rate image.

        Converts photon rates to detected electrons including QE,
        dark current, CIC, and read noise.

        Args:
            image_rate: Incident photon rate array in ph/s/pixel.
            exposure_time: Exposure time in seconds.
            prng_key: JAX PRNG key (required, no default).

        Returns:
            Detected electrons array, same shape as image_rate.
        """
        ...

    def add_source_electrons(
        self,
        image_rate: Array,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Poisson-sample incident photons and convert to electrons via QE.

        This is the source-dependent half of detector readout: photon
        arrival statistics and quantum-efficiency selection. Dark current,
        CIC, and read noise are handled separately by
        :meth:`add_noise_electrons` so that multi-source exposures do not
        double-count the source-independent noise floor.

        Args:
            image_rate: Incident photon rate array in ph/s/pixel.
            exposure_time: Exposure time in seconds.
            prng_key: JAX PRNG key.

        Returns:
            Photo-electron counts, same shape as image_rate.
        """
        key_phot, key_qe = jax.random.split(prng_key, 2)
        inc_photons = jax.random.poisson(key_phot, image_rate * exposure_time)
        return jax.random.binomial(key_qe, inc_photons, self.quantum_efficiency)

    @abc.abstractmethod
    def add_noise_electrons(
        self,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Source-independent detector noise (dark + CIC + read).

        Each call draws a fresh noise realization of shape ``self.shape``;
        consumers add exactly one such draw per exposure regardless of
        how many sources were co-added via :meth:`add_source_electrons`.

        Args:
            exposure_time: Exposure time in seconds.
            prng_key: JAX PRNG key.

        Returns:
            Noise-electron array of shape ``self.shape``.
        """
        ...


# -- Pure noise simulation functions -----------------------------------------


def simulate_dark_current(
    dark_current_rate: float,
    exposure_time: ArrayLike,
    shape: tuple[int, int],
    prng_key: Array,
) -> Array:
    """Draw dark current electrons from a Poisson distribution.

    Args:
        dark_current_rate: Dark current rate in electrons/s/pixel.
        exposure_time: Exposure time in seconds.
        shape: Detector shape (ny, nx).
        prng_key: PRNG key.

    Returns:
        Dark current electrons, shape (ny, nx).
    """
    return jax.random.poisson(prng_key, dark_current_rate * exposure_time, shape=shape)


def simulate_cic(
    cic_rate: float,
    num_frames: ArrayLike,
    shape: tuple[int, int],
    prng_key: Array,
) -> Array:
    """Draw clock-induced charge electrons from a Poisson distribution.

    Args:
        cic_rate: CIC rate in electrons/pixel/frame.
        num_frames: Number of frames (kept as float for JIT safety).
        shape: Detector shape (ny, nx).
        prng_key: PRNG key.

    Returns:
        CIC electrons, shape (ny, nx).
    """
    return jax.random.poisson(prng_key, cic_rate * num_frames, shape=shape)


def simulate_read_noise(
    read_noise: float,
    num_frames: ArrayLike,
    shape: tuple[int, int],
    prng_key: Array,
) -> Array:
    """Draw read noise from a Gaussian distribution.

    Total read noise sigma = sqrt(num_frames) * read_noise_per_read.

    Args:
        read_noise: Read noise in electrons/pixel/read.
        num_frames: Number of frames.
        shape: Detector shape (ny, nx).
        prng_key: PRNG key.

    Returns:
        Read noise electrons, shape (ny, nx).
    """
    sigma = read_noise * jnp.sqrt(num_frames)
    return sigma * jax.random.normal(prng_key, shape=shape)


# -- Concrete implementations -----------------------------------------------


@final
class SimpleDetector(AbstractDetector):
    """Detector with constant QE and minimal noise sources.

    Suitable for broadband imager studies where wavelength-dependent
    QE variation is not important and CIC/read noise are negligible.
    """

    pixel_scale: float
    quantum_efficiency: float
    dark_current_rate: float
    read_noise_electrons: float
    cic_rate: float
    frame_time: float
    read_time: float
    dqe: float
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        pixel_scale: float,
        shape: tuple[int, int],
        quantum_efficiency: float = 1.0,
        dark_current_rate: float = 0.0,
        read_noise_electrons: float = 0.0,
        cic_rate: float = 0.0,
        frame_time: float = 1.0,
        read_time: float = 0.05,
        dqe: float = 0.0,
    ) -> None:
        """Create a simple constant-QE detector."""
        self.pixel_scale = pixel_scale
        self.shape = shape
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate = dark_current_rate
        self.read_noise_electrons = read_noise_electrons
        self.cic_rate = cic_rate
        self.frame_time = frame_time
        self.read_time = read_time
        self.dqe = dqe

    def get_qe(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant QE, ignoring wavelength."""
        return self.quantum_efficiency

    def scalar_noise_rate(self, n_pix: ArrayLike, t_photon: ArrayLike) -> ArrayLike:
        """Combined dark + CIC noise variance rate.

        Read noise is not included here as it scales per-read, not per-second.
        Callers add (read_noise^2 * n_reads) / t_exp separately.
        """
        dark_variance_rate = self.dark_current_rate * n_pix
        cic_variance_rate = self.cic_rate * n_pix / t_photon
        return dark_variance_rate + cic_variance_rate

    def add_noise_electrons(
        self,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Dark current only -- SimpleDetector has no CIC or read noise."""
        return simulate_dark_current(
            self.dark_current_rate, exposure_time, self.shape, prng_key
        )

    def add_noise(
        self,
        image_rate: Array,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Full detector readout: source electrons + dark current."""
        key_src, key_noise = jax.random.split(prng_key, 2)
        source = self.add_source_electrons(image_rate, exposure_time, key_src)
        noise = self.add_noise_electrons(exposure_time, key_noise)
        return source + noise


@final
class Detector(AbstractDetector):
    """Full detector model with dark current, CIC, and read noise.

    Suitable for detailed noise simulations where all detector noise
    sources matter.  Uses Poisson statistics for dark/CIC and Gaussian
    for read noise, matching the coronagraphoto convention.

    Warning: ``num_frames = jnp.ceil(exposure_time / frame_time)`` is
    kept as a float. Never cast it to int inside JIT -- that triggers a
    ConcretizationTypeError when exposure_time is traced.
    """

    pixel_scale: float
    quantum_efficiency: float
    dark_current_rate: float
    read_noise_electrons: float
    cic_rate: float
    frame_time: float
    read_time: float
    dqe: float
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        pixel_scale: float,
        shape: tuple[int, int],
        quantum_efficiency: float = 1.0,
        dark_current_rate: float = 0.0,
        read_noise_electrons: float = 0.0,
        cic_rate: float = 0.0,
        frame_time: float = 1.0,
        read_time: float = 0.05,
        dqe: float = 0.0,
    ) -> None:
        """Create a full detector with all noise sources."""
        self.pixel_scale = pixel_scale
        self.shape = shape
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate = dark_current_rate
        self.read_noise_electrons = read_noise_electrons
        self.cic_rate = cic_rate
        self.frame_time = frame_time
        self.read_time = read_time
        self.dqe = dqe

    def get_qe(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant QE, ignoring wavelength."""
        return self.quantum_efficiency

    def scalar_noise_rate(self, n_pix: ArrayLike, t_photon: ArrayLike) -> ArrayLike:
        """Combined dark + CIC noise variance rate."""
        dark_variance_rate = self.dark_current_rate * n_pix
        cic_variance_rate = self.cic_rate * n_pix / t_photon
        return dark_variance_rate + cic_variance_rate

    def add_noise_electrons(
        self,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Dark current + CIC + read noise (source-independent)."""
        key_dark, key_cic, key_read = jax.random.split(prng_key, 3)
        dark = simulate_dark_current(
            self.dark_current_rate, exposure_time, self.shape, key_dark
        )
        # num_frames stays as a traced float -- never cast to int
        num_frames = jnp.ceil(exposure_time / self.frame_time)
        cic = simulate_cic(self.cic_rate, num_frames, self.shape, key_cic)
        read = simulate_read_noise(
            self.read_noise_electrons, num_frames, self.shape, key_read
        )
        return dark + cic + read

    def add_noise(
        self,
        image_rate: Array,
        exposure_time: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Full detector readout: source electrons + all noise sources."""
        key_src, key_noise = jax.random.split(prng_key, 2)
        source = self.add_source_electrons(image_rate, exposure_time, key_src)
        noise = self.add_noise_electrons(exposure_time, key_noise)
        return source + noise
