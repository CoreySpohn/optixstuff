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

    pixel_scale_arcsec: AbstractVar[float]
    """Detector plate scale in arcsec/pixel."""

    quantum_efficiency: AbstractVar[float]
    """Baseline quantum efficiency as a fraction in [0, 1]."""

    dark_current_rate_e_per_s: AbstractVar[float]
    """Dark current rate in electrons/pixel/second."""

    read_noise_e: AbstractVar[float]
    """Read noise in electrons RMS per pixel per read."""

    clock_induced_charge_rate_e_per_frame: AbstractVar[float]
    """Clock-induced charge in electrons/pixel/frame."""

    frame_time_s: AbstractVar[float]
    """Integration time per frame/read in seconds."""

    read_time_s: AbstractVar[float]
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
    def readout(
        self,
        image_rate: Array,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Apply stochastic noise realization to a photon rate image.

        Converts photon rates to detected electrons including QE,
        dark current, CIC, and read noise.

        Args:
            image_rate: Incident photon rate array in ph/s/pixel.
            exposure_time_s: ExposureConfig time in seconds.
            prng_key: JAX PRNG key (required, no default).

        Returns:
            Detected electrons array, same shape as image_rate.
        """
        ...

    def readout_source_electrons(
        self,
        image_rate: Array,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Poisson-sample incident photons and convert to electrons via QE.

        This is the source-dependent half of detector readout: photon
        arrival statistics and quantum-efficiency selection. Dark current,
        CIC, and read noise are handled separately by
        :meth:`readout_noise_electrons` so that multi-source exposures do not
        double-count the source-independent noise floor.

        Args:
            image_rate: Incident photon rate array in ph/s/pixel.
            exposure_time_s: ExposureConfig time in seconds.
            prng_key: JAX PRNG key.

        Returns:
            Photo-electron counts, same shape as image_rate.
        """
        key_phot, key_qe = jax.random.split(prng_key, 2)
        inc_photons = jax.random.poisson(key_phot, image_rate * exposure_time_s)
        return jax.random.binomial(key_qe, inc_photons, self.quantum_efficiency)

    def readout_source_electrons_thinned(
        self,
        image_rate: Array,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Fast equivalent of :meth:`readout_source_electrons` via Poisson thinning.

        Distributionally identical to the explicit Poisson-then-Binomial
        chain (Poisson thinning theorem: ``Binomial(Poisson(L), p) ~
        Poisson(L * p)``), but ~3x faster because it skips the Binomial
        draw and never materialises the intermediate photon count. Use in
        performance-critical paths (animation rendering, yield runs)
        when the photon count is not needed downstream.

        The marginal distribution of returned electrons is identical to
        :meth:`readout_source_electrons`. The two methods produce
        different specific realisations even with the same key.

        Args:
            image_rate: Incident photon rate array in ph/s/pixel.
            exposure_time_s: ExposureConfig time in seconds.
            prng_key: JAX PRNG key.

        Returns:
            Photo-electron counts, same shape as image_rate.
        """
        return jax.random.poisson(
            prng_key, image_rate * exposure_time_s * self.quantum_efficiency
        )

    def noise_variance(self, image_rate: Array, exposure_time_s: ArrayLike) -> Array:
        """Deterministic per-pixel total noise variance in electrons^2.

        The expected variance at each pixel for an incident photon-rate image
        -- the deterministic companion to :meth:`readout`. Combines source shot
        noise (Poisson on detected electrons) with the source-independent floor
        (dark current, clock-induced charge, read noise)::

            N = QE * rate * t
                + dark_rate * t
                + CIC_rate * n_frames
                + read_noise^2 * n_frames

        with ``n_frames = ceil(t / frame_time_s)``. Use ``1 / noise_variance(...)``
        as inverse-variance weights for least-squares spectral extraction or its
        GLS covariance, where the shot term carries the wavelength dependence.

        Args:
            image_rate: Incident photon rate [ph/s/pixel], any shape.
            exposure_time_s: Exposure time in seconds.

        Returns:
            Per-pixel noise variance [electrons^2], same shape as image_rate.
        """
        n_frames = jnp.ceil(exposure_time_s / self.frame_time_s)
        shot = self.quantum_efficiency * image_rate * exposure_time_s
        dark = self.dark_current_rate_e_per_s * exposure_time_s
        cic = self.clock_induced_charge_rate_e_per_frame * n_frames
        read = self.read_noise_e**2 * n_frames
        return shot + dark + cic + read

    @abc.abstractmethod
    def readout_noise_electrons(
        self,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Source-independent detector noise (dark + CIC + read).

        Each call draws a fresh noise realization of shape ``self.shape``;
        consumers add exactly one such draw per exposure regardless of
        how many sources were co-added via :meth:`readout_source_electrons`.

        Args:
            exposure_time_s: ExposureConfig time in seconds.
            prng_key: JAX PRNG key.

        Returns:
            Noise-electron array of shape ``self.shape``.
        """
        ...


# -- Pure noise simulation functions -----------------------------------------


def dark_current(
    dark_current_rate_e_per_s: float,
    exposure_time_s: ArrayLike,
    shape: tuple[int, int],
    prng_key: Array,
) -> Array:
    """Draw dark current electrons from a Poisson distribution.

    Args:
        dark_current_rate_e_per_s: Dark current rate in electrons/s/pixel.
        exposure_time_s: ExposureConfig time in seconds.
        shape: Detector shape (ny, nx).
        prng_key: PRNG key.

    Returns:
        Dark current electrons, shape (ny, nx).
    """
    return jax.random.poisson(
        prng_key, dark_current_rate_e_per_s * exposure_time_s, shape=shape
    )


def clock_induced_charge(
    clock_induced_charge_rate_e_per_frame: float,
    num_frames: ArrayLike,
    shape: tuple[int, int],
    prng_key: Array,
) -> Array:
    """Draw clock-induced charge electrons from a Poisson distribution.

    Args:
        clock_induced_charge_rate_e_per_frame: CIC rate in electrons/pixel/frame.
        num_frames: Number of frames (kept as float for JIT safety).
        shape: Detector shape (ny, nx).
        prng_key: PRNG key.

    Returns:
        CIC electrons, shape (ny, nx).
    """
    return jax.random.poisson(
        prng_key, clock_induced_charge_rate_e_per_frame * num_frames, shape=shape
    )


def read_noise(
    read_noise_e: float,
    num_frames: ArrayLike,
    shape: tuple[int, int],
    prng_key: Array,
) -> Array:
    """Draw read noise from a Gaussian distribution.

    Total read noise sigma = sqrt(num_frames) * read_noise_per_read.

    Args:
        read_noise_e: Read noise in electrons/pixel/read.
        num_frames: Number of frames.
        shape: Detector shape (ny, nx).
        prng_key: PRNG key.

    Returns:
        Read noise electrons, shape (ny, nx).
    """
    sigma = read_noise_e * jnp.sqrt(num_frames)
    return sigma * jax.random.normal(prng_key, shape=shape)


# -- Concrete implementations -----------------------------------------------


@final
class IdealDetector(AbstractDetector):
    """Detector with constant QE and minimal noise sources.

    Suitable for broadband imager studies where wavelength-dependent
    QE variation is not important and CIC/read noise are negligible.
    """

    pixel_scale_arcsec: float
    quantum_efficiency: float
    dark_current_rate_e_per_s: float
    read_noise_e: float
    clock_induced_charge_rate_e_per_frame: float
    frame_time_s: float
    read_time_s: float
    dqe: float
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        pixel_scale_arcsec: float,
        shape: tuple[int, int],
        quantum_efficiency: float = 1.0,
        dark_current_rate_e_per_s: float = 0.0,
        read_noise_e: float = 0.0,
        clock_induced_charge_rate_e_per_frame: float = 0.0,
        frame_time_s: float = 1.0,
        read_time_s: float = 0.05,
        dqe: float = 0.0,
    ) -> None:
        """Create a simple constant-QE detector."""
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.shape = shape
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate_e_per_s = dark_current_rate_e_per_s
        self.read_noise_e = read_noise_e
        self.clock_induced_charge_rate_e_per_frame = (
            clock_induced_charge_rate_e_per_frame
        )
        self.frame_time_s = frame_time_s
        self.read_time_s = read_time_s
        self.dqe = dqe

    def get_qe(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant QE, ignoring wavelength."""
        return self.quantum_efficiency

    def scalar_noise_rate(self, n_pix: ArrayLike, t_photon: ArrayLike) -> ArrayLike:
        """Combined dark + CIC noise variance rate.

        Read noise is not included here as it scales per-read, not per-second.
        Callers add (read_noise_e^2 * n_reads) / t_exp separately.
        """
        dark_variance_rate = self.dark_current_rate_e_per_s * n_pix
        cic_variance_rate = (
            self.clock_induced_charge_rate_e_per_frame * n_pix / t_photon
        )
        return dark_variance_rate + cic_variance_rate

    def readout_noise_electrons(
        self,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Dark current only -- IdealDetector has no CIC or read noise."""
        return dark_current(
            self.dark_current_rate_e_per_s, exposure_time_s, self.shape, prng_key
        )

    def readout(
        self,
        image_rate: Array,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Full detector readout: source electrons + dark current."""
        key_src, key_noise = jax.random.split(prng_key, 2)
        source = self.readout_source_electrons(image_rate, exposure_time_s, key_src)
        noise = self.readout_noise_electrons(exposure_time_s, key_noise)
        return source + noise

    def __repr__(self) -> str:
        """One-line summary of shape, plate scale, QE, and dark current."""
        ny, nx = self.shape
        return (
            f"IdealDetector({ny}x{nx} @ {self.pixel_scale_arcsec:.3g} arcsec/px, "
            f"QE={self.quantum_efficiency:.2f}, "
            f"dark={self.dark_current_rate_e_per_s:.2g} e-/s/px)"
        )


@final
class Detector(AbstractDetector):
    """Full detector model with dark current, CIC, and read noise.

    Suitable for detailed noise simulations where all detector noise
    sources matter.  Uses Poisson statistics for dark/CIC and Gaussian
    for read noise, matching the coronagraphoto convention.

    Warning: ``num_frames = jnp.ceil(exposure_time_s / frame_time_s)`` is
    kept as a float. Never cast it to int inside JIT -- that triggers a
    ConcretizationTypeError when exposure_time_s is traced.
    """

    pixel_scale_arcsec: float
    quantum_efficiency: float
    dark_current_rate_e_per_s: float
    read_noise_e: float
    clock_induced_charge_rate_e_per_frame: float
    frame_time_s: float
    read_time_s: float
    dqe: float
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        pixel_scale_arcsec: float,
        shape: tuple[int, int],
        quantum_efficiency: float = 1.0,
        dark_current_rate_e_per_s: float = 0.0,
        read_noise_e: float = 0.0,
        clock_induced_charge_rate_e_per_frame: float = 0.0,
        frame_time_s: float = 1.0,
        read_time_s: float = 0.05,
        dqe: float = 0.0,
    ) -> None:
        """Create a full detector with all noise sources."""
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.shape = shape
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate_e_per_s = dark_current_rate_e_per_s
        self.read_noise_e = read_noise_e
        self.clock_induced_charge_rate_e_per_frame = (
            clock_induced_charge_rate_e_per_frame
        )
        self.frame_time_s = frame_time_s
        self.read_time_s = read_time_s
        self.dqe = dqe

    def get_qe(self, wavelength_nm: ArrayLike) -> ArrayLike:
        """Return constant QE, ignoring wavelength."""
        return self.quantum_efficiency

    def scalar_noise_rate(self, n_pix: ArrayLike, t_photon: ArrayLike) -> ArrayLike:
        """Combined dark + CIC noise variance rate."""
        dark_variance_rate = self.dark_current_rate_e_per_s * n_pix
        cic_variance_rate = (
            self.clock_induced_charge_rate_e_per_frame * n_pix / t_photon
        )
        return dark_variance_rate + cic_variance_rate

    def readout_noise_electrons(
        self,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Dark current + CIC + read noise (source-independent)."""
        key_dark, key_cic, key_read = jax.random.split(prng_key, 3)
        dark_e = dark_current(
            self.dark_current_rate_e_per_s, exposure_time_s, self.shape, key_dark
        )
        # num_frames stays as a traced float -- never cast to int
        num_frames = jnp.ceil(exposure_time_s / self.frame_time_s)
        cic_e = clock_induced_charge(
            self.clock_induced_charge_rate_e_per_frame, num_frames, self.shape, key_cic
        )
        read_e = read_noise(self.read_noise_e, num_frames, self.shape, key_read)
        return dark_e + cic_e + read_e

    def readout(
        self,
        image_rate: Array,
        exposure_time_s: ArrayLike,
        prng_key: Array,
    ) -> Array:
        """Full detector readout: source electrons + all noise sources."""
        key_src, key_noise = jax.random.split(prng_key, 2)
        source = self.readout_source_electrons(image_rate, exposure_time_s, key_src)
        noise = self.readout_noise_electrons(exposure_time_s, key_noise)
        return source + noise

    def __repr__(self) -> str:
        """One-line summary of shape, plate scale, QE, and noise sources."""
        ny, nx = self.shape
        return (
            f"Detector({ny}x{nx} @ {self.pixel_scale_arcsec:.3g} arcsec/px, "
            f"QE={self.quantum_efficiency:.2f}, "
            f"dark={self.dark_current_rate_e_per_s:.2g} e-/s/px, "
            f"RN={self.read_noise_e:.2g} e-/read, "
            f"CIC={self.clock_induced_charge_rate_e_per_frame:.2g} e-/frame, "
            f"frame_time_s={self.frame_time_s:.3g} s)"
        )
