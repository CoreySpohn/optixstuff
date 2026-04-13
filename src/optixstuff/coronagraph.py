"""Coronagraph abstractions."""

import abc

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from jax.typing import ArrayLike
from jaxtyping import Array


class AbstractCoronagraph(eqx.Module):
    """Abstract interface for coronagraph performance models.

    Provides both scalar performance curves (for ETC use) and 2D PSF
    generation (for image simulation). Implementations can be backed by
    pre-computed interpolation tables (yippy), physical wavefront
    propagation, or analytical models.

    All wavelength arguments are in nanometres throughout.
    All separations are in lambda/D units.
    """

    pixel_scale_lod: AbstractVar[float]
    """Native pixel scale in lambda/D per pixel."""

    IWA: AbstractVar[float]
    """Inner working angle in lambda/D."""

    OWA: AbstractVar[float]
    """Outer working angle in lambda/D."""

    # ------------------------------------------------------------------
    # Scalar interface -- consumed by jaxEDITH and yield estimators
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def throughput(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Core (off-axis planet) throughput.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Fractional throughput in [0, 1].
        """
        ...

    @abc.abstractmethod
    def core_area(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Photometric aperture area in (lambda/D)^2.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Core area in (lambda/D)^2.
        """
        ...

    @abc.abstractmethod
    def core_mean_intensity(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Mean stellar intensity within the photometric aperture.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Mean stellar leakage intensity in (lambda/D)^-2.
        """
        ...

    @abc.abstractmethod
    def occulter_transmission(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Off-axis (sky/zodi) transmission through the occulter.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Fractional sky transmission in [0, 1].
        """
        ...

    # ------------------------------------------------------------------
    # Image interface -- consumed by coronagraphoto
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def on_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """On-axis (stellar leakage) PSF.

        Returns the coronagraphic PSF for an on-axis point source,
        normalized to unit stellar flux before the coronagraph.

        Args:
            wavelength_nm: Wavelength in nanometres.
            pixel_scale_rad: Output pixel scale in radians/pixel.
            npixels: Output array side length in pixels. Must be a
                Python int (not a JAX array) as it determines the
                output shape at compile time.

        Returns:
            2D float array of shape (npixels, npixels).
        """
        ...

    @abc.abstractmethod
    def off_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        separation_lod: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Off-axis PSF at a given angular separation.

        Args:
            wavelength_nm: Wavelength in nanometres.
            separation_lod: Source separation in lambda/D.
            pixel_scale_rad: Output pixel scale in radians/pixel.
            npixels: Output array side length in pixels. Must be a
                Python int (not a JAX array) as it determines the
                output shape at compile time.

        Returns:
            2D float array of shape (npixels, npixels).
        """
        ...


class AbstractScalarOnlyCoronagraph(AbstractCoronagraph):
    """Base for ETC-only coronagraph models that lack 2D PSF generation.

    Stubs out the image interface with zero arrays so the class satisfies
    AbstractCoronagraph without requiring a full optical model.
    """

    def on_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Return a zero PSF (not implemented for scalar-only models)."""
        return jnp.zeros((npixels, npixels))

    def off_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        separation_lod: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Return a zero PSF (not implemented for scalar-only models)."""
        return jnp.zeros((npixels, npixels))
