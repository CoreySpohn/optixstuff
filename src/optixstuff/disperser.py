"""Disperser hardware descriptors for integral field spectrographs.

optixstuff owns the descriptor: the interface plus a cheap closed-form
scalar/ETC face. The heavy render logic (building the forward operator) lives
in coronachrome. This mirrors the coronagraph split, so jaxedith and yield
tools can read IFS hardware info without importing the render engine.
"""

import abc

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class AbstractDisperser(eqx.Module):
    """Interface for a dispersing IFS element (lenslet array, slicer, MSA).

    Only the scalar/ETC face is defined here. Render geometry lives in
    coronachrome and dispatches on the concrete descriptor type.
    """

    @abc.abstractmethod
    def spectral_resolution(self, wavelength_nm):
        """Resolving power R = lambda / dlambda at the given wavelength."""

    @abc.abstractmethod
    def spectral_sampling(self):
        """Detector pixels per resolution element."""

    @abc.abstractmethod
    def n_pix_spread(self, wavelength_min_nm, wavelength_max_nm):
        """Detector pixels a single spaxel spectrum spans across a band."""

    @abc.abstractmethod
    def throughput(self, wavelength_nm):
        """Disperser optical throughput in [0, 1] at the given wavelength."""


def _polyval_deriv(coeffs, x):
    """Evaluate the derivative of a descending-order polynomial at x."""
    n = coeffs.shape[0]
    if n <= 1:
        return jnp.zeros_like(jnp.asarray(x, dtype=float))
    powers = jnp.arange(n - 1, 0, -1)
    return jnp.polyval(coeffs[:-1] * powers, x)


class LensletDisperser(AbstractDisperser):
    """Lenslet-array IFS disperser (CRISPY heritage).

    Config only. The render geometry (IR build) is performed by coronachrome,
    which reads these fields. Scalar/ETC methods derive from
    ``dispersion_coeffs`` + ``pix_per_reselt`` so the dispersion model is the
    single source of truth.

    ``psflet_params[0]`` is the PSFlet core width in detector pixels (Gaussian
    ``sigma`` or Moffat ``alpha``); any trailing entries are dimensionless shape
    parameters (e.g. Moffat ``beta``). ``psflet_ref_nm`` is the wavelength at
    which that core width is specified: a diffraction-limited spot scales as
    ``lambda f / D``, so at fixed pixel scale coronachrome scales the core width
    by ``lambda / psflet_ref_nm`` per wavelength (the shape parameters do not
    scale).
    """

    pitch_m: float
    pixsize_m: float
    angle_rad: float
    lam_ref_nm: float
    pix_per_reselt: float
    dispersion_coeffs: Array = eqx.field(converter=jnp.asarray)
    psflet_params: Array = eqx.field(converter=jnp.asarray)
    psflet_ref_nm: float
    grid_kind: str = eqx.field(static=True)
    n_lenslets: int = eqx.field(static=True)
    psflet_kind: str = eqx.field(static=True)
    detector_shape: tuple[int, int] = eqx.field(static=True)
    throughput_value: float = 1.0

    def _dispersion_px(self, wavelength_nm):
        """Spectral-axis detector offset [px] for the wavelength(s)."""
        u = jnp.log(jnp.asarray(wavelength_nm, dtype=float) / self.lam_ref_nm)
        return jnp.polyval(self.dispersion_coeffs, u)

    def spectral_resolution(self, wavelength_nm):
        """R = (local px per unit log-lambda) / pixels-per-resolution-element."""
        u = jnp.log(jnp.asarray(wavelength_nm, dtype=float) / self.lam_ref_nm)
        local = jnp.abs(_polyval_deriv(self.dispersion_coeffs, u))
        return local / self.pix_per_reselt

    def spectral_sampling(self):
        """Detector pixels per resolution element."""
        return self.pix_per_reselt

    def n_pix_spread(self, wavelength_min_nm, wavelength_max_nm):
        """Spectral trace length [px] across a band, plus a PSFlet-width margin."""
        span = jnp.abs(
            self._dispersion_px(wavelength_max_nm)
            - self._dispersion_px(wavelength_min_nm)
        )
        return span + self.psflet_params[0]

    def throughput(self, wavelength_nm):
        """Constant throughput in v1."""
        return self.throughput_value * jnp.ones_like(
            jnp.asarray(wavelength_nm, dtype=float)
        )
