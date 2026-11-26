"""Disperser hardware descriptors for integral field spectrographs.

optixstuff owns the descriptor: the interface plus a cheap closed-form
scalar/ETC face. The heavy render logic (building the forward operator) lives
in coronachrome. This mirrors the coronagraph split, so analytic and yield
tools can read IFS hardware info without importing the render engine.
"""

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from optixstuff.optical_elements import AbstractOpticalElement, ConstantThroughput


class AbstractDisperser(eqx.Module):
    """Interface for a dispersing IFS element (lenslet array, slicer, MSA).

    Only the scalar/ETC face is defined here. Render geometry lives in
    coronachrome and dispatches on the concrete descriptor type.
    """

    @abc.abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the host wavelength breaks [nm] for this model.

        Returns:
            Break wavelengths in nm as a 1D array; empty when the model declares none.
        """

    @abc.abstractmethod
    def spectral_resolution(self, wavelength_nm):
        """Resolving power R = lambda / dlambda at the given wavelength.

        Args:
            wavelength_nm: Wavelength in nm.
        """

    @abc.abstractmethod
    def spectral_sampling(self):
        """Detector pixels per resolution element."""

    @abc.abstractmethod
    def n_pix_spread(self, low_nm, high_nm):
        """Detector pixels a single spaxel spectrum spans across a band.

        Args:
            low_nm: Short edge of the band in nm.
            high_nm: Long edge of the band in nm.
        """

    @abc.abstractmethod
    def throughput(self, wavelength_nm):
        """Disperser optical throughput in [0, 1] at the given wavelength.

        Args:
            wavelength_nm: Wavelength in nm.
        """


def _polyval_deriv(coeffs, x):
    """Evaluate the derivative of a descending-order polynomial at x."""
    n = coeffs.shape[0]
    if n <= 1:
        return jnp.zeros_like(x)
    powers = jnp.arange(n - 1, 0, -1, dtype=jnp.int32)
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

    ``sky_pitch_arcsec`` is the lenslet pitch projected on sky, a plain angle
    in arcseconds (the focal-plane cube a render layer consumes lives on a
    fixed angular grid, so no reference wavelength is involved). It makes the
    spatial sampling an instrument property: the render layer derives
    focal-plane pixels per lenslet as the ratio of this pitch to the cube's
    angular plate scale instead of taking a free knob. ``None`` means
    unspecified.

    ``psflet_pack_path`` is a reference to a frozen PSFlet template pack file
    (used with ``psflet_kind="template"``), the way ``throughput_element``
    carries the blaze curve: config only, the render layer loads it.

    Attributes:
        pitch_m: Lenslet pitch in meters.
        pixsize_m: Detector pixel size in meters.
        angle_rad: Rotation of the lenslet layout relative to the detector, in
            radians.
        lam_ref_nm: Reference wavelength in nm of the dispersion polynomial.
        pix_per_reselt: Detector pixels per spectral resolution element.
        dispersion_coeffs: Descending polynomial coefficients giving the detector
            offset in pixels as a function of ``ln(lambda / lam_ref_nm)``.
        psflet_params: PSFlet core width in detector pixels, then any dimensionless
            shape parameters.
        psflet_ref_nm: Wavelength in nm at which the core width is specified.
        grid_kind: Lenslet lattice, ``"square"`` or ``"hex"``.
        n_lenslets: Number of lenslets along each side of the array.
        psflet_kind: Kind of PSFlet model, for example ``"template"``.
        detector_shape: Detector pixel shape (ny, nx).
        throughput_element: Optical element giving the disperser throughput.
        sky_pitch_arcsec: Lenslet pitch projected on the sky in arcsec, or None.
        psflet_pack_path: Path to a frozen PSFlet template pack, or None.
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
    throughput_element: AbstractOpticalElement = ConstantThroughput(1.0)
    sky_pitch_arcsec: float | None = None
    psflet_pack_path: str | None = eqx.field(static=True, default=None)

    def spectral_breaks_nm(self) -> np.ndarray:
        """The smooth dispersion law adds no breaks to the throughput.

        Returns:
            Break wavelengths in nm of the throughput element.
        """
        return self.throughput_element.spectral_breaks_nm()

    def dispersion_px(self, wavelength_nm):
        """Spectral-axis detector offset [px] for the wavelength(s).

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            Spectral-axis detector offset in pixels.
        """
        wavelength_nm = jnp.asarray(wavelength_nm)
        u = jnp.log(wavelength_nm / self.lam_ref_nm)
        return jnp.polyval(self.dispersion_coeffs, u)

    def spectral_resolution(self, wavelength_nm):
        """R = (local px per unit log-lambda) / pixels-per-resolution-element.

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            Resolving power R, dimensionless.
        """
        if isinstance(wavelength_nm, (tuple, list)):
            wavelength_nm = jnp.asarray(wavelength_nm)
        dtype = jnp.result_type(
            wavelength_nm,
            self.lam_ref_nm,
            self.dispersion_coeffs,
            self.pix_per_reselt,
            1.0,
        )
        wavelength_nm = jnp.asarray(wavelength_nm, dtype=dtype)
        u = jnp.log(wavelength_nm / self.lam_ref_nm)
        local = jnp.abs(_polyval_deriv(self.dispersion_coeffs, u))
        return local / self.pix_per_reselt

    def spectral_sampling(self):
        """Detector pixels per resolution element.

        Returns:
            Detector pixels per spectral resolution element.
        """
        return self.pix_per_reselt

    def n_pix_spread(self, low_nm, high_nm):
        """Spectral trace length [px] across a band, plus a PSFlet-width margin.

        Args:
            low_nm: Short edge of the band in nm.
            high_nm: Long edge of the band in nm.

        Returns:
            Spectral trace length in detector pixels plus the PSFlet core width.
        """
        span = jnp.abs(self.dispersion_px(high_nm) - self.dispersion_px(low_nm))
        return span + self.psflet_params[0]

    def throughput(self, wavelength_nm):
        """Disperser optical throughput in [0, 1], shaped like wavelength_nm.

        Delegates to the composed throughput element (ConstantThroughput by
        default; TransmissiveElement holding a Transmission for a tabulated
        blaze curve) and broadcasts to the wavelength shape so the output
        shape is canonical regardless of the backing element.

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            Throughput in [0, 1], shaped like ``wavelength_nm``.
        """
        w = jnp.asarray(wavelength_nm)
        return self.throughput_element.get_throughput(w) * jnp.ones_like(
            w, dtype=jnp.result_type(w, 1.0)
        )
