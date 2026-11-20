"""Coronagraph performance records and images on explicit shared grids."""

import abc
from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar

from optixstuff.grids import (
    Frame,
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
    resample,
)
from optixstuff.spectral import Band, spectral_breaks


def _on_grid(image: Image, grid: PixelGrid) -> Image:
    """Resample using the product's frame; reconcile different lod frames via sky.

    Measured cell integrals with piecewise-constant reconstruction collect by
    footprint. Optical point samples integrate their declared reconstruction,
    including on the native pitch. Transmission samples the reconstruction.
    """
    source = image.grid
    target = grid
    measured_cells = (
        image.measure == Measure.integral() and image.reconstruction == "nearest"
    )
    if measured_cells and source == target:
        return image
    if image.measure != Measure.intensive():
        image = image.integrated()
    if source.unit != target.unit or (
        source.unit == "lod" and source.frame != target.frame
    ):
        if source.unit == "lod":
            source = source.to_arcsec(source.frame.reference_wavelength_nm)
        if target.unit == "lod":
            target = target.to_arcsec(target.frame.reference_wavelength_nm)
    native = replace(image, grid=source)
    sampled = (
        integrate_footprints(native, footprint_overlap(source, target))
        if measured_cells
        else resample(native, target)
    )
    return replace(sampled, grid=grid)


class ProductGrids(NamedTuple):
    """Native grids for source PSFs, stellar maps and sky maps at one wavelength."""

    source_grid: PixelGrid
    stellar_grid: PixelGrid
    sky_grid: PixelGrid


class SourcePSFs(NamedTuple):
    """Point-source flux fractions and one admission decision per source."""

    image: Image
    valid: jax.Array


class SceneRender(NamedTuple):
    """Scene flux on the output grid and support for all illuminated positions."""

    image: Image
    valid: jax.Array


class AbstractCoronagraph(eqx.Module):
    """The coronagraph slot: a declared frame and calibrated wavelength interval.

    The slot carries the image capability only (see AbstractImagingCoronagraph).
    Summary curves are a characterization record, PerformanceCurves, read with
    PerformanceCurves.evaluate; no simulation reads them from the slot.
    Separations use the native lambda/D frame. Wavelength selects a band;
    monochromatic tables have no further chromatic dependence.
    """

    frame: AbstractVar[Frame]
    bandpass: AbstractVar[Band | None]
    """Calibrated wavelength interval in nm, or None for unknown coverage."""

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the edges of the calibrated interval, when declared."""
        band = self.bandpass
        return np.empty(0) if band is None else np.array([band.low_nm, band.high_nm])

    def band(self, wavelength_nm):
        """Select the performance model at a concrete wavelength."""
        return self


class AbstractImagingCoronagraph(AbstractCoronagraph):
    """Point-source, stellar and uniform extended-illumination image products."""

    @abc.abstractmethod
    def at_wavelength(self, wavelength_nm) -> ProductGrids:
        """Return the native product grids without rendering images."""

    def point_energy_fraction(self, x_lod, y_lod):
        """Sum the native PSF of each point source, returning NaN outside support."""
        x, y = jnp.broadcast_arrays(jnp.asarray(x_lod), jnp.asarray(y_lod))
        psfs = self.source_psfs(
            self.frame.reference_wavelength_nm, x.ravel(), y.ravel()
        )
        energy = jnp.sum(psfs.image.integrated().data, axis=(-2, -1))
        return jnp.where(psfs.valid, energy, jnp.nan).reshape(x.shape)

    def aperture_samples(self, wavelength_nm, separations_lod):
        """Native PSFs and positions used to calibrate the aperture record."""
        psfs = self.source_psfs(
            wavelength_nm, separations_lod, jnp.zeros_like(separations_lod)
        )
        return tuple(
            (
                replace(
                    psfs.image, data=jnp.where(valid, data, jnp.nan), leading_axes=()
                ),
                (sep, 0.0),
            )
            for data, valid, sep in zip(
                psfs.image.data, psfs.valid, separations_lod, strict=True
            )
        )

    def separation_map(self):
        """Native off-axis pixel radii in lambda/D, including the full grid basis."""
        return self.at_wavelength(
            self.frame.reference_wavelength_nm
        ).source_grid.radius()

    def on_axis_psf(self, wavelength_nm, grid: PixelGrid) -> Image:
        """Point-star cell-integrated stellar flux fraction on grid, shape (ny,nx)."""
        return self.stellar_map(wavelength_nm, 0.0, grid=grid)

    def off_axis_psf(self, wavelength_nm, separation_lod, grid: PixelGrid) -> Image:
        """Cell-integrated stellar flux fractions on grid for a source along +x."""
        result = self.source_psfs(
            wavelength_nm, jnp.atleast_1d(separation_lod), jnp.zeros(1), grid=grid
        )
        return replace(
            result.image,
            data=jnp.where(result.valid[0], result.image.data[0], jnp.nan),
            leading_axes=(),
        )

    def stellar_frame(self, wavelength_nm) -> Frame:
        """Native stellar-diameter frame from the selected product metadata."""
        return self.at_wavelength(wavelength_nm).stellar_grid.frame

    @abc.abstractmethod
    def stellar_map(
        self, wavelength_nm, stellar_diam_lod, *, grid: PixelGrid | None = None
    ) -> Image:
        """Finite-star flux fractions; table grid=None retains native samples."""

    @abc.abstractmethod
    def source_psfs(
        self, wavelength_nm, x_lod, y_lod, *, grid: PixelGrid | None = None
    ) -> SourcePSFs:
        """Source fractions; table grid=None retains native samples and metadata."""

    @abc.abstractmethod
    def uniform_sky_map(self, wavelength_nm, *, grid: PixelGrid) -> Image:
        """Dimensionless, intensive transmission sampled on the requested grid."""


class AbstractSceneCoronagraph(AbstractImagingCoronagraph):
    """Imaging model with a calibrated extended-scene redistribution operator."""

    @abc.abstractmethod
    def render_scene(
        self, scene: Image, wavelength_nm, *, grid: PixelGrid
    ) -> SceneRender:
        """Redistribute per-pixel flux; only illuminated sources require support."""


class AbstractTableCoronagraph(AbstractImagingCoronagraph):
    """Native Image tables resampled once through optixstuff.grids.resample.

    For stellar_map and source_psfs, grid=None returns the native samples and
    their product grid without integration. An explicit grid requests pixel
    integrals, even at native pitch. This distinction preserves the native
    reconstruction needed for aperture measurements and avoids integrating
    twice when a consumer subsequently samples onto detector pixels.

    Tables with a separate stellar grid expose its frame via stellar_frame.
    """

    sky_trans: AbstractVar[Image]
    grid: AbstractVar[PixelGrid]

    @property
    def frame(self):
        """Performance and source coordinates use the native off-axis frame."""
        return self.grid.frame

    def at_wavelength(self, wavelength_nm):
        """Tables with separate stellar sampling override the stellar grid."""
        return ProductGrids(self.grid, self.grid, self.sky_trans.grid)

    @property
    def psf_shape(self):
        """Native image shape (ny,nx)."""
        return self.grid.shape

    @abc.abstractmethod
    def _native_stellar_map(self, stellar_diam_lod) -> Image:
        """Supply native stellar samples with their stellar product grid."""

    @abc.abstractmethod
    def _native_source_psfs(self, x_lod, y_lod) -> SourcePSFs:
        """Supply native PSFs with their source admission mask."""

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid=None):
        """Per-pixel stellar fraction on grid, using the stellar product's frame."""
        image = self._native_stellar_map(stellar_diam_lod)
        return image if grid is None else _on_grid(image, grid)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Per-pixel source fractions on grid; x indexes columns, y indexes rows."""
        result = self._native_source_psfs(x_lod, y_lod)
        return (
            result
            if grid is None
            else SourcePSFs(_on_grid(result.image, grid), result.valid)
        )

    def uniform_sky_map(self, wavelength_nm, *, grid):
        """Sample dimensionless transmission without multiplying by pixel area."""
        return _on_grid(self.sky_trans, grid)


class MultiBandCoronagraph(AbstractCoronagraph):
    """Dispatch by declared bandpass, in input order at shared wavelengths.

    frame uses the reference band; band(wavelength_nm) selects the model that
    covers a wavelength. Monochromatic tables retain their existing lack of
    additional wavelength dependence.
    """

    bands: tuple
    reference_wavelength_nm: float = eqx.field(static=True)

    def __init__(self, bands, *, reference_wavelength_nm=None):
        """Store one model per band, selecting the first as reference by default."""
        self.bands = tuple(bands)
        if not self.bands:
            raise ValueError("Need one or more coronagraph models")
        if any(model.bandpass is None for model in self.bands):
            raise ValueError("Each dispatched coronagraph must declare its bandpass")
        self.reference_wavelength_nm = float(
            self.bands[0].bandpass.center_nm
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )

    def band(self, wavelength_nm):
        """Return the first calibrated model covering the concrete wavelength."""
        if wavelength_nm is None:
            wavelength_nm = self.reference_wavelength_nm
        wave = float(wavelength_nm)
        for model in self.bands:
            if model.bandpass.low_nm <= wave <= model.bandpass.high_nm:
                return model
        raise ValueError(f"No coronagraph bandpass covers {wave} nm")

    @property
    def bandpass(self):
        """Span all declared band edges."""
        return Band(
            min(band.bandpass.low_nm for band in self.bands),
            max(band.bandpass.high_nm for band in self.bands),
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Union the child model breaks and every dispatch interval edge."""
        edges = [
            (model.bandpass.low_nm, model.bandpass.high_nm) for model in self.bands
        ]
        return np.union1d(edges, spectral_breaks(*self.bands))

    @property
    def frame(self):
        """Native frame for the reference performance record."""
        return self.band(None).frame
