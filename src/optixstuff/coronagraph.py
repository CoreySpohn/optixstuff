"""The coronagraph slot and its image products on explicit shared grids."""

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


def on_grid(image: Image, grid: PixelGrid) -> Image:
    """Put a native product on a requested grid.

    Resample using the product's frame; reconcile different lod frames via sky.

    Measured cell integrals with piecewise-constant reconstruction collect by
    footprint. Optical point samples integrate their declared reconstruction,
    including on the native pitch. Transmission samples the reconstruction.

    Args:
        image: Native product Image.
        grid: Requested PixelGrid.

    Returns:
        The product on ``grid``.
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
    """Native grids for source PSFs, stellar maps and sky maps at one wavelength.

    Attributes:
        source_grid: Native grid of the source PSFs.
        stellar_grid: Native grid of the stellar maps.
        sky_grid: Native grid of the uniform sky map.
    """

    source_grid: PixelGrid
    stellar_grid: PixelGrid
    sky_grid: PixelGrid


class SourcePSFs(NamedTuple):
    """Point-source flux fractions and one admission decision per source.

    valid is False for a source outside the implementor's source domain (the
    positions its data tabulate or its optical model supports), and only then.
    The image of such a source is finite and exactly zero: the light beyond
    the data is not modeled, so it contributes nothing, and it is flagged for
    that source alone. The stamp of the native grid is not the source domain.

    Attributes:
        image: Flux fractions of the sources, with one leading axis over sources.
        valid: Boolean per source, False where the source is outside the source
            domain.
    """

    image: Image
    valid: jax.Array


class SceneRender(NamedTuple):
    """Scene flux on the output grid and support for all illuminated positions.

    image is always finite. valid is False only when lit flux lies outside
    the union of the operators' domains, and that flux contributes zero.

    Attributes:
        image: Scene flux on the output grid.
        valid: Boolean support flag, False only where lit flux lies outside the
            union of the operators' domains.
    """

    image: Image
    valid: jax.Array


class AbstractCoronagraph(eqx.Module):
    """The coronagraph slot: a declared frame and calibrated wavelength interval.

    The slot carries the image capability only (see AbstractImagingCoronagraph).
    Summary curves are a characterization record, PerformanceCurves, read with
    PerformanceCurves.evaluate; no simulation reads them from the slot.
    Separations use the native lambda/D frame. Wavelength selects a band;
    monochromatic tables have no further chromatic dependence.

    Attributes:
        frame: Definition of lambda/D for the coronagraph's native coordinates.
        bandpass: Calibrated wavelength interval in nm, or None for unknown
            coverage.
    """

    frame: AbstractVar[Frame]
    bandpass: AbstractVar[Band | None]

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the edges of the calibrated interval, when declared.

        Returns:
            The two edges of the bandpass in nm, or an empty array when no bandpass is
            declared.
        """
        band = self.bandpass
        return np.empty(0) if band is None else np.array([band.low_nm, band.high_nm])

    def coronagraph_at(self, wavelength_nm):
        """Select the performance model at a concrete wavelength.

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            The coronagraph model for that wavelength; here, the instance itself.
        """
        return self


class AbstractImagingCoronagraph(AbstractCoronagraph):
    """Point-source, stellar and uniform extended-illumination image products.

    Every product renders only what the coronagraph data hold. A source outside
    the source domain gives SourcePSFs.valid False with a finite all-zero image,
    never NaN and never an extrapolated or reflected one. Reads are
    gradient-safe, so a zero beyond the domain has an exact zero gradient.
    """

    @abc.abstractmethod
    def at_wavelength(self, wavelength_nm) -> ProductGrids:
        """Return the native product grids without rendering images.

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            The native grids for the source PSFs, stellar maps and sky map.
        """

    def on_axis_psf(self, wavelength_nm, grid: PixelGrid) -> Image:
        """Point-star cell-integrated stellar flux fraction on grid, shape (ny,nx).

        Args:
            wavelength_nm: Wavelength in nm.
            grid: Grid the Image is returned on.

        Returns:
            Stellar flux fraction per cell for a point star on axis.
        """
        return self.stellar_map(wavelength_nm, 0.0, grid=grid)

    def off_axis_psf(self, wavelength_nm, separation_lod, grid: PixelGrid) -> Image:
        """Cell-integrated stellar flux fractions on grid for a source along +x.

        A separation outside the source domain gives zeros, not NaN.

        Args:
            wavelength_nm: Wavelength in nm.
            separation_lod: Source separation along +x in lambda/D.
            grid: Grid the Image is returned on.

        Returns:
            Flux fraction per cell for the off-axis source.
        """
        result = self.source_psfs(
            wavelength_nm, jnp.atleast_1d(separation_lod), jnp.zeros(1), grid=grid
        )
        return replace(
            result.image,
            data=jnp.where(result.valid[0], result.image.data[0], 0.0),
            leading_axes=(),
        )

    @abc.abstractmethod
    def stellar_map(
        self, wavelength_nm, stellar_diameter_lod, *, grid: PixelGrid | None = None
    ) -> Image:
        """Finite-star flux fractions; table grid=None retains native samples.

        Args:
            wavelength_nm: Wavelength in nm.
            stellar_diameter_lod: Angular diameter of the star in lambda/D.
            grid: Grid the Image is returned on; None keeps the native samples
                (table implementations).

        Returns:
            Stellar flux fraction per cell.
        """

    @abc.abstractmethod
    def source_psfs(
        self, wavelength_nm, x_lod, y_lod, *, grid: PixelGrid | None = None
    ) -> SourcePSFs:
        """Source fractions; table grid=None retains native samples and metadata.

        Args:
            wavelength_nm: Wavelength in nm.
            x_lod: Source x positions in lambda/D.
            y_lod: Source y positions in lambda/D.
            grid: Grid the Image is returned on; None keeps the native samples
                (table implementations).

        Returns:
            Flux fraction per cell for each source and the per-source admission mask.
        """

    @abc.abstractmethod
    def uniform_sky_map(self, wavelength_nm, *, grid: PixelGrid) -> Image:
        """Dimensionless, intensive transmission sampled on the requested grid.

        Args:
            wavelength_nm: Wavelength in nm.
            grid: Grid the Image is sampled on.

        Returns:
            Dimensionless, intensive transmission for uniform illumination.
        """


class AbstractSceneCoronagraph(AbstractImagingCoronagraph):
    """Imaging model with a calibrated extended-scene redistribution operator."""

    @abc.abstractmethod
    def render_scene(
        self, scene: Image, wavelength_nm, *, grid: PixelGrid
    ) -> SceneRender:
        """Redistribute per-pixel flux; only illuminated sources require support.

        Args:
            scene: Per-pixel flux Image of the scene.
            wavelength_nm: Wavelength in nm.
            grid: Output grid.

        Returns:
            The redistributed flux on ``grid`` and its support flag.
        """


class AbstractTableCoronagraph(AbstractImagingCoronagraph):
    """Native Image tables resampled once through optixstuff.grids.resample.

    For stellar_map and source_psfs, grid=None returns the native samples and
    their product grid without integration. An explicit grid requests pixel
    integrals, even at native pitch. This distinction preserves the native
    reconstruction needed for aperture measurements and avoids integrating
    twice when a consumer subsequently samples onto detector pixels.

    Tables with a separate stellar grid declare it in at_wavelength.

    Attributes:
        sky_trans: Dimensionless, intensive transmission Image for uniform
            illumination, on its own native grid.
        grid: Native PixelGrid of the source PSFs and stellar maps.
    """

    sky_trans: AbstractVar[Image]
    grid: AbstractVar[PixelGrid]

    @property
    def frame(self):
        """Performance and source coordinates use the native off-axis frame."""
        return self.grid.frame

    def at_wavelength(self, wavelength_nm):
        """Tables with separate stellar sampling override the stellar grid.

        Args:
            wavelength_nm: Wavelength in nm; ignored by monochromatic tables.

        Returns:
            The native grids for the source PSFs, stellar maps and sky map.
        """
        return ProductGrids(self.grid, self.grid, self.sky_trans.grid)

    @property
    def psf_shape(self):
        """Native image shape (ny,nx)."""
        return self.grid.shape

    @abc.abstractmethod
    def _native_stellar_map(self, stellar_diameter_lod) -> Image:
        """Supply native stellar samples with their stellar product grid."""

    @abc.abstractmethod
    def _native_source_psfs(self, x_lod, y_lod) -> SourcePSFs:
        """Supply native PSFs with their source admission mask."""

    def stellar_map(self, wavelength_nm, stellar_diameter_lod, *, grid=None):
        """Per-pixel stellar fraction on grid, using the stellar product's frame.

        Args:
            wavelength_nm: Wavelength in nm; ignored by monochromatic tables.
            stellar_diameter_lod: Angular diameter of the star in lambda/D.
            grid: Grid the Image is returned on; None keeps the native samples.

        Returns:
            Stellar flux fraction per pixel.
        """
        image = self._native_stellar_map(stellar_diameter_lod)
        return image if grid is None else on_grid(image, grid)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Per-pixel source fractions on grid; x indexes columns, y indexes rows.

        Args:
            wavelength_nm: Wavelength in nm; ignored by monochromatic tables.
            x_lod: Source x positions in lambda/D.
            y_lod: Source y positions in lambda/D.
            grid: Grid the Image is returned on; None keeps the native samples.

        Returns:
            Flux fraction per pixel for each source and the per-source admission mask.
        """
        result = self._native_source_psfs(x_lod, y_lod)
        return (
            result
            if grid is None
            else SourcePSFs(on_grid(result.image, grid), result.valid)
        )

    def uniform_sky_map(self, wavelength_nm, *, grid):
        """Sample dimensionless transmission without multiplying by pixel area.

        Args:
            wavelength_nm: Wavelength in nm; ignored by monochromatic tables.
            grid: Grid the Image is sampled on.

        Returns:
            Dimensionless transmission on ``grid``.
        """
        return on_grid(self.sky_trans, grid)


class MultiBandCoronagraph(AbstractCoronagraph):
    """Dispatch by declared bandpass, in input order at shared wavelengths.

    frame uses the reference band; coronagraph_at(wavelength_nm) selects the model that
    covers a wavelength. Monochromatic tables retain their existing lack of
    additional wavelength dependence.

    Attributes:
        coronagraphs: Coronagraph models, one per band, in dispatch order.
        reference_wavelength_nm: Wavelength in nm whose model supplies ``frame``.
    """

    coronagraphs: tuple
    reference_wavelength_nm: float = eqx.field(static=True)

    def __init__(self, coronagraphs, *, reference_wavelength_nm=None):
        """Store one model per band, selecting the first as reference by default.

        Args:
            coronagraphs: Models that each declare a bandpass.
            reference_wavelength_nm: Reference wavelength in nm; the center of the
                first model's bandpass when None.
        """
        self.coronagraphs = tuple(coronagraphs)
        if not self.coronagraphs:
            raise ValueError("Need one or more coronagraph models")
        if any(model.bandpass is None for model in self.coronagraphs):
            raise ValueError("Each dispatched coronagraph must declare its bandpass")
        self.reference_wavelength_nm = float(
            self.coronagraphs[0].bandpass.center_nm
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )

    def coronagraph_at(self, wavelength_nm):
        """Return the first calibrated model covering the concrete wavelength.

        Args:
            wavelength_nm: Wavelength in nm; the reference wavelength when None.

        Returns:
            The first model whose bandpass covers the wavelength.
        """
        if wavelength_nm is None:
            wavelength_nm = self.reference_wavelength_nm
        wave = float(wavelength_nm)
        for model in self.coronagraphs:
            if model.bandpass.low_nm <= wave <= model.bandpass.high_nm:
                return model
        raise ValueError(f"No coronagraph bandpass covers {wave} nm")

    @property
    def bandpass(self):
        """Span all declared band edges."""
        return Band(
            min(model.bandpass.low_nm for model in self.coronagraphs),
            max(model.bandpass.high_nm for model in self.coronagraphs),
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Union the child model breaks and every dispatch interval edge.

        Returns:
            Sorted union of the models' breaks and the bandpass edges, in nm.
        """
        edges = [
            (model.bandpass.low_nm, model.bandpass.high_nm)
            for model in self.coronagraphs
        ]
        return np.union1d(edges, spectral_breaks(*self.coronagraphs))

    @property
    def frame(self):
        """Native frame for the reference performance record."""
        return self.coronagraph_at(None).frame
