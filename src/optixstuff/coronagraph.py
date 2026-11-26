"""The coronagraph slot and its image products on explicit shared grids."""

import abc
import math
from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar
from hwoutils.numerics import COARSEST

from optixstuff.grids import (
    Frame,
    Image,
    Measure,
    PixelGrid,
    _footprint_integral,
    _reads_footprint,
    footprint_overlap,
    integrate_footprints,
    resample,
)
from optixstuff.spectral import Band, same_wavelength, spectral_breaks

# Under-blur grows with the difference of squared fractional widths. For a 10%
# product, a 12.5% band has about 0.19 of the under-blur of a 20% band.
WIDE_BAND_FACTOR = 1.25

# Fractional widths use a relative slack of four float64 eps to allow
# rounding in the host width calculation at its limit.
BAND_COMPARE_RTOL = 4 * float(np.finfo(np.float64).eps)


def _within(value, limit):
    """Whether ``value <= limit`` up to ``BAND_COMPARE_RTOL`` of the limit."""
    return value <= limit + BAND_COMPARE_RTOL * abs(limit)


def _wavelength_le(a, b):
    """Compare both dispatch coordinates on the common float32 wavelength grid."""
    a, b = COARSEST(a), COARSEST(b)
    return a <= b or same_wavelength(a, b)


def on_grid(image: Image, grid: PixelGrid) -> Image:
    """Put a native product on a requested grid.

    Resample using the product's frame; reconcile different lod frames via sky.

    Measured cell integrals with piecewise-constant reconstruction collect by
    footprint. A point density is read as the integral of its declared
    reconstruction over each pixel's footprint, at every pitch including the
    native one, and returns a cell integral in the area the density is per
    (grid units squared of the native grid, or steradians). For nearest,
    linear, cubic and cubic_spline this read is axis-aligned only and raises
    NotImplementedError for a rotated or sheared target; ``resample`` still
    samples a density at the centers of a rotated grid. The nearest, linear
    and cubic reads integrate the zero-extended kernel sum and are not clipped
    to the table: a grid that covers the table collects the table sum, and
    pixels within the kernel's reach beyond the table receive light (a linear
    read at the native pitch puts up to 1/8 of an edge sample in the next
    pixel out); ``coverage`` supports none of those pixels. The cubic_spline
    and band-limited reads are restricted to the table footprint. A
    band-limited density becomes cell integrals and takes the box integral of
    ``resample``, which is the same footprint integral. An intensive map, such
    as a transmission, is not integrated: ``resample`` samples its declared
    reconstruction at the pixel centers.

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
    footprint = _reads_footprint(image.measure, image.reconstruction)
    if image.measure != Measure.intensive() and not footprint:
        image = image.integrated()
    if source.unit != target.unit or (
        source.unit == "lod" and source.frame != target.frame
    ):
        if source.unit == "lod":
            source = source.to_arcsec(source.frame.reference_wavelength_nm)
        if target.unit == "lod":
            target = target.to_arcsec(target.frame.reference_wavelength_nm)
    if footprint:
        return replace(_footprint_integral(image, source, target), grid=grid)
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
    Separations use the native lambda/D frame. A flat band average already
    includes the chromatic blur across its product band (Stark and Krist 2019,
    "Standardized Coronagraph Parameters for Input into Yield Calculations").

    Attributes:
        frame: Definition of lambda/D for the coronagraph's native coordinates.
        bandpass: Calibrated wavelength interval in nm, or None for unknown
            coverage.
        band_averaged: True for a band-averaged product scaled once per observing
            band at its center, False for a single-wavelength response scaled
            at each node.
    """

    frame: AbstractVar[Frame]
    bandpass: AbstractVar[Band | None]
    band_averaged: AbstractVar[bool]

    def scale_wavelength_nm(self, wavelength_nm, band_center_nm):
        """Select the observing band center for a band average or the node otherwise.

        Args:
            wavelength_nm: Node wavelength in nm.
            band_center_nm: Center of the exposure's observing band in nm.

        Returns:
            The wavelength setting the angular scale of the native lambda/D grid.
        """
        return band_center_nm if self.band_averaged else wavelength_nm

    def check_bands(self, bands, *, allow_wide_bands=False) -> None:
        """Check that the product supplies enough chromatic blur in each band.

        Args:
            bands: Concrete observing bands.
            allow_wide_bands: Accept a wider observing band with the product's blur.

        Raises:
            ValueError: An observing band's fractional width exceeds the allowed
                multiple of the band-averaged product's fractional width.
        """
        product = self.bandpass
        if not self.band_averaged or product is None or allow_wide_bands:
            return
        product_width = (
            0.0
            if same_wavelength(product.low_nm, product.high_nm)
            else product.width_nm / product.center_nm
        )
        for band in bands:
            width = band.width_nm / band.center_nm
            if _within(width, WIDE_BAND_FACTOR * product_width):
                continue
            message = (
                f"Observing band {band.low_nm:.12g}-{band.high_nm:.12g} nm has "
                f"fractional width {width:.12g}. The band-averaged product's "
                f"fractional width is {product_width:.12g}. "
            )
            if product_width == 0:
                raise ValueError(message + "No positive-width tile fits this product.")

            # Equal log spans have equal fractional widths. Check the actual tile
            # edges too, since a rounded edge can put a tile just over the limit.
            n = max(
                1,
                math.floor(
                    math.log1p(band.width_nm / band.low_nm)
                    / math.log1p(product.width_nm / product.low_nm)
                ),
            )
            tiles = band.channels(n, "log")
            while not all(
                _within(t.width_nm / t.center_nm, product_width) for t in tiles
            ):
                n += 1
                tiles = band.channels(n, "log")
            intervals = ", ".join(f"{t.low_nm:.12g}-{t.high_nm:.12g} nm" for t in tiles)
            raise ValueError(message + f"Split into these tiles: {intervals}.")

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

    @property
    @abc.abstractmethod
    def scene_grid(self) -> PixelGrid:
        """The grid of source cells ``render_scene`` reads, in lambda/D.

        A scene on another grid is resampled onto this one, so a caller that
        builds cell integrals here passes every source the operator models. It
        may reach beyond the stored PSF grid (``ProductGrids.source_grid``) when
        the source domain does.
        """

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

        The transmission is read at the pixel centers, not averaged over each
        pixel. Against the pixel mean, the center sample differs by at most
        about 0.5 percent in an aperture of radius 0.7 lambda/D on the public
        coronagraph packages tested, and by much more in single pixels on a
        steep occulter edge or a hard field stop.

        Args:
            wavelength_nm: Wavelength in nm; ignored by monochromatic tables.
            grid: Grid the Image is sampled on.

        Returns:
            Dimensionless transmission on ``grid``.
        """
        return on_grid(self.sky_trans, grid)


class MultiBandCoronagraph(AbstractCoronagraph):
    """Dispatch by declared bandpass, in input order at shared wavelengths.

    Band-averaged products are dispatched once per observing band to a child
    covering both edges. Single-wavelength products are dispatched per node.
    The reference band's child supplies the frame.

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
        if len({model.band_averaged for model in self.coronagraphs}) > 1:
            children = []
            for model in self.coronagraphs:
                band = model.bandpass
                interval = (
                    "None" if band is None else f"{band.low_nm:g}-{band.high_nm:g} nm"
                )
                children.append(
                    f"{type(model).__name__} (bandpass={interval}, "
                    f"band_averaged={model.band_averaged})"
                )
            raise ValueError(
                "Cannot mix band-averaged and single-wavelength "
                f"coronagraph children: {', '.join(children)}"
            )
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
            if _wavelength_le(model.bandpass.low_nm, wave) and _wavelength_le(
                wave, model.bandpass.high_nm
            ):
                return model
        raise ValueError(f"No coronagraph bandpass covers {wave} nm")

    @property
    def band_averaged(self):
        """Whether all children hold band-averaged products."""
        return self.coronagraphs[0].band_averaged

    def coronagraph_for_band(self, band):
        """Select the first band average covering both edges, or the center model.

        Both edges use the float32 dispatch grid before computing the midpoint
        for a single-wavelength model.

        Args:
            band: Concrete observing band.

        Returns:
            The covering child for a band average or the band-center child for
            single-wavelength products.

        Raises:
            ValueError: No child covers the required interval.
        """
        if not self.band_averaged:
            edges = np.asarray((band.low_nm, band.high_nm), dtype=COARSEST)
            return self.coronagraph_at(edges.mean(dtype=np.float64))
        for model in self.coronagraphs:
            interval = model.bandpass
            if _wavelength_le(interval.low_nm, band.low_nm) and _wavelength_le(
                band.high_nm, interval.high_nm
            ):
                return model
        message = (
            f"No coronagraph bandpass covers the observing band "
            f"{band.low_nm:.12g}-{band.high_nm:.12g} nm"
        )
        boundaries = sorted(
            {
                edge
                for model in self.coronagraphs
                for edge in (model.bandpass.low_nm, model.bandpass.high_nm)
                if band.low_nm < edge < band.high_nm
            }
        )
        if boundaries:
            edges = ", ".join(f"{edge:.12g} nm" for edge in boundaries)
            message += f", split the observing band at {edges}"
        raise ValueError(message)

    def check_bands(self, bands, *, allow_wide_bands=False) -> None:
        """Require each observing band to fit one child's band average.

        Args:
            bands: Concrete observing bands.
            allow_wide_bands: Forward the width opt-in to the covering child.
                The child must still cover both edges.
        """
        if self.band_averaged:
            for band in bands:
                self.coronagraph_for_band(band).check_bands(
                    (band,), allow_wide_bands=allow_wide_bands
                )

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
