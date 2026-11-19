"""Coronagraph performance records and images on explicit shared grids."""

import abc
from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar

from optixstuff.apertures import DEFAULT_APERTURE_RULE, RecordRule
from optixstuff.grids import (
    Frame,
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
    resample,
)
from optixstuff.performance import PerformanceCurves, SampledProfile
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
    """Performance in a declared wavelength/diameter frame.

    Separations use the performance frame. Resolved diameters use the native
    stellar frame of a model that supplies finite-star calibration.
    Leakage queries accept stellar_diam_lod=0.0 for the stored point-source
    record. Finite diameters use prepared image weights. Models without them
    raise ValueError for concrete nonzero diameters and return NaN for traced
    nonzero diameters.
    Wavelength selects a band; monochromatic tables have no further chromatic
    dependence. time_d is always days from the scene epoch.
    """

    frame: AbstractVar[Frame]
    bandpass: AbstractVar[Band | None]
    """Calibrated wavelength interval in nm, or None for unknown coverage."""
    outer_working_angle_lod: AbstractVar[float | None]

    @abc.abstractmethod
    def curves(
        self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"
    ) -> PerformanceCurves:
        """Return the prepared performance record for an explicit aperture rule."""

    def prepared(
        self, rule: RecordRule, *, leakage_rule="aperture_mean"
    ) -> tuple["AbstractCoronagraph", PerformanceCurves]:
        """Return a model and its usable record, preparing on the host if supported."""
        return self, self.curves(rule, leakage_rule=leakage_rule)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the edges of the calibrated interval, when declared."""
        band = self.bandpass
        return np.empty(0) if band is None else np.array([band.low_nm, band.high_nm])

    def band(self, wavelength_nm):
        """Select the performance model at a concrete wavelength."""
        return self

    def throughput(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        stellar_diam_lod=0.0,
        rule=DEFAULT_APERTURE_RULE,
        leakage_rule="aperture_mean",
        variance=False,
    ):
        """Planet flux fraction using weights, or squared weights for variance."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return (
            self.band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate(
                "throughput_variance" if variance else "throughput", separation_lod
            )
        )

    def core_area(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        stellar_diam_lod=0.0,
        rule=DEFAULT_APERTURE_RULE,
        leakage_rule="aperture_mean",
        variance=False,
    ):
        """Collecting area in (lambda/D)^2, using squared weights for variance."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return (
            self.band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate(
                "core_area_variance_lod2" if variance else "core_area_lod2",
                separation_lod,
            )
        )

    def leakage_fraction(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        stellar_diam_lod=0.0,
        rule=DEFAULT_APERTURE_RULE,
        leakage_rule="aperture_mean",
        variance=False,
    ):
        """Stellar flux fraction using weights, or squared weights for variance."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.band(wavelength_nm)._leakage_query(
            "leakage_fraction_variance" if variance else "leakage_fraction",
            separation_lod,
            wavelength_nm,
            stellar_diam_lod,
            rule,
            leakage_rule,
        )

    def leakage_density_lod2(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        stellar_diam_lod=0.0,
        rule=DEFAULT_APERTURE_RULE,
        leakage_rule="aperture_mean",
        variance=False,
    ):
        """Stellar fraction per collecting or variance area in (lambda/D)^2."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.band(wavelength_nm)._leakage_query(
            "leakage_density_variance_lod2" if variance else "leakage_density_lod2",
            separation_lod,
            wavelength_nm,
            stellar_diam_lod,
            rule,
            leakage_rule,
        )

    def raw_contrast(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        stellar_diam_lod=0.0,
        rule=DEFAULT_APERTURE_RULE,
        leakage_rule="aperture_mean",
    ):
        """Rule-defined stellar intensity divided by planet intensity."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.band(wavelength_nm)._leakage_query(
            "raw_contrast",
            separation_lod,
            wavelength_nm,
            stellar_diam_lod,
            rule,
            leakage_rule,
        )

    def _leakage_query(
        self, name, separation, wavelength, diameter, rule, leakage_rule
    ):
        """Point-source fallback; models without prepared images reject finite stars.

        A concrete nonzero diameter raises ValueError. A traced nonzero value
        returns NaN, so jit/vmap cannot silently substitute a point source.
        """
        if not isinstance(diameter, jax.core.Tracer) and float(diameter) != 0:
            raise ValueError("finite stellar diameter requires a prepared stellar map")
        value = self.curves(rule, leakage_rule=leakage_rule).evaluate(name, separation)
        if not isinstance(diameter, jax.core.Tracer):
            return value
        return jnp.where(jnp.asarray(diameter) == 0, value, jnp.nan)

    @abc.abstractmethod
    def uniform_sky_response(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Annular mean of the uniform-sky map in the performance frame."""


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

    def leakage_density_map(
        self,
        rules=None,
        *,
        stellar_diam_lod=0.0,
        leakage_rule="aperture_mean",
        variance=False,
    ):
        """Project density per (lambda/D)^2 on the native separation map.

        Explicit prepared rules are stacked on the last axis. Diameter uses
        the stellar product's native lambda/D, as in stellar_map.
        """
        radius = self.separation_map()
        if rules is None:
            return self.leakage_density_lod2(
                radius,
                stellar_diam_lod=stellar_diam_lod,
                leakage_rule=leakage_rule,
                variance=variance,
            )
        return jnp.stack(
            [
                self.leakage_density_lod2(
                    radius,
                    rule=r,
                    variance=variance,
                    stellar_diam_lod=stellar_diam_lod,
                    leakage_rule=leakage_rule,
                )
                for r in rules
            ],
            axis=-1,
        )

    def separation_map(self):
        """Native off-axis pixel radii in lambda/D, including the full grid basis."""
        return self.at_wavelength(
            self.frame.reference_wavelength_nm
        ).source_grid.radius()

    def throughput_map(self, rules=None, *, variance=False):
        """Project throughput, stacking explicitly prepared rules on the last axis."""
        radius = self.separation_map()
        if rules is None:
            return self.throughput(radius, variance=variance)
        return jnp.stack(
            [self.throughput(radius, rule=r, variance=variance) for r in rules], axis=-1
        )

    def core_area_map(self, rules=None, *, variance=False):
        """Project aperture area, stacking prepared rules on the last axis."""
        radius = self.separation_map()
        if rules is None:
            return self.core_area(radius, variance=variance)
        return jnp.stack(
            [self.core_area(radius, rule=r, variance=variance) for r in rules], axis=-1
        )

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

    Finite-star leakage uses _prepared_leakage_weights, built at construction
    with aperture operators for each supported record. Tables
    without prepared stellar-image weights retain the point-source contract.
    Tables with a separate stellar grid expose its frame via stellar_frame.
    """

    uniform_sky_profile: AbstractVar[SampledProfile]
    sky_trans: AbstractVar[Image]
    grid: AbstractVar[PixelGrid]

    @property
    def frame(self):
        """Performance and source coordinates use the native off-axis frame."""
        return self.grid.frame

    def at_wavelength(self, wavelength_nm):
        """Tables with separate stellar sampling override the stellar grid."""
        return ProductGrids(self.grid, self.grid, self.sky_trans.grid)

    def uniform_sky_response(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Read dimensionless transmission from this band's radial sky profile."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        profile = self.uniform_sky_profile
        frame = self.frame
        if frame != profile.frame:
            # A single angular separation spans different native lambda/D counts.
            ratio = frame.lod_to_arcsec(
                frame.reference_wavelength_nm
            ) / profile.frame.lod_to_arcsec(profile.frame.reference_wavelength_nm)
            separation_lod = jnp.asarray(separation_lod) * ratio
        return profile.evaluate(separation_lod)

    def _prepared_leakage_weights(self, rule, leakage_rule):
        """Return native-image weights, or None for a table without stellar maps."""
        return None

    def _leakage_query(
        self, name, separation, wavelength, diameter, rule, leakage_rule
    ):
        record = self.curves(rule, leakage_rule=leakage_rule)
        point = record.evaluate(name, separation)
        if not isinstance(diameter, jax.core.Tracer) and float(diameter) == 0:
            return point
        operators = self._prepared_leakage_weights(rule, leakage_rule)
        if operators is None:
            return super()._leakage_query(
                name, separation, wavelength, diameter, rule, leakage_rule
            )

        def finite(d):
            image = self.stellar_map(wavelength, d, grid=None).integrated().data
            aperture, aperture_variance, ring = operators
            is_variance = name in (
                "leakage_fraction_variance",
                "leakage_density_variance_lod2",
            )
            operator = aperture_variance if is_variance else aperture
            leakage = jnp.einsum("syx,yx->s", operator, image)
            if name in ("leakage_fraction", "leakage_fraction_variance"):
                samples = leakage
            else:
                area = jnp.asarray(
                    record.core_area_variance_lod2
                    if is_variance
                    else record.core_area_lod2
                )
                samples = (
                    jnp.einsum("syx,yx->s", ring, image)
                    if ring is not None
                    else jnp.where(area > 0, leakage / jnp.where(area > 0, area, 1), 0)
                )
                if name == "raw_contrast":
                    tau = jnp.asarray(record.throughput)
                    samples = jnp.where(
                        tau > 0, samples * area / jnp.where(tau > 0, tau, 1), 1e20
                    )
            return record.evaluate_samples(name, separation, samples).astype(
                jnp.asarray(point).dtype
            )

        if not isinstance(diameter, jax.core.Tracer):
            return finite(diameter)
        return jax.lax.cond(
            jnp.asarray(diameter) == 0, lambda d: jnp.asarray(point), finite, diameter
        )

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


class ScalarCoronagraph(AbstractCoronagraph):
    """Constant performance calibrated for a declared aperture rule."""

    frame: Frame = eqx.field(static=True)
    bandpass: Band | None
    _records: tuple
    sky_transmission: float
    outer_working_angle_lod: float | None

    def __init__(
        self,
        *,
        grid,
        throughput_fraction,
        aperture_area_lod2,
        stellar_leakage_density_lod2,
        sky_transmission=1.0,
        bandpass=None,
        rule=DEFAULT_APERTURE_RULE,
        outer_working_angle_lod=None,
    ):
        """Calibrate one constant record over the radial extent of the supplied grid."""
        if grid.unit != "lod":
            raise ValueError("native grid must use lod")
        self.frame = grid.frame
        self.bandpass = bandpass
        self.sky_transmission = sky_transmission
        self.outer_working_angle_lod = outer_working_angle_lod
        tau, area, density = (
            throughput_fraction,
            aperture_area_lod2,
            stellar_leakage_density_lod2,
        )
        self._records = (
            PerformanceCurves(
                [0.0],
                grid.frame,
                rule,
                [tau],
                [area],
                [density * area],
                [density],
                [density * area / tau if tau > 0 else 1e20],
                0.0 if tau > 0 else float("nan"),
                min(grid.shape) * grid.step / 2,
                throughput_variance=[tau],
                core_area_variance_lod2=[area],
                leakage_fraction_variance=[density * area],
                leakage_density_variance_lod2=[density],
            ),
        )

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Read an eagerly prepared constant record; never build inside tracing."""
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    def uniform_sky_response(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Constant dimensionless sky transmission."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return jnp.full_like(jnp.asarray(separation_lod) * 1.0, self.sky_transmission)


class MultiBandCoronagraph(AbstractCoronagraph):
    """Dispatch by declared bandpass, in input order at shared wavelengths.

    curves() and frame use the reference band; band(wavelength_nm).curves(rule)
    selects another record. Convenience curves keep their optional wavelength
    argument and dispatch to that band's record. Monochromatic tables retain
    their existing lack of additional wavelength dependence.
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

    @property
    def outer_working_angle_lod(self):
        """Reference band's design outer working angle, or None when unknown."""
        return self.band(self.reference_wavelength_nm).outer_working_angle_lod

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Reference band's cached performance record."""
        return self.band(self.reference_wavelength_nm).curves(
            rule, leakage_rule=leakage_rule
        )

    def prepared(
        self, rule: RecordRule, *, leakage_rule="aperture_mean"
    ) -> tuple[AbstractCoronagraph, PerformanceCurves]:
        """Return the reference band's prepared model and record."""
        return self.band(None).prepared(rule, leakage_rule=leakage_rule)

    def uniform_sky_response(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Selected band's sky transmission."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.band(wavelength_nm).uniform_sky_response(
            separation_lod, wavelength_nm, time_d=time_d
        )
