"""Coronagraph performance records and images on explicit shared grids."""

import abc
import math
from dataclasses import replace
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.grids import Frame, Image, PixelGrid, resample
from hwoutils.spectral import Band

from optixstuff.apertures import DEFAULT_APERTURE_RULE, ApertureRule
from optixstuff.performance import PerformanceCurves


def _on_grid(image: Image, grid: PixelGrid) -> Image:
    """Resample using the product's frame; reconcile different lod frames via sky.

    There is no shortcut for an identical grid: native samples are point
    densities under the declared reconstruction, and the target pixels are
    box integrals of it, so the result must be continuous in the target pitch.
    """
    source = image.grid
    target = grid
    if source.unit != target.unit or (
        source.unit == "lod" and source.frame != target.frame
    ):
        if source.unit == "lod":
            source = source.to_arcsec(source.frame.reference_wavelength_nm)
        if target.unit == "lod":
            target = target.to_arcsec(target.frame.reference_wavelength_nm)
    sampled = resample(replace(image, grid=source), target)
    return replace(sampled, grid=grid)


def _convolve_quadrants(flux, psf_datacube, grid: PixelGrid):
    """Fold sources about grid.center, with odd shared axes counted exactly once.

    Cube axes are (source_y, source_x, y, x). Positive source indices start
    at ceil(c); even grids have half-integer offsets and disjoint quadrants.
    Odd grids share the zero-offset axes, supplied by the positive quadrant;
    reflected negative halves receive one leading zero on each shared axis.
    """
    cx, cy = grid.center_x, grid.center_y
    if not grid.is_geometric:
        raise ValueError("quarter datacubes require a geometric grid center")
    sx, sy = math.ceil(cx), math.ceil(cy)
    padx, pady = int(cx == sx), int(cy == sy)
    quadrants = (
        flux[sy:, sx:],
        jnp.pad(flux[sy:, :sx][:, ::-1], ((0, 0), (padx, 0))),
        jnp.pad(flux[:sy, :sx][::-1, ::-1], ((pady, 0), (padx, 0))),
        jnp.pad(flux[:sy, sx:][::-1, :], ((pady, 0), (0, 0))),
    )
    images = jnp.einsum("qij,ijyx->qyx", jnp.stack(quadrants), psf_datacube)
    return images[0] + images[1, :, ::-1] + images[2, ::-1, ::-1] + images[3, ::-1, :]


class AbstractCoronagraph(eqx.Module):
    """Optics on a native lod grid with its explicit wavelength/diameter frame.

    Source coordinates and stellar diameters are in each native product's
    lambda/D. Image output sampling is defined entirely by the target grid.
    Leakage queries accept stellar_diam_lod=0.0 for the stored point-source
    record. Finite diameters use prepared image weights. Models without them
    raise ValueError for concrete nonzero diameters and return NaN for traced
    nonzero diameters.
    Wavelength selects a band; monochromatic tables have no further chromatic
    dependence. time_d is always days from the scene epoch.
    """

    grid: AbstractVar[PixelGrid]
    bandpass: AbstractVar[Band | None]
    """Calibrated wavelength interval in nm, or None for unknown coverage."""
    outer_working_angle_lod: AbstractVar[float | None]

    @abc.abstractmethod
    def curves(
        self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"
    ) -> PerformanceCurves:
        """Return the prepared performance record for an explicit aperture rule."""

    def prepared(
        self, rule: ApertureRule, *, leakage_rule="aperture_mean"
    ) -> tuple["AbstractCoronagraph", PerformanceCurves]:
        """Return a model and its usable record, preparing on the host if supported."""
        return self, self.curves(rule, leakage_rule=leakage_rule)

    def _performance_band(self, wavelength_nm):
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
    ):
        """Weighted planet flux fraction, independent of stellar diameter."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return (
            self._performance_band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate("throughput", separation_lod)
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
    ):
        """Aperture area in (lambda/D)^2, independent of stellar diameter."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return (
            self._performance_band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate("core_area_lod2", separation_lod)
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
    ):
        """Absolute stellar flux fraction integrated with the rule's weights."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self._performance_band(wavelength_nm)._leakage_query(
            "leakage_fraction",
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
    ):
        """Stellar flux fraction per (lambda/D)^2 under the named averaging rule."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self._performance_band(wavelength_nm)._leakage_query(
            "leakage_density_lod2",
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
        return self._performance_band(wavelength_nm)._leakage_query(
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

    def leakage_density_map(
        self, rules=None, *, stellar_diam_lod=0.0, leakage_rule="aperture_mean"
    ):
        """Project density per (lambda/D)^2 on the native separation map.

        Explicit prepared rules are stacked on the last axis. Diameter uses
        the stellar product's native lambda/D, as in stellar_map.
        """
        radius = self.separation_map()
        if rules is None:
            return self.leakage_density_lod2(
                radius, stellar_diam_lod=stellar_diam_lod, leakage_rule=leakage_rule
            )
        return jnp.stack(
            [
                self.leakage_density_lod2(
                    radius,
                    rule=r,
                    stellar_diam_lod=stellar_diam_lod,
                    leakage_rule=leakage_rule,
                )
                for r in rules
            ],
            axis=-1,
        )

    def separation_map(self):
        """Native off-axis pixel radii in lambda/D, including the full grid basis."""
        x, y = self.grid.coordinates()
        return jnp.hypot(x, y)

    def throughput_map(self, rules=None):
        """Project throughput, stacking explicitly prepared rules on the last axis."""
        radius = self.separation_map()
        if rules is None:
            return self.throughput(radius)
        return jnp.stack([self.throughput(radius, rule=r) for r in rules], axis=-1)

    def core_area_map(self, rules=None):
        """Project aperture area, stacking prepared rules on the last axis."""
        radius = self.separation_map()
        if rules is None:
            return self.core_area(radius)
        return jnp.stack([self.core_area(radius, rule=r) for r in rules], axis=-1)

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Read dimensionless sky transmission when the record carries that profile.

        Backends with a separate sky product may override this concrete helper.
        """
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return (
            self._performance_band(wavelength_nm)
            .curves()
            .evaluate("occulter_transmission", separation_lod)
        )

    def on_axis_psf(self, wavelength_nm, grid: PixelGrid) -> Image:
        """Point-star per_pixel stellar_flux_fraction on grid, shape (ny,nx)."""
        return self.stellar_map(wavelength_nm, 0.0, grid=grid)

    def off_axis_psf(self, wavelength_nm, separation_lod, grid: PixelGrid) -> Image:
        """One +x source, per_pixel stellar_flux_fraction on grid, shape (ny,nx)."""
        image = self.source_psfs(
            wavelength_nm, jnp.atleast_1d(separation_lod), jnp.zeros(1), grid=grid
        )
        return replace(image, data=image.data[0], leading_axes=())

    def stellar_frame(self, wavelength_nm) -> Frame:
        """Return the native frame for stellar diameters from model metadata.

        Models with a separate stellar product grid override this method.
        Scalar and point-source models use their native grid frame.
        """
        return self.grid.frame

    @abc.abstractmethod
    def stellar_map(
        self, wavelength_nm, stellar_diam_lod, *, grid: PixelGrid | None = None
    ) -> Image:
        """Finite-star per_pixel fractions; table grid=None retains native samples."""

    @abc.abstractmethod
    def source_psfs(
        self, wavelength_nm, x_lod, y_lod, *, grid: PixelGrid | None = None
    ) -> Image:
        """Source fractions; table grid=None retains native samples and metadata."""

    @abc.abstractmethod
    def background_transmission(self, wavelength_nm, *, grid: PixelGrid) -> Image:
        """Dimensionless transmission, per_unit_area to preserve sampled amplitude."""

    @abc.abstractmethod
    def extended_scene(
        self, scene_map: Image, wavelength_nm, *, grid: PixelGrid, rotation_deg=0.0
    ) -> Image:
        """Redistribute per-pixel scene flux; retain quantity on the target grid.

        scene_map is a 2D per_pixel Image. rotation_deg is a concrete CCW
        angle in the scene's coordinate plane; orient grids for sky parity.
        """


class AbstractTableCoronagraph(AbstractCoronagraph):
    """Native Image tables resampled once through hwoutils.grids.resample.

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

    sky_trans: AbstractVar[Image]
    psf_datacube: AbstractVar[object]

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
            image = self.stellar_map(wavelength, d, grid=None).data
            aperture, ring = operators
            leakage = jnp.einsum("syx,yx->s", aperture, image)
            area, tau = (
                jnp.asarray(record.core_area_lod2),
                jnp.asarray(record.throughput),
            )
            density = (
                jnp.einsum("syx,yx->s", ring, image)
                if ring is not None
                else jnp.where(area > 0, leakage / jnp.where(area > 0, area, 1), 0)
            )
            values = {
                "leakage_fraction": leakage,
                "leakage_density_lod2": density,
                "raw_contrast": jnp.where(
                    tau > 0, density * area / jnp.where(tau > 0, tau, 1), 1e20
                ),
            }
            return record.evaluate_samples(name, separation, values[name]).astype(
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

    def _native_stellar_map(self, stellar_diam_lod) -> Image:
        """Supply native stellar samples when the table supports stellar images."""
        raise NotImplementedError("table has no native stellar image provider")

    def _native_source_psfs(self, x_lod, y_lod) -> Image:
        """Supply native PSFs when the table supports point-source images."""
        raise NotImplementedError("table has no native source image provider")

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid=None):
        """Per-pixel stellar fraction on grid, using the stellar product's frame."""
        image = self._native_stellar_map(stellar_diam_lod)
        return image if grid is None else _on_grid(image, grid)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Per-pixel source fractions on grid; x indexes columns, y indexes rows."""
        image = self._native_source_psfs(x_lod, y_lod)
        return image if grid is None else _on_grid(image, grid)

    def background_transmission(self, wavelength_nm, *, grid):
        """Sample dimensionless transmission without multiplying by pixel area."""
        return _on_grid(self.sky_trans, grid)

    def extended_scene(self, scene_map, wavelength_nm, *, grid, rotation_deg=0.0):
        """Convolve scene flux with full or grid-centred quarter PSF tables."""
        if self.psf_datacube is None:
            raise ValueError("extended_scene requires psf_datacube")
        if scene_map.measure != "per_pixel" or scene_map.data.ndim != 2:
            raise ValueError("scene_map must be a two-dimensional per_pixel Image")
        if rotation_deg != 0:
            angle = math.radians(rotation_deg)
            c, s = math.cos(angle), math.sin(angle)
            (a, b), (d, e) = scene_map.grid.basis
            rotated = replace(
                scene_map.grid,
                basis=((c * a - s * d, c * b - s * e), (s * a + c * d, s * b + c * e)),
            )
            scene_map = replace(scene_map, grid=rotated)
        src = _on_grid(scene_map, self.grid).data
        ny, nx = self.grid.shape
        shape = self.psf_datacube.shape[:2]
        if shape == (ny, nx):
            native = jnp.einsum("ij,ijyx->yx", src, self.psf_datacube)
        elif shape == ((ny + 1) // 2, (nx + 1) // 2):
            native = _convolve_quadrants(src, self.psf_datacube, self.grid)
        else:
            raise ValueError(
                f"psf_datacube source shape {shape} must be full "
                f"{(ny, nx)} or quarter {((ny + 1) // 2, (nx + 1) // 2)}"
            )
        return _on_grid(
            Image(native, self.grid, (), "per_pixel", scene_map.quantity), grid
        )


class AbstractScalarCoronagraph(AbstractCoronagraph):
    """Analytic records for ETC use; image methods are explicitly zero placeholders.

    Constants describe one calibrated rule. Without a PSF, another aperture's
    encircled flux cannot be inferred: subclasses supply records for each rule.
    """

    grid: PixelGrid
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
        """Prepare one constant record from explicitly unit-labelled constants."""
        if grid.unit != "lod":
            raise ValueError("native grid must use lod")
        self.grid = grid
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
            ),
        )

    @abc.abstractmethod
    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Read an eagerly prepared constant record; never build inside tracing."""
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid=None):
        """Zero per-pixel stellar fraction; scalar models do not synthesize PSFs."""
        grid = self.grid if grid is None else grid
        return Image(
            jnp.zeros(grid.shape), grid, (), "per_pixel", "stellar_flux_fraction"
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Zero source PSFs, shape (source,ny,nx), for a scalar-only model."""
        grid = self.grid if grid is None else grid
        k = jnp.atleast_1d(x_lod).shape[0]
        return Image(
            jnp.zeros((k, *grid.shape)),
            grid,
            ("source",),
            "per_pixel",
            "stellar_flux_fraction",
        )

    def background_transmission(self, wavelength_nm, *, grid):
        """Zero sampled transmission placeholder for scalar-only models."""
        return Image(jnp.zeros(grid.shape), grid, (), "per_unit_area", "transmission")

    def extended_scene(self, scene_map, wavelength_nm, *, grid, rotation_deg=0.0):
        """Zero scene-flux placeholder, retaining the input quantity."""
        return Image(jnp.zeros(grid.shape), grid, (), "per_pixel", scene_map.quantity)


class ScalarCoronagraph(AbstractScalarCoronagraph):
    """Constant performance for a declared aperture rule, independent of wavelength.

    The one-knot record has tau=throughput_fraction, A=aperture_area_lod2,
    density=stellar_leakage_density_lod2, ell=density*A and C=ell/tau.
    A positive constant curve first has positive throughput at zero (iwa=0).
    Data extent describes the supplied grid; a design OWA is separate.
    """

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Select the constant record built by AbstractScalarCoronagraph."""
        return super().curves(rule, leakage_rule=leakage_rule)

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Constant dimensionless sky transmission."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return jnp.full_like(jnp.asarray(separation_lod) * 1.0, self.sky_transmission)


class MultiBandCoronagraph(AbstractCoronagraph):
    """Nearest-band dispatch at a concrete wavelength, with no spectral interpolation.

    curves() and grid use the reference band; band(wavelength_nm).curves(rule)
    selects another record. Convenience curves keep their optional wavelength
    argument and dispatch to that band's record. Monochromatic tables retain
    their existing lack of additional wavelength dependence.
    """

    band_centers_nm: tuple = eqx.field(static=True)
    bands: tuple
    reference_wavelength_nm: float = eqx.field(static=True)

    def __init__(self, band_centers_nm, bands, *, reference_wavelength_nm=None):
        """Store one model per band, selecting the first as reference by default."""
        self.band_centers_nm = tuple(float(c) for c in band_centers_nm)
        self.bands = tuple(bands)
        if not self.bands or len(self.bands) != len(self.band_centers_nm):
            raise ValueError("Need one or more bands, with one center per band")
        self.reference_wavelength_nm = float(
            self.band_centers_nm[0]
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )

    def band(self, wavelength_nm):
        """Return the nearest band; wavelength is a concrete host scalar."""
        index = min(
            range(len(self.bands)),
            key=lambda i: abs(self.band_centers_nm[i] - float(wavelength_nm)),
        )
        return self.bands[index]

    @property
    def bandpass(self):
        """Span all declared band edges, or None if any band has unknown coverage."""
        if any(band.bandpass is None for band in self.bands):
            return None
        return Band(
            min(band.bandpass.lo_nm for band in self.bands),
            max(band.bandpass.hi_nm for band in self.bands),
        )

    @property
    def quadrature_knots_nm(self):
        """Nearest-band dispatch breakpoints: midpoints between band centres [nm]."""
        centers = sorted(self.band_centers_nm)
        return tuple((a + b) / 2 for a, b in pairwise(centers))

    def _performance_band(self, wavelength_nm):
        return self.band(
            self.reference_wavelength_nm if wavelength_nm is None else wavelength_nm
        )

    @property
    def grid(self):
        """Reference band's native grid."""
        return self.band(self.reference_wavelength_nm).grid

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
        self, rule: ApertureRule, *, leakage_rule="aperture_mean"
    ) -> tuple[AbstractCoronagraph, PerformanceCurves]:
        """Return the reference band's prepared model and record."""
        return self._performance_band(None).prepared(rule, leakage_rule=leakage_rule)

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Selected band's sky transmission."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self._performance_band(wavelength_nm).occulter_transmission(
            separation_lod, wavelength_nm, time_d=time_d
        )

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid=None):
        """Selected band's stellar Image on grid."""
        return self._performance_band(wavelength_nm).stellar_map(
            wavelength_nm, stellar_diam_lod, grid=grid
        )

    def stellar_frame(self, wavelength_nm) -> Frame:
        """Return the selected band's native stellar frame."""
        return self._performance_band(wavelength_nm).stellar_frame(wavelength_nm)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Selected band's source Images on grid."""
        return self._performance_band(wavelength_nm).source_psfs(
            wavelength_nm, x_lod, y_lod, grid=grid
        )

    def background_transmission(self, wavelength_nm, *, grid):
        """Selected band's dimensionless transmission Image on grid."""
        return self.band(wavelength_nm).background_transmission(
            wavelength_nm, grid=grid
        )

    def extended_scene(self, scene_map, wavelength_nm, *, grid, rotation_deg=0.0):
        """Selected band's scene redistribution on grid."""
        return self.band(wavelength_nm).extended_scene(
            scene_map, wavelength_nm, grid=grid, rotation_deg=rotation_deg
        )
