"""Coronagraph performance records and images on explicit shared grids."""

import abc
import math
from dataclasses import replace
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.grids import Image, PixelGrid, resample
from yippy import Circular, PerformanceCurves

_DEFAULT_RULE = Circular(0.7)


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
    ny, nx = grid.shape
    cx, cy = grid.center_x, grid.center_y
    if (cx, cy) != ((nx - 1) / 2, (ny - 1) / 2):
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
    Wavelength selects a band; monochromatic tables have no further chromatic
    dependence. time_d is always days from the scene epoch.
    """

    grid: AbstractVar[PixelGrid]
    outer_working_angle_lod: AbstractVar[float | None]

    @abc.abstractmethod
    def curves(
        self, rule=_DEFAULT_RULE, *, leakage_rule="aperture_mean"
    ) -> PerformanceCurves:
        """Return the prepared performance record for an explicit aperture rule."""

    def _performance_band(self, wavelength_nm):
        return self

    def throughput(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        rule=_DEFAULT_RULE,
        leakage_rule="aperture_mean",
    ):
        """Weighted planet flux fraction, from the selected rule's record."""
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
        rule=_DEFAULT_RULE,
        leakage_rule="aperture_mean",
    ):
        """Aperture or noise-equivalent area in (lambda/D)^2."""
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
        rule=_DEFAULT_RULE,
        leakage_rule="aperture_mean",
    ):
        """Absolute stellar flux fraction integrated with the rule's weights."""
        return (
            self._performance_band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate("leakage_fraction", separation_lod)
        )

    def leakage_density_lod2(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        rule=_DEFAULT_RULE,
        leakage_rule="aperture_mean",
    ):
        """Stellar flux fraction per (lambda/D)^2 under the named averaging rule."""
        return (
            self._performance_band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate("leakage_density_lod2", separation_lod)
        )

    def raw_contrast(
        self,
        separation_lod,
        wavelength_nm=None,
        *,
        time_d=0.0,
        rule=_DEFAULT_RULE,
        leakage_rule="aperture_mean",
    ):
        """Rule-defined stellar intensity divided by planet intensity."""
        return (
            self._performance_band(wavelength_nm)
            .curves(rule, leakage_rule=leakage_rule)
            .evaluate("raw_contrast", separation_lod)
        )

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Read dimensionless sky transmission when the record carries that profile.

        Backends with a separate sky product may override this concrete helper.
        """
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

    @abc.abstractmethod
    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid: PixelGrid) -> Image:
        """Finite-star per_pixel stellar_flux_fraction, shape (ny,nx)."""

    @abc.abstractmethod
    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid: PixelGrid) -> Image:
        """Per_pixel stellar_flux_fraction PSFs, shape (source,ny,nx)."""

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
    """Native Image tables resampled once through hwoutils.grids.resample."""

    sky_trans: AbstractVar[Image]
    psf_datacube: AbstractVar[object]

    @property
    def psf_shape(self):
        """Native image shape (ny,nx)."""
        return self.grid.shape

    @abc.abstractmethod
    def stellar_intens(self, stellar_diam_lod) -> Image:
        """Native per-pixel stellar fraction Image, retaining its stellar grid."""

    @abc.abstractmethod
    def create_psfs(self, x_lod, y_lod) -> Image:
        """Native source PSFs Image with leading axis 'source'."""

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid):
        """Per-pixel stellar fraction on grid, using the stellar product's frame."""
        return _on_grid(self.stellar_intens(stellar_diam_lod), grid)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid):
        """Per-pixel source fractions on grid; x indexes columns, y indexes rows."""
        return _on_grid(self.create_psfs(x_lod, y_lod), grid)

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
        rule=_DEFAULT_RULE,
        outer_working_angle_lod=None,
    ):
        """Prepare one constant record from explicitly unit-labelled constants."""
        if grid.unit != "lod":
            raise ValueError("native grid must use lod")
        self.grid = grid
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
    def curves(self, rule=_DEFAULT_RULE, *, leakage_rule="aperture_mean"):
        """Read an eagerly prepared constant record; never build inside tracing."""
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid):
        """Zero per-pixel stellar fraction; scalar models do not synthesize PSFs."""
        return Image(
            jnp.zeros(grid.shape), grid, (), "per_pixel", "stellar_flux_fraction"
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid):
        """Zero source PSFs, shape (source,ny,nx), for a scalar-only model."""
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

    def curves(self, rule=_DEFAULT_RULE, *, leakage_rule="aperture_mean"):
        """Select the constant record built by AbstractScalarCoronagraph."""
        return super().curves(rule, leakage_rule=leakage_rule)

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Constant dimensionless sky transmission."""
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

    def curves(self, rule=_DEFAULT_RULE, *, leakage_rule="aperture_mean"):
        """Reference band's cached performance record."""
        return self.band(self.reference_wavelength_nm).curves(
            rule, leakage_rule=leakage_rule
        )

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Selected band's sky transmission."""
        return self._performance_band(wavelength_nm).occulter_transmission(
            separation_lod, wavelength_nm, time_d=time_d
        )

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid):
        """Selected band's stellar Image on grid."""
        return self.band(wavelength_nm).stellar_map(
            wavelength_nm, stellar_diam_lod, grid=grid
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid):
        """Selected band's source Images on grid."""
        return self.band(wavelength_nm).source_psfs(
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
