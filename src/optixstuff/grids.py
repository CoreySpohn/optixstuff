"""Static image geometry and arrays with explicit sampling metadata.

Arrays are (..., y, x); named coordinates are (x, y). Pixel (row j, column i)
is centered at (i, j), so the array corner is (-0.5, -0.5). Coordinates obey
u = B (p - c). No array or device value is stored in a PixelGrid.
"""

import functools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from operator import index
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import arcsec2rad
from hwoutils.conversions import lambda_d_to_arcsec
from hwoutils.leaves import as_leaf
from hwoutils.map_coordinates import map_coordinates
from hwoutils.radial import get_center


class _XY(NamedTuple):
    x: object
    y: object


def _traced(*values) -> bool:
    """Whether any value, or any entry of a nested sequence, is a JAX tracer."""
    for value in values:
        if isinstance(value, jax.core.Tracer):
            return True
        if isinstance(value, (tuple, list)) and _traced(*value):
            return True
    return False


def _cos_sin(angle_deg):
    """Cosine and sine of a degree angle; host floats unless the angle is traced."""
    if isinstance(angle_deg, jax.core.Tracer):
        angle = jnp.radians(angle_deg)
        return jnp.cos(angle), jnp.sin(angle)
    angle = math.radians(angle_deg)
    return math.cos(angle), math.sin(angle)


@dataclass(frozen=True)
class Frame:
    """Definition of one lambda/D, including the declared diameter convention.

    The convention labels which physical diameter is supplied; no conversion
    between inscribed and circumscribed diameters is inferred.
    coordinate_model='paraxial_entrance_tilt' (the only supported model)
    identifies small entrance-pupil angular tilts, theta = s * lambda / D
    in radians for a coordinate s in lambda/D. No tangent or sine projection
    is applied. The default retains this linear angular conversion.

    Attributes:
        reference_wavelength_nm: Wavelength in nm at which the frame is defined.
        diameter_m: Telescope diameter in meters, in the stated convention.
        diameter_convention: ``"circumscribed"`` or ``"inscribed"``; labels which
            diameter ``diameter_m`` is.
        coordinate_model: Angular coordinate model; only
            ``"paraxial_entrance_tilt"`` is supported.
    """

    reference_wavelength_nm: float
    diameter_m: float
    diameter_convention: str
    coordinate_model: str = "paraxial_entrance_tilt"

    def __post_init__(self):
        """Store finite positive host floats and validate the diameter label."""
        for name in ("reference_wavelength_nm", "diameter_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.diameter_convention not in ("circumscribed", "inscribed"):
            raise ValueError("diameter_convention must be circumscribed or inscribed")
        if self.coordinate_model not in ("paraxial_entrance_tilt",):
            raise ValueError("coordinate_model must be 'paraxial_entrance_tilt'")

    def lod_to_arcsec(self, wavelength_nm):
        """Size of one lambda/D in arcsec; wavelength_nm may be traced.

        Args:
            wavelength_nm: Wavelength in nm, scalar or array.

        Returns:
            Angular size of one lambda/D in arcsec, shaped like ``wavelength_nm``.
        """
        return lambda_d_to_arcsec(1.0, wavelength_nm, self.diameter_m)


@dataclass(frozen=True)
class SkyOrientation:
    """Position angle of detector +y, north through east, and handedness.

    Let c=cos(pa_y_deg), s=sin(pa_y_deg). Detector +y has sky direction
    (E,N)=(s,c); the orthogonal +x direction is parity*(c,-s). Thus
    x=parity*(E*c-N*s), y=E*s+N*c, with determinant parity. For parity +1,
    x=E*c-N*s; for the inside-sphere parity -1, x=-E*c+N*s. The inverse
    is the transpose since both detector axes are orthonormal.

    pa_y_deg may be a traced value inside a JAX transformation. Such an
    orientation is only for use within that transformation: it cannot be
    compared with another traced angle and must not be returned in a grid.

    Attributes:
        pa_y_deg: Position angle of detector +y, north through east, in degrees.
        parity: Handedness of the detector axes, -1 (default) or +1.
    """

    pa_y_deg: float
    parity: int = -1

    def __post_init__(self):
        """Canonicalize host values and reject undefined orientations."""
        if not isinstance(self.pa_y_deg, jax.core.Tracer):
            object.__setattr__(self, "pa_y_deg", float(self.pa_y_deg))
            if not math.isfinite(self.pa_y_deg):
                raise ValueError("orientation needs finite pa_y_deg")
        object.__setattr__(self, "parity", index(self.parity))
        if self.parity not in (-1, 1):
            raise ValueError("orientation needs parity -1 or +1")

    def sky_to_detector(self, east, north):
        """Map east/north offsets to named detector (x, y) offsets.

        Args:
            east: East offsets in the sky plane.
            north: North offsets in the sky plane.

        Returns:
            Named (x, y) detector offsets, in the unit of the inputs.
        """
        c, s = _cos_sin(self.pa_y_deg)
        east, north = jnp.asarray(east), jnp.asarray(north)
        return _XY(self.parity * (east * c - north * s), east * s + north * c)

    def detector_to_sky(self, x, y):
        """Return named (east, north) offsets in the sky plane.

        Args:
            x: Detector x offsets.
            y: Detector y offsets.

        Returns:
            Named (east, north) offsets, in the unit of the inputs.
        """
        c, s = _cos_sin(self.pa_y_deg)
        x, y = jnp.asarray(x), jnp.asarray(y)
        return _Sky(self.parity * c * x + s * y, -self.parity * s * x + c * y)


class _Sky(NamedTuple):
    east: object
    north: object


def _checked_wavelength(wavelength_nm):
    """Return a concrete positive float, or a traced wavelength unchanged."""
    if isinstance(wavelength_nm, jax.core.Tracer):
        return wavelength_nm
    wavelength_nm = float(wavelength_nm)
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive and finite")
    return wavelength_nm


@jax.tree_util.register_static
@dataclass(frozen=True)
class PixelGrid:
    """Hashable host geometry, with basis mapping pixel offsets to unit.

    shape is (ny, nx), center_x/center_y are pixel-center coordinates, and
    basis is a nonsingular 2 by 2 tuple. frame is mandatory for unit='lod'.
    orientation, when present, relates the unit plane to the east/north sky.

    Inside a JAX transformation to_lod and to_arcsec accept a traced
    wavelength, and the returned grid then carries traced basis entries
    (shape, centers, unit and frame stay host values). Such a grid is for use
    within that transformation: resample builds its operators from the traced
    scale; do not return it or compare it with another grid.

    Attributes:
        shape: Array shape (ny, nx).
        center_x: Column coordinate of the optical axis, in pixels.
        center_y: Row coordinate of the optical axis, in pixels.
        basis: 2 by 2 matrix mapping pixel offsets (dx, dy) to offsets in ``unit``,
            in ``unit`` per pixel.
        unit: One of ``"pupil_diameter"``, ``"meter"``, ``"lod"``, ``"arcsec"``,
            ``"pixel"`` or ``"au"``.
        frame: Frame defining lambda/D; required when ``unit`` is ``"lod"``.
        orientation: Relation of the unit plane to the east/north sky, or None.
    """

    shape: tuple[int, int]
    center_x: float
    center_y: float
    basis: tuple[tuple[float, float], tuple[float, float]]
    unit: str
    frame: Frame | None = None
    orientation: SkyOrientation | None = None

    def __post_init__(self):
        """Canonicalize host geometry and validate units and invertibility."""
        shape = tuple(index(n) for n in self.shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("shape must contain positive (ny, nx)")
        if _traced(self.basis):
            if len(self.basis) != 2 or any(len(row) != 2 for row in self.basis):
                raise ValueError("basis must have shape (2, 2)")
            basis = tuple(tuple(row) for row in self.basis)
        else:
            basis = np.asarray(self.basis, dtype=np.float64)
            if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
                raise ValueError("basis must be finite with shape (2, 2)")
            determinant = basis[0, 0] * basis[1, 1] - basis[0, 1] * basis[1, 0]
            if not np.isfinite(determinant) or determinant == 0:
                raise ValueError("basis must be nonsingular with finite determinant")
            basis = tuple(tuple(map(float, row)) for row in basis)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "basis", basis)
        for name in ("center_x", "center_y"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("centers must be finite")
            object.__setattr__(self, name, value)
        if self.unit not in ("pupil_diameter", "meter", "lod", "arcsec", "pixel", "au"):
            raise ValueError("unknown grid unit")
        if self.unit == "lod" and self.frame is None:
            raise ValueError("a lod grid requires a frame")
        if self.frame is not None and not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")
        if self.orientation is not None and not isinstance(
            self.orientation, SkyOrientation
        ):
            raise TypeError("orientation must be a SkyOrientation")

    @classmethod
    def regular(cls, shape, step, unit, *, center=None, frame=None, orientation=None):
        """Make an axis-aligned isotropic grid; center defaults to geometric (x,y).

        Args:
            shape: Array shape (ny, nx).
            step: Pixel spacing in ``unit``, positive and finite.
            unit: Grid unit, as for the ``unit`` attribute.
            center: Optical axis (x, y) in pixels; the geometric center when None.
            frame: Frame defining lambda/D; required for ``"lod"``.
            orientation: Relation of the unit plane to the sky, or None.

        Returns:
            The axis-aligned isotropic grid.
        """
        if center is None:
            cy, cx = get_center(shape)
            center = (cx, cy)
        step = float(step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("step must be positive and finite")
        return cls(
            shape,
            center[0],
            center[1],
            ((step, 0.0), (0.0, step)),
            unit,
            frame,
            orientation,
        )

    @classmethod
    def pupil(cls, npix):
        """Square grid spanning one pupil diameter, with geometric center.

        Args:
            npix: Number of pixels along each axis.

        Returns:
            Grid in units of the pupil diameter, with pixel spacing 1 / ``npix``.
        """
        npix = index(npix)
        if npix < 1:
            raise ValueError("npix must be positive")
        return cls.regular((npix, npix), 1 / npix, "pupil_diameter")

    @classmethod
    def focal(cls, npix, step_lod, frame):
        """Square focal grid in lambda/D, with geometric center.

        Args:
            npix: Number of pixels along each axis.
            step_lod: Pixel spacing in lambda/D.
            frame: Frame defining lambda/D.

        Returns:
            Square grid in lambda/D.
        """
        return cls.regular((npix, npix), step_lod, "lod", frame=frame)

    @classmethod
    def from_corner_count(
        cls, shape, center_x, center_y, basis, unit, frame=None, orientation=None
    ):
        """Construct from corner counts, subtracting 0.5 from each center count.

        Arguments match PixelGrid; only center_x and center_y change meaning
        here. For example, a header count 128 becomes center coordinate 127.5.

        Args:
            shape: Array shape (ny, nx).
            center_x: Column center as a corner count, in pixels (a header value);
                0.5 is subtracted.
            center_y: Row center as a corner count, in pixels; 0.5 is subtracted.
            basis: 2 by 2 matrix mapping pixel offsets to offsets in ``unit``.
            unit: Grid unit, as for the ``unit`` attribute.
            frame: Frame defining lambda/D; required for ``"lod"``.
            orientation: Relation of the unit plane to the sky, or None.

        Returns:
            The grid with pixel-center coordinates.
        """
        return cls(
            shape,
            float(center_x) - 0.5,
            float(center_y) - 0.5,
            basis,
            unit,
            frame,
            orientation,
        )

    def coordinates(self):
        """Return named (x, y) arrays of shape (ny, nx), in this grid's unit.

        Returns:
            Named (x, y) arrays of shape (ny, nx) in the grid's unit.
        """
        rows, columns = jnp.indices(
            self.shape, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32
        )
        return self.pixel_to_unit(columns, rows)

    def corners(self) -> np.ndarray:
        """Return the four outer (x, y) corners, starting at pixel (-0.5, -0.5).

        The corners follow increasing column index first, then increasing row
        index. Like the grid metadata, the result uses host precision.

        Returns:
            Array of shape (4, 2) of (x, y) corners in the grid's unit.
        """
        ny, nx = self.shape
        pixels = np.array(
            [(-0.5, -0.5), (nx - 0.5, -0.5), (nx - 0.5, ny - 0.5), (-0.5, ny - 0.5)]
        )
        return (pixels - (self.center_x, self.center_y)) @ np.asarray(self.basis).T

    def extent(self) -> tuple[float, float, float, float]:
        """Return outer edges (x0, x1, y0, y1) in increasing pixel-index order.

        Reflected axes have descending edges. Resample rotated or sheared
        grids before drawing with imshow.

        Returns:
            Outer edges (x0, x1, y0, y1) in the grid's unit.
        """
        if self.basis[0][1] != 0 or self.basis[1][0] != 0:
            raise ValueError("resample a rotated or sheared grid before display")
        first, _, last, _ = self.corners()
        return float(first[0]), float(last[0]), float(first[1]), float(last[1])

    def radius(self) -> jax.Array:
        """Distance from the optical axis at each pixel center, in grid units.

        Returns:
            Array of shape (ny, nx) of distances in the grid's unit.
        """
        return jnp.hypot(*self.coordinates())

    def pixel_to_unit(self, x, y):
        """Map named pixel-center coordinates (x, y) to offsets in unit.

        Args:
            x: Pixel-center column coordinates.
            y: Pixel-center row coordinates.

        Returns:
            Named (x, y) offsets from the optical axis in the grid's unit.
        """
        dx, dy = jnp.asarray(x) - self.center_x, jnp.asarray(y) - self.center_y
        (a, b), (c, d) = self.basis
        return _XY(a * dx + b * dy, c * dx + d * dy)

    def unit_to_pixel(self, x, y):
        """Invert the basis and center, returning named pixel coordinates (x,y).

        Args:
            x: Offsets along x in the grid's unit.
            y: Offsets along y in the grid's unit.

        Returns:
            Named (x, y) pixel coordinates.
        """
        (a, b), (c, d) = self.basis
        det = a * d - b * c
        x, y = jnp.asarray(x), jnp.asarray(y)
        return _XY(
            (d * x - b * y) / det + self.center_x, (a * y - c * x) / det + self.center_y
        )

    @property
    def is_geometric(self) -> bool:
        """Whether the center is exactly ((nx-1)/2, (ny-1)/2), in pixels.

        Tolerance is zero: integer and half-integer centers are exactly
        representable, and even a small displacement breaks pixel-flip
        symmetry. This does not constrain the basis or orientation.
        """
        cy, cx = get_center(self.shape)
        return self.center_x == cx and self.center_y == cy

    @property
    def step(self) -> float:
        """Positive scalar spacing; raise for rotated or anisotropic bases."""
        (a, b), (c, d) = self.basis
        if b != 0 or c != 0 or a != d or a <= 0:
            raise ValueError("step requires a positive isotropic axis-aligned basis")
        return a

    @property
    def pixel_area(self) -> float:
        """Absolute basis determinant, in unit squared."""
        (a, b), (c, d) = self.basis
        return abs(a * d - b * c)

    def cell_area(self, area: str) -> float:
        """Return cell area in grid units squared or, for an arcsec grid, sr.

        Args:
            area: ``"grid"`` for grid units squared or ``"sr"`` for steradians.

        Returns:
            Area of one cell in the requested denominator.
        """
        if area == "grid":
            return self.pixel_area
        if area != "sr":
            raise ValueError("area must be 'grid' or 'sr'")
        if self.unit != "arcsec":
            raise ValueError("steradian cell area requires an arcsec grid")
        return self.pixel_area * arcsec2rad**2

    def to_arcsec(self, wavelength_nm) -> "PixelGrid":
        """Convert a lod grid at a wavelength to arcsec geometry.

        The basis scales by lambda/D at wavelength_nm, preserving centers and
        retaining the frame. A concrete wavelength gives host geometry; a
        traced one gives a grid with traced basis entries for use inside the
        transformation (see the class docstring).

        Args:
            wavelength_nm: Wavelength in nm; may be traced.

        Returns:
            The grid in arcsec.
        """
        if self.unit != "lod":
            raise ValueError("to_arcsec requires a lod grid")
        wavelength_nm = _checked_wavelength(wavelength_nm)
        scale = self.frame.lod_to_arcsec(wavelength_nm)
        basis = tuple(tuple(value * scale for value in row) for row in self.basis)
        return PixelGrid(
            self.shape,
            self.center_x,
            self.center_y,
            basis,
            "arcsec",
            self.frame,
            self.orientation,
        )

    def to_lod(self, wavelength_nm, frame: Frame) -> "PixelGrid":
        """Convert an arcsec grid to lambda/D geometry in the given frame.

        A traced wavelength gives a grid with traced basis entries for use
        inside the transformation (see the class docstring).

        Args:
            wavelength_nm: Wavelength in nm; may be traced.
            frame: Frame defining lambda/D.

        Returns:
            The grid in lambda/D.
        """
        if self.unit != "arcsec":
            raise ValueError("to_lod requires an arcsec grid")
        wavelength_nm = _checked_wavelength(wavelength_nm)
        scale = frame.lod_to_arcsec(wavelength_nm)
        basis = tuple(tuple(value / scale for value in row) for row in self.basis)
        return PixelGrid(
            self.shape,
            self.center_x,
            self.center_y,
            basis,
            "lod",
            frame,
            self.orientation,
        )

    def is_native(self, wavelength_nm):
        """Whether this is a lod grid at its frame's reference wavelength.

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            True for a lod grid whose frame reference wavelength equals the
            wavelength.
        """
        return (
            self.unit == "lod" and wavelength_nm == self.frame.reference_wavelength_nm
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class GridArray:
    """Array (..., y, x) with a static grid and names for every leading axis.

    Construction validates shape and names. PyTree reconstruction bypasses
    construction checks, as JAX may insert placeholders or map away an axis.
    When returning a container with a changed rank, supply the new axis names.

    Attributes:
        data: Array of shape (..., ny, nx).
        grid: PixelGrid of the last two axes.
        leading_axes: One name per leading axis of ``data``.
    """

    data: jax.Array
    grid: PixelGrid
    leading_axes: tuple[str, ...]

    def __post_init__(self):
        """Convert data and validate spatial shape and leading-axis labels."""
        object.__setattr__(self, "data", jnp.asarray(self.data))
        object.__setattr__(self, "leading_axes", tuple(self.leading_axes))
        if self.data.ndim < 2 or self.data.shape[-2:] != self.grid.shape:
            raise ValueError("data spatial shape must match grid.shape")
        if len(self.leading_axes) != self.data.ndim - 2:
            raise ValueError("leading_axes count must match data leading dimensions")
        if any(not isinstance(name, str) or not name for name in self.leading_axes):
            raise ValueError("leading_axes must contain nonempty strings")

    def tree_flatten(self):
        """Expose data, retaining all other dataclass fields as static metadata.

        Returns:
            The one-tuple of leaves holding the data, and the static metadata tuple.
        """
        metadata = tuple(getattr(self, field.name) for field in fields(self)[1:])
        return (self.data,), metadata

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Rebuild transformed containers without converting JAX placeholders.

        Args:
            metadata: Static fields from ``tree_flatten``, in field order after
                ``data``.
            arrays: One-tuple holding the data array.

        Returns:
            The rebuilt container.
        """
        result = object.__new__(cls)
        for attribute, value in zip(fields(cls), (arrays[0], *metadata), strict=True):
            object.__setattr__(result, attribute.name, value)
        return result


@jax.tree_util.register_static
@dataclass(frozen=True)
class Measure:
    """Spatial meaning of a value and, for a density, its area denominator.

    Attributes:
        kind: ``"integral"`` (value integrated over the cell), ``"density"`` (point
            sample per unit area) or ``"intensive"`` (point sample with no area
            denominator).
        area: Area denominator of a density, ``"grid"`` or ``"sr"``; None
            otherwise.
    """

    kind: str
    area: str | None = None

    def __post_init__(self):
        """Require a denominator exactly when the stored values are densities."""
        if self.kind == "density":
            if self.area not in ("grid", "sr"):
                raise ValueError("density area must be 'grid' or 'sr'")
        elif self.kind not in ("integral", "intensive") or self.area is not None:
            raise ValueError("expected an integral, density or intensive measure")

    @classmethod
    def integral(cls) -> "Measure":
        """Value integrated over the cell.

        Returns:
            The integral measure.
        """
        return cls("integral")

    @classmethod
    def density(cls, area: str) -> "Measure":
        """Point sample of density per grid unit squared or per steradian.

        Args:
            area: ``"grid"`` for per grid unit squared or ``"sr"`` for per steradian.

        Returns:
            The density measure.
        """
        return cls("density", area)

    @classmethod
    def intensive(cls) -> "Measure":
        """Point sample with no area denominator, such as transmission.

        Returns:
            The intensive measure.
        """
        return cls("intensive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Image(GridArray):
    """Image samples with their spatial measure and numerator unit.

    Reconstruction specifies nearest, linear, Keys cubic, cubic spline, or
    band-limited interpolation. Samples beyond the array are zero: an image ends
    where its data end. The local kernels (nearest, linear and cubic) sum those
    zero-extended samples and are not clipped, so their reconstruction reaches
    past the array by the kernel's reach; ``cubic_spline`` and ``band_limited``
    are restricted to the footprint ``[-1/2, N - 1/2]`` of each axis.

    ``cubic_spline`` is the interpolating cubic B-spline of the samples with the
    not-a-knot end condition: the coefficients are a linear map of the samples
    alone (a prefilter, the inverse of the tridiagonal 1/6, 2/3, 1/6 system
    closed by continuity of the third derivative at the second and the
    second-to-last sample), and the reconstruction is restricted to the
    footprint ``[-1/2, N - 1/2]`` of each axis. It reproduces cubic
    polynomials exactly over the whole footprint. Every coefficient depends on
    every sample, with a weight that falls by about 0.268 per cell, so like
    ``band_limited`` it has no finite reach. It suits products that are
    band-limited at the table pitch, including tables cut inside a bright
    field; it rings at a hard edge and is not for zero-filled masks.

    Attributes:
        measure: Spatial meaning of the values.
        quantity: Numerator unit of the values, without a per-steradian
            denominator.
        reconstruction: Interpolation, ``"nearest"``, ``"linear"``, ``"cubic"``,
            ``"cubic_spline"`` or ``"band_limited"``.
    """

    measure: Measure
    quantity: str
    reconstruction: str = "linear"

    def __post_init__(self):
        """Validate array geometry and the interpretation of image values."""
        super().__post_init__()
        if not isinstance(self.measure, Measure):
            raise ValueError("measure must be a Measure")
        if self.measure.kind == "density":
            self.grid.cell_area(self.measure.area)
        _validate_reconstruction(self.reconstruction)
        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise ValueError("quantity must be a nonempty unit string")
        if re.search(r"/\s*sr\b|\bsr\s*(?:\^|\*\*)?\{?-1\b", self.quantity):
            raise ValueError("quantity must contain only the numerator unit")

    def integrated(self) -> "Image":
        """Convert a point density to a cell integral by the midpoint rule.

        Returns:
            The Image with an integral measure; itself when already integral.
        """
        area = self._integration_area()
        if area is None:
            return self
        return self._with_measure(self.data * area, Measure.integral())

    def _integration_area(self):
        """Area factor for density samples, or None for existing integrals."""
        if self.measure.kind == "intensive":
            raise ValueError("an intensive image has no cell integral")
        if self.measure.kind == "integral":
            return None
        return self.grid.cell_area(self.measure.area)

    def density(self, area: str) -> "Image":
        """Express values as density per unit area, averaging cell integrals.

        Args:
            area: ``"grid"`` for per grid unit squared or ``"sr"`` for per steradian.

        Returns:
            The Image with a density measure.
        """
        measure = Measure.density(area)
        target_area = self.grid.cell_area(area)
        if self.measure == measure:
            return self
        if self.measure.kind == "intensive":
            raise ValueError("an intensive image has no area denominator")
        if self.measure.kind == "integral":
            values = self.data / target_area
        else:
            values = self.data * (self.grid.cell_area(self.measure.area) / target_area)
        return self._with_measure(values, measure)

    def _with_measure(self, data, measure):
        # vmap can temporarily remove leading dimensions from the data.
        metadata = (
            self.grid,
            self.leading_axes,
            measure,
            self.quantity,
            self.reconstruction,
        )
        return Image.tree_unflatten(metadata, (data,))


def _validate_reconstruction(reconstruction):
    """Reject unknown reconstruction names before tracing numeric operations."""
    if reconstruction not in _RECONSTRUCTION_MARGIN:
        raise ValueError(
            "reconstruction must be nearest, linear, cubic, cubic_spline, or "
            "band_limited"
        )


def _same_orientation(first, second):
    """Equality that never converts a traced angle to a host bool."""
    return first is second or (
        not _traced(first.pa_y_deg, second.pa_y_deg) and first == second
    )


def _pixel_map(source, target):
    """Compose the target-to-source linear map through the sky plane.

    The map is a NumPy array for host grids and a JAX array when a basis or an
    orientation angle is traced.
    """
    if source.unit != target.unit:
        raise ValueError("mapping requires matching grid units")
    if source.unit == "lod" and source.frame != target.frame:
        raise ValueError("lod mapping requires equal frames")
    if (source.orientation is None) != (target.orientation is None):
        raise ValueError("mapping requires both orientations or neither")
    angles = (
        ()
        if source.orientation is None
        else (source.orientation.pa_y_deg, target.orientation.pa_y_deg)
    )
    xp = jnp if _traced(source.basis, target.basis, angles) else np

    def orientation_matrix(orientation):
        c, s = _cos_sin(orientation.pa_y_deg)
        return xp.array([[orientation.parity * c, -orientation.parity * s], [s, c]])

    transform = xp.eye(2)
    if source.orientation is not None and not _same_orientation(
        source.orientation, target.orientation
    ):
        transform = (
            orientation_matrix(source.orientation)
            @ orientation_matrix(target.orientation).T
        )
    return xp.linalg.solve(
        xp.asarray(source.basis), transform @ xp.asarray(target.basis)
    )


def _band_limited_matrix(n_source, positions, width, pad_factor, dtype, integrate):
    """Build one target-by-source DFT interpolation or box-integration matrix.

    Positions and width may be traced; the matrix is then built with JAX
    operations and is differentiable in both. Host inputs give a NumPy matrix.
    """
    xp = jnp if _traced(positions, width) else np
    positions = xp.asarray(positions)
    samples = np.arange(n_source, dtype=dtype)
    n_padded = math.ceil(pad_factor * n_source)
    frequencies = np.fft.fftfreq(n_padded).astype(dtype)
    phase = np.asarray(2j * np.pi, dtype=np.result_type(dtype, np.complex64))
    if integrate:
        left = xp.clip(positions - width / 2, -0.5, n_source - 0.5)
        right = xp.clip(positions + width / 2, -0.5, n_source - 0.5)
        length = (right - left).astype(dtype)
        positions = ((left + right) / 2).astype(dtype)
        weights = (length / width)[:, None] * xp.sinc(frequencies * length[:, None])
    else:
        weights = ((positions >= -0.5) & (positions <= n_source - 0.5))[:, None]
        positions = xp.clip(positions, -0.5, n_source - 0.5).astype(dtype)
    evaluate = xp.exp(phase * positions[:, None] * frequencies)
    evaluate = evaluate * weights
    analyze = np.exp(-phase * frequencies[:, None] * samples)
    return (evaluate @ analyze) / n_padded


def _box_antiderivative(t, xp):
    """Integral of the nearest (unit box) kernel from -inf to t, piecewise in t."""
    return xp.where(t < -0.5, 0.0, xp.where(t < 0.5, t + 0.5, 1.0))


def _hat_antiderivative(t, xp):
    """Integral of the linear kernel from -inf to t, piecewise in t."""
    return xp.where(
        t < -1.0,
        0.0,
        xp.where(
            t < 0.0,
            0.5 * (t + 1.0) ** 2,
            xp.where(t < 1.0, 1.0 - 0.5 * (1.0 - t) ** 2, 1.0),
        ),
    )


def _keys_antiderivative(t, xp):
    """Integral of the Keys cubic (a = -1/2) from -inf to t, piecewise in t."""
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    return xp.where(
        t < -2.0,
        0.0,
        xp.where(
            t < -1.0,
            t4 / 8 + 5 * t3 / 6 + 2 * t2 + 2 * t + 2 / 3,
            xp.where(
                t < 0.0,
                0.5 + t - 5 * t3 / 6 - 3 * t4 / 8,
                xp.where(
                    t < 1.0,
                    0.5 + t - 5 * t3 / 6 + 3 * t4 / 8,
                    xp.where(
                        t < 2.0, 1 / 3 + 2 * t - 2 * t2 + 5 * t3 / 6 - t4 / 8, 1.0
                    ),
                ),
            ),
        ),
    )


def _bspline(t, xp):
    """Cubic B-spline kernel, piecewise in t."""
    t2 = t * t
    t3 = t2 * t
    return xp.where(
        t < -2.0,
        0.0,
        xp.where(
            t < -1.0,
            (2.0 + t) ** 3 / 6,
            xp.where(
                t < 0.0,
                2 / 3 - t2 - t3 / 2,
                xp.where(
                    t < 1.0,
                    2 / 3 - t2 + t3 / 2,
                    xp.where(t < 2.0, (2.0 - t) ** 3 / 6, 0.0),
                ),
            ),
        ),
    )


def _bspline_antiderivative(t, xp):
    """Integral of the cubic B-spline from -inf to t, piecewise in t."""
    t3 = t * t * t
    t4 = t3 * t
    return xp.where(
        t < -2.0,
        0.0,
        xp.where(
            t < -1.0,
            (2.0 + t) ** 4 / 24,
            xp.where(
                t < 0.0,
                0.5 + 2 * t / 3 - t3 / 3 - t4 / 8,
                xp.where(
                    t < 1.0,
                    0.5 + 2 * t / 3 - t3 / 3 + t4 / 8,
                    xp.where(t < 2.0, 1.0 - (2.0 - t) ** 4 / 24, 1.0),
                ),
            ),
        ),
    )


# The antiderivatives are polynomials in t on each piece, with no sign or
# absolute value, so their derivative at a knot (a pixel edge on a sample) is
# the kernel value there.
_LOCAL_ANTIDERIVATIVE = {
    "nearest": _box_antiderivative,
    "linear": _hat_antiderivative,
    "cubic": _keys_antiderivative,
}


@functools.lru_cache(maxsize=32)
def _not_a_knot_prefilter(n_source):
    """Map from N samples to the N + 4 cubic B-spline coefficients c_-2..c_N+1.

    c_-1..c_N solve the interpolation conditions (c_n-1 + 4 c_n + c_n+1) / 6 =
    d_n and the not-a-knot conditions (zero fourth difference of the
    coefficients about knots 1 and N - 2, a continuous third derivative there).
    The ghosts c_-2 and c_N+1 continue the end pieces, so the expansion is the
    not-a-knot spline, end pieces extended, on [-1, N]. Host float64.
    """
    if n_source < 4:
        raise ValueError("cubic_spline needs at least 4 samples along each axis")
    m = n_source + 2  # unknowns c_-1 .. c_N at columns 0 .. N+1
    system = np.zeros((m, m))
    rows = np.arange(n_source)
    system[rows, rows] = 1 / 6
    system[rows, rows + 1] = 4 / 6
    system[rows, rows + 2] = 1 / 6
    fourth = np.array([-1.0, 4.0, -6.0, 4.0, -1.0])
    system[n_source, 0:5] = fourth
    system[n_source + 1, m - 5 : m] = fourth
    rhs = np.zeros((m, n_source))
    rhs[rows, rows] = 1.0
    inner = np.linalg.solve(system, rhs)
    low = fourth[1:] @ inner[0:4]
    high = fourth[1:][::-1] @ inner[m - 4 : m]
    prefilter = np.vstack([low, inner, high])
    prefilter.setflags(write=False)
    return prefilter


def _cubic_spline_matrix(n_source, positions, width, dtype, integrate):
    """Target-by-source matrix of a not-a-knot cubic spline along one axis.

    With ``integrate`` each row integrates the spline over [p - w/2, p + w/2]
    clipped to the footprint [-1/2, N - 1/2], in source pixels; otherwise it
    samples the spline at p, zero outside the footprint. Positions and width
    may be traced; the matrix is then built with JAX operations.
    """
    xp = jnp if _traced(positions, width) else np
    positions = xp.asarray(positions)
    knots = np.arange(-2, n_source + 2, dtype=np.float64)
    if integrate:
        low = xp.clip(positions - width / 2, -0.5, n_source - 0.5)
        high = xp.clip(positions + width / 2, -0.5, n_source - 0.5)
        basis = _bspline_antiderivative(
            high[:, None] - knots, xp
        ) - _bspline_antiderivative(low[:, None] - knots, xp)
    else:
        inside = (positions >= -0.5) & (positions <= n_source - 0.5)
        basis = xp.where(inside[:, None], _bspline(positions[:, None] - knots, xp), 0.0)
    prefilter = _not_a_knot_prefilter(n_source)
    if xp is np:
        return (basis @ prefilter).astype(dtype)
    return basis.astype(dtype) @ jnp.asarray(prefilter, dtype=dtype)


def _footprint_matrix(n_source, positions, width, reconstruction, dtype):
    """Integral of each source sample's cardinal function over each target pixel.

    Rows are target pixels [p - w/2, p + w/2] along one axis and columns are
    source samples, in source pixels. The local kernels sum zero-extended
    samples and are not clipped, so a covering grid collects the sum of the
    samples exactly; the cubic spline is restricted to the footprint.
    """
    if reconstruction == "cubic_spline":
        return _cubic_spline_matrix(n_source, positions, width, dtype, True)
    antiderivative = _LOCAL_ANTIDERIVATIVE[reconstruction]
    xp = jnp if _traced(positions, width) else np
    positions = xp.asarray(positions)
    samples = np.arange(n_source, dtype=np.float64)
    high = positions[:, None] + width / 2 - samples
    low = positions[:, None] - width / 2 - samples
    matrix = antiderivative(high, xp) - antiderivative(low, xp)
    return matrix.astype(dtype)


def _require_axis_aligned(name, source_grid, target_grid, mapping):
    """Raise for a rotated or sheared basis or map.

    A traced basis keeps the diagonal form it was built with, so a map traced
    only through the bases is accepted. Two different orientations with a
    traced angle may hold any rotation, which cannot be checked on the host,
    so they raise.
    """
    source, target = source_grid.orientation, target_grid.orientation
    if (
        source is not None
        and not _same_orientation(source, target)
        and _traced(source.pa_y_deg, target.pa_y_deg)
    ):
        raise NotImplementedError(
            f"{name} reconstruction does not support a traced orientation angle "
            "that differs between the grids; give both grids one orientation"
        )
    for matrix in (source_grid.basis, target_grid.basis, mapping):
        if _traced(matrix) or isinstance(matrix, jax.core.Tracer):
            continue  # a traced basis keeps the diagonal form it was built with
        matrix = np.asarray(matrix)
        tolerance = 32 * np.finfo(float).eps * np.max(np.abs(matrix))
        if np.any(np.abs(matrix[[0, 1], [1, 0]]) > tolerance):
            raise NotImplementedError(
                f"{name} reconstruction does not support rotated or sheared "
                "grid bases or composed maps"
            )


def _axis_positions(source_grid, target_grid, mapping):
    """Target pixel centers in source pixels and pixel widths, for x then y."""
    traced = _traced(source_grid.basis, target_grid.basis)
    xp = jnp if traced else np
    axes = []
    for axis, source_center, target_center in (
        (0, source_grid.center_x, target_grid.center_x),
        (1, source_grid.center_y, target_grid.center_y),
    ):
        ratio = mapping[axis, axis]
        positions = (
            xp.arange(target_grid.shape[1 - axis]) - target_center
        ) * ratio + source_center
        axes.append((source_grid.shape[1 - axis], positions, abs(ratio)))
    return axes


# Density images whose ``on_grid`` read is the footprint integral of the
# declared reconstruction. A band-limited density is absent because its
# present read (cell integrals, then the box integral of ``resample``) is
# already the footprint integral of its reconstruction.
_FOOTPRINT_READS = frozenset({"nearest", "linear", "cubic", "cubic_spline"})


def _reads_footprint(measure, reconstruction):
    """Whether ``on_grid`` integrates the reconstruction over each pixel.

    The one predicate for the read and its margin: ``on_grid`` routes by it
    and ``_read_margins`` gives such a read the sampling margin, measured from
    the pixel corners.
    """
    return measure.kind == "density" and reconstruction in _FOOTPRINT_READS


def _footprint_integral(image, source_grid, target_grid):
    """Integral of a density's declared reconstruction over each target pixel.

    The density is reconstructed from its point samples and integrated over the
    footprint of every target pixel, in the area the density is per: a
    ``grid`` density in the units of ``image.grid`` and an ``sr`` density in
    steradians, with the source cell area taken as ``Image.integrated`` takes
    it. ``source_grid`` is ``image.grid`` expressed in the target's unit. The
    result is a cell integral that keeps the reconstruction name. The read is
    axis-aligned only: a rotated or sheared map raises NotImplementedError.
    """
    area = image._integration_area()
    mapping = _pixel_map(source_grid, target_grid)
    _require_axis_aligned(image.reconstruction, source_grid, target_grid, mapping)
    data = jnp.asarray(image.data, dtype=jnp.result_type(image.data, 1.0))
    wx, wy = (
        jnp.asarray(
            _footprint_matrix(
                n, positions, width, image.reconstruction, data.real.dtype
            ),
            dtype=data.dtype,
        )
        for n, positions, width in _axis_positions(source_grid, target_grid, mapping)
    )
    metadata = (
        target_grid,
        image.leading_axes,
        Measure.integral(),
        image.quantity,
        image.reconstruction,
    )
    return Image.tree_unflatten(metadata, ((wy @ data @ wx.T) * area,))


_RECONSTRUCTION_MARGIN = {
    "nearest": 0.0,
    "linear": 0.5,
    "cubic": 2.0,
    "cubic_spline": 2.0,
    "band_limited": 2.0,
}


def reconstruction_margin(name: str) -> float:
    """Reach of a reconstruction at an edge, in source pixels.

    Within this distance of the source footprint the reconstruction mixes in
    the zero beyond the data: 0 for nearest, 0.5 for linear, 2 for the Keys
    cubic, the cubic spline and band-limited interpolation. For the first three
    it is the kernel's support, so a reconstruction outside it is exact. The
    cubic spline and the band-limited kernel have no finite support, so for
    them the margin marks the support boundary, not an accuracy bound: a
    band-limited edge rings with an algebraic tail (resampling a constant field
    is off by about 1e-2 eight pixels in), and the coefficients of a cubic
    spline depend on samples at every distance with a weight that falls by
    about 0.268 per cell (its end condition moves a pixel by up to about 13
    percent 2 to 3 cells from the table edge and under 0.1 percent beyond 8).

    This is the reach when the product is sampled at target centers. A read
    that may deposit instead (a cell integral onto pixels at least as large as
    the cells) reaches further: pass ``reconstruction`` and ``measure`` to
    ``coverage`` for the margin of the read that ``resample`` performs.

    Args:
        name: Reconstruction name.

    Returns:
        The margin in source pixels.
    """
    _validate_reconstruction(name)
    return _RECONSTRUCTION_MARGIN[name]


# Kernel half-support in target pixels, for the deposit form of a read.
_DEPOSIT_HALF_SUPPORT = {"nearest": 0.5, "linear": 1.0, "cubic": 2.0}


def _deposits(source_grid, target_grid, measure, reconstruction):
    """Whether ``resample`` deposits cell integrals instead of sampling.

    True for a cell-integral measure, a nearest, linear or cubic
    reconstruction, and target pixels at least as large as source cells. The
    answer is a host bool for host geometry and a traced boolean when the
    pixel areas are traced.
    """
    if measure.kind != "integral" or reconstruction not in _DEPOSIT_HALF_SUPPORT:
        return False
    return target_grid.pixel_area / source_grid.pixel_area >= 1.0


def _read_margins(source_grid, target_grid, reconstruction, measure, mapping):
    """Margins along the source x and y axes of the read ``resample`` performs.

    Sampling reaches the reconstruction margin. Depositing reaches the kernel
    half-support k in target pixels, so a pixel whose extent along a source
    axis is g cells reaches ``(k - 1/2) g - 1/2`` cells past its edge, never
    less than the sampling margin. g is the row sum of absolute values of the
    target-to-source map, which is the bounding extent of a rotated pixel.
    """
    _validate_reconstruction(reconstruction)
    sampling = _RECONSTRUCTION_MARGIN[reconstruction]
    if (
        _reads_footprint(measure, reconstruction)
        or reconstruction not in _DEPOSIT_HALF_SUPPORT
    ):
        return sampling, sampling
    deposit = _deposits(source_grid, target_grid, measure, reconstruction)
    traced = _traced(deposit, mapping)
    if not traced and not deposit:
        return sampling, sampling
    half_support = _DEPOSIT_HALF_SUPPORT[reconstruction]
    xp = jnp if traced else np
    extent = xp.sum(xp.abs(xp.asarray(mapping)), axis=1)
    margins = xp.maximum(sampling, (half_support - 0.5) * extent - 0.5)
    margins = xp.where(deposit, margins, sampling)
    return margins[0], margins[1]


# Reconstructions whose read depends on samples at every distance.
_NO_FINITE_REACH = frozenset({"cubic_spline", "band_limited"})


def _all_valid(xp, valid, first_x, last_x, first_y, last_y):
    """Whether every cell of each inclusive index box is inside and valid.

    A summed-area table of ``valid`` counts the valid cells of each box,
    clipped to the array; the box is all valid when that count equals its
    unclipped area, so cells beyond the array count as missing.
    """
    ny, nx = valid.shape
    counts = xp.cumsum(xp.cumsum(xp.asarray(valid, dtype=xp.int32), axis=0), axis=1)
    table = xp.pad(counts, ((1, 0), (1, 0)))
    x0, x1 = (xp.clip(i, 0, nx).astype(xp.int32) for i in (first_x, last_x + 1))
    y0, y1 = (xp.clip(i, 0, ny).astype(xp.int32) for i in (first_y, last_y + 1))
    count = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    area = (last_x - first_x + 1) * (last_y - first_y + 1)
    return count == area


def coverage(
    source_grid: PixelGrid,
    target_grid: PixelGrid,
    margin: float = 0.0,
    *,
    reconstruction: str | None = None,
    measure: Measure | None = None,
    valid=None,
):
    """Boolean (ny, nx) of target pixels wholly inside the source footprint.

    A target pixel is covered when all four corners of its footprint map into
    the source footprint shrunk by a margin in source pixels on every side, so
    a pixel that straddles the edge is not covered. Orientation and basis are
    handled as in resample. The result is static geometry: a host array for
    host grids and a traced array when a basis is traced.

    The margin is either given or derived from the read. Pass ``margin``
    (for example ``reconstruction_margin(name)``) to exclude the pixels whose
    sampled value depends on the zero beyond the data. Pass ``reconstruction``
    and ``measure`` together for the margin of the read ``resample`` performs
    on these grids: the reconstruction margin when the read samples, and the
    larger reach of the deposit form when a cell integral lands on pixels at
    least as large as the cells, computed per source axis from the pixel's
    extent along it (so it holds under rotation and for non-square pixels).
    The deposit margin is ``max(m, (k - 1/2) g - 1/2)`` cells for a pixel of g
    cells and kernel half-support k in pixels (1/2 nearest, 1 linear, 2
    cubic). The cubic spline and band-limited reconstructions have no finite
    reach, so their margin marks the support boundary as in
    ``reconstruction_margin``; they never deposit, so their margin is the
    sampling margin for every measure. ``on_grid`` reads every density by the
    footprint integral of its reconstruction, over the whole pixel (a
    band-limited one through the box integral of ``resample``), so pass the
    native density measure for that read: its margin is the sampling margin
    measured from the pixel corners, which is what ``coverage`` tests. The
    integral measure of the ``on_grid`` output gives the deposit rule for
    nearest, linear and cubic, which is never too small and may be a ring too
    wide. An intensive map is sampled at the pixel centers, and its margin is
    the sampling margin.

    ``valid`` narrows the footprint to the source cells that hold data, for a
    product whose array is wider than its data (zero-filled where ``valid`` is
    False). A target pixel is then covered only when, in addition, no cell
    without data lies within the margin of its footprint: the bounding box of
    its four mapped corners, grown by the margin on each source axis, touches
    only cells where ``valid`` is True. For nearest, linear and cubic, whose
    kernels have finite reach, the read of a covered pixel is then the read of
    the unmasked product. ``cubic_spline`` and ``band_limited`` have no finite
    reach, so a mask cannot bound their read, and ``valid`` raises for them.

    Args:
        source_grid: Grid of the source footprint.
        target_grid: Grid whose pixels are tested.
        margin: Source pixels by which the source footprint is shrunk on every
            side. Must be zero when ``reconstruction`` and ``measure`` are
            given.
        reconstruction: Reconstruction of the read, given with ``measure``.
        measure: Measure of the product that is read, given with
            ``reconstruction``.
        valid: Optional boolean of ``source_grid.shape``, True where a source
            cell holds data; a host or JAX array. Requires ``reconstruction``
            and ``measure``. None means every cell holds data.

    Returns:
        Boolean array of shape ``target_grid.shape``, True where a target pixel is
        covered.

    Raises:
        ValueError: If only one of ``reconstruction`` and ``measure`` is given,
            or either is given with a nonzero ``margin``; if ``valid`` is given
            without them, with a reconstruction of no finite reach, or with a
            shape other than the source grid's.
    """
    if (reconstruction is None) != (measure is None):
        raise ValueError("pass reconstruction and measure together")
    if reconstruction is not None and margin != 0:
        raise ValueError(
            "margin must be zero when reconstruction and measure are given"
        )
    if valid is not None:
        if reconstruction is None:
            raise ValueError("valid requires reconstruction and measure")
        if reconstruction in _NO_FINITE_REACH:
            raise ValueError(
                f"{reconstruction} has no finite reach, so a data mask cannot "
                "bound its read; use a reconstruction of finite reach (nearest, "
                "linear or cubic) for a masked product"
            )
        if tuple(np.shape(valid)) != tuple(source_grid.shape):
            raise ValueError("valid must have the source grid's shape")
    mapping = _pixel_map(source_grid, target_grid)
    if reconstruction is None:
        margin_x = margin_y = margin
    else:
        margin_x, margin_y = _read_margins(
            source_grid, target_grid, reconstruction, measure, mapping
        )
    traced = _traced(source_grid.basis, target_grid.basis, mapping)
    if valid is not None and not isinstance(valid, np.ndarray):
        traced = traced or _traced(valid)
        if not traced:
            valid = np.asarray(valid)
    xp = jnp if traced else np
    rows, columns = xp.indices(target_grid.shape)
    dx = columns - target_grid.center_x
    dy = rows - target_grid.center_y
    ny, nx = source_grid.shape
    tolerance = 1e-9
    covered = xp.ones(target_grid.shape, dtype=bool)
    xs, ys = [], []
    for corner_x in (-0.5, 0.5):
        for corner_y in (-0.5, 0.5):
            u, v = dx + corner_x, dy + corner_y
            x = mapping[0][0] * u + mapping[0][1] * v + source_grid.center_x
            y = mapping[1][0] * u + mapping[1][1] * v + source_grid.center_y
            covered = (
                covered
                & (x >= -0.5 + margin_x - tolerance)
                & (x <= nx - 0.5 - margin_x + tolerance)
                & (y >= -0.5 + margin_y - tolerance)
                & (y <= ny - 0.5 - margin_y + tolerance)
            )
            xs.append(x)
            ys.append(y)
    if valid is None:
        return covered
    # Cells whose footprint [j - 1/2, j + 1/2] meets the grown bounding box,
    # with the same slack as the corner test.
    xs, ys = xp.stack(xs), xp.stack(ys)
    first_x = xp.floor(xs.min(axis=0) - margin_x + tolerance - 0.5) + 1
    last_x = xp.ceil(xs.max(axis=0) + margin_x - tolerance + 0.5) - 1
    first_y = xp.floor(ys.min(axis=0) - margin_y + tolerance - 0.5) + 1
    last_y = xp.ceil(ys.max(axis=0) + margin_y - tolerance + 0.5) - 1
    return covered & _all_valid(xp, valid, first_x, last_x, first_y, last_y)


def resample(
    image: Image,
    target_grid: PixelGrid,
    *,
    reconstruction: str | None = None,
    pad_factor: float = 2,
) -> Image:
    """Resample in one affine interpolation, respecting centers and full bases.

    Both grids must have the same unit (and equal frames for lod). Convert
    units explicitly first. If both orientations are supplied, coordinates
    pass through the sky plane; if only one is supplied, orientation is unknown
    and must be reconciled explicitly before resampling.

    reconstruction overrides the image's declared reconstruction for this
    operation only. nearest, linear, and cubic use the existing kernels.
    reconstruction and pad_factor must be static when supplied under jit.
    Density and intensive values sample the source reconstruction
    at target pixel centers; values beyond the source array are zero.

    For these three kernels, per-pixel flux uses two forms,
    chosen on the host from the pixel areas. When target pixels are smaller
    than source pixels (refinement) the flux is converted to a density, sampled at
    target centers and multiplied by the target pixel area, which is the
    model of ``transforms.resample_flux`` with explicit centers and bases:
    the result is smooth, and the sum is conserved up to sampling error
    (exactly for an interior source at an integer refinement). When the
    target pixels are as large or larger (coarsening, equal scale, or a
    rotation at equal scale) each source pixel's flux is deposited onto the
    target with the transpose of the interpolation kernel; its partition of
    unity conserves the sum to roundoff whenever the target contains every
    nonzero source pixel's stencil. Cropping loses flux in both forms.
    Use ``integrate_footprints`` to integrate a piecewise-constant field.

    cubic_spline uses the not-a-knot cubic spline of the samples, restricted
    to the footprint [-1/2, N-1/2] on each axis (see ``Image``). Density and
    intensive values sample it at target centers. A cell integral is converted
    to density samples (data / source_pixel_area), and the spline of those is
    integrated over each target pixel's footprint, clipped at the boundary;
    this one form holds at every pixel scale, with no switch to deposition.
    Both bases and the composed map must be diagonal, as for band_limited.

    band_limited uses the interpolant of a zero-padded array:
    data are assumed zero outside the source array. Each padded axis has
    P=ceil(pad_factor*N) samples (default 2; 1 means no padding). Padding
    controls zero-extension accuracy inside the footprint [-1/2,N-1/2] on
    each axis. The interpolant is restricted to that footprint and is exactly
    zero outside it, regardless of the padded period. Point samples at the
    boundary retain the interpolant value; box integrals clip at the boundary.
    Both bases and the composed target-to-source map must be diagonal;
    rotation or shear raises NotImplementedError. Reflections are supported.

    Per-axis derivation, in source-pixel coordinates (sinc(z)=sin(pi*z)/(pi*z)):
    c_k = sum_n d_n exp(-2*pi*i*k*n/P) / P, with FFT-ordered k.
    A_jn = sum_k exp(2*pi*i*k*(t_j-n)/P) / P evaluates the density.
    Intersect [t-w/2,t+w/2] with [-1/2,N-1/2] to get [a,b].
    Let ell=max(b-a,0), m=(a+b)/2; empty intersections contribute zero.
    Integral_a^b exp(2*pi*i*k*u/P) du = ell*exp(2*pi*i*k*m/P)*sinc(k*ell/P).
    Thus B_jn=sum_k (ell/w)*sinc(k*ell/P)*exp(2*pi*i*k*(m-n)/P)/P.
    For densities and intensive values return A_y @ data @ A_x.T at target centers.
    For cell integrals, density samples are data / source_pixel_area; integrating
    yields target_pixel_area/source_pixel_area * B_y @ data @ B_x.T.
    Real inputs use the cosine Nyquist convention separately on each even
    axis; complex inputs retain the FFT-ordered signed Nyquist mode.

    Matrices are built on the host from static geometry, in the input's real
    and complex precision, then applied with JAX; integer inputs are promoted
    to floating point. float32 has a rounding floor (typically 1e-5 for small
    grids, increasing with phase arguments and matrix size); use float64 for
    precision photometry. No positivity clamp or flux renormalization is used.
    Footprint clipping creates a value jump unless the interpolant vanishes
    at the edge: the restricted field is no longer globally band-limited.
    A well-sampled Gaussian with negligible edges makes this cost small; for
    sigma=0.8, center=(1,0), a 24x24 grid at 0.5 has its nearest edge 6.25 sigma
    away (analytic edge/peak <3.3e-9, omitted Gaussian mass <2.1e-10).
    With pad_factor=2 in float64, the reconstructed jump at (6,0) is about
    -5.15e-7 of peak; the clipped integral differs from the sample sum by
    3.58e-9 relative. float32 adds its rounding floor.
    This is not a bound for undersampled PSFs or bright/truncated wings:
    their edge jump and discarded interpolant tails can appreciably change
    flux. Total flux is the integral over the footprint, not in general the
    source sample sum. Coarsening aliases the density, so subsequent
    reconstruction need not conserve flux.

    The output retains the source reconstruction,
    regardless of an override, except that a band_limited source becomes linear
    when the target Nyquist region cannot contain its source frequency box.
    If p_source=M*p_target+b,
    target frequencies are M.T*f_source; the condition is each column's
    absolute sum <= 1 (for a diagonal M, each absolute spacing ratio <= 1).
    Coarsening either axis fails this guarantee even if pixel area decreases.

    Args:
        image: Image to resample.
        target_grid: Grid of the result, in the same unit as ``image.grid``.
        reconstruction: Reconstruction for this operation; the Image's own when
            None.
        pad_factor: Padding factor of the band-limited zero extension, at least
            one; ignored for other reconstructions.

    Returns:
        The Image on ``target_grid``.
    """
    source_grid = image.grid
    selected = image.reconstruction if reconstruction is None else reconstruction
    _validate_reconstruction(selected)
    mapping = _pixel_map(source_grid, target_grid)
    traced = _traced(source_grid.basis, target_grid.basis)
    output_reconstruction = image.reconstruction
    # A traced scale cannot pick the output reconstruction on the host; the
    # source reconstruction is kept.
    if (
        output_reconstruction == "band_limited"
        and not traced
        and np.any(np.sum(np.abs(mapping), axis=0) > 1 + 32 * np.finfo(float).eps)
    ):
        output_reconstruction = "linear"
    metadata = (
        target_grid,
        image.leading_axes,
        image.measure,
        image.quantity,
        output_reconstruction,
    )
    integral = image.measure.kind == "integral"
    area_ratio = target_grid.pixel_area / source_grid.pixel_area
    deposit = _deposits(source_grid, target_grid, image.measure, selected)
    data = jnp.asarray(image.data, dtype=jnp.result_type(image.data, 1.0))
    if selected == "cubic_spline":
        _require_axis_aligned(selected, source_grid, target_grid, mapping)
        wx, wy = (
            jnp.asarray(
                _cubic_spline_matrix(n, positions, width, data.real.dtype, integral),
                dtype=data.dtype,
            )
            for n, positions, width in _axis_positions(
                source_grid, target_grid, mapping
            )
        )
        return Image.tree_unflatten(metadata, (wy @ data @ wx.T,))
    if selected == "band_limited":
        _require_axis_aligned(selected, source_grid, target_grid, mapping)
        pad_factor = float(pad_factor)
        if not math.isfinite(pad_factor) or pad_factor < 1:
            raise ValueError("pad_factor must be finite and at least 1")
        matrices = []
        for n_source, positions, width in _axis_positions(
            source_grid, target_grid, mapping
        ):
            matrix = _band_limited_matrix(
                n_source,
                positions,
                width,
                pad_factor,
                data.real.dtype,
                integral,
            )
            # A real cosine choice on EACH axis avoids a mixed Nyquist term.
            if not jnp.iscomplexobj(data):
                matrix = matrix.real
            matrices.append(jnp.asarray(matrix, dtype=data.dtype))
        wx, wy = matrices
        data = wy @ data @ wx.T
        if integral:
            data = data * area_ratio
        return Image.tree_unflatten(metadata, (data,))
    order = {"nearest": 0, "linear": 1, "cubic": 3}[selected]

    def kernel(deposit_flux):
        scale = area_ratio if integral and not deposit_flux else 1.0
        # Sampling maps target centers into source pixels; deposition maps
        # source centers into target pixels and applies the kernel's transpose.
        from_grid, to_grid = (
            (source_grid, target_grid) if deposit_flux else (target_grid, source_grid)
        )
        x, y = from_grid.coordinates()
        if not _same_orientation(from_grid.orientation, to_grid.orientation):
            east, north = from_grid.orientation.detector_to_sky(x, y)
            x, y = to_grid.orientation.sky_to_detector(east, north)
        pixels = to_grid.unit_to_pixel(x, y)

        def interpolate(plane):
            if deposit_flux:

                def sample_target(target):
                    return map_coordinates(target, (pixels.y, pixels.x), order=order)

                _, deposit = jax.vjp(
                    sample_target, jnp.zeros(target_grid.shape, dtype=plane.dtype)
                )
                return deposit(plane)[0]
            return map_coordinates(
                plane,
                (pixels.y, pixels.x),
                order=order,
                mode="constant",
            )

        for _ in range(data.ndim - 2):
            interpolate = jax.vmap(interpolate)
        return interpolate(data) * scale

    if integral:
        if traced:
            # The host cannot compare traced pixel areas: build both forms and
            # keep the one the areas select.
            data = jnp.where(deposit, kernel(True), kernel(False))
        else:
            data = kernel(deposit)
    else:
        data = kernel(False)
    return Image.tree_unflatten(metadata, (data,))


def _clip(polygon, axis, bound, sign):
    """Clip a convex polygon to sign * (coordinate - bound) >= 0."""
    if not polygon:
        return []
    clipped = []
    previous = polygon[-1]
    prev_distance = sign * (previous[axis] - bound)
    for current in polygon:
        distance = sign * (current[axis] - bound)
        if (distance >= 0) != (prev_distance >= 0):
            fraction = prev_distance / (prev_distance - distance)
            crossing = [
                previous[k] + fraction * (current[k] - previous[k]) for k in (0, 1)
            ]
            crossing[axis] = bound
            clipped.append(tuple(crossing))
        if distance >= 0:
            clipped.append(current)
        previous, prev_distance = current, distance
    return clipped


def _polygon_overlaps(source, target, mapping):
    """Yield target index, source index and fractional area in source pixels."""
    ny, nx = source.shape
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    offsets = (corners - [target.center_x, target.center_y]) @ mapping.T
    offsets += [source.center_x, source.center_y]
    for row, col in np.ndindex(target.shape):
        polygon = offsets + mapping @ [col, row]
        lo_x, lo_y = np.floor(polygon.min(axis=0) + 0.5).astype(int)
        hi_x, hi_y = np.ceil(polygon.max(axis=0) + 0.5).astype(int)
        for y in range(max(0, lo_y), min(ny, hi_y)):
            for x in range(max(0, lo_x), min(nx, hi_x)):
                # Clipping in local coordinates keeps the shoelace sum accurate.
                clipped = [tuple(p) for p in polygon - [x, y]]
                for axis in (0, 1):
                    clipped = _clip(clipped, axis, -0.5, 1)
                    clipped = _clip(clipped, axis, 0.5, -1)
                if len(clipped) >= 3:
                    area = (
                        abs(
                            math.fsum(
                                p[0] * q[1] - p[1] * q[0]
                                for p, q in zip(
                                    clipped, clipped[1:] + clipped[:1], strict=True
                                )
                            )
                        )
                        / 2
                    )
                    if area:
                        yield row * target.shape[1] + col, y * nx + x, area


def _axis_overlaps(n_source, n_target, source_center, target_center, step):
    """Lengths of overlapping intervals measured in source pixels."""
    src = np.arange(n_source)
    edges = (np.arange(n_target + 1) - 0.5 - target_center) * step + source_center
    left = np.minimum(edges[1:], edges[:-1])
    right = np.maximum(edges[1:], edges[:-1])
    return np.maximum(
        0, np.minimum(right[:, None], src + 0.5) - np.maximum(left[:, None], src - 0.5)
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class CellOverlap:
    """Reusable overlap operator with static geometry and numeric weight leaves.

    Construct with footprint_overlap, which stores the weights as strong JAX
    leaves (indices int32, fractions in the active float dtype). The matrix
    shape is (cells, source pixels).

    Attributes:
        source: Source grid.
        target: Target grid, or None for a sequence of grids.
        shape: Matrix shape (cells, source pixels).
        flat_output: Whether the result has a flat cell axis.
        blocks: Block factors (by, bx) for aligned tiles, or None.
        axes: Pair of one-dimensional overlap matrices for a diagonal map, or None.
        sparse: Triples of cell index, source pixel index and covered fraction, or
            None.
    """

    source: PixelGrid = field(metadata={"static": True})
    target: PixelGrid | None = field(metadata={"static": True})
    shape: tuple[int, int] = field(metadata={"static": True})
    flat_output: bool = field(metadata={"static": True})
    # Aligned tiles use block sums. Other diagonal maps use two 1D matrices.
    # Affine or disjoint grids use sparse (cell, source pixel, fraction) triples.
    blocks: tuple[int, int] | None = field(metadata={"static": True}, default=None)
    axes: tuple | None = None
    sparse: tuple | None = None


def footprint_overlap(
    source: PixelGrid, cells: Sequence[PixelGrid] | PixelGrid
) -> CellOverlap:
    """Compute exact overlaps on the host for static grids in a common frame.

    A single grid produces a gridded result. A sequence concatenates the cells
    of its grids in row-major order. Overlapping cells collect independently.

    Args:
        source: Source grid.
        cells: Target grid, or a sequence of grids whose cells are concatenated.

    Returns:
        The reusable overlap operator.
    """
    flat = not isinstance(cells, PixelGrid)
    targets = tuple(cells) if flat else (cells,)
    shape = (sum(math.prod(t.shape) for t in targets), math.prod(source.shape))
    target = targets[0] if len(targets) == 1 else None
    offset = 0
    rows, cols, fractions = [], [], []
    for grid in targets:
        mapping = _pixel_map(source, grid)
        if target is not None:
            # Compare stored bases directly. Recovering the integer ratio by
            # division can miss exact block alignment at decimal plate scales.
            by, bx = (n // m for n, m in zip(source.shape, grid.shape, strict=True))
            if (
                source.orientation == grid.orientation
                and (by * grid.shape[0], bx * grid.shape[1]) == source.shape
                and (grid.center_x + 0.5) * bx == source.center_x + 0.5
                and (grid.center_y + 0.5) * by == source.center_y + 0.5
                and np.array_equal(np.asarray(source.basis) * [bx, by], grid.basis)
            ):
                return CellOverlap(source, target, shape, flat, blocks=(by, bx))
        if mapping[0, 1] == 0 and mapping[1, 0] == 0:
            sx, sy = np.diag(mapping)
            wx = _axis_overlaps(
                source.shape[1], grid.shape[1], source.center_x, grid.center_x, sx
            )
            wy = _axis_overlaps(
                source.shape[0], grid.shape[0], source.center_y, grid.center_y, sy
            )
            if target is not None:
                return CellOverlap(
                    source, target, shape, flat, axes=(as_leaf(wy), as_leaf(wx))
                )
            # Form only products of nonzero intervals for a sequence of grids.
            ty, y = np.nonzero(wy)
            tx, x = np.nonzero(wx)
            rows.extend((offset + ty[:, None] * grid.shape[1] + tx).ravel())
            cols.extend((y[:, None] * source.shape[1] + x).ravel())
            fractions.extend((wy[ty, y, None] * wx[tx, x]).ravel())
        else:
            for row, col, area in _polygon_overlaps(source, grid, mapping):
                rows.append(offset + row)
                cols.append(col)
                fractions.append(area)
        offset += math.prod(grid.shape)
    sparse = (
        as_leaf(np.asarray(rows, dtype=np.int32)),
        as_leaf(np.asarray(cols, dtype=np.int32)),
        as_leaf(np.asarray(fractions, dtype=np.float64)),
    )
    return CellOverlap(source, target, shape, flat, sparse=sparse)


def integrate_footprints(image: Image, overlap: CellOverlap) -> Image | jax.Array:
    """Collect integral or density samples as a piecewise-constant field.

    A target pixel receives the covered fraction of each source integral.
    Outside the source, the field is zero. Leading dimensions are preserved.
    A sequence of grids returns a final, flat cell axis.

    Args:
        image: Integral or density Image on the overlap's source grid.
        overlap: Operator from ``footprint_overlap``.

    Returns:
        Integral Image on the target grid, or a plain array with a flat cell axis
        for a sequence of grids.
    """
    if image.grid != overlap.source:
        raise ValueError("image grid must match the overlap source grid")
    area = image._integration_area()
    data = jnp.asarray(image.data, dtype=jnp.result_type(image.data, 1.0))
    leading = data.shape[:-2]
    if overlap.blocks is not None:
        by, bx = overlap.blocks
        ny, nx = overlap.target.shape
        data = data.reshape(*leading, ny, by, nx, bx).sum(axis=(-3, -1))
    elif overlap.axes is not None:
        wy, wx = (jnp.asarray(w, dtype=data.dtype) for w in overlap.axes)
        data = wy @ data @ wx.T
    else:
        rows, cols, fractions = overlap.sparse
        weighted = data.reshape(*leading, overlap.shape[1])[..., cols] * jnp.asarray(
            fractions, dtype=data.dtype
        )
        data = jnp.zeros((*leading, overlap.shape[0]), dtype=data.dtype)
        data = data.at[..., rows].add(weighted)
    if area is not None:
        data = data * area
    if overlap.flat_output:
        return data.reshape(*leading, overlap.shape[0])
    metadata = (
        overlap.target,
        image.leading_axes,
        Measure.integral(),
        image.quantity,
        "nearest",
    )
    return Image.tree_unflatten(
        metadata, (data.reshape(*leading, *overlap.target.shape),)
    )
