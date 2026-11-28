"""Measured detector images, their exposures and aligned host records."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import mas_to_arcsec
from hwoutils.leaves import as_bulk_leaf, is_traced
from hwoutils.precision import like
from hwoutils.provenance import Stamps

from optixstuff.exposure import Exposure
from optixstuff.grids import Image, SkyOrientation

ELECTRON_COUNT_QUANTITIES = ("electron", "electrons")
"""Image quantity spellings that name electron counts."""

_AXES = ("draw", "event")
_LEADING_AXES = (("band",), ("event", "band"), ("draw", "band"), (*_AXES, "band"))


class ExposureData(eqx.Module):
    """Measured detector values, their quality arrays and the exposure.

    Attributes:
        image: Values with axes drawn from (draw, event, band, y, x). Floating
            values are standard. Integral electron counts may be stored as
            integers when the producer omits variance.
        variance: Per-pixel variance of the stored values, in the square of
            the image quantity, including every noise term the producer models.
            None delegates the variance to the consumer's noise model.
        valid: Boolean usability mask with the same shape as the image.
        flags: Optional unsigned quality bits, one value per pixel.
        exposure: Exposure, batched only when the image has an event axis.
        flag_names: Names of the flag bits in order from least significant.
        invalid_flags: Names of the bits that make a pixel unusable.
        built_at: Static build facts made with `hwoutils.provenance.build_stamps`.
    """

    image: Image
    variance: jax.Array | None
    valid: jax.Array
    flags: jax.Array | None
    exposure: Exposure
    flag_names: tuple[str, ...] = eqx.field(static=True, default=())
    invalid_flags: tuple[str, ...] = eqx.field(static=True, default=())
    built_at: Stamps = eqx.field(static=True, default=())

    def __init__(
        self,
        image,
        exposure,
        *,
        variance=None,
        valid=None,
        flags=None,
        flag_names=(),
        invalid_flags=(),
        built_at=(),
    ):
        """Store the image, the exposure and aligned quality arrays.

        Args:
            image: Band image without orientation on its grid.
            exposure: Exposure matching the image's bands and events.
            variance: Optional array of pixel variances at full image shape.
            valid: Optional boolean mask at pixel or full image shape.
            flags: Optional unsigned flag array at full image shape.
            flag_names: Bit names in order from least significant.
            invalid_flags: Flag names that invalidate a pixel.
            built_at: Static build facts shared by all draws and events.
        """
        pixels = image.data
        if not isinstance(pixels, jax.Array) or getattr(pixels, "weak_type", False):
            image = replace(image, data=as_bulk_leaf(pixels))
        self.image = image
        self.exposure = exposure
        self.variance = (
            None if variance is None else jnp.asarray(variance, dtype=image.data.dtype)
        )
        self.flag_names = tuple(flag_names)
        self.invalid_flags = tuple(invalid_flags)
        self.built_at = tuple(built_at)
        self.flags = None if flags is None else jnp.asarray(flags)
        if (
            self.flags is not None
            and getattr(flags, "dtype", self.flags.dtype) != self.flags.dtype
        ):
            raise ValueError("flags dtype must be enabled in JAX")

        if valid is not None:
            valid = jnp.asarray(valid)
            if valid.shape == image.grid.shape:
                valid = jnp.broadcast_to(valid, image.data.shape)
            if valid.shape != image.data.shape or valid.dtype != jnp.bool_:
                raise ValueError("valid must be boolean at pixel or full image shape")
        if self.flags is None:
            self.valid = (
                jnp.ones(image.data.shape, dtype=jnp.bool_) if valid is None else valid
            )
        elif valid is None or is_traced(valid) or is_traced(self.flags):
            self.valid = self._flag_valid()
        else:
            self.valid = valid

    def _flag_mask(self) -> int:
        """Validate the named bit policy and return its integer mask."""
        if self.flags is None:
            if self.flag_names or self.invalid_flags:
                raise ValueError("flag_names and invalid_flags require flags")
            return 0
        if not jnp.issubdtype(self.flags.dtype, jnp.unsignedinteger):
            raise ValueError("flags must have an unsigned integer dtype")
        for name, names in (
            ("flag_names", self.flag_names),
            ("invalid_flags", self.invalid_flags),
        ):
            if any(not isinstance(bit, str) or not bit for bit in names) or len(
                set(names)
            ) != len(names):
                raise ValueError(f"{name} must contain unique nonempty names")
        if len(self.flag_names) > jnp.iinfo(self.flags.dtype).bits:
            raise ValueError("flag_names exceeds the number of bits in flags dtype")
        if not set(self.invalid_flags).issubset(self.flag_names):
            raise ValueError("invalid_flags must name bits in flag_names")
        return sum(
            1 << i
            for i, name in enumerate(self.flag_names)
            if name in self.invalid_flags
        )

    def _flag_valid(self) -> jax.Array:
        """Evaluate the configured flag policy for every pixel."""
        mask = jnp.asarray(self._flag_mask(), dtype=self.flags.dtype)
        return (self.flags & mask) == 0

    def __check_init__(self):
        """Enforce array geometry and quality rules at construction."""
        image = self.image
        if image.grid.orientation is not None:
            raise ValueError(
                "image.grid.orientation must be None; roll lives on exposure"
            )
        if image.leading_axes not in _LEADING_AXES:
            raise ValueError(
                "image.leading_axes must be ordered as (draw, event, band)"
            )
        if image.data.shape[-3] != len(self.exposure.bands):
            raise ValueError("image band axis must match exposure.bands")
        if not jnp.issubdtype(image.data.dtype, jnp.floating) and not (
            jnp.issubdtype(image.data.dtype, jnp.integer)
            and image.quantity in ELECTRON_COUNT_QUANTITIES
            and image.measure.kind == "integral"
            and self.variance is None
        ):
            raise ValueError(
                "image.data must be floating unless it holds integral electron "
                "counts with variance=None"
            )
        for field in ("variance", "valid", "flags"):
            array = getattr(self, field)
            if array is not None and array.shape != image.data.shape:
                raise ValueError(f"{field} shape must match image.data")
        if self.valid.dtype != jnp.bool_:
            raise ValueError("valid must have boolean dtype")
        if "event" in image.leading_axes:
            n_event = image.data.shape[image.leading_axes.index("event")]
            if any(
                leaf.shape[:1] != (n_event,) for leaf in jax.tree.leaves(self.exposure)
            ):
                raise ValueError("exposure leaves must match the leading event axis")
        mask = self._flag_mask()
        if self.flags is not None and not (
            is_traced(self.flags) or is_traced(self.valid)
        ):
            if not np.array_equal(self.valid, (np.asarray(self.flags) & mask) == 0):
                raise ValueError(
                    "valid disagrees with the flags and invalid_flags policy"
                )

    def _arcsec_grid(self) -> None:
        """Require detector coordinates in arcseconds."""
        if self.image.grid.unit != "arcsec":
            raise ValueError("coordinate methods require image.grid.unit='arcsec'")

    def _sky_grid(self) -> None:
        """Require one pointing for the sky transform."""
        if "event" in self.image.leading_axes:
            raise ValueError(
                "use take(index, axis='event') before mapping sky coordinates"
            )
        self._arcsec_grid()

    def target_px(self) -> jax.Array:
        """Locate the commanded target on the detector.

        Returns:
            A pixel (x, y) pair, or one pair per stacked event.
        """
        self._arcsec_grid()
        grid = self.image.grid
        offset = self.exposure.pointing.offset_mas
        if offset is not None:
            offset = mas_to_arcsec(offset)
            return jnp.stack(
                grid.unit_to_pixel(offset[..., 0], offset[..., 1]), axis=-1
            )
        center = like((grid.center_x, grid.center_y), self.exposure.pointing.roll_deg)
        return jnp.broadcast_to(center, (*self.exposure.pointing.roll_deg.shape, 2))

    def sky_to_pixel(self, east_arcsec, north_arcsec) -> tuple[jax.Array, jax.Array]:
        """Map target-relative sky coordinates to detector pixel centers.

        Args:
            east_arcsec: East of the target in arcseconds.
            north_arcsec: North of the target in arcseconds.

        Returns:
            Pixel column and row coordinates.
        """
        self._sky_grid()
        offsets = self.exposure.pointing.detector_offsets_arcsec(
            east_arcsec, north_arcsec
        )
        return self.image.grid.unit_to_pixel(*offsets)

    def pixel_to_sky(self, x_px, y_px) -> tuple[jax.Array, jax.Array]:
        """Map detector pixel centers to target-relative sky coordinates.

        Args:
            x_px: Pixel column coordinate.
            y_px: Pixel row coordinate.

        Returns:
            East and north of the target in arcseconds.
        """
        self._sky_grid()
        x, y = self.image.grid.pixel_to_unit(x_px, y_px)
        pointing = self.exposure.pointing
        if pointing.offset_mas is not None:
            dx, dy = mas_to_arcsec(pointing.offset_mas)
            x, y = x - dx, y - dy
        return SkyOrientation(pointing.roll_deg).detector_to_sky(x, y)

    def take(self, index: int, *, axis: str = "event") -> "ExposureData":
        """Select an event or draw and remove that axis from the image.

        Args:
            index: Position on the named axis, with negative indices from the end.
            axis: An existing `"event"` or `"draw"` axis.

        Returns:
            The selected measurement.
        """
        leading = self.image.leading_axes
        if axis not in _AXES or axis not in leading:
            raise ValueError(
                f"image.leading_axes has no event or draw axis named {axis!r}"
            )
        position = leading.index(axis)
        index = range(self.image.data.shape[position])[index]
        selection = (slice(None),) * position + (index,)
        image = replace(
            self.image,
            data=self.image.data[selection],
            leading_axes=leading[:position] + leading[position + 1 :],
        )
        exposure = (
            jax.tree.map(lambda leaf: leaf[index], self.exposure)
            if axis == "event"
            else self.exposure
        )
        return ExposureData(
            image,
            exposure,
            variance=None if self.variance is None else self.variance[selection],
            valid=self.valid[selection],
            flags=None if self.flags is None else self.flags[selection],
            flag_names=self.flag_names,
            invalid_flags=self.invalid_flags,
            built_at=self.built_at,
        )


def _require_same_exposure_layout(
    first: Exposure, current: Exposure, item: int
) -> None:
    """Check exposure metadata and array layout without comparing leaf values."""
    for field, left, right in (
        ("bands", first.bands, current.bands),
        ("instrument", first.instrument, current.instrument),
        ("epoch", first.epoch, current.epoch),
        ("plan presence", first.plan is None, current.plan is None),
        (
            "offset_mas presence",
            first.pointing.offset_mas is None,
            current.pointing.offset_mas is None,
        ),
    ):
        if left != right:
            raise ValueError(f"item {item}: exposure {field} differs")
    left_leaves, left_structure = jax.tree.flatten(first)
    right_leaves, right_structure = jax.tree.flatten(current)
    if left_structure != right_structure:
        raise ValueError(f"item {item}: exposure pytree structure differs")
    for left, right in zip(left_leaves, right_leaves, strict=True):
        if left.shape != right.shape:
            raise ValueError(f"item {item}: exposure leaf shape differs")
        if left.dtype != right.dtype:
            raise ValueError(f"item {item}: exposure leaf dtype differs")


def _require_homogeneous(first: ExposureData, current: ExposureData, item: int) -> None:
    """Check that two measurements have the same image and quality layout."""
    for field in ("grid", "measure", "quantity", "reconstruction", "leading_axes"):
        if getattr(first.image, field) != getattr(current.image, field):
            raise ValueError(f"item {item}: image.{field} differs")
    for field in ("shape", "dtype"):
        if getattr(first.image.data, field) != getattr(current.image.data, field):
            raise ValueError(f"item {item}: image.data {field} differs")
    for field in ("variance", "flags"):
        if (getattr(first, field) is None) != (getattr(current, field) is None):
            raise ValueError(f"item {item}: {field} presence differs")
    for field in ("flag_names", "invalid_flags", "built_at"):
        if getattr(first, field) != getattr(current, field):
            raise ValueError(f"item {item}: {field} differs")
    if first.flags is not None and first.flags.dtype != current.flags.dtype:
        raise ValueError(f"item {item}: flags dtype differs")
    _require_same_exposure_layout(first.exposure, current.exposure, item)


def stack_exposure_data(
    items: Sequence[ExposureData], *, axis: str = "event"
) -> ExposureData:
    """Stack compatible measurements on an outer event or draw axis.

    Args:
        items: Nonempty sequence with matching image, quality and exposure layouts.
            Draws must also share the same exposure values.
        axis: `"event"` for distinct exposures or `"draw"` for repeat measurements.

    Returns:
        Measurements with the new axis. Event stacks batch the exposure leaves.
    """
    if not items:
        raise ValueError("items must contain at least one measurement")
    if axis not in _AXES:
        raise ValueError("item 0: axis must be 'event' or 'draw'")
    first = items[0]
    for item, current in enumerate(items):
        leading = current.image.leading_axes
        if axis in leading or (axis == "event" and "draw" in leading):
            raise ValueError(
                f"item {item}: cannot stack {axis!r} over image.leading_axes {leading}"
            )
        if item:
            _require_homogeneous(first, current, item)
            if axis == "draw":
                equal = eqx.tree_equal(first.exposure, current.exposure)
                if is_traced(equal) or not equal:
                    raise ValueError(
                        f"item {item}: draw exposure must be equal on concrete data"
                    )

    exposure = first.exposure
    if axis == "event":
        exposure = jax.tree.map(
            lambda *leaves: jnp.stack(leaves), *(item.exposure for item in items)
        )
    image = replace(
        first.image,
        data=jnp.stack([item.image.data for item in items]),
        leading_axes=(axis, *first.image.leading_axes),
    )
    return ExposureData(
        image,
        exposure,
        variance=None
        if first.variance is None
        else jnp.stack([item.variance for item in items]),
        valid=jnp.stack([item.valid for item in items]),
        flags=None
        if first.flags is None
        else jnp.stack([item.flags for item in items]),
        flag_names=first.flag_names,
        invalid_flags=first.invalid_flags,
        built_at=first.built_at,
    )


@dataclass(frozen=True)
class HostRecords:
    """Host values aligned with measured events or draws.

    Attributes:
        axis: `"event"` or `"draw"`.
        columns: Ordered mapping from column names to tuples of host values.
    """

    axis: str
    columns: Mapping[str, tuple]

    def __post_init__(self):
        """Store immutable columns, requiring one host value per row."""
        if self.axis not in _AXES:
            raise ValueError("axis must be 'event' or 'draw'")
        columns = {name: tuple(values) for name, values in self.columns.items()}
        if len({len(values) for values in columns.values()}) > 1:
            raise ValueError("columns must have equal lengths")
        if any(not isinstance(name, str) for name in columns):
            raise ValueError("columns must have string names")
        if any(
            value is not None and not isinstance(value, (str, int, float, bool))
            for values in columns.values()
            for value in values
        ):
            raise ValueError(
                "columns must contain host strings, numbers, booleans or None"
            )
        object.__setattr__(self, "columns", MappingProxyType(columns))

    def __len__(self) -> int:
        """Count the rows shared by the columns.

        Returns:
            The number of rows, zero for a table without columns.
        """
        return len(next(iter(self.columns.values()), ()))

    def row(self, index: int) -> dict:
        """Read one row as a dictionary.

        Args:
            index: Row position, with negative indices from the end.

        Returns:
            The values at this row, in column order.
        """
        index = range(len(self))[index]
        return {name: values[index] for name, values in self.columns.items()}

    @classmethod
    def concatenate(cls, records: Sequence["HostRecords"]) -> "HostRecords":
        """Join tables of the same axis and ordered columns.

        Args:
            records: Nonempty sequence of compatible tables.

        Returns:
            A table with rows in input order.
        """
        if not records:
            raise ValueError("records must contain at least one table")
        first = records[0]
        names = tuple(first.columns)
        for index, table in enumerate(records[1:], start=1):
            if table.axis != first.axis:
                raise ValueError(f"records item {index}: axis differs")
            if tuple(table.columns) != names:
                raise ValueError(f"records item {index}: columns differ")
        return cls(
            first.axis,
            {
                name: tuple(value for table in records for value in table.columns[name])
                for name in names
            },
        )

    def require_aligned(self, data: ExposureData) -> None:
        """Require one row per item on this axis of the measurement.

        Args:
            data: The measurement the rows describe.
        """
        axes = data.image.leading_axes
        size = data.image.data.shape[axes.index(self.axis)] if self.axis in axes else 1
        if len(self) != size:
            raise ValueError(
                f"{self.axis} axis has {size} items but records has {len(self)} rows"
            )
