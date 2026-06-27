"""Primary mirror abstractions."""

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar


class AbstractPrimary(eqx.Module):
    """Abstract interface for a primary aperture.

    Any concrete implementation must provide the diameter and collecting
    area of the primary mirror as scalar values in SI units. These are
    consumed by exposure time calculators and simulation tools alike.
    """

    diameter_m: AbstractVar[float]
    """Primary mirror diameter in metres."""

    area_m2: AbstractVar[float]
    """Effective collecting area in square metres."""


class SimplePrimary(AbstractPrimary):
    """A simple circular primary mirror with a central obscuration.

    Args:
        diameter_m: Primary mirror diameter in metres.
        obscuration: Linear obscuration fraction (0 = no obscuration).
        shape_factor: Fraction of unobscured area that is collecting
            (accounts for struts, segment gaps, etc.). Default 1.0.
    """

    _diameter_m: float
    obscuration: float
    shape_factor: float

    def __init__(
        self,
        diameter_m: float,
        obscuration: float = 0.0,
        shape_factor: float = 1.0,
    ) -> None:
        """Create a simple circular primary mirror."""
        self._diameter_m = diameter_m
        self.obscuration = obscuration
        self.shape_factor = shape_factor

    @property
    def diameter_m(self) -> float:
        """Primary mirror diameter in metres."""
        return self._diameter_m

    @property
    def area_m2(self) -> float:
        """Effective collecting area in square metres."""
        r = self._diameter_m / 2.0
        gross_area = jnp.pi * r**2
        return gross_area * (1.0 - self.obscuration**2) * self.shape_factor

    def __repr__(self) -> str:
        """One-line summary of diameter, obscuration, and effective area."""
        return (
            f"SimplePrimary(D={self._diameter_m:.3g} m, "
            f"obs={self.obscuration:.3g}, "
            f"shape_factor={self.shape_factor:.3g}, "
            f"area={float(self.area_m2):.3g} m^2)"
        )


def _hex_axial_coords(n_rings: int) -> list[tuple[int, int]]:
    """Axial (q, r) centres of a hex-packed n-ring layout.

    Yields ``1 + 3 n (n + 1)`` coordinates, centre first.
    """
    coords = [(0, 0)]
    for q in range(-n_rings, n_rings + 1):
        for r in range(-n_rings, n_rings + 1):
            if max(abs(q), abs(r), abs(q + r)) <= n_rings and not (q == 0 and r == 0):
                coords.append((q, r))
    return coords


class SegmentedPrimary(AbstractPrimary):
    """A segmented hex primary that carries pupil geometry, not just scalars.

    Beyond the diameter and collecting area every primary provides, this
    describes the segment layout -- ring count, segment count, gap, shape -- so a
    diffraction backend (e.g. dLux) can build the pupil from it. optixstuff only
    DESCRIBES the geometry; it does not propagate wavefronts.

    Args:
        diameter_m: Circumscribing diameter in metres.
        area_m2: Effective collecting area in square metres (gap/fill corrected).
        n_rings: Number of segment rings around the centre segment.
        n_segments: Total segment count (``1 + 3 n (n + 1)`` for a full layout).
        segment_gap_m: Inter-segment optical gap in metres.
        segment_shape: Segment shape; only ``"hexagon"`` is supported today.
    """

    _diameter_m: float
    _area_m2: float
    segment_gap_m: float
    n_rings: int = eqx.field(static=True)
    n_segments: int = eqx.field(static=True)
    segment_shape: str = eqx.field(static=True)

    def __init__(
        self,
        diameter_m: float,
        area_m2: float,
        n_rings: int,
        n_segments: int,
        segment_gap_m: float,
        segment_shape: str = "hexagon",
    ) -> None:
        """Create a segmented hex primary from its geometry."""
        self._diameter_m = diameter_m
        self._area_m2 = area_m2
        self.segment_gap_m = segment_gap_m
        self.n_rings = n_rings
        self.n_segments = n_segments
        self.segment_shape = segment_shape

    @property
    def diameter_m(self) -> float:
        """Circumscribing diameter in metres."""
        return self._diameter_m

    @property
    def area_m2(self) -> float:
        """Effective collecting area in square metres."""
        return self._area_m2

    @property
    def segment_flat_to_flat_m(self) -> float:
        """Segment flat-to-flat size in metres for the circumscribing layout."""
        return self._diameter_m / (2 * self.n_rings + 1)

    @property
    def segment_centres_m(self):
        """``(n_segments, 2)`` array of segment centre (x, y) positions in metres."""
        if self.segment_shape != "hexagon":
            msg = f"segment_centres_m undefined for shape {self.segment_shape!r}"
            raise NotImplementedError(msg)
        axial = _hex_axial_coords(self.n_rings)
        spacing = self.segment_flat_to_flat_m + self.segment_gap_m
        qs = jnp.array([q for q, _ in axial])
        rs = jnp.array([r for _, r in axial])
        xs = spacing * (qs + rs / 2.0)
        ys = spacing * (jnp.sqrt(3.0) / 2.0) * rs
        return jnp.stack([xs, ys], axis=-1)

    def __repr__(self) -> str:
        """One-line summary of diameter, segment layout, and effective area."""
        return (
            f"SegmentedPrimary(D={self._diameter_m:.3g} m, "
            f"{self.n_segments} {self.segment_shape} segs, "
            f"{self.n_rings} rings, gap={self.segment_gap_m * 1e3:.3g} mm, "
            f"area={float(self._area_m2):.3g} m^2)"
        )
