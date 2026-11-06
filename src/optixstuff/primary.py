"""Primary mirror abstractions."""

from math import sqrt
from pathlib import Path

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
    """Primary mirror diameter in meters."""

    area_m2: AbstractVar[float]
    """Effective collecting area in square meters."""


class SimplePrimary(AbstractPrimary):
    """A simple circular primary mirror with a central obscuration.

    The obscuration is the inner-to-outer diameter ratio, so the annular
    collecting area is ``pi * (diameter_m / 2)**2 * (1 - obscuration**2)``
    times ``shape_factor``. :meth:`from_yip` converts the YIP ``OBSCURED``
    area fraction to this linear ratio by taking its square root.

    Args:
        diameter_m: Primary mirror diameter in meters.
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

    @classmethod
    def from_yip(cls, yip, shape_factor: float = 1.0) -> "SimplePrimary":
        """Build an aperture from YIP diameter and obscured area metadata.

        Args:
            yip: YIP directory or stellar-intensity FITS path, a
                YippyCoronagraph, or a yippy Coronagraph/EqxCoronagraph.
                Objects must retain the header or diameter metadata.
                Paths read only the header, without loading PSF arrays.
            shape_factor: Fraction of remaining area that collects light.

        Returns:
            An aperture with diameter in meters and linear obscuration.

        Raises:
            ValueError: If D or OBSCURED is missing or None. Some yippy
                EqxCoronagraph versions discard D; use the original YIP
                path or yippy Coronagraph in that case.
        """
        from astropy.io import fits
        from yippy.header import HeaderData

        from optixstuff.yippy_coronagraph import YippyCoronagraph

        if isinstance(yip, YippyCoronagraph):
            yip = yip._backend
        if isinstance(yip, (str, Path)):
            path = Path(yip)
            if path.is_dir():
                path = path / "stellar_intens.fits"
            raw_header = fits.getheader(path, 0)
            for key in ("D", "OBSCURED"):
                if raw_header.get(key) is None:
                    raise ValueError(f"YIP header is missing {key}")
            header = HeaderData.from_fits_header(raw_header)
        else:
            header = getattr(yip, "header", None)

        if header is not None:
            diameter = getattr(header, "diameter", None)
            obscured = getattr(header, "obscured", None)
        else:
            diameter = getattr(yip, "diameter_m", None)
            obscured = getattr(yip, "frac_obscured", None)
        if diameter is None:
            raise ValueError(
                "YIP metadata is missing D (diameter); use the original YIP "
                "path or yippy Coronagraph if the backend discarded it"
            )
        if obscured is None:
            raise ValueError(
                "YIP metadata is missing OBSCURED (obscured area fraction)"
            )
        diameter_m = (
            float(diameter.to_value("m"))
            if hasattr(diameter, "to_value")
            else float(diameter)
        )
        return cls(diameter_m, obscuration=sqrt(obscured), shape_factor=shape_factor)

    @property
    def diameter_m(self) -> float:
        """Primary mirror diameter in meters."""
        return self._diameter_m

    @property
    def area_m2(self) -> float:
        """Effective collecting area in square meters."""
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
    """Axial (q, r) centers of a hex-packed n-ring layout.

    Yields ``1 + 3 n (n + 1)`` coordinates, center first.
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
        diameter_m: Circumscribing diameter in meters.
        area_m2: Effective collecting area in square meters (gap/fill corrected).
        n_rings: Number of segment rings around the center segment.
        n_segments: Total segment count (``1 + 3 n (n + 1)`` for a full layout).
        segment_gap_m: Inter-segment optical gap in meters.
        segment_shape: Segment shape; only ``"hexagon"`` is supported today.
    """

    _diameter_m: float
    _area_m2: float
    segment_gap_m: float
    segment_point_to_point_m: float | None
    inscribed_diameter_m: float | None
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
        segment_point_to_point_m: float | None = None,
        inscribed_diameter_m: float | None = None,
    ) -> None:
        """Create a segmented hex primary from its geometry."""
        self._diameter_m = diameter_m
        self._area_m2 = area_m2
        self.segment_gap_m = segment_gap_m
        self.segment_point_to_point_m = segment_point_to_point_m
        self.inscribed_diameter_m = inscribed_diameter_m
        self.n_rings = n_rings
        self.n_segments = n_segments
        self.segment_shape = segment_shape

    @property
    def diameter_m(self) -> float:
        """Circumscribing diameter in meters."""
        return self._diameter_m

    @property
    def area_m2(self) -> float:
        """Effective collecting area in square meters."""
        return self._area_m2

    @property
    def segment_flat_to_flat_m(self) -> float:
        """Segment flat-to-flat size in meters.

        Exact (``point_to_point * sqrt(3) / 2``) when the segment size is
        given; otherwise the legacy circumscribing approximation
        ``diameter / (2 n_rings + 1)``.
        """
        if self.segment_point_to_point_m is not None:
            return self.segment_point_to_point_m * jnp.sqrt(3.0) / 2.0
        return self._diameter_m / (2 * self.n_rings + 1)

    @property
    def segment_pitch_m(self) -> float:
        """Center-to-center distance of adjacent segments (flat-to-flat + gap)."""
        return self.segment_flat_to_flat_m + self.segment_gap_m

    @property
    def segment_centers_m(self):
        """``(n_segments, 2)`` segment center (x, y) positions in meters.

        A flat-top hexagonal lattice (flats up, points along x), the
        convention of the HWO EAC baseline pupils: the center segment is
        first, and adjacent centers are one ``segment_pitch_m`` apart.
        """
        if self.segment_shape != "hexagon":
            msg = f"segment_centers_m undefined for shape {self.segment_shape!r}"
            raise NotImplementedError(msg)
        axial = _hex_axial_coords(self.n_rings)
        pitch = self.segment_pitch_m
        qs = jnp.array([q for q, _ in axial])
        rs = jnp.array([r for _, r in axial])
        xs = pitch * (jnp.sqrt(3.0) / 2.0) * (qs + rs)
        ys = pitch * (rs - qs) / 2.0
        return jnp.stack([xs, ys], axis=-1)

    def __repr__(self) -> str:
        """One-line summary of diameter, segment layout, and effective area."""
        return (
            f"SegmentedPrimary(D={self._diameter_m:.3g} m, "
            f"{self.n_segments} {self.segment_shape} segs, "
            f"{self.n_rings} rings, gap={self.segment_gap_m * 1e3:.3g} mm, "
            f"area={float(self._area_m2):.3g} m^2)"
        )
