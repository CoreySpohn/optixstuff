"""Primary apertures with explicit diameter conventions and collecting areas."""

import abc
from math import pi, sqrt
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.grids import Frame


class AbstractPrimary(eqx.Module):
    """Aperture geometry in metres, independent of its optical consumers."""

    collecting_area_m2: AbstractVar[float]

    @abc.abstractmethod
    def diameter(self, convention: str) -> float:
        """Return metres for static 'circumscribed' or 'inscribed'; else raise."""

    def frame(self, reference_wavelength_nm, convention="circumscribed") -> Frame:
        """Build static lambda/D geometry from this primary's declared diameter."""
        return Frame(reference_wavelength_nm, self.diameter(convention), convention)


class SimplePrimary(AbstractPrimary):
    """An aperture with independently supplied diameters and collecting area.

    All dimensions are metres. For a filled circular aperture only, the
    inscribed diameter may be omitted and defaults to the circumscribed one.
    Area is supplied independently; it never determines either diameter.
    """

    circumscribed_diameter_m: float
    inscribed_diameter_m: float
    collecting_area_m2: float

    def __init__(
        self, *, circumscribed_diameter_m, collecting_area_m2, inscribed_diameter_m=None
    ):
        """Store both diameter conventions and the independently measured area."""
        self.circumscribed_diameter_m = circumscribed_diameter_m
        self.inscribed_diameter_m = (
            circumscribed_diameter_m
            if inscribed_diameter_m is None
            else inscribed_diameter_m
        )
        self.collecting_area_m2 = collecting_area_m2

    def diameter(self, convention):
        """Return the explicitly selected diameter in metres."""
        if convention == "circumscribed":
            return self.circumscribed_diameter_m
        if convention == "inscribed":
            return self.inscribed_diameter_m
        raise ValueError(f"Unknown diameter convention: {convention!r}")

    @classmethod
    def from_yip(cls, yip, shape_factor=1.0):
        """Read D and D_INSC directly; area is pi*D**2/4*(1-OBSCURED)*factor.

        Accept a YIP directory/FITS path or an object retaining its header.
        Backends which discarded D_INSC cannot reconstruct it from an area.
        """
        from astropy.io import fits
        from yippy.header import HeaderData

        from optixstuff.yippy_coronagraph import YippyCoronagraph

        if isinstance(yip, YippyCoronagraph):
            yip = yip._backend
        if isinstance(yip, (str, Path)):
            path = Path(yip)
            raw = fits.getheader(
                path / "stellar_intens.fits" if path.is_dir() else path
            )
            for key in ("D", "D_INSC", "OBSCURED"):
                if raw.get(key) is None:
                    raise ValueError(f"YIP header is missing {key}")
            header = HeaderData.from_fits_header(raw)
        else:
            header = getattr(yip, "header", None)
        values = []
        for key, attr in (
            ("D", "diameter"),
            ("D_INSC", "diameter_inscribed"),
            ("OBSCURED", "obscured"),
        ):
            value = getattr(header, attr, None)
            if value is None:
                raise ValueError(f"YIP metadata is missing {key}; use the original YIP")
            values.append(
                float(value.to_value("m"))
                if hasattr(value, "to_value")
                else float(value)
            )
        circ, insc, obscured = values
        return cls(
            circumscribed_diameter_m=circ,
            inscribed_diameter_m=insc,
            collecting_area_m2=pi * circ**2 / 4 * (1 - obscured) * shape_factor,
        )

    def __repr__(self):
        """Summarize the two diameter conventions and collecting area."""
        return (
            f"SimplePrimary(D_circ={self.circumscribed_diameter_m:.3g} m, "
            f"D_insc={self.inscribed_diameter_m:.3g} m, "
            f"area={self.collecting_area_m2:.3g} m^2)"
        )


def _hex_axial_coords(n_rings):
    """Axial centres of a complete hexagonal layout, centre first."""
    return [(0, 0)] + [
        (q, r)
        for q in range(-n_rings, n_rings + 1)
        for r in range(-n_rings, n_rings + 1)
        if max(abs(q), abs(r), abs(q + r)) <= n_rings and (q, r) != (0, 0)
    ]


class SegmentedPrimary(AbstractPrimary):
    """Complete rings of flat-top hexagons, including the centre segment.

    Write a=segment_point_to_point_m/2, p=sqrt(3)*a+gap and n=n_rings.
    The outer-ring centre at (sqrt(3)*n*p/2, n*p/2) plus vertex (a,0)
    has the maximum radius: D_circ=2*hypot(sqrt(3)*n*p/2+a,n*p/2).
    D_insc is the largest centred circle inside the scalloped outer outline,
    bridging gaps at the perimeter and ignoring internal segment gaps. For
    n>0 its nearest reentrant vertices have x=sqrt(3)*n*p/2+a/2;
    odd rings have a gap bridge at y=0, even rings have a vertex at
    y=sqrt(3)*a/2. Thus D_insc=2*hypot(x,0 or sqrt(3)*a/2).
    For n=0 a single hexagon has D_circ=2*a and D_insc=sqrt(3)*a.
    Area=(1+3*n*(n+1))*6*(sqrt(3)/4)*a**2, the sum of six equilateral
    triangles per segment. Gaps alter pitch and extent, not segment area.
    """

    segment_point_to_point_m: float
    segment_gap_m: float
    collecting_area_m2: float
    circumscribed_diameter_m: float
    inscribed_diameter_m: float
    n_rings: int = eqx.field(static=True)
    n_segments: int = eqx.field(static=True)
    segment_shape: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_rings,
        segment_point_to_point_m,
        segment_gap_m=0.0,
        segment_shape="hexagon",
    ):
        """Derive diameters and area from the full hexagonal segment geometry."""
        if segment_shape != "hexagon":
            raise ValueError("segment_shape must be hexagon")
        if int(n_rings) != n_rings or n_rings < 0:
            raise ValueError("n_rings must be a nonnegative integer")
        if segment_point_to_point_m <= 0 or segment_gap_m < 0:
            raise ValueError("segment size must be positive and gap nonnegative")
        self.n_rings = int(n_rings)
        self.n_segments = 1 + 3 * self.n_rings * (self.n_rings + 1)
        self.segment_shape = segment_shape
        self.segment_point_to_point_m = segment_point_to_point_m
        self.segment_gap_m = segment_gap_m
        a, n = segment_point_to_point_m / 2, self.n_rings
        p = sqrt(3) * a + segment_gap_m
        self.circumscribed_diameter_m = (
            2 * ((sqrt(3) * n * p / 2 + a) ** 2 + (n * p / 2) ** 2) ** 0.5
        )
        self.inscribed_diameter_m = (
            sqrt(3) * a
            if n == 0
            else 2
            * (
                (sqrt(3) * n * p / 2 + a / 2) ** 2
                + (sqrt(3) * a / 2 if n % 2 == 0 else 0) ** 2
            )
            ** 0.5
        )
        self.collecting_area_m2 = self.n_segments * 3 * sqrt(3) / 2 * a**2

    def diameter(self, convention):
        """Return metres under one of the two explicit diameter conventions."""
        if convention == "circumscribed":
            return self.circumscribed_diameter_m
        if convention == "inscribed":
            return self.inscribed_diameter_m
        raise ValueError(f"Unknown diameter convention: {convention!r}")

    @property
    def segment_flat_to_flat_m(self):
        """Twice the hexagon apothem, sqrt(3)/2 times vertex-to-vertex size."""
        return self.segment_point_to_point_m * sqrt(3) / 2

    @property
    def segment_pitch_m(self):
        """Neighbouring segment centre distance in metres."""
        return self.segment_flat_to_flat_m + self.segment_gap_m

    @property
    def segment_centers_m(self):
        """Flat-top lattice (n_segments, 2) coordinates, ordered (x,y)."""
        axial = jnp.asarray(_hex_axial_coords(self.n_rings))
        q, r = axial[:, 0], axial[:, 1]
        return self.segment_pitch_m * jnp.stack(
            (sqrt(3) / 2 * (q + r), (r - q) / 2), axis=-1
        )

    def __repr__(self):
        """Summarize aperture extents, segment count and collecting area."""
        return (
            f"SegmentedPrimary(D_circ={self.circumscribed_diameter_m:.3g} m, "
            f"D_insc={self.inscribed_diameter_m:.3g} m, "
            f"{self.n_segments} hexagon segs, {self.n_rings} rings, "
            f"area={self.collecting_area_m2:.3g} m^2)"
        )
