"""Wavelength bands and the band quadrature rule.

Build quadrature on the host from band edges, declared breaks and the field
the response is read over. Evaluate integrands at its nodes and reduce the
values with JAX transformations.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from operator import index
from typing import Protocol, Self, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from optixstuff.grids import Image

__all__ = [
    "Band",
    "Quadrature",
    "SpectralBreaks",
    "Transmission",
    "rebin_bands",
    "spectral_breaks",
    "trapezoid_weights",
]


@runtime_checkable
class SpectralBreaks(Protocol):
    """Model declaring wavelength breaks needed for piecewise integration."""

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return concrete break wavelengths [nm], or an empty 1D array.

        Returns:
            Break wavelengths in nm as a 1D array; empty when the model declares none.
        """
        ...


def _break_array(breaks_nm):
    breaks = np.asarray(breaks_nm, dtype=float)
    if breaks.ndim != 1 or not np.all(np.isfinite(breaks)):
        raise ValueError("spectral breaks must be a finite 1D array")
    return breaks


def spectral_breaks(*models: SpectralBreaks) -> np.ndarray:
    """Return the sorted union of model-declared spectral breaks [nm] on the host.

    Args:
        *models: Models that each declare ``spectral_breaks_nm()``.

    Returns:
        Sorted unique break wavelengths in nm.
    """
    tables = []
    for model in models:
        if not isinstance(model, SpectralBreaks) or not callable(
            model.spectral_breaks_nm
        ):
            raise TypeError(
                f"{type(model).__name__} must implement spectral_breaks_nm()"
            )
        tables.append(_break_array(model.spectral_breaks_nm()))
    return np.unique(np.concatenate(tables)) if tables else np.empty(0)


@dataclass(frozen=True)
class Band:
    """Positive wavelength interval [nm], stored as hashable host floats.

    Attributes:
        low_nm: Short edge of the band in nm.
        high_nm: Long edge of the band in nm, at least ``low_nm``.
    """

    low_nm: float
    high_nm: float

    def __post_init__(self):
        """Canonicalize concrete edges and reject reversed or unbounded bands."""
        object.__setattr__(self, "low_nm", float(self.low_nm))
        object.__setattr__(self, "high_nm", float(self.high_nm))
        if not (0 < self.low_nm <= self.high_nm < float("inf")):
            raise ValueError("Band edges must be finite and satisfy 0 < lo <= hi")

    @property
    def center_nm(self) -> float:
        """Arithmetic band center [nm]."""
        return self.low_nm + self.width_nm / 2

    @property
    def geometric_center_nm(self) -> float:
        """Geometric band center sqrt(low_nm * high_nm) [nm]."""
        return math.sqrt(self.low_nm * self.high_nm)

    @property
    def width_nm(self) -> float:
        """Full band width [nm]."""
        return self.high_nm - self.low_nm

    def channels(self, n: int, spacing: str = "linear") -> tuple["Band", ...]:
        """Partition into n bands using one shared edge table, including endpoints.

        Args:
            n: Number of channels.
            spacing: ``"linear"`` or ``"log"`` edge spacing.

        Returns:
            The ``n`` contiguous bands covering this band.
        """
        n = index(n)
        if n < 1:
            raise ValueError("n must be a positive integer")
        if spacing == "linear":
            edges = np.linspace(self.low_nm, self.high_nm, n + 1)
        elif spacing == "log":
            edges = np.geomspace(self.low_nm, self.high_nm, n + 1)
        else:
            raise ValueError("spacing must be 'linear' or 'log'")
        edges[0], edges[-1] = self.low_nm, self.high_nm
        return tuple(Band(lo, hi) for lo, hi in pairwise(edges))


def _overlap_fractions(src_lo, src_hi, dst_lo, dst_hi):
    """Fraction of each source interval inside each destination interval.

    All bounds are 1D arrays in a common unit, with nonnegative source widths.
    The result has shape (n_dst, n_src); disjoint or empty intervals give zero.
    """
    src_lo, src_hi, dst_lo, dst_hi = map(jnp.asarray, (src_lo, src_hi, dst_lo, dst_hi))
    lo = jnp.maximum(dst_lo[:, None], src_lo[None])
    hi = jnp.minimum(dst_hi[:, None], src_hi[None])
    widths = src_hi - src_lo
    return jnp.maximum(hi - lo, 0) / jnp.where(widths > 0, widths, 1)[None]


def rebin_bands(values, src_bands, dst_bands, axis=0):
    """Rebin band integrals using exact Band overlaps, preserving units.

    Assumes constant spectral density within each source Band. Full destination
    coverage preserves the total integral to floating-point precision. Source
    and destination bands must have positive widths. An Image
    retains its spatial grid, measure, quantity and reconstruction; axis must
    name a leading (non-spatial) dimension in that case.

    Args:
        values: Band integrals, an array or an Image, with a band axis.
        src_bands: Bands of the input values.
        dst_bands: Bands of the output.
        axis: Band axis of ``values``; must be a leading axis for an Image.

    Returns:
        Band integrals over ``dst_bands``, in the same units and type as the input.
    """
    image = values if isinstance(values, Image) else None
    data = image.data if image is not None else jnp.asarray(values)
    axis = axis % data.ndim
    if image is not None and axis >= data.ndim - 2:
        raise ValueError("spectral axis must be a leading Image axis")
    if data.shape[axis] != len(src_bands):
        raise ValueError("source bands must match the spectral axis")
    if any(b.width_nm == 0 for b in (*src_bands, *dst_bands)):
        raise ValueError("overlap rebinning requires positive-width bands")
    src_lo = jnp.asarray([b.low_nm for b in src_bands])
    src_hi = jnp.asarray([b.high_nm for b in src_bands])
    dst_lo = jnp.asarray([b.low_nm for b in dst_bands])
    dst_hi = jnp.asarray([b.high_nm for b in dst_bands])
    frac = _overlap_fractions(src_lo, src_hi, dst_lo, dst_hi)
    out = jnp.moveaxis(
        jnp.tensordot(frac, jnp.moveaxis(data, axis, 0), axes=1), 0, axis
    )
    return replace(image, data=out) if image is not None else out


def trapezoid_weights(x):
    """Integration weights along the last axis of an increasing grid x.

    ``sum(weights * y, axis=-1)`` is the composite trapezoid integral, with
    half-width endpoint cells and centered interior widths. Leading axes
    are independent grids. A singleton has zero weight (zero integration
    span); an empty axis raises. Concrete inputs must be finite and strictly
    increasing; under JAX tracing the caller must maintain these conditions.

    Args:
        x: Increasing grid, with leading axes as independent grids.

    Returns:
        Trapezoid weights with the shape of ``x``.
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, 1.0))
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty last axis")
    if not isinstance(x, jax.core.Tracer):
        values = np.asarray(x)
        if not np.all(np.isfinite(values)) or np.any(np.diff(values, axis=-1) <= 0):
            raise ValueError("x must be finite and strictly increasing")
    half = jnp.diff(x, axis=-1) / 2
    zero = jnp.zeros_like(x[..., :1])
    return jnp.concatenate((half, zero), axis=-1) + jnp.concatenate(
        (zero, half), axis=-1
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Transmission:
    """Tabulated dimensionless response, linear in wavelength, zero outside.

    Construct with concrete, matching 1D arrays of at least two samples:
    positive strictly increasing ``wavelength_nm`` and finite nonnegative
    ``transmission``. Values above one and identically zero responses are
    allowed (relative responses and opaque elements). Values at endpoints
    retain the tabulated response, with no extrapolated wings.

    The wavelength table retains host edge precision. Evaluation uses the
    configured JAX precision. Both arrays are dynamic PyTree leaves for
    jit, vmap and parameter gradients.
    Transformed tables must preserve the construction constraints. A photon
    spectrum multiplied by this response integrates detected photons; no
    energy-to-photon weighting or normalization is implicit.

    Attributes:
        wavelength_nm: Strictly increasing wavelengths in nm of the table.
        transmission: Dimensionless response at those wavelengths.
    """

    wavelength_nm: jax.Array
    transmission: jax.Array

    def __post_init__(self):
        """Preserve host wavelength geometry and validate the response table."""
        wave = np.asarray(self.wavelength_nm, dtype=float)
        response = np.asarray(jnp.asarray(self.transmission, dtype=float))
        if wave.ndim != 1 or wave.size < 2 or response.shape != wave.shape:
            raise ValueError("tables must share shape (N,) with N >= 2")
        if (
            not np.all(np.isfinite(wave))
            or np.any(wave <= 0)
            or np.any(np.diff(wave) <= 0)
        ):
            raise ValueError("wavelength_nm must be finite, positive and increasing")
        if not np.all(np.isfinite(response)) or np.any(response < 0):
            raise ValueError("transmission must be finite and nonnegative")
        # Retain the supplied break coordinates even when evaluations use float32.
        with jax.enable_x64():
            object.__setattr__(self, "wavelength_nm", jnp.asarray(wave))
        object.__setattr__(self, "transmission", jnp.asarray(response, dtype=float))

    @classmethod
    def from_band(cls, band: Band) -> "Transmission":
        """Unit response between the two supplied wavelength edges [nm].

        Args:
            band: Band whose edges bound the response.

        Returns:
            A Transmission equal to one between the band edges.
        """
        return cls([band.low_nm, band.high_nm], [1.0, 1.0])

    def __call__(self, wavelength_nm):
        """Evaluate at scalar or array wavelengths [nm], with zero extrapolation.

        Args:
            wavelength_nm: Wavelength in nm, scalar or array.

        Returns:
            Dimensionless response, shaped like ``wavelength_nm``, zero outside the
            table.
        """
        return jnp.interp(
            wavelength_nm,
            jnp.asarray(self.wavelength_nm, dtype=float),
            jnp.asarray(self.transmission, dtype=float),
            left=0.0,
            right=0.0,
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """All breakpoints [nm], including support endpoints for zero extension.

        Returns:
            The tabulated wavelengths in nm.
        """
        return np.asarray(self.wavelength_nm)

    @property
    def low_nm(self):
        """Lower edge of the tabulated support [nm]."""
        return self.wavelength_nm[0]

    @property
    def high_nm(self):
        """Upper edge of the tabulated support [nm]."""
        return self.wavelength_nm[-1]

    @property
    def mean_nm(self):
        """Response-weighted mean wavelength [nm], NaN for an opaque table.

        Exact linear-segment area and quadratic first moment; this is the
        detected mean for a flat incident photon flux density.
        """
        wave = jnp.asarray(self.wavelength_nm, dtype=float)
        response = jnp.asarray(self.transmission, dtype=float)
        a, b = wave[:-1], wave[1:]
        ta, tb = response[:-1], response[1:]
        area = jnp.sum((b - a) * (ta + tb) / 2)
        moment = jnp.sum((b - a) * ((2 * a + b) * ta + (a + 2 * b) * tb) / 6)
        return moment / area

    def integrate(self, band: Band, fn=None, *, order: int = 4, breaks_nm=()):
        """Integrate response times fn over the supplied edges, in fn units times nm.

        Construct the rule on the host from this table and any additional
        breaks in fn. With no fn the result is the equivalent width [nm].
        A scalar-wavelength fn may return any fixed shape.

        Args:
            band: Wavelength interval of the integral.
            fn: Optional function of one wavelength in nm; the response alone when
                None.
            order: Quadrature order per panel.
            breaks_nm: Extra break wavelengths in nm for ``fn``.

        Returns:
            The integral in units of ``fn`` times nm, or the equivalent width in nm.
        """
        rule = Quadrature.build((band,), self, order=order, breaks_nm=breaks_nm)
        if fn is None:
            return rule.integrate(self(rule.nodes_nm))[0]
        values = jax.vmap(lambda w: self(w) * fn(w))(rule.nodes_nm[0])
        return rule.integrate(values[None])[0]

    def tree_flatten(self):
        """Expose both table arrays as dynamic leaves.

        Returns:
            The wavelength and transmission arrays as leaves, and None for metadata.
        """
        return (self.wavelength_nm, self.transmission), None

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Rebuild transformed tables without host validation.

        Args:
            metadata: Auxiliary data from ``tree_flatten``; unused.
            arrays: The wavelength and transmission arrays.

        Returns:
            The rebuilt Transmission.
        """
        result = object.__new__(cls)
        object.__setattr__(result, "wavelength_nm", arrays[0])
        object.__setattr__(result, "transmission", arrays[1])
        return result


NODES_PER_CYCLE = 3
"""Nodes ``Quadrature.build`` adds to a piece per chromatic cycle it holds."""


def _field_radii(bands, models, field_lod):
    """Field radius [lambda/D] of each band: supplied, or the largest declared."""
    if field_lod is None:
        declared = [
            model.spectral_field_lod
            for model in models
            if callable(getattr(model, "spectral_field_lod", None))
        ]
        radii = [max((float(fn(b)) for fn in declared), default=0.0) for b in bands]
        radii = np.asarray(radii, dtype=float)
    else:
        radii = np.asarray(field_lod, dtype=float)
        if radii.ndim == 0:
            radii = np.full(len(bands), float(radii))
        if radii.shape != (len(bands),):
            raise ValueError("field_lod must be a number or one value per band")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("field_lod must be finite and nonnegative")
    return radii


def _piece_order(order, cycles):
    """Nodes of a piece: ``order`` without cycles, else raised to an even count."""
    if cycles == 0:
        return order
    return max(order, 2 * math.ceil((order + NODES_PER_CYCLE * cycles) / 2))


class Quadrature(eqx.Module):
    """Shared wavelength nodes of shape (W, Q) with one row of plain weights.

    Summed with values at the nodes, ``weights_nm`` integrate those values
    over each band, in value units times nm: each piece of a band carries
    Gauss-Legendre weights that sum to its width. The rule never depends on a
    source spectrum. A rough tabulated spectrum sampled at the nodes is
    integrated only as well as its samples represent it; declare its knots as
    breaks for an exact integral.

    ``pieces`` records the layout of a built rule: for each band a tuple of
    ``(low_nm, high_nm, order)``, one entry per piece between declared breaks.
    It is ``None`` for a rule constructed by hand. ``pieces`` and ``bands``
    are static, so ``layout_digest`` can be read inside a traced program.

    The record is built on the host; nodes and weights are JAX leaves. Build
    with x64 enabled for float64 leaves. Padding has zero weight and sits at
    the band center. A monochromatic channel carries one integrated rate with
    unit weight.

    Attributes:
        bands: Wavelength bands, one per row.
        nodes_nm: Node wavelengths in nm, shape (W, Q).
        weights_nm: Quadrature weights in nm, shape (W, Q); zero on padding.
        pieces: Per band, the ``(low_nm, high_nm, order)`` of each piece, or
            None for a rule constructed by hand.
    """

    bands: tuple[Band, ...] = eqx.field(static=True)
    nodes_nm: jax.Array
    weights_nm: jax.Array
    pieces: tuple[tuple[tuple[float, float, int], ...], ...] | None = eqx.field(
        static=True
    )

    def __init__(self, bands, nodes_nm, weights_nm, pieces=None):
        """Store nodes, weights and, for a built rule, the piece layout.

        Args:
            bands: Wavelength bands, one per row of nodes.
            nodes_nm: Node wavelengths in nm, shape (W, Q).
            weights_nm: Weights in nm, shape (W, Q).
            pieces: Per band, the ``(low_nm, high_nm, order)`` of each piece, or
                None for a rule constructed by hand.
        """
        self.bands = tuple(bands)
        self.nodes_nm = jnp.asarray(nodes_nm)
        self.weights_nm = jnp.asarray(weights_nm)
        self.pieces = (
            None
            if pieces is None
            else tuple(
                tuple((float(lo), float(hi), index(n)) for lo, hi, n in row)
                for row in pieces
            )
        )

    def __check_init__(self):
        """Validate the rectangular tables, the weight signs and the layout."""
        shape = self.nodes_nm.shape
        if self.nodes_nm.ndim != 2 or shape[0] != len(self.bands) or not shape[1]:
            raise ValueError(
                "each band must have a nonempty row of nodes, shape (W, Q)"
            )
        if self.weights_nm.shape != shape:
            raise ValueError("weights_nm must match nodes_nm, shape (W, Q)")
        if not isinstance(self.weights_nm, jax.core.Tracer) and bool(
            jnp.any(self.weights_nm < 0)
        ):
            raise ValueError("weights must be nonnegative")
        if self.pieces is not None and len(self.pieces) != len(self.bands):
            raise ValueError("pieces must hold one row per band")
        # A rule constructed by hand has no layout to check.
        for band, row in zip(self.bands, self.pieces or (), strict=False):
            edges = [lo for lo, _, _ in row] + [row[-1][1]] if row else []
            joined = all(a[1] == b[0] for a, b in pairwise(row))
            if not edges or (edges[0], edges[-1]) != (band.low_nm, band.high_nm):
                raise ValueError("the pieces of a band must span it")
            if not joined or any(n < 1 for *_, n in row):
                raise ValueError("pieces must be contiguous with positive orders")
            if sum(n for *_, n in row) > shape[1]:
                raise ValueError("the pieces of a band hold more nodes than Q")

    @classmethod
    def build(
        cls,
        bands: Sequence[Band],
        *models: SpectralBreaks,
        order: int = 4,
        breaks_nm=(),
        field_lod=0.0,
    ) -> Self:
        """Build a Gauss-Legendre rule on each piece between declared breaks.

        Everything here is concrete host data.

        Declared breaks, from ``models`` and ``breaks_nm``, split a band into
        pieces. Declare a break for any factor that is evaluated at the nodes
        and has a kink or a feature narrower than about the band width over
        ``order``; such a feature is otherwise integrated to a few percent at
        any practical order.

        The node count of a piece is set by the response it is read through.
        A pixel at separation s (in lambda/D at the band center c) sees the
        diffraction pattern slide across a piece from a to b by
        ``cycles = s * c * (1 / a - 1 / b)`` chromatic cycles. With
        ``field_lod`` the largest separation read, a piece that holds no
        cycles gets exactly ``order`` nodes; otherwise it gets
        ``max(order, 2 * ceil((order + NODES_PER_CYCLE * cycles) / 2))``.
        A ``field_lod`` of zero, the default, is right for a response read at
        one band-center separation for every node, and gives ``order`` nodes
        per piece.

        Args:
            bands: Wavelength bands to integrate over.
            *models: Models whose ``spectral_breaks_nm()`` are declared as breaks.
            order: Nodes per piece that holds no chromatic cycles.
            breaks_nm: Extra break wavelengths in nm.
            field_lod: Largest separation, in lambda/D at the band center, at which
                the response is read: a number, one value per band, or None for
                the largest value a model declares through
                ``spectral_field_lod(band)`` (zero when none does).

        Returns:
            The Quadrature holding nodes, weights and the piece layout.
        """
        order = index(order)
        if order < 1:
            raise ValueError("order must be a positive integer")
        bands = tuple(bands)
        if not bands:
            raise ValueError("at least one band is required")
        breaks = np.union1d(spectral_breaks(*models), _break_array(breaks_nm))
        radii = _field_radii(bands, models, field_lod)
        legendre = {}
        pieces, rows = [], []
        for band, radius in zip(bands, radii, strict=True):
            if band.width_nm == 0:
                raise ValueError("use Quadrature.monochromatic for a zero-width band")
            start = np.searchsorted(breaks, band.low_nm, side="right")
            stop = np.searchsorted(breaks, band.high_nm, side="left")
            edges = np.concatenate(([band.low_nm], breaks[start:stop], [band.high_nm]))
            layout, waves, weights = [], [], []
            for lo, hi in pairwise(edges):
                cycles = radius * band.center_nm * (1 / lo - 1 / hi)
                n = _piece_order(order, cycles)
                if n not in legendre:
                    legendre[n] = leggauss(n)
                nodes, unit = legendre[n]
                length = hi - lo
                waves.append(lo + length * ((nodes + 1) / 2))
                weights.append(length * unit / 2)
                layout.append((lo, hi, n))
            pieces.append(tuple(layout))
            rows.append((np.concatenate(waves), np.concatenate(weights)))

        q = max(wave.size for wave, _ in rows)
        waves = np.array([[band.center_nm] * q for band in bands])
        weights = np.zeros_like(waves)
        for i, (wave, weight) in enumerate(rows):
            waves[i, : wave.size] = wave
            weights[i, : wave.size] = weight
        return cls(bands, waves, weights, pieces)

    @classmethod
    def monochromatic(cls, wavelength_nm: float) -> Self:
        """Select an already integrated rate at one wavelength, with unit weight.

        Args:
            wavelength_nm: Wavelength in nm.

        Returns:
            A one-band, one-node Quadrature with unit weight.
        """
        band = Band(wavelength_nm, wavelength_nm)
        return cls((band,), [[band.low_nm]], [[1.0]])

    def weights(self, name: str = "response") -> jax.Array:
        """Return the (W, Q) weights in nm.

        Args:
            name: The row name; ``"response"`` is the only row.

        Returns:
            Weights of shape (W, Q), in nm.
        """
        if name != "response":
            raise KeyError(f"no weights named {name!r}; the only row is 'response'")
        return self.weights_nm

    @property
    def active(self) -> jax.Array:
        """Boolean (W, Q) mask of real nodes; False marks zero-weight padding."""
        return self.weights_nm != 0

    @property
    def digest(self) -> str:
        """Hex digest of the bands, nodes and weights, for cache keys.

        Host-side only: the leaves must be concrete.
        """
        state = hashlib.sha256()
        for band in self.bands:
            state.update(np.array([band.low_nm, band.high_nm]).tobytes())
        state.update(np.asarray(self.nodes_nm, dtype=float).tobytes())
        state.update(np.asarray(self.weights_nm, dtype=float).tobytes())
        return state.hexdigest()[:16]

    @property
    def layout_digest(self) -> str | None:
        """Hex digest of the bands and pieces, or None for a hand-made rule.

        Computed from static fields only, so it is available inside a traced
        program.
        """
        if self.pieces is None:
            return None
        bands = tuple((band.low_nm, band.high_nm) for band in self.bands)
        return hashlib.sha256(repr((bands, self.pieces)).encode()).hexdigest()[:16]

    def integrate(self, values: jax.Array) -> jax.Array:
        """Sum values of shape (W, Q, ...) with the weights into (W, ...).

        Values on padding are ignored, including NaN.

        Args:
            values: Values at the nodes, shape (W, Q, ...).

        Returns:
            The integral over each band, shape (W, ...).
        """
        values = jnp.asarray(values)
        if values.shape[:2] != self.nodes_nm.shape:
            raise ValueError("values must have leading shape (W, Q)")
        active = self.active.reshape(self.active.shape + (1,) * (values.ndim - 2))
        values = jnp.where(active, values, 0)
        return jnp.einsum("wq,wq...->w...", self.weights_nm, values)

    def mean(self, values: jax.Array) -> jax.Array:
        """Return the weighted mean over each positive-width band.

        The integral is divided by the sum of the weights, the band width.

        Args:
            values: Values at the nodes, shape (W, Q, ...).

        Returns:
            The weighted mean over each band, shape (W, ...).
        """
        if any(band.width_nm == 0 for band in self.bands):
            raise ValueError("a monochromatic integrated rate has no band-width mean")
        integral = self.integrate(values)
        total = self.weights_nm.sum(-1)
        return integral / total.reshape((-1,) + (1,) * (integral.ndim - 1))

    def band(self, i: int) -> Self:
        """Select a single band, preserving its row, padding, and wavelength axis.

        Args:
            i: Index of the band.

        Returns:
            A Quadrature holding only that band.
        """
        return type(self)(
            (self.bands[i],),
            self.nodes_nm[i][None],
            self.weights_nm[i][None],
            None if self.pieces is None else (self.pieces[i],),
        )
