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
    "Reduction",
    "Resolution",
    "SpectralBreaks",
    "SpectralTable",
    "Transmission",
    "digest_knots",
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


def digest_knots(knots_nm, method: str) -> str:
    """Return the knot digest of a tabulated spectrum: its method and its knots.

    The digest has the form ``"<method>:<hash>"``. A table computes it once, on
    the host, from its concrete knots and the name of its interpolation, and
    keeps it as a static field, so a rule can be looked up for the table inside
    a traced program. ``Resolution.build`` recomputes the hash from the knots
    to catch a digest that no longer matches them.

    Args:
        knots_nm: Concrete, strictly increasing table wavelengths in nm.
        method: Name of the interpolation between the knots, such as
            ``"linear"`` or ``"cubic"``; it must not contain a colon.

    Returns:
        The digest string.
    """
    if ":" in method:
        raise ValueError("the interpolation method name must not contain ':'")
    return f"{method}:{_knot_hash(_knot_array(knots_nm))}"


def _knot_array(knots_nm):
    if isinstance(knots_nm, jax.core.Tracer):
        raise TypeError(
            "the knots of a spectral table must be concrete; construct the "
            "table on the host, outside any traced function"
        )
    knots = np.asarray(knots_nm, dtype=float)
    if (
        knots.ndim != 1
        or knots.size < 2
        or not np.all(np.isfinite(knots))
        or np.any(np.diff(knots) <= 0)
    ):
        raise ValueError(
            "the knots of a spectral table must be a finite, strictly "
            "increasing 1D array of at least two wavelengths"
        )
    return knots


def _knot_hash(knots):
    return hashlib.sha256(knots.tobytes()).hexdigest()[:16]


@runtime_checkable
class SpectralTable(Protocol):
    """A tabulated source spectrum that a ``Reduction`` integrates exactly.

    The table is linear in its values through its own interpolant, so the
    integral of the source against each node's Lagrange polynomial is a fixed
    linear map of the values. The protocol carries what the host build needs:

    - ``spectral_breaks_nm()``: the concrete knots. The first and the last are
      the ends of the interval on which the table is defined.
    - ``knot_digest``: a static ``digest_knots(knots, method)`` computed at
      construction from the knots as the table stores them, or None for a
      table constructed under a trace (no rule can be built for it). It is
      computed once, so a knot leaf replaced afterwards leaves it stale;
      ``Resolution.build`` catches that.
    - ``spectral_degree``: the polynomial degree of the source between knots
      (1 for a linear table, 3 for a cubic one, the sum for a product).

    Optionally, ``spectral_interpolant()`` returns the table's own
    interpolation as a function ``f(values, wavelength_nm)``, linear in the
    values of shape (T,). With it the build returns the ``"table"`` kind, which
    acts on the values; without it the ``"fine"`` kind, which acts on the
    source evaluated at ``Reduction.fine_nm``.

    Attributes:
        knot_digest: Static digest of the knots and the interpolation method,
            or None for a table constructed under a trace.
        spectral_degree: Polynomial degree of the source between knots.
    """

    knot_digest: str | None
    spectral_degree: int

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the concrete knots [nm] of the table.

        Returns:
            Strictly increasing knot wavelengths in nm.
        """
        ...


def _lagrange(nodes, points):
    """Lagrange basis of the nodes at the points, shape (nodes, points)."""
    basis = np.ones((nodes.size, points.size))
    for i, node in enumerate(nodes):
        for other in np.delete(nodes, i):
            basis[i] *= (points - other) / (node - other)
    return basis


def _cells_rule(edges, n):
    """Gauss-Legendre points and weights with n points on every cell of edges."""
    x, w = leggauss(n)
    length = np.diff(edges)[:, None]
    points = edges[:-1, None] + length * (x + 1) / 2
    return points.ravel(), (length * w / 2).ravel()


def _compose(interpolant, size, moments, points):
    """Per-piece matrices (moment @ interpolation) by transposing the interpolant.

    ``moments`` (P, Q, M) and ``points`` (P, M) are padded to common sizes.
    Returns, for each piece, the first table column it reads and its float64
    block over the columns from there to the last it reads.
    """

    def rows(moment, wave):
        transpose = jax.linear_transpose(
            lambda values: moment @ interpolant(values, wave), jnp.zeros(size)
        )
        return jax.vmap(lambda row: transpose(row)[0])(jnp.eye(moment.shape[0]))

    compose = jax.jit(rows)
    blocks = []
    with jax.enable_x64():
        for moment, wave in zip(moments, points, strict=True):
            dense = np.asarray(compose(jnp.asarray(moment), jnp.asarray(wave)))
            used = np.flatnonzero(np.any(dense != 0, axis=0))
            first, last = (used[0], used[-1] + 1) if used.size else (0, 1)
            blocks.append((first, dense[:, first:last]))
    return blocks


_RULE_CACHE: dict = {}
_RULE_CACHE_SIZE = 64


class Reduction(eqx.Module):
    """Node-effective values of one tabulated source on one band rule.

    On a piece with nodes ``lambda_k``, weights ``w_k`` and Lagrange
    polynomials ``l_k``, a source ``phi`` has the node-effective value
    ``phibar_k = (1 / w_k) * integral of phi * l_k``. Summed with the weights
    and a response ``R`` at the nodes it gives the integral of ``phi * R`` over
    the piece, exactly for any ``R`` of degree below the piece's order and with
    no error from the source, however rough. A tabulated source is linear in
    its values, so ``phibar`` is a fixed linear map of them, built on the host
    from the nodes and the knots and never from a value.

    Three kinds, recorded in the static ``kind``:

    - ``"identity"``: every knot inside the bands is a piece edge (and no
      piece has fewer nodes than the source's degree), so a Gauss rule on each
      piece is already exact. ``apply`` takes the source at the
      nodes, (W, Q, ...).
    - ``"table"``: ``apply`` takes the table values, (T, ...), and applies one
      block per piece to the columns that piece reads.
    - ``"fine"``: ``apply`` takes the source evaluated at ``fine_nm``,
      (P, M, ...), for a source that is not one table.

    Every kind returns (W, Q, ...) on the quadrature's nodes. A padded node
    copies the first real node of its band, so a ratio of two node rows stays
    finite there. The record belongs to one quadrature and one knot grid, named
    by the static ``layout_digest`` and ``knot_digest``; build it with
    ``Reduction.build`` or, for a set of tables, ``Resolution.build``.

    Attributes:
        layout_digest: ``layout_digest`` of the quadrature the record was built for.
        knot_digest: Digest of the knot grid the record was built for.
        kind: ``"identity"``, ``"table"`` or ``"fine"``.
        slots: Static (W, Q) index of each node into the per-piece values
            (into the node values for the identity kind).
        blocks: Per-piece matrices, (P, Q_max, T_max) for the table kind and
            (P, Q_max, M_max) for the fine kind; None for the identity kind.
        columns: Table columns each piece reads, (P, T_max); table kind only.
        fine_nm: Wavelengths in nm where the fine kind reads the source,
            (P, M_max); fine kind only.
        n_knots: Number of table values the table kind takes.
    """

    layout_digest: str = eqx.field(static=True)
    knot_digest: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    slots: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    blocks: jax.Array | None = None
    columns: jax.Array | None = None
    fine_nm: jax.Array | None = None
    n_knots: int = eqx.field(static=True, default=0)

    def __check_init__(self):
        """Require arrays that match the kind."""
        arrays = {
            "identity": (False, False, False),
            "table": (True, True, False),
            "fine": (True, False, True),
        }
        if self.kind not in arrays:
            raise ValueError("kind must be 'identity', 'table' or 'fine'")
        present = tuple(
            x is not None for x in (self.blocks, self.columns, self.fine_nm)
        )
        if present != arrays[self.kind]:
            raise ValueError(f"the {self.kind!r} kind holds the wrong arrays")

    @classmethod
    def build(
        cls,
        quadrature: Quadrature,
        knots_nm,
        *,
        degree: int,
        interpolant=None,
        knot_digest: str | None = None,
    ) -> Self:
        """Build the operator of one table grid on a built quadrature, on the host.

        Runs in float64 on concrete data and refuses tracers. Every piece of
        every band must lie inside the table, from its first knot to its last;
        a piece outside it, or one that straddles its end, is refused, because
        the table says nothing there. End the band at the table end, or load
        the table over the band.

        The moments use ``ceil((order + degree) / 2)`` Gauss points on every
        cell between knots, which is exact for a source of polynomial degree
        ``degree`` between knots. Where every knot is a piece edge and no piece
        has fewer nodes than ``degree``, the Gauss rule of each piece is
        already exact and the build returns the identity kind. With
        ``interpolant`` the moments are composed with the interpolation by
        transposing that function, so the scheme is never restated here.

        Args:
            quadrature: A rule from ``Quadrature.build`` (its ``pieces`` set).
            knots_nm: Concrete, strictly increasing table wavelengths in nm.
            degree: Polynomial degree of the source between knots.
            interpolant: The source's interpolation ``f(values, wavelength_nm)``,
                linear in values of shape (T,); None for the fine kind.
            knot_digest: Digest of the knots; ``digest_knots(knots_nm,
                f"degree {degree}")`` when None.

        Returns:
            The Reduction for this quadrature and knot grid.
        """
        if quadrature.pieces is None:
            raise ValueError(
                "a hand-made quadrature has no piece layout; build it with "
                "Quadrature.build"
            )
        if any(
            isinstance(x, jax.core.Tracer)
            for x in (quadrature.nodes_nm, quadrature.weights_nm)
        ):
            raise TypeError("build the operator on the host, outside any trace")
        degree = index(degree)
        if degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        knots = _knot_array(knots_nm)
        if knot_digest is None:
            knot_digest = digest_knots(knots, f"degree {degree}")
        for band, row in zip(quadrature.bands, quadrature.pieces, strict=True):
            for lo, hi, _ in row:
                if lo < knots[0] or hi > knots[-1]:
                    word, end = (
                        ("begins", knots[0]) if lo < knots[0] else ("ends", knots[-1])
                    )
                    raise ValueError(
                        f"band {band.low_nm:g} to {band.high_nm:g} nm: the piece "
                        f"{lo:g} to {hi:g} nm leaves the table, which {word} at "
                        f"{end:.10g} nm. End the band at the table end, or load "
                        "the table over the band."
                    )
        key = (
            quadrature.layout_digest,
            knot_digest,
            degree,
            interpolant is not None,
            quadrature.nodes_nm.dtype.name,
        )
        cached = _RULE_CACHE.get(key)
        if cached is None:
            cached = _build_arrays(quadrature, knots, degree, interpolant)
            if len(_RULE_CACHE) >= _RULE_CACHE_SIZE:
                _RULE_CACHE.pop(next(iter(_RULE_CACHE)))
            _RULE_CACHE[key] = cached
        kind, slots, arrays = cached
        # Leaves follow the configured precision, like the quadrature's own.
        return cls(
            quadrature.layout_digest,
            knot_digest,
            kind,
            slots,
            **{name: jnp.asarray(value) for name, value in arrays.items()},
            n_knots=knots.size if kind == "table" else 0,
        )

    @property
    def nbytes(self) -> int:
        """Bytes held by the operator's arrays."""
        return sum(
            x.nbytes for x in (self.blocks, self.columns, self.fine_nm) if x is not None
        )

    def apply(self, values: jax.Array) -> jax.Array:
        """Return node-effective values, (W, Q, ...), from the source's values.

        Args:
            values: The source in the form the kind takes: at the quadrature's
                nodes, (W, Q, ...), for the identity kind; the table values,
                (T, ...), for the table kind; at ``fine_nm``, (P, M, ...), for the
                fine kind. Trailing axes are carried through.

        Returns:
            Node-effective values on the quadrature's nodes, (W, Q, ...).
        """
        values = jnp.asarray(values)
        slots = jnp.asarray(self.slots)
        if self.kind == "identity":
            if values.shape[:2] != slots.shape:
                raise ValueError(f"values must have leading shape {slots.shape}")
            flat = values.reshape((-1, *values.shape[2:]))
            return flat[slots]
        if self.kind == "table":
            if values.shape[:1] != (self.n_knots,):
                raise ValueError(f"values must have leading shape ({self.n_knots},)")
            per_piece = jnp.einsum(
                "pqt,pt...->pq...", self.blocks, values[self.columns]
            )
        else:
            if values.shape[:2] != self.fine_nm.shape:
                raise ValueError(f"values must have leading shape {self.fine_nm.shape}")
            per_piece = jnp.einsum("pqm,pm...->pq...", self.blocks, values)
        flat = per_piece.reshape((-1, *per_piece.shape[2:]))
        return flat[slots]


def _build_arrays(quadrature, knots, degree, interpolant):
    """Host arrays of a Reduction: its kind, its slots and its padded blocks."""
    nodes = np.asarray(quadrature.nodes_nm, dtype=float)
    weights = np.asarray(quadrature.weights_nm, dtype=float)
    n_bands, n_nodes = nodes.shape
    identity = True
    pieces = []  # (band, first node, nodes, weights, cell edges)
    for w, row in enumerate(quadrature.pieces):
        start = 0
        for lo, hi, n in row:
            inside = knots[(knots > lo) & (knots < hi)]
            # Gauss on a piece is exact for the source times a response of
            # degree n - 1 only while the source's degree is at most n.
            identity &= inside.size == 0 and degree <= n
            stop = start + n
            edges = np.concatenate(([lo], inside, [hi]))
            pieces.append(
                (w, start, nodes[w, start:stop], weights[w, start:stop], edges)
            )
            start = stop
    counts = [sum(n for *_, n in row) for row in quadrature.pieces]
    if identity:
        slots = [
            [w * n_nodes + (j if j < counts[w] else 0) for j in range(n_nodes)]
            for w in range(n_bands)
        ]
        return "identity", _tuple(slots), {}
    q_max = max(p[2].size for p in pieces)
    moments, points = [], []
    for _, _, piece_nodes, piece_weights, edges in pieces:
        n = piece_nodes.size
        x, wx = _cells_rule(edges, math.ceil((n + degree) / 2))
        moments.append(_lagrange(piece_nodes, x) * wx / piece_weights[:, None])
        points.append(x)
    m_max = max(x.size for x in points)
    blocks = np.zeros((len(pieces), q_max, m_max))
    fine = np.empty((len(pieces), m_max))
    for p, (moment, x) in enumerate(zip(moments, points, strict=True)):
        blocks[p, : moment.shape[0], : x.size] = moment
        fine[p, : x.size] = x
        fine[p, x.size :] = x[0]
    piece_of = {}
    for p, (w, start, piece_nodes, *_) in enumerate(pieces):
        for r in range(piece_nodes.size):
            piece_of[w, start + r] = p * q_max + r
    slots = [
        [piece_of[w, j if j < counts[w] else 0] for j in range(n_nodes)]
        for w in range(n_bands)
    ]
    if interpolant is None:
        return "fine", _tuple(slots), {"blocks": blocks, "fine_nm": fine}
    composed = _compose(interpolant, knots.size, blocks, fine)
    t_max = max(block.shape[1] for _, block in composed)
    table_blocks = np.zeros((len(pieces), q_max, t_max))
    columns = np.zeros((len(pieces), t_max), dtype=int)
    for p, (first, block) in enumerate(composed):
        table_blocks[p, :, : block.shape[1]] = block
        columns[p] = np.clip(first + np.arange(t_max), 0, knots.size - 1)
    return "table", _tuple(slots), {"blocks": table_blocks, "columns": columns}


def _tuple(rows):
    return tuple(tuple(int(x) for x in row) for row in rows)


class Resolution(eqx.Module):
    """The ``Reduction`` of every tabulated source an exposure or term reads.

    One rule per pair of a quadrature and a table grid, found by the two
    static digests, so a source cannot be integrated by an operator built for
    another table or another layout. ``quadratures`` lists the rules it was
    built for: the exposure's own, then any auxiliary one (such as the
    calibration band of a disk). ``Resolution()`` is empty.

    A rule belongs to the full quadrature: ``Quadrature.band(i)`` has its own
    layout digest and no rule. A route that works band by band computes the
    node values once on the full rule and carries the rows of each band.

    Attributes:
        quadratures: The quadratures the rules were built for.
        keys: Static ``(layout_digest, knot_digest)`` of each rule.
        rules: The Reductions, in the order of ``keys``.
    """

    quadratures: tuple[Quadrature, ...] = ()
    keys: tuple[tuple[str, str], ...] = eqx.field(static=True, default=())
    rules: tuple[Reduction, ...] = ()

    def __check_init__(self):
        """Require one rule per key, and keys that name their rule."""
        if len(self.keys) != len(self.rules):
            raise ValueError("a Resolution needs one rule per key")
        for (layout, knots), rule in zip(self.keys, self.rules, strict=True):
            if (rule.layout_digest, rule.knot_digest) != (layout, knots):
                raise ValueError("each key must be the digests of its rule")

    @classmethod
    def build(
        cls, quadratures: Quadrature | Sequence[Quadrature], *tables: SpectralTable
    ) -> Self:
        """Build one rule per pair of a quadrature and a table, on the host.

        Checks every table first: its digest must exist (a table constructed
        under a trace has none) and match its concrete knots. Rules are cached
        on the digests, so a runner that builds an exposure per roll pays for
        each operator once. Tables that share a knot grid share a rule.

        Args:
            quadratures: The exposure's quadrature, or a sequence of quadratures
                (the exposure's first, then any auxiliary one).
            *tables: Tabulated sources implementing ``SpectralTable``.

        Returns:
            The Resolution holding every rule.
        """
        if isinstance(quadratures, Quadrature):
            quadratures = (quadratures,)
        quadratures = tuple(quadratures)
        rules = {}
        for table in tables:
            name = type(table).__name__
            if not isinstance(table, SpectralTable):
                raise TypeError(f"{name} does not implement SpectralTable")
            digest = table.knot_digest
            if digest is None:
                raise ValueError(
                    f"{name} was constructed under a trace and has no knot "
                    "digest; construct the table on the host"
                )
            knots = _knot_array(table.spectral_breaks_nm())
            if digest.rpartition(":")[2] != _knot_hash(knots):
                raise ValueError(
                    f"the knot digest of {name} does not match its knots: a knot "
                    "grid was replaced after construction; rebuild the table"
                )
            get = getattr(table, "spectral_interpolant", None)
            interpolant = get() if callable(get) else None
            for quadrature in quadratures:
                key = (quadrature.layout_digest, digest)
                if key in rules:
                    continue
                try:
                    rules[key] = Reduction.build(
                        quadrature,
                        knots,
                        degree=table.spectral_degree,
                        interpolant=interpolant,
                        knot_digest=digest,
                    )
                except ValueError as error:
                    raise ValueError(f"{name}: {error}") from None
        return cls(quadratures, tuple(rules), tuple(rules.values()))

    def rule(self, quadrature: Quadrature, table: SpectralTable) -> Reduction:
        """Return the rule of a table on a quadrature, found by static digests.

        Works inside a traced program, since both digests are static, and
        raises at trace time when there is no rule.

        Args:
            quadrature: The quadrature whose nodes the values are wanted on.
            table: The tabulated source.

        Returns:
            The Reduction built for that pair.
        """
        layout = quadrature.layout_digest
        if layout is None:
            raise LookupError(
                "a hand-made quadrature has no piece layout and no rule; build "
                "it with Quadrature.build"
            )
        digest = getattr(table, "knot_digest", None)
        name = type(table).__name__
        if digest is None:
            raise LookupError(
                f"{name} has no knot digest (it was constructed under a trace); "
                "construct the table on the host"
            )
        for key, rule in zip(self.keys, self.rules, strict=True):
            if key == (layout, digest):
                return rule
        raise LookupError(
            f"no rule for {name} on this quadrature: build the Resolution with "
            "Resolution.build over this quadrature and the table. A band slice "
            "has no rule of its own; carry the rows of the full rule's values"
        )
