"""Wavelength bands and quadrature with named weights for rough host spectra.

Build quadrature on the host from band edges, declared breaks and the rough
spectra its weights absorb. Evaluate the smooth remainder at its nodes and
reduce the values with JAX transformations.
"""

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise, product
from operator import index
from typing import Protocol, Self, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from numpy.polynomial.legendre import leggauss

from optixstuff.grids import Image

__all__ = [
    "Band",
    "Probe",
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
        """Return concrete break wavelengths [nm], or an empty 1D array."""
        ...


def _break_array(breaks_nm):
    breaks = np.asarray(breaks_nm, dtype=float)
    if breaks.ndim != 1 or not np.all(np.isfinite(breaks)):
        raise ValueError("spectral breaks must be a finite 1D array")
    return breaks


def spectral_breaks(*models: SpectralBreaks) -> np.ndarray:
    """Return the sorted union of model-declared spectral breaks [nm] on the host."""
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
    """Positive wavelength interval [nm], stored as hashable host floats."""

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
        """Partition into n bands using one shared edge table, including endpoints."""
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
        """Unit response between the two supplied wavelength edges [nm]."""
        return cls([band.low_nm, band.high_nm], [1.0, 1.0])

    def __call__(self, wavelength_nm):
        """Evaluate at scalar or array wavelengths [nm], with zero extrapolation."""
        return jnp.interp(
            wavelength_nm,
            jnp.asarray(self.wavelength_nm, dtype=float),
            jnp.asarray(self.transmission, dtype=float),
            left=0.0,
            right=0.0,
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """All breakpoints [nm], including support endpoints for zero extension."""
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
        """
        rule = Quadrature.build((band,), self, order=order, breaks_nm=breaks_nm)
        if fn is None:
            return rule.integrate(self(rule.nodes_nm))[0]
        values = jax.vmap(lambda w: self(w) * fn(w))(rule.nodes_nm[0])
        return rule.integrate(values[None])[0]

    def tree_flatten(self):
        """Expose both table arrays as dynamic leaves."""
        return (self.wavelength_nm, self.transmission), None

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Rebuild transformed tables without host validation."""
        result = object.__new__(cls)
        object.__setattr__(result, "wavelength_nm", arrays[0])
        object.__setattr__(result, "transmission", arrays[1])
        return result


_REFERENCE_ORDER = 8
_REFERENCE_PANELS = 16
_ENVELOPE_SAFETY = 1.1
# A weight below zero by more than round-off of the largest one is a real negative.
_NEGATIVE_WEIGHT_RTOL = 1e-12


@dataclass(frozen=True)
class Probe:
    """Smooth integrand family whose worst quadrature error the build stores.

    ``fn(wavelength_nm, **parameters)`` returns the factor a consumer will
    evaluate at the nodes and sum with the named weights: a contrast for
    star weights, a thermal photon spectrum for response weights. Each entry
    of ``ranges`` is the admissible ``(low, high)`` interval of one traced
    parameter, sampled at ``samples`` evenly spaced values; several ranges
    form a grid. The error at one parameter value can vanish while the error
    nearby is large, so the stored figure is the maximum over the ranges.
    """

    fn: Callable[..., ArrayLike]
    weights: str = "response"
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    samples: int = 17

    def parameter_grid(self) -> list[dict[str, float]]:
        """Return every sampled parameter combination, one dict per call."""
        if index(self.samples) < 2:
            raise ValueError("samples must be at least 2")
        axes = []
        for name, (low, high) in self.ranges.items():
            if not (np.isfinite(low) and np.isfinite(high) and low <= high):
                raise ValueError(f"range of {name} must be finite with low <= high")
            axes.append(np.linspace(low, high, index(self.samples)))
        return [
            dict(zip(self.ranges, map(float, point), strict=True))
            for point in product(*axes)
        ]


def _host_values(fn, wavelength_nm, label, *, nonnegative=True, **parameters):
    """Evaluate a host callable in float64, shaped like the wavelengths."""
    with jax.enable_x64():
        values = np.asarray(fn(wavelength_nm, **parameters), dtype=float)
    values = np.broadcast_to(values, wavelength_nm.shape)
    if not np.all(np.isfinite(values)) or (nonnegative and np.any(values < 0)):
        requirement = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {requirement} across every band")
    return values


def _reference_rule(edges, knots, order):
    """Composite Gauss-Legendre rule resolving every knot inside each panel.

    Returns wavelengths [nm], weights [nm] and the panel index of each point.
    """
    nodes, weights = leggauss(order)
    waves, widths, owners = [], [], []
    for i, (lo, hi) in enumerate(pairwise(edges)):
        inside = knots[(knots > lo) & (knots < hi)]
        cells = np.union1d(np.linspace(lo, hi, _REFERENCE_PANELS + 1), inside)
        cells[0], cells[-1] = lo, hi
        lengths = np.diff(cells)
        waves.append((cells[:-1, None] + lengths[:, None] * (nodes + 1) / 2).ravel())
        widths.append((lengths[:, None] * weights / 2).ravel())
        owners.append(np.full(waves[-1].size, i))
    return np.concatenate(waves), np.concatenate(widths), np.concatenate(owners)


def _gauss_nodes(t, measure, order):
    """Gauss nodes on [-1, 1] of a discrete measure, by the Stieltjes recurrence.

    The monic orthogonal polynomials are rescaled at each step, which leaves
    the recurrence coefficients unchanged and keeps high orders in range.
    A measure that vanishes everywhere falls back to Gauss-Legendre nodes.
    """
    if not np.any(measure > 0):
        return leggauss(order)[0]
    alpha, beta = np.zeros(order), np.zeros(order)
    previous, current, previous_norm = np.zeros_like(t), np.ones_like(t), 1.0
    for k in range(order):
        norm = np.sum(measure * current**2)
        if not norm > 0:
            raise ValueError("the node measure supports fewer points than order")
        alpha[k] = np.sum(measure * t * current**2) / norm
        beta[k] = norm / previous_norm
        following = (t - alpha[k]) * current - (beta[k] if k else 0.0) * previous
        scale = np.max(np.abs(following)) or 1.0
        previous, current = current / scale, following / scale
        previous_norm = norm / scale**2
    off_diagonal = np.sqrt(beta[1:])
    jacobi = np.diag(alpha) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return np.linalg.eigvalsh(jacobi)


def _lagrange_basis(nodes, t):
    """Lagrange cardinal polynomials of the nodes at t, shape (n, t.size)."""
    basis = np.ones((nodes.size, t.size))
    for i, node in enumerate(nodes):
        for other in np.delete(nodes, i):
            basis[i] *= (t - other) / (node - other)
    return basis


class Quadrature(eqx.Module):
    """Shared wavelength nodes of shape (W, Q) with a named set of weight rows.

    ``weights("response")`` is always present: summed with values at the
    nodes it integrates the instrument response times those values over each
    band, in value units times nm. With no response supplied the response is
    one and these are plain Gauss-Legendre weights that sum to the band width.
    Each further name belongs to one rough host spectrum (a tabulated star,
    a zodiacal table): ``weights(name)`` integrates response times that
    spectrum times the values, so the spectrum is never sampled at the nodes
    and its lines cost no nodes. The values are the smooth remainder, for
    example a planet contrast under ``"star"``.

    The record is built on the host; nodes and weights are JAX leaves. Build
    with x64 enabled for float64 leaves. Padding has zero weight under every
    name and sits at the band center. A monochromatic channel carries one
    integrated rate with unit weight.
    """

    bands: tuple[Band, ...] = eqx.field(static=True)
    nodes_nm: jax.Array
    weight_table: dict[str, jax.Array]
    envelope_error: dict[str, jax.Array]

    def __init__(self, bands, nodes_nm, weights, envelope_error=None):
        """Store nodes with one weight array, or a mapping of named arrays.

        A bare array is the ``"response"`` weights. ``envelope_error`` maps a
        probe name to relative errors of shape (W,).
        """
        self.bands = tuple(bands)
        self.nodes_nm = jnp.asarray(nodes_nm)
        if not isinstance(weights, Mapping):
            weights = {"response": weights}
        self.weight_table = {
            str(name): jnp.asarray(weights[name]) for name in sorted(weights)
        }
        errors = envelope_error or {}
        self.envelope_error = {
            str(name): jnp.asarray(errors[name]) for name in sorted(errors)
        }

    def __check_init__(self):
        """Validate the rectangular node table and the sign of every weight."""
        shape = self.nodes_nm.shape
        if "response" not in self.weight_table:
            raise ValueError("the weight table must contain 'response'")
        if self.nodes_nm.ndim != 2 or shape[0] != len(self.bands) or not shape[1]:
            raise ValueError(
                "each band must have a nonempty row of nodes, shape (W, Q)"
            )
        for name, table in self.weight_table.items():
            if table.shape != shape:
                raise ValueError(f"{name} weights must match nodes_nm, shape (W, Q)")
            if not isinstance(table, jax.core.Tracer) and bool(jnp.any(table < 0)):
                raise ValueError(f"{name} weights must be nonnegative")
        for name, error in self.envelope_error.items():
            if error.shape != shape[:1]:
                raise ValueError(f"{name} envelope error must have shape (W,)")

    @classmethod
    def build(
        cls,
        bands: Sequence[Band],
        *models: SpectralBreaks,
        order: int = 4,
        breaks_nm=(),
        response: Callable[[ArrayLike], ArrayLike] | None = None,
        spectra: Mapping[str, Callable[[ArrayLike], ArrayLike]] | None = None,
        probes: Mapping[str, Probe] | None = None,
    ) -> Self:
        """Build an order-point rule on each panel between declared breaks.

        Everything here is concrete host data.

        Declared breaks, from ``models`` and ``breaks_nm``, split a band into
        panels with ``order`` nodes each. Declare a break for any factor that
        is evaluated at the nodes and has a kink or a feature narrower than
        about the band width over ``order``; such a feature is otherwise
        integrated to a few percent at any practical order.

        ``response`` and each entry of ``spectra`` are callables of wavelength
        [nm] returning finite nonnegative values. They are absorbed into the
        weights instead of being sampled at the nodes, so their own
        ``spectral_breaks_nm()`` knots, when they declare any, refine only
        the reference rule that computes the weights and add no nodes. The
        nodes are the Gauss nodes of the response times the first spectrum
        (of the response alone without spectra), found by the Stieltjes
        recurrence on that reference rule. The weights of that first spectrum
        are then exact for polynomials up to degree ``2 * order - 1``, and
        every other name is exact up to degree ``order - 1``.

        Guidance on ``order``: 4 serves contrast-form terms under the star
        weights. Flux-form terms under the response weights, such as thermal
        emission, need 6 to 8 in bands 20 percent or wider.

        ``probes`` names integrand families whose error is stored in
        ``envelope_error``: the largest relative difference from the
        reference rule over each probe's parameter ranges, times a safety
        factor of 1.1, with shape (W,).

        Raises:
            ValueError: If any weight is negative, which happens when the
                node spectrum is so steep that the nodes crowd one end of a
                panel. Lower the order or declare a break.
        """
        order = index(order)
        if order < 1:
            raise ValueError("order must be a positive integer")
        bands = tuple(bands)
        if not bands:
            raise ValueError("at least one band is required")
        spectra = dict(spectra or {})
        probes = dict(probes or {})
        if "response" in spectra:
            raise ValueError("'response' names the response weights, not a spectrum")
        hosts = {"response": response, **spectra}
        for name, probe in probes.items():
            if probe.weights not in hosts:
                raise ValueError(f"probe {name} names unknown weights {probe.weights}")
        breaks = np.union1d(spectral_breaks(*models), _break_array(breaks_nm))
        band_edges = []
        for band in bands:
            if band.width_nm == 0:
                raise ValueError("use Quadrature.monochromatic for a zero-width band")
            start = np.searchsorted(breaks, band.low_nm, side="right")
            stop = np.searchsorted(breaks, band.high_nm, side="left")
            band_edges.append(
                np.concatenate(([band.low_nm], breaks[start:stop], [band.high_nm]))
            )

        weighted = response is not None or bool(spectra)
        references = []
        if weighted or probes:
            knots = spectral_breaks(
                *(h for h in hosts.values() if isinstance(h, SpectralBreaks))
            )
            references = [
                _reference_rule(edges, knots, max(_REFERENCE_ORDER, order + 2))
                for edges in band_edges
            ]
            fine = np.concatenate([wave for wave, _, _ in references])
            factors = {
                name: np.ones_like(fine)
                if host is None
                else _host_values(host, fine, f"the {name} host")
                for name, host in hosts.items()
            }
            splits = np.cumsum([wave.size for wave, _, _ in references])[:-1]
            # Per band and name: reference weight times response times spectrum.
            measures = [
                {
                    name: width
                    * factors["response"][part]
                    * (1.0 if name == "response" else factors[name][part])
                    for name in hosts
                }
                for (_, width, _), part in zip(
                    references,
                    np.split(np.arange(fine.size), splits),
                    strict=True,
                )
            ]

        node_name = next(iter(spectra), "response")
        legendre_nodes, legendre_weights = leggauss(order)
        rows = []
        for i, edges in enumerate(band_edges):
            lengths = np.diff(edges)
            if not weighted:
                wave = edges[:-1, None] + lengths[:, None] * ((legendre_nodes + 1) / 2)
                weight = lengths[:, None] * legendre_weights / 2
                rows.append((wave.ravel(), {"response": weight.ravel()}))
                continue
            fine_wave, _, owner = references[i]
            wave = np.empty((lengths.size, order))
            table = {name: np.empty((lengths.size, order)) for name in hosts}
            for p, (lo, length) in enumerate(zip(edges[:-1], lengths, strict=True)):
                cell = owner == p
                t = 2 * (fine_wave[cell] - lo) / length - 1
                nodes = _gauss_nodes(t, measures[i][node_name][cell], order)
                basis = _lagrange_basis(nodes, t)
                wave[p] = lo + length * (nodes + 1) / 2
                for name in hosts:
                    table[name][p] = basis @ measures[i][name][cell]
            for name, values in table.items():
                floor = -_NEGATIVE_WEIGHT_RTOL * np.max(np.abs(values))
                if np.any(values < floor):
                    raise ValueError(
                        f"{name} weights are negative in band {i} at order {order}; "
                        "lower the order or declare a break"
                    )
                table[name] = np.maximum(values, 0.0).ravel()
            rows.append((wave.ravel(), table))

        q = max(wave.size for wave, _ in rows)
        waves = np.array([[band.center_nm] * q for band in bands])
        weights = {name: np.zeros_like(waves) for name in rows[0][1]}
        for i, (wave, table) in enumerate(rows):
            waves[i, : wave.size] = wave
            for name, values in table.items():
                weights[name][i, : wave.size] = values

        errors = {}
        for name, probe in probes.items():
            worst = np.zeros(len(bands))
            for parameters in probe.parameter_grid():
                for i, (wave, table) in enumerate(rows):
                    label = f"probe {name}"
                    exact = np.sum(
                        measures[i][probe.weights]
                        * _host_values(
                            probe.fn,
                            references[i][0],
                            label,
                            nonnegative=False,
                            **parameters,
                        )
                    )
                    value = np.sum(
                        table[probe.weights]
                        * _host_values(
                            probe.fn, wave, label, nonnegative=False, **parameters
                        )
                    )
                    error = abs(value - exact) / (abs(exact) or 1.0)
                    worst[i] = max(worst[i], error)
            errors[name] = _ENVELOPE_SAFETY * worst
        return cls(bands, waves, weights, errors)

    @classmethod
    def monochromatic(cls, wavelength_nm: float) -> Self:
        """Select an already integrated rate at one wavelength, with unit weight."""
        band = Band(wavelength_nm, wavelength_nm)
        return cls((band,), [[band.low_nm]], [[1.0]])

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the stored weight rows, sorted; always includes 'response'."""
        return tuple(self.weight_table)

    def weights(self, name: str = "response") -> jax.Array:
        """Return the (W, Q) weights of one name, in host-spectrum units times nm."""
        try:
            return self.weight_table[name]
        except KeyError:
            raise KeyError(
                f"no weights named {name!r}; this rule holds {self.names}"
            ) from None

    @property
    def active(self) -> jax.Array:
        """Boolean (W, Q) mask of real nodes; False marks zero-weight padding."""
        return jnp.any(jnp.stack([w != 0 for w in self.weight_table.values()]), axis=0)

    @property
    def digest(self) -> str:
        """Hex digest of the bands, nodes and named weights, for cache keys.

        Host-side only: the leaves must be concrete.
        """
        state = hashlib.sha256()
        for band in self.bands:
            state.update(np.array([band.low_nm, band.high_nm]).tobytes())
        state.update(np.asarray(self.nodes_nm, dtype=float).tobytes())
        for name, table in self.weight_table.items():
            state.update(name.encode())
            state.update(np.asarray(table, dtype=float).tobytes())
        return state.hexdigest()[:16]

    def integrate(self, values: jax.Array, weights: str = "response") -> jax.Array:
        """Sum values of shape (W, Q, ...) with the named weights into (W, ...).

        The values exclude whatever the named weights absorbed: the response
        always, and the host spectrum of that name.
        """
        values = jnp.asarray(values)
        if values.shape[:2] != self.nodes_nm.shape:
            raise ValueError("values must have leading shape (W, Q)")
        table = self.weights(weights)
        active = (table != 0).reshape(table.shape + (1,) * (values.ndim - 2))
        return jnp.einsum("wq,wq...->w...", table, jnp.where(active, values, 0))

    def mean(self, values: jax.Array, weights: str = "response") -> jax.Array:
        """Return the weighted mean over each positive-width band.

        The integral is divided by the sum of the named weights, which is the
        band width for a unit response.
        """
        if any(band.width_nm == 0 for band in self.bands):
            raise ValueError("a monochromatic integrated rate has no band-width mean")
        integral = self.integrate(values, weights)
        total = self.weights(weights).sum(-1)
        return integral / total.reshape((-1,) + (1,) * (integral.ndim - 1))

    def band(self, i: int) -> Self:
        """Select a single band, preserving its row, padding, and wavelength axis."""
        return type(self)(
            (self.bands[i],),
            self.nodes_nm[i][None],
            {name: table[i][None] for name, table in self.weight_table.items()},
            {name: error[i][None] for name, error in self.envelope_error.items()},
        )
