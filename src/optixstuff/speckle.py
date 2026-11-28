"""Time-varying stellar leakage excess on explicit focal-plane grids.

The deterministic coronagraph supplies the nominal leakage floor. Speckle
Images contain only the excess over that floor, as per-pixel fractions of
unocculted stellar flux; coherent interference can make the excess negative.
"""

import abc
import math
from dataclasses import replace
from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar
from hwoutils.epoch import Epoch, elapsed_seconds
from hwoutils.precision import like
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.spectral import Quadrature


def _dtype_probe(value):
    """Stand in for a nested list or tuple when inferring a working dtype.

    A sequence of Python numbers is typed weakly, like a Python scalar; any
    other sequence is typed by its NumPy array.

    Args:
        value: An observation input as supplied by the caller.

    Returns:
        The value itself, or a weak Python scalar or NumPy array of the
        sequence's dtype.
    """
    if not isinstance(value, (list, tuple)):
        return value
    flat = np.ravel(np.asarray(value, dtype=object))
    if all(type(x) in (bool, int, float, complex) for x in flat):
        if any(type(x) is complex for x in flat):
            return 0j
        if any(type(x) is float for x in flat):
            return 0.0
        return 0
    return np.asarray(value)


class FocalResponse(eqx.Module):
    """Complex focal amplitudes and their photometric reference.

    e_nom is in sqrt(photons) per sqrt(grid unit area), e.g.
    sqrt(photons) / (lambda/D) on a lod grid. G has the same units
    per unit of its corresponding command or mode coefficient. Their shapes
    are (y, x) and (m, y, x), or (w, y, x) and (w, m, y, x) for a table
    of wavelength nodes. input_energy is the total incident photon
    count, scalar or one count per wavelength node.
    Passing ``wavelength_law="opd"`` reproduces the earlier behavior.

    Attributes:
        e_nom: Nominal complex focal amplitude, shape (y, x) or (w, y, x), in
            sqrt(photons) per sqrt(grid unit area).
        G: Complex sensitivity per command or mode coefficient, shape (m, y, x) or
            (w, m, y, x), in the units of ``e_nom`` per coefficient unit.
        grid: PixelGrid the amplitudes are sampled on.
        input_energy: Total incident photon count, scalar or one per wavelength
            node; positive and finite.
        reference_wavelength_nm: Grid frame's reference wavelength in nm.
            For an OPD response, ``G`` is specified at this wavelength.
        wavelength_law: For ``"opd"``, G is the sensitivity to an optical path
            difference. The phase sensitivity at wavelength lambda is
            ``G * reference_wavelength_nm / lambda``. For ``"amplitude"``, G is
            achromatic and used as given at every wavelength. With ``None``,
            the consumer reads the tabulated arrays at their wavelength nodes
            without scaling.
    """

    e_nom: Array = eqx.field(converter=jnp.asarray)
    G: Array = eqx.field(converter=jnp.asarray)
    grid: PixelGrid
    input_energy: Array = eqx.field(converter=jnp.asarray)
    reference_wavelength_nm: float = eqx.field(static=True, converter=float)
    wavelength_law: Literal["opd", "amplitude"] | None = eqx.field(
        static=True, kw_only=True
    )

    def __check_init__(self):
        """Validate the sampled arrays and their reference."""
        if self.e_nom.ndim not in (2, 3) or self.e_nom.shape[-2:] != self.grid.shape:
            raise ValueError("e_nom must have shape (y, x) or (w, y, x) on grid")
        if (
            self.G.ndim != self.e_nom.ndim + 1
            or self.G.shape[:-3] != self.e_nom.shape[:-2]
            or self.G.shape[-2:] != self.grid.shape
        ):
            raise ValueError("G must have shape (m, y, x) or (w, m, y, x) on grid")
        if self.e_nom.ndim == 3:
            if self.wavelength_law is not None:
                raise ValueError("pass wavelength_law=None for a tabulated response")
        elif not isinstance(self.wavelength_law, str) or self.wavelength_law not in (
            "opd",
            "amplitude",
        ):
            raise ValueError(
                'pass wavelength_law="opd" or "amplitude" for a single-plane response'
            )
        if self.input_energy.shape not in ((), self.e_nom.shape[:-2]):
            raise ValueError("input_energy must be scalar or match the wavelength axis")
        if not isinstance(self.input_energy, jax.core.Tracer):
            energy = np.asarray(self.input_energy)
            if not np.all(np.isfinite(energy) & (energy > 0)):
                raise ValueError("input_energy must be positive and finite")
        if (
            self.grid.frame is not None
            and self.reference_wavelength_nm != self.grid.frame.reference_wavelength_nm
        ):
            raise ValueError("reference_wavelength_nm must match grid.frame")

    def per_pixel(self, *, pixels=None) -> tuple[Array, Array]:
        """Return (e_nom, G) in sqrt(photons) per pixel and per command unit.

        If given, pixels is a (rows, columns) index pair on the grid.

        Args:
            pixels: Optional (rows, columns) index pair on the grid selecting pixels.

        Returns:
            The pair ``(e_nom, G)`` in sqrt(photons) per pixel and per command unit,
            at the selected pixels if given.
        """
        e_nom, G = self.e_nom, self.G
        if pixels is not None:
            rows, cols = pixels
            e_nom, G = e_nom[..., rows, cols], G[..., rows, cols]
        scale = self._pixel_amplitude_scale()
        return e_nom * scale, G * scale

    def _pixel_amplitude_scale(self) -> Array:
        return jnp.sqrt(like(self.grid.pixel_area, self.e_nom))


def _sensitivity_scale(response, wavelengths):
    """Scale a single-plane sensitivity on its native lambda/D grid."""
    if response.wavelength_law == "opd":
        return response.reference_wavelength_nm / wavelengths
    if response.wavelength_law == "amplitude":
        return jnp.ones_like(wavelengths)
    raise ValueError("a tabulated response must be read at its wavelength nodes")


class AbstractSpeckleField(eqx.Module):
    """Interface for deterministic time-driven speckles, including replayed cubes.

    A field runs on its own clock. Every method takes ``elapsed_s``: seconds
    since the field's clock origin, keyword-only, converted to the active
    dtype and used as given. Implementations never read ``epoch_d`` and never
    subtract a time origin.

    ``epoch`` is the field's own day zero, static host metadata: the field is
    hardware, so its clock never reads a scene epoch. ``epoch_d`` places the
    clock origin in days from ``epoch``. It is a static host binary64 float
    (``eqx.field(static=True, converter=float)``), never a leaf, and finite.
    A caller states an event time on the field's epoch with
    ``exposure.on_epoch(field.epoch).time_d``.

    ``elapsed_s(time_d)`` is the only conversion from a time on ``epoch`` and is
    not overridden. With ``epoch_d == 0`` it returns ``time_d * d2s`` in the
    input precision. With a nonzero ``epoch_d`` it subtracts in binary64 and
    requires ``time_d`` to be binary64: a Python number, a NumPy float64 or
    integer array (subtracted on the host), or a JAX float64 array
    (subtracted on the device when x64 is enabled). A concrete float64
    JAX array can also be subtracted on the host when x64 is off.
    A reduced-precision floating input raises ``PrecisionError``, at trace
    time under ``jit``. Use the original times to form ``elapsed_s`` on the
    host or to reconstruct the binary64 input. A traced float64 time
    requires x64 execution.

    A NumPy array of either byte order is accepted in a concrete (eager)
    call, where the subtraction runs on the host. Under ``jit`` JAX itself
    misreads a non-native byte-order array as it enters the compiled
    function, before ``elapsed_s`` runs, so pass native float64 there (for
    example ``time_d.astype(np.float64)``) or form ``elapsed_s`` on the host.

    The resolution of ``elapsed_s`` itself is the caller's: float32 seconds
    resolve ``elapsed_s * 2**-24``.

    All wavelengths are nm. Implementations own their native lod grid and its
    diameter/wavelength Frame. They return Images, leaving detector
    resampling to the consumer. The realization is fixed at construction,
    preserving temporal correlation across exposures without a per-call
    random key.

    Attributes:
        grid: Native lambda/D sampling, including center, basis and Frame.
        epoch: Day zero of the field's clock; static host metadata.
        epoch_d: Clock origin in days from ``epoch``; static host
            metadata, read only by ``elapsed_s``.
    """

    grid: AbstractVar[PixelGrid]
    epoch: AbstractVar[Epoch]
    epoch_d: AbstractVar[float]

    def _working_dtype(self, *values):
        """Type weak observation inputs from the field and explicit inputs."""
        leaves = (
            leaf
            for leaf in jax.tree.leaves(self)
            if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)
        )
        values = tuple(_dtype_probe(value) for value in values)
        dtype = jnp.finfo(jnp.result_type(*leaves, *values, 1.0)).dtype
        return jax.dtypes.canonicalize_dtype(dtype)

    def elapsed_s(self, time_d: ArrayLike) -> Array:
        """Convert a time stated on ``epoch`` to seconds on this field's clock.

        The refusal is decided from static facts (the dtype of ``time_d`` and
        the host value of ``epoch_d``), so it holds at trace time inside
        ``jit``, ``vmap`` and ``grad``.

        Args:
            time_d: Time in days from ``epoch``, scalar or array.
                Binary64 (a Python number, a NumPy float64 or integer array, or
                a JAX float64 array) when ``epoch_d`` is
                nonzero. Either byte order in a concrete call; native float64
                under ``jit``, where JAX misreads a byte-swapped array before
                this method runs.

        Returns:
            Seconds since the field's clock origin. Explicit float32 and
            float64 relative times keep their available precision. Weak
            relative times use the field's precision. Host absolute epoch
            subtraction uses the construction dtype for the returned offset.

        Raises:
            TypeError: If ``epoch_d`` is not a static host float.
            ValueError: If a nonzero epoch receives an unsupported time type.
            PrecisionError: If a nonzero epoch receives a reduced-precision
                floating time, or a traced float64 time cannot run in float64.
        """
        return elapsed_seconds(
            time_d, origin_d=self.epoch_d, reference=self, what="elapsed_s"
        )

    def data_mask(self) -> Array | None:
        """Cells of ``grid`` that hold data, or None when every cell does.

        A field whose array is wider than its data (data on a list of pixels,
        scattered into a zero array) returns a boolean of ``grid.shape``,
        True where it holds data; its ``realize`` is 0.0 where the mask is
        False. The mask may be a JAX array. A consumer passes it to
        ``optixstuff.grids.coverage`` as ``valid``, so the pixels whose read
        reaches a cell without data are not supported.

        Returns:
            None by default: every cell of ``grid`` holds data.
        """
        return None

    @abc.abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the host wavelength breaks [nm] for this model.

        Returns:
            Break wavelengths in nm as a 1D array; empty when the model declares none.
        """

    @abc.abstractmethod
    def realize(self, *, wavelength_nm: ArrayLike, elapsed_s: ArrayLike = 0.0) -> Image:
        """Return cell-integrated stellar_flux_fraction excess on the field's grid.

        The nominal coronagraph leakage must not be included again. Concrete
        implementations document their wavelength geometry and leading axes.

        Args:
            wavelength_nm: Wavelength in nm, scalar or array.
            elapsed_s: Seconds since the field's clock origin.

        Returns:
            Stellar flux fraction excess per cell on the field's grid.
        """

    def realize_average(
        self,
        *,
        wavelength_nm: ArrayLike,
        exposure_s: ArrayLike,
        start_elapsed_s: ArrayLike = 0.0,
        n_sub: int = 8,
    ) -> Image:
        """Time-average realized Images at n_sub exposure midpoints.

        Samples ``realize`` at ``start_elapsed_s + (index + 0.5) * step_s``
        with ``step_s = exposure_s / n_sub``, all in seconds on the field's
        clock. This second-order midpoint rule averages intensities, not
        coefficients. Only one image accumulator is held, irrespective of
        n_sub. All image metadata and wavelength leading axes are retained.

        Args:
            wavelength_nm: Wavelength in nm, scalar or array.
            exposure_s: Exposure duration in seconds.
            start_elapsed_s: Start of the exposure in seconds since the field's
                clock origin.
            n_sub: Number of exposure midpoints averaged.

        Returns:
            Time-averaged stellar flux fraction excess per cell on the field's grid.

        Raises:
            ValueError: If n_sub is not a positive integer.
        """
        if int(n_sub) < 1 or int(n_sub) != n_sub:
            raise ValueError(f"n_sub must be a positive integer, got {n_sub}")
        n_sub = int(n_sub)
        dtype = self._working_dtype(wavelength_nm, exposure_s, start_elapsed_s)
        wavelength = jnp.asarray(wavelength_nm, dtype=dtype)
        start = jnp.asarray(start_elapsed_s, dtype=dtype)
        step = jnp.asarray(exposure_s, dtype=dtype) / n_sub

        def sample(index):
            return self.realize(
                wavelength_nm=wavelength,
                elapsed_s=start + (jnp.asarray(index, dtype=dtype) + 0.5) * step,
            )

        def accumulate(index, total):
            return total + sample(index).data

        first = sample(0)
        total = jax.lax.fori_loop(1, n_sub, accumulate, first.data)
        return replace(first, data=total / n_sub)


class LinearSpeckleField(AbstractSpeckleField):
    """Abstract coefficient-driven family with one linear field law.

    The response holds the nominal focal amplitude and its sensitivity in
    density units. Squared per-pixel amplitudes divided by the response's
    input energy (unocculted photon count) are per-pixel stellar flux fractions.

    The response's ``wavelength_law`` sets the sensitivity at wavelength lambda.
    A single-plane response holds e_nom fixed. Native coordinates
    u = basis @ (pixel - center) describe lambda/D AT THE NODE wavelength,
    with angular coordinates u * grid.frame.lod_to_arcsec(lambda). The same
    native pixel spans lambda/lambda_ref times its reference angular size.
    Geometric dilation alone preserves its per-pixel flux fraction. The returned
    grid remains self.grid: a wavelength stack has one static lod grid, not one
    static angular grid. For each concrete node, attach grid.to_arcsec(node)
    before shared resample. This uses the same Frame conversion as native
    coronagraph products; their stored reference grid is not itself a request
    to rescale a monochromatic coronagraph table.

    ``epoch`` is a required static keyword. ``epoch_d`` is placement metadata,
    a static host binary64 float counted from ``epoch``, keyword-only, default
    zero. ``coefficients(*, elapsed_s)`` is a function
    of the field's own clock and does not read ``epoch_d``. Deep-contrast
    accuracy also requires x64 arrays; this class does not change global JAX
    precision.

    Attributes:
        response: Focal amplitudes and their photometric reference.
        epoch: Day zero of the field's clock; static, keyword-only, required.
        epoch_d: Clock origin in days from ``epoch``; static host
            metadata, keyword-only, default zero.
    """

    response: FocalResponse
    epoch: Epoch = eqx.field(static=True, kw_only=True)
    epoch_d: float = eqx.field(static=True, converter=float, default=0.0, kw_only=True)
    _response_ndim: ClassVar[int] = 2

    def spectral_breaks_nm(self) -> np.ndarray:
        """The wavelength dependence is smooth throughout the band.

        Returns:
            An empty array.
        """
        return np.empty(0)

    @property
    def grid(self) -> PixelGrid:
        """The native focal grid of the response."""
        return self.response.grid

    def __check_init__(self):
        """Validate native geometry and photometric ingredient shapes eagerly."""
        if self.grid.unit != "lod":
            raise ValueError("grid must use lod with its own Frame")
        if self.response.e_nom.ndim != self._response_ndim:
            raise ValueError("response has the wrong number of wavelength axes")
        if not isinstance(self.epoch, Epoch):
            raise TypeError(f"epoch must be an Epoch, got {type(self.epoch).__name__}")
        if not math.isfinite(self.epoch_d):
            raise ValueError("epoch_d must be finite")

    @abc.abstractmethod
    def coefficients(self, *, elapsed_s: ArrayLike) -> Array:
        """Return eps, shape ``(*time_shape, n_modes)``, at ``elapsed_s``.

        ``elapsed_s`` is seconds since the clock origin. Coefficient units are
        those used to define G (for example nm of wavefront error). The
        implementation reads elapsed seconds at the working precision.

        Args:
            elapsed_s: Seconds since the field's clock origin, any shape.

        Returns:
            Coefficients eps of shape ``(*time_shape, n_modes)``.
        """

    def realize(self, *, wavelength_nm: ArrayLike, elapsed_s: ArrayLike = 0.0) -> Image:
        """Evaluate ``[2 Re(conj(e_nom) G eps) + |G eps|^2] / input_energy``.

        Positive wavelength_nm may be scalar or an array, including (W, Q)
        quadrature nodes. Output shape is
        ``(*wavelength_shape, *time_shape, ny, nx)``: wavelength and time
        arrays form an outer product. Leading axes are ('band', 'quadrature')
        for (W, Q), ('wavelength',) for a vector, otherwise wavelength_0, ...;
        times use time_0, ... .
        The cross-term form avoids cancellation of nearly equal intensities.

        Args:
            wavelength_nm: Positive wavelength in nm, scalar or array, including (W, Q)
                quadrature nodes.
            elapsed_s: Seconds since the field's clock origin, scalar or array.

        Returns:
            Stellar flux fraction excess per cell, shape
            ``(*wavelength_shape, *time_shape, ny, nx)``.
        """
        dtype = self._working_dtype(wavelength_nm, elapsed_s)
        wavelengths = jnp.asarray(wavelength_nm, dtype=dtype)
        eps = self.coefficients(elapsed_s=jnp.asarray(elapsed_s, dtype=dtype))
        # Contract densities first to avoid a per-realization copy of G.
        pixel_scale = self.response._pixel_amplitude_scale()
        e_nom = self.response.e_nom * pixel_scale
        delta_e = jnp.einsum("...m,myx->...yx", eps, self.response.G) * pixel_scale
        scale = _sensitivity_scale(self.response, wavelengths)
        delta_e = scale.reshape((*scale.shape, *((1,) * delta_e.ndim))) * delta_e
        delta = 2 * jnp.real(jnp.conj(e_nom) * delta_e) + jnp.abs(delta_e) ** 2
        if wavelengths.ndim == 2:
            wave_axes = ("band", "quadrature")
        elif wavelengths.ndim == 1:
            wave_axes = ("wavelength",)
        else:
            wave_axes = tuple(f"wavelength_{i}" for i in range(wavelengths.ndim))
        time_axes = tuple(f"time_{i}" for i in range(eps.ndim - 1))
        return Image(
            delta / self.response.input_energy,
            self.grid,
            wave_axes + time_axes,
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
        )

    def broadened(self, *, quadrature: Quadrature, elapsed_s: ArrayLike = 0.0) -> Image:
        """Integrate realized nodes with unit-sum weights in each band.

        The response weights sum to the represented band width in nm.
        Dividing by that nonzero sum gives a uniform-wavelength band mean,
        in per-pixel stellar flux fractions, shape ``(W, *time_shape, ny, nx)``.
        Every node is evaluated; a quadratic law at one band-center
        wavelength cannot replace this integral.

        This mean is at fixed native lambda/D coordinates. For a fixed
        angular detector, resample EACH node using its wavelength first,
        then integrate with the stellar spectrum and throughput at the nodes.
        This helper neither applies a spectrum nor produces Rates.

        Args:
            quadrature: Wavelength nodes and weights of the bands.
            elapsed_s: Seconds since the field's clock origin, scalar or array.

        Returns:
            Band-mean stellar flux fraction excess per cell, shape
            ``(W, *time_shape, ny, nx)``.
        """
        nodes = self.realize(wavelength_nm=quadrature.nodes_nm, elapsed_s=elapsed_s)
        mean = quadrature.mean(nodes.data)
        return replace(nodes, data=mean, leading_axes=("band", *nodes.leading_axes[2:]))
