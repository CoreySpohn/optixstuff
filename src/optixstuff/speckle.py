"""Time-varying stellar leakage excess on explicit focal-plane grids.

The deterministic coronagraph supplies the nominal leakage floor. Speckle
Images contain only the excess over that floor, as per-pixel fractions of
unocculted stellar flux; coherent interference can make the excess negative.
"""

import abc
import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.constants import s2d
from hwoutils.grids import Image, PixelGrid
from hwoutils.spectral import Quadrature
from jax.typing import ArrayLike
from jaxtyping import Array


class AbstractSpeckleField(eqx.Module):
    """Interface for deterministic time-driven speckles, including replayed cubes.

    All wavelengths are nm and timestamps are days from the scene epoch.
    Implementations own their native lod grid and its diameter/wavelength
    Frame. They return Images, leaving detector resampling to the consumer.
    The realization is fixed at construction, preserving temporal correlation
    across exposures without a per-call random key.
    """

    grid: AbstractVar[PixelGrid]
    """Native lambda/D sampling, including center, basis and Frame."""

    epoch_d: AbstractVar[float]
    """Clock origin in days from the scene epoch, subtracted by implementations."""

    @abc.abstractmethod
    def realize(self, *, wavelength_nm: ArrayLike, time_d: ArrayLike = 0.0) -> Image:
        """Return per_pixel stellar_flux_fraction excess on the field's grid.

        The nominal coronagraph leakage must not be included again. Concrete
        implementations document their wavelength geometry and leading axes.
        """

    def realize_average(
        self,
        *,
        wavelength_nm: ArrayLike,
        exposure_s: ArrayLike,
        start_time_d: ArrayLike = 0.0,
        n_sub: int = 8,
    ) -> Image:
        """Time-average realized Images at n_sub exposure midpoints.

        exposure_s is seconds; start_time_d is days from the scene epoch.
        This second-order midpoint rule averages intensities, not coefficients.
        Only one image accumulator is held, irrespective of n_sub. All image
        metadata and wavelength leading axes are retained. Enable JAX x64 when
        sub-exposure offsets need more precision than the absolute day clock.

        Raises:
            ValueError: If n_sub is not a positive integer.
        """
        if int(n_sub) < 1 or int(n_sub) != n_sub:
            raise ValueError(f"n_sub must be a positive integer, got {n_sub}")
        n_sub = int(n_sub)
        start = jnp.asarray(start_time_d, dtype=float)
        step = jnp.asarray(exposure_s, dtype=float) * s2d / n_sub

        def sample(index):
            return self.realize(
                wavelength_nm=wavelength_nm, time_d=start + (index + 0.5) * step
            )

        def accumulate(index, total):
            return total + sample(index).data

        first = sample(0)
        total = jax.lax.fori_loop(1, n_sub, accumulate, first.data)
        return replace(first, data=total / n_sub)


class LinearSpeckleField(AbstractSpeckleField):
    """Abstract coefficient-driven family with one linear field law.

    e_nom is the complex nominal focal amplitude at reference_wavelength_nm,
    shape (ny, nx). G is the complex amplitude sensitivity per coefficient
    unit, shape (n_modes, ny, nx). Amplitudes are in sqrt(energy per pixel):
    their squared magnitudes divided by input_energy (unocculted energy)
    are per-pixel stellar flux fractions. A producer of intensity-density
    amplitudes must multiply BOTH e_nom and G by sqrt(grid.pixel_area) first.
    No further pixel-area factor or peak normalization is applied here.

    At wavelength lambda, the phase sensitivity is G * lambda_ref / lambda;
    e_nom is held fixed. Native coordinates u = basis @ (pixel - center)
    describe lambda/D AT THE NODE wavelength, with angular coordinates
    u * grid.frame.lod_to_arcsec(lambda). Thus the same native pixel spans
    lambda/lambda_ref times its reference angular size, without changing its
    per-pixel flux fraction for geometric dilation alone. The returned grid
    remains self.grid: a wavelength stack has one static lod grid, not one
    static angular grid. For each concrete node, attach grid.to_arcsec(node)
    before shared resample. This uses the same Frame conversion as native
    coronagraph products; their stored reference grid is not itself a request
    to rescale a monochromatic coronagraph table.

    epoch_d is a static host binary64 float. Implement coefficients(time_d)
    by subtracting epoch_d before reducing precision, then convert days with
    hwoutils constants if needed. A float32 absolute timestamp has already
    lost information; enable JAX x64 and promote to float64 BEFORE adding a
    small offset to an absolute epoch on device. Deep-contrast accuracy also
    requires x64 arrays; this class does not change global JAX precision.
    """

    e_nom: Array = eqx.field(converter=jnp.asarray)
    G: Array = eqx.field(converter=jnp.asarray)
    input_energy: Array = eqx.field(converter=jnp.asarray)
    grid: PixelGrid
    reference_wavelength_nm: float = eqx.field(static=True, converter=float)
    epoch_d: float = eqx.field(static=True, converter=float, default=0.0, kw_only=True)

    def __check_init__(self):
        """Validate native geometry and photometric ingredient shapes eagerly."""
        if self.grid.unit != "lod":
            raise ValueError("grid must use lod with its own Frame")
        if self.e_nom.shape != self.grid.shape:
            raise ValueError("e_nom must have shape grid.shape (ny, nx)")
        if self.G.ndim != 3 or self.G.shape[1:] != self.grid.shape:
            raise ValueError("G must have shape (n_modes, ny, nx)")
        if self.input_energy.ndim != 0:
            raise ValueError("input_energy must be scalar")
        if not isinstance(self.input_energy, jax.core.Tracer):
            energy = float(self.input_energy)
            if not math.isfinite(energy) or energy <= 0:
                raise ValueError("input_energy must be positive and finite")
        if self.reference_wavelength_nm != self.grid.frame.reference_wavelength_nm:
            raise ValueError("reference_wavelength_nm must match grid.frame")
        if not math.isfinite(self.epoch_d):
            raise ValueError("epoch_d must be finite")

    @abc.abstractmethod
    def coefficients(self, time_d: ArrayLike) -> Array:
        """Return eps, shape (*time_shape, n_modes), at absolute scene days.

        The implementation forms elapsed time time_d - epoch_d. Coefficient
        units are those used to define G (for example nm of wavefront error).
        """

    def realize(self, *, wavelength_nm: ArrayLike, time_d: ArrayLike = 0.0) -> Image:
        """Evaluate [2 Re(conj(e_nom) G eps) + |G eps|^2] / input_energy.

        Positive wavelength_nm may be scalar or an array, including (W, Q)
        quadrature nodes. Output shape is (*wavelength_shape, *time_shape,
        ny, nx): wavelength and time arrays form an outer product. Leading
        axes are ('band', 'quadrature') for (W, Q), ('wavelength',) for a
        vector, otherwise wavelength_0, ...; times use time_0, ... .
        The cross-term form avoids cancellation of nearly equal intensities.
        """
        wavelengths = jnp.asarray(wavelength_nm)
        eps = self.coefficients(time_d)
        delta_e = jnp.einsum("...m,myx->...yx", eps, self.G)
        scale = self.reference_wavelength_nm / wavelengths
        delta_e = scale.reshape((*scale.shape, *((1,) * delta_e.ndim))) * delta_e
        delta = 2 * jnp.real(jnp.conj(self.e_nom) * delta_e) + jnp.abs(delta_e) ** 2
        if wavelengths.ndim == 2:
            wave_axes = ("band", "quadrature")
        elif wavelengths.ndim == 1:
            wave_axes = ("wavelength",)
        else:
            wave_axes = tuple(f"wavelength_{i}" for i in range(wavelengths.ndim))
        time_axes = tuple(f"time_{i}" for i in range(eps.ndim - 1))
        return Image(
            delta / self.input_energy,
            self.grid,
            wave_axes + time_axes,
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
            outside="reflect",
        )

    def broadened(self, *, quadrature: Quadrature, time_d: ArrayLike = 0.0) -> Image:
        """Integrate realized nodes with unit-sum weights in each band.

        Quadrature.weights_nm sum to the represented band width in nm.
        Dividing by that nonzero sum gives a uniform-wavelength band mean,
        in per-pixel stellar flux fractions, shape (W, *time_shape, ny, nx).
        Every node is evaluated; a quadratic law at one band-center
        wavelength cannot replace this integral.

        This mean is at fixed native lambda/D coordinates. For a fixed
        angular detector, resample EACH node using its wavelength first,
        then integrate with the stellar spectrum and throughput at the nodes.
        This helper neither applies a spectrum nor produces PhotonRates.
        """
        nodes = self.realize(wavelength_nm=quadrature.nodes_nm, time_d=time_d)
        integral = quadrature.integrate(nodes.data)
        width = jnp.sum(quadrature.weights_nm, axis=1)
        mean = integral / width.reshape((-1, *((1,) * (integral.ndim - 1))))
        return replace(nodes, data=mean, leading_axes=("band", *nodes.leading_axes[2:]))
