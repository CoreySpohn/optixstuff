"""One photon-to-electron contract for detector statistics and sampling."""

import abc
from dataclasses import fields
from math import isfinite
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.spectral import Quadrature
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.exposure import FramePlan
from optixstuff.rates import ElectronRates, PhotonRates


def _check_range(name, value, low, high=float("inf")):
    """Reject concrete nonfinite or out-of-range construction parameters."""
    value = jnp.asarray(value)
    if not isinstance(value, jax.core.Tracer) and not bool(
        jnp.all(jnp.isfinite(value) & (value >= low) & (value <= high))
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


class AbstractDetector(eqx.Module):
    """Focal-plane hardware; exposure timing belongs to a separate FramePlan.

    Noise parameters are per pixel, QE and survival are fractions, and
    ``excess_noise_factor_sq`` is a variance multiplier, at least one.
    Pixel scale, shape and read overlap are static PyTree geometry fields.
    Subclasses supply QE; statistics and sampling share the electron contract.
    """

    pixel_scale_arcsec: float = eqx.field(static=True, converter=float)
    shape: tuple[int, int] = eqx.field(static=True)
    survival_fraction: Array = eqx.field(default=1.0, converter=jnp.asarray)
    dark_current_e_per_pixel_s: Array = eqx.field(default=0.0, converter=jnp.asarray)
    clock_induced_charge_e_per_pixel_frame: Array = eqx.field(
        default=0.0, converter=jnp.asarray
    )
    read_noise_e_rms: Array = eqx.field(default=0.0, converter=jnp.asarray)
    excess_noise_factor_sq: Array = eqx.field(default=1.0, converter=jnp.asarray)
    read_time_s: Array = eqx.field(default=0.0, converter=jnp.asarray)
    overlaps_read: bool = eqx.field(default=False, static=True)

    def __check_init__(self):
        """Validate host geometry, survival and per-pixel hardware parameters."""
        if not isfinite(self.pixel_scale_arcsec) or self.pixel_scale_arcsec <= 0:
            raise ValueError("pixel_scale_arcsec must be finite and positive")
        _check_range("survival_fraction", self.survival_fraction, 0.0, 1.0)
        for name in (
            "dark_current_e_per_pixel_s",
            "clock_induced_charge_e_per_pixel_frame",
            "read_noise_e_rms",
            "read_time_s",
        ):
            _check_range(name, getattr(self, name), 0.0)
        _check_range("excess_noise_factor_sq", self.excess_noise_factor_sq, 1.0)

    @abc.abstractmethod
    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return QE in [0, 1], with the same shape as ``wavelength_nm``."""
        ...

    def quadrature_knots_nm(self) -> Array:
        """Wavelengths [nm] where the QE is not smooth, for band quadrature.

        A consumer building a ``Quadrature`` splits its bands at these points so
        that each node interval sees a smooth QE. A constant QE has none.
        """
        return jnp.zeros((0,))

    def electrons(self, rates: PhotonRates, quadrature: Quadrature) -> ElectronRates:
        """Integrate survival * QE * photons/s/nm into electrons/s per field.

        This is the only place detector yield is applied anywhere in the
        library. Inputs have shape ``(W, Q, *spatial)``; outputs have shape
        ``(W, *spatial)``, including scalar-zero absent contributions.
        QE is evaluated at every node before dimensional integration.
        """
        shape = rates.total().shape or quadrature.nodes_nm.shape
        yield_fraction = self.survival_fraction * self.quantum_efficiency(
            quadrature.nodes_nm
        )
        yield_fraction = jnp.reshape(
            yield_fraction, yield_fraction.shape + (1,) * (len(shape) - 2)
        )
        return ElectronRates(
            **{
                field.name: quadrature.integrate(
                    yield_fraction * jnp.broadcast_to(getattr(rates, field.name), shape)
                )
                for field in fields(rates)
            }
        )

    def mean_and_variance(
        self,
        electron_rates: ElectronRates,
        plan: FramePlan,
        n_pix: ArrayLike = 1.0,
        *,
        sum_bands: bool = False,
    ) -> tuple[Array, Array]:
        """Return electron count mean and variance, each ``(W, *spatial)``.

        Rates are already in electrons/s. Aperture rates use their effective
        pixel count ``n_pix``; images use one. Dark and CIC contribute to
        the mean and are multiplied by F2 in the variance; Gaussian read
        noise contributes only variance. With ``sum_bands=True``, sum axis
        zero only, returning ``spatial`` (a scalar for aperture rates).
        Bands and pixels are independent in this model.
        """
        mean = (
            electron_rates.total() + n_pix * self.dark_current_e_per_pixel_s
        ) * plan.live_time_s + (
            n_pix * self.clock_induced_charge_e_per_pixel_frame * plan.n_frames
        )
        variance = self.excess_noise_factor_sq * mean + (
            n_pix * self.read_noise_e_rms**2 * plan.n_frames
        )
        if sum_bands:
            return jnp.sum(mean, axis=0), jnp.sum(variance, axis=0)
        return mean, variance

    def draw(self, key: Array, electron_rates: ElectronRates, plan: FramePlan) -> Array:
        """Sample independent pixels/bands, shape ``(W, *spatial)``, n_pix=1.

        Draw source plus dark as Poisson, and CIC as an independent Poisson.
        Their sum N has mean and variance lambda. For F2=1 return N plus
        independent Gaussian read noise with variance sigma_r^2*n_frames.
        For F2>1 set theta=F2-1 and draw G|N ~ Gamma(N/theta, scale=theta);
        G=0 exactly when N=0. This sums independent unit-mean electron gains.
        E[G|N]=N, Var(G|N)=N*theta, so E[G]=lambda and
        Var(G)=E[N*theta]+Var(N)=lambda*(theta+1)=F2*lambda.
        Independent read noise adds sigma_r^2*n_frames to the variance.

        This continuous gain model matches the specified first two moments;
        it omits saturation, thresholds, gain-register location effects,
        interpixel correlations, and digitization. It is not a full sensor
        response model. Yield has already been applied by ``electrons``.
        """
        key_source, key_cic, key_gain, key_read = jax.random.split(key, 4)
        source_mean = (
            electron_rates.total() + self.dark_current_e_per_pixel_s
        ) * plan.live_time_s
        source = jax.random.poisson(key_source, source_mean)
        cic = jax.random.poisson(
            key_cic,
            self.clock_induced_charge_e_per_pixel_frame * plan.n_frames,
            shape=source.shape,
        )
        counts = (source + cic).astype(source_mean.dtype)

        def amplified(counts):
            # Guarded so the unused branch stays finite when a batched
            # ``cond`` evaluates both sides at F2 = 1.
            excess = self.excess_noise_factor_sq - 1.0
            theta = jnp.where(excess > 0, excess, 1.0)
            gain = jax.random.gamma(key_gain, jnp.maximum(counts, 1.0) / theta)
            return jnp.where(counts > 0, theta * gain, 0.0)

        signal = jax.lax.cond(
            self.excess_noise_factor_sq > 1.0, amplified, lambda n: n, counts
        )
        read = (
            self.read_noise_e_rms
            * jnp.sqrt(plan.n_frames)
            * jax.random.normal(key_read, shape=source.shape, dtype=source_mean.dtype)
        )
        return signal + read


@final
class IdealDetector(AbstractDetector):
    """QE-one detector with no dark, CIC, read, or excess noise."""

    def __init__(
        self,
        *,
        pixel_scale_arcsec: float,
        shape: tuple[int, int],
        survival_fraction: ArrayLike = 1.0,
        read_time_s: ArrayLike = 0.0,
        overlaps_read: bool = False,
    ):
        """Create ideal hardware; survival and read overhead remain configurable."""
        super().__init__(
            pixel_scale_arcsec=pixel_scale_arcsec,
            shape=shape,
            survival_fraction=survival_fraction,
            read_time_s=read_time_s,
            overlaps_read=overlaps_read,
        )

    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return ones with the input wavelength shape."""
        return jnp.ones_like(jnp.asarray(wavelength_nm), dtype=jnp.result_type(float))

    def draw(self, key: Array, electron_rates: ElectronRates, plan: FramePlan) -> Array:
        """Draw only Poisson source electrons of shape ``(W, *spatial)``."""
        return jax.random.poisson(key, electron_rates.total() * plan.live_time_s)

    def __repr__(self) -> str:
        """Summarize detector geometry and ideal yield."""
        ny, nx = self.shape
        return (
            f"IdealDetector({ny}x{nx} @ {self.pixel_scale_arcsec:.3g} arcsec/px, "
            f"QE=1.00, survival={self.survival_fraction:.3g})"
        )


@final
class Detector(AbstractDetector):
    """Constant or tabulated QE with Poisson, gamma gain, and Gaussian noise."""

    _qe: Array
    _qe_curve: tuple[Array, Array] | None

    def __init__(
        self,
        *,
        pixel_scale_arcsec: float,
        shape: tuple[int, int],
        qe: ArrayLike = 1.0,
        qe_curve: tuple[ArrayLike, ArrayLike] | None = None,
        survival_fraction: ArrayLike = 1.0,
        dark_current_e_per_pixel_s: ArrayLike = 0.0,
        clock_induced_charge_e_per_pixel_frame: ArrayLike = 0.0,
        read_noise_e_rms: ArrayLike = 0.0,
        excess_noise_factor_sq: ArrayLike = 1.0,
        read_time_s: ArrayLike = 0.0,
        overlaps_read: bool = False,
    ):
        """Create hardware with constant ``qe`` or a measured ``qe_curve``.

        ``qe_curve=(wavelengths_nm, efficiencies)`` overrides ``qe``.
        Interpolation is linear and QE is zero outside the measured range.
        Concrete QE and survival outside [0, 1] raise at construction;
        traced values must satisfy the same contract.
        """
        self._qe = jnp.asarray(qe)
        self._qe_curve = (
            None if qe_curve is None else tuple(jnp.asarray(x) for x in qe_curve)
        )
        super().__init__(
            pixel_scale_arcsec=pixel_scale_arcsec,
            shape=shape,
            survival_fraction=survival_fraction,
            dark_current_e_per_pixel_s=dark_current_e_per_pixel_s,
            clock_induced_charge_e_per_pixel_frame=clock_induced_charge_e_per_pixel_frame,
            read_noise_e_rms=read_noise_e_rms,
            excess_noise_factor_sq=excess_noise_factor_sq,
            read_time_s=read_time_s,
            overlaps_read=overlaps_read,
        )

    def __check_init__(self):
        """Validate concrete QE values and measured wavelength samples."""
        _check_range("QE", self._qe, 0.0, 1.0)
        if self._qe.ndim != 0:
            raise ValueError("constant QE must be scalar")
        if self._qe_curve is not None:
            wavelengths_nm, qe = self._qe_curve
            if (
                wavelengths_nm.ndim != 1
                or qe.shape != wavelengths_nm.shape
                or qe.size < 2
            ):
                raise ValueError(
                    "QE curve requires matching 1D arrays with >=2 samples"
                )
            _check_range("QE", qe, 0.0, 1.0)
            if not isinstance(wavelengths_nm, jax.core.Tracer) and not bool(
                jnp.all(jnp.isfinite(wavelengths_nm))
                & jnp.all(jnp.diff(wavelengths_nm) > 0)
            ):
                raise ValueError(
                    "QE wavelengths must be finite and strictly increasing"
                )

    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return constant or linearly interpolated QE at every input wavelength."""
        wavelength_nm = jnp.asarray(wavelength_nm)
        if self._qe_curve is None:
            return jnp.broadcast_to(self._qe, wavelength_nm.shape)
        wavelengths_nm, qe = self._qe_curve
        return jnp.interp(wavelength_nm, wavelengths_nm, qe, left=0.0, right=0.0)

    def quadrature_knots_nm(self) -> Array:
        """Tabulated QE wavelengths [nm]; empty for a constant QE.

        The linear interpolant has a kink at every table point, and the zero
        extension outside the table has a jump at both ends.
        """
        if self._qe_curve is None:
            return jnp.zeros((0,))
        return self._qe_curve[0]

    def __repr__(self) -> str:
        """Summarize geometry, yield, and noise parameters."""
        ny, nx = self.shape
        qe_str = "spectral" if self._qe_curve is not None else f"{self._qe:.2f}"
        return (
            f"Detector({ny}x{nx} @ {self.pixel_scale_arcsec:.3g} arcsec/px, "
            f"QE={qe_str}, survival={self.survival_fraction:.3g}, "
            f"dark={self.dark_current_e_per_pixel_s:.2g} e-/s/px, "
            f"RN={self.read_noise_e_rms:.2g} e-/read, "
            f"CIC={self.clock_induced_charge_e_per_pixel_frame:.2g} e-/frame, "
            f"F2={self.excess_noise_factor_sq:.3g})"
        )
