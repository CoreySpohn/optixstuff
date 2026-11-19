"""One photon-to-electron contract for detector statistics and sampling."""

import abc
from math import isfinite
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.exposure import FramePlan
from optixstuff.grids import PixelGrid
from optixstuff.rates import Rates


def _check_range(name, value, low, high=float("inf")):
    """Reject concrete nonfinite or out-of-range construction parameters."""
    value = jnp.asarray(value)
    if not isinstance(value, jax.core.Tracer) and not bool(
        jnp.all(jnp.isfinite(value) & (value >= low) & (value <= high))
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def _check_electrons(rates):
    """Require band-integrated electron rates, the detector statistics' input."""
    if rates.carrier != "electron" or rates.spectral:
        raise ValueError(f"rates must be in electron/s; got {rates.unit}")


def _per_pixel_rate(rates):
    """Total electron/s of per-pixel rates, the only kind draw can sample."""
    _check_electrons(rates)
    if rates.n_pix is not None or rates.variance is not None:
        raise ValueError("draw samples per-pixel rates, not a weighted aperture")
    return rates.total()


class AbstractDetector(eqx.Module):
    """Focal-plane hardware; exposure timing belongs to a separate FramePlan.

    Noise parameters are per pixel, QE and survival are fractions, and
    ``excess_noise_factor_sq`` is a variance multiplier, at least one.
    Pixel scale, shape and read overlap are static PyTree geometry fields;
    ``grid`` is the PixelGrid they define.
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

    @property
    def grid(self) -> PixelGrid:
        """Centered detector pixel grid in arcsec, without a sky orientation."""
        return PixelGrid.regular(self.shape, self.pixel_scale_arcsec, "arcsec")

    @abc.abstractmethod
    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return QE in [0, 1], with the same shape as ``wavelength_nm``."""
        ...

    @abc.abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the detector's quantum efficiency [nm]."""

    def electrons(self, rates: Rates) -> Rates:
        """Integrate survival * QE * photon/s/nm into electron/s per field.

        This is the only place detector yield is applied anywhere in the
        library. Inputs have shape ``(W, Q, *spatial)``; outputs have shape
        ``(W, *spatial)``, including scalar-zero absent contributions.
        QE is evaluated at every node before dimensional integration.
        An aperture's ``n_pix`` is kept and its ``variance`` rates are
        converted the same way.
        """
        if rates.carrier != "photon" or not rates.spectral:
            raise ValueError(f"rates must be in photon/s/nm; got {rates.unit}")
        quadrature = rates.quadrature
        shape = rates.total().shape
        yield_fraction = self.survival_fraction * self.quantum_efficiency(
            quadrature.nodes_nm
        )
        yield_fraction = jnp.reshape(
            yield_fraction, yield_fraction.shape + (1,) * (len(shape) - 2)
        )
        return Rates(
            quadrature.bands,
            "electron",
            n_pix=rates.n_pix,
            variance=None if rates.variance is None else self.electrons(rates.variance),
            **{
                name: quadrature.integrate(
                    yield_fraction * jnp.broadcast_to(getattr(rates, name), shape)
                )
                for name in rates.component_names()
            },
        )

    def mean_and_variance(
        self, rates: Rates, plan: FramePlan, *, sum_bands: bool = False
    ) -> tuple[Array, Array]:
        """Return electron count mean and variance, each ``(W, *spatial)``.

        Rates are in electron/s. A scalar plan applies to all
        bands. A (W,) field applies along axis zero and must match that axis.
        Dark current, CIC and read noise are paid once per pixel:
        ``rates.n_pix`` pixels, or one when it is None (per-pixel rates).
        For a weighted aperture, the rates are summed with w and
        ``rates.variance`` with w**2, each with its own sum of weights in
        ``n_pix``. Without variance rates the variance follows the mean
        rates; variance rates without ``n_pix`` use the mean's pixel count.
        F2 scales source, dark and CIC variance. Read noise adds independent
        per-frame variance. With ``sum_bands=True``, sum axis
        zero only, returning ``spatial`` (a scalar for aperture rates).
        Bands and pixels are independent in this model.
        """
        _check_electrons(rates)

        def poisson_counts(rate, pixels):
            live_time, frames = plan._for_bands(rates.bands, rate.ndim)
            counts = (rate + pixels * self.dark_current_e_per_pixel_s) * live_time + (
                pixels * self.clock_induced_charge_e_per_pixel_frame * frames
            )
            return counts, frames

        n_pix = 1.0 if rates.n_pix is None else rates.n_pix
        mean, frames = poisson_counts(rates.total(), n_pix)
        variance_pixels = n_pix
        shot_variance = mean
        if rates.variance is not None:
            if rates.variance.n_pix is not None:
                variance_pixels = rates.variance.n_pix
            shot_variance, frames = poisson_counts(
                rates.variance.total(), variance_pixels
            )
        variance = self.excess_noise_factor_sq * shot_variance + (
            variance_pixels * self.read_noise_e_rms**2 * frames
        )
        if sum_bands:
            return jnp.sum(mean, axis=0), jnp.sum(variance, axis=0)
        return mean, variance

    def log_prob(
        self,
        observed: ArrayLike,
        rates: Rates,
        plan: FramePlan,
        *,
        mask: ArrayLike | None = None,
    ) -> Array:
        """Return the log density of an observed frame given expected electron rates.

        ``observed`` is in electrons with shape ``(*batch, W, *spatial)``, the
        shape of ``mean_and_variance(rates, plan)`` behind optional leading
        batch axes. The result sums the per-pixel log densities over the
        trailing axes of that shape and keeps ``*batch``. ``mask`` (True
        uses the pixel) broadcasts against the mean. To batch over rates or
        a detector, ``vmap`` the call. Pixels and bands are independent, as in
        ``mean_and_variance``, and every term is normalized, constants included.

        Two forms, chosen from the hardware parameters:

        * No read noise and unit excess factor (the photon-counting case, always
          for ``IdealDetector``): the Poisson pmf of the mean ``lambda`` from
          ``mean_and_variance``, ``y log(lambda) - lambda - log(y!)``, with
          ``y`` the count. This is the observation density at low counts and
          sums to one over ``y = 0, 1, 2, ...``.
        * Otherwise: the Gaussian density with the mean and variance of
          ``mean_and_variance``, ``-(y - m)^2 / (2 v) - log(2 pi v) / 2``. This is
          the moment-matched approximation of the Poisson plus gain plus
          read-noise draw (the first two moments of ``draw`` are exact); it is
          the same normalized Gaussian as a per-pixel Gaussian with
          ``sigma = sqrt(v)``. It is not exact at low counts. The exact
          Poisson-Gaussian convolution is not summed here.
        """
        mean, variance = self.mean_and_variance(rates, plan)
        observed = jnp.asarray(observed, dtype=mean.dtype)
        poisson = (self.read_noise_e_rms == 0) & (self.excess_noise_factor_sq == 1)
        # The branch not taken is evaluated on safe values so gradients stay finite.
        safe_variance = jnp.where(poisson | (variance <= 0), 1.0, variance)
        gauss = -0.5 * (observed - mean) ** 2 / safe_variance - 0.5 * jnp.log(
            2.0 * jnp.pi * safe_variance
        )
        safe_mean = jnp.where(poisson, mean, 1.0)
        safe_obs = jnp.where(poisson, observed, 0.0)
        positive = safe_mean > 0
        log_mean = jnp.log(jnp.where(positive, safe_mean, 1.0))
        count_term = jnp.where(
            positive,
            safe_obs * log_mean,
            jnp.where(safe_obs == 0, 0.0, -jnp.inf),
        )
        pmf = count_term - safe_mean - jsp.gammaln(safe_obs + 1.0)
        terms = jnp.where(poisson, pmf, gauss)
        if mask is not None:
            terms = jnp.where(jnp.broadcast_to(mask, terms.shape), terms, 0.0)
        return jnp.sum(terms, axis=tuple(range(-mean.ndim, 0)))

    def draw(self, key: Array, rates: Rates, plan: FramePlan) -> Array:
        """Sample independent pixels/bands, shape ``(W, *spatial)``, n_pix=1.

        A (W,) plan field must match the leading axis of the rates.
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
        rate = _per_pixel_rate(rates)
        live_time, frames = plan._for_bands(rates.bands, rate.ndim)
        key_source, key_cic, key_gain, key_read = jax.random.split(key, 4)
        source_mean = (rate + self.dark_current_e_per_pixel_s) * live_time
        source = jax.random.poisson(key_source, source_mean)
        cic = jax.random.poisson(
            key_cic,
            self.clock_induced_charge_e_per_pixel_frame * frames,
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
            * jnp.sqrt(frames)
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

    def spectral_breaks_nm(self) -> np.ndarray:
        """Unit QE has no wavelength breaks."""
        return np.empty(0)

    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return ones with the input wavelength shape."""
        return jnp.ones_like(jnp.asarray(wavelength_nm), dtype=jnp.result_type(float))

    def draw(self, key: Array, rates: Rates, plan: FramePlan) -> Array:
        """Draw only Poisson source electrons of shape ``(W, *spatial)``."""
        rate = _per_pixel_rate(rates)
        live_time, _ = plan._for_bands(rates.bands, rate.ndim)
        return jax.random.poisson(key, rate * live_time)

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

    def spectral_breaks_nm(self) -> np.ndarray:
        """Tabulated QE wavelengths [nm]; empty for a constant QE.

        The linear interpolant has a kink at every table point, and the zero
        extension outside the table has a jump at both ends.
        """
        if self._qe_curve is None:
            return np.empty(0)
        return np.asarray(self._qe_curve[0])

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
