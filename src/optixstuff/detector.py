"""One photon-to-electron contract for detector statistics and sampling."""

import abc
from math import isfinite
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from hwoutils.leaves import as_float_leaf
from jax.typing import ArrayLike
from jaxtyping import Array

from optixstuff.exposure import FramePlan
from optixstuff.grids import PixelGrid
from optixstuff.rates import RATE_FLOOR_RTOL, Rates, floor_rate
from optixstuff.spectral import Knots, as_knots


def _check_range(name, value, low, high=float("inf")):
    """Reject concrete nonfinite or out-of-range construction parameters."""
    value = jnp.asarray(value)
    if not isinstance(value, jax.core.Tracer) and not bool(
        jnp.all(jnp.isfinite(value) & (value >= low) & (value <= high))
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def _per_band_pixels(n_pix, ndim):
    """Align a scalar or (W,) pixel count with ndim-dimensional rates, band first."""
    if n_pix is None:
        return 1.0
    return (
        n_pix.reshape(n_pix.shape + (1,) * (ndim - n_pix.ndim)) if n_pix.ndim else n_pix
    )


def _check_electrons(rates):
    """Require band-integrated electron rates, the detector statistics' input."""
    if rates.carrier != "electron" or rates.spectral:
        raise ValueError(f"rates must be in electron/s; got {rates.unit}")


def _floored_total(rates):
    """Total rate floored at zero against its frame peak, and the flag beyond it.

    The frame peak is the largest summed absolute component rate over the
    spatial axes of each band; a read with no spatial axes is its own frame.
    """
    total = rates.total()
    magnitude = jnp.broadcast_to(
        sum(jnp.abs(getattr(rates, name)) for name in rates.component_names()),
        total.shape,
    )
    spatial = tuple(range(1, total.ndim))
    scale = jnp.max(magnitude, axis=spatial, keepdims=True) if spatial else magnitude
    return floor_rate(total, scale)


def _floored(rates):
    """Rates whose totals are floored, and the flag of a total beyond the floor."""
    total, bad = _floored_total(rates)
    variance = rates.variance
    if variance is not None:
        variance, bad_variance = _floored(variance)
        bad = bad | bad_variance
    floored = Rates(
        rates.bands,
        rates.carrier,
        planet=total,
        n_pix=rates.n_pix,
        variance=variance,
        supported=rates.supported,
        static_supported=rates.static_supported,
    )
    return floored, bad


def _per_pixel_rate(rates):
    """Total electron/s of per-pixel rates, the only kind draw can sample.

    The total is floored at zero; a total below the floor raises, because a
    Poisson draw of a negative mean returns -1 without notice.
    """
    _check_electrons(rates)
    if rates.n_pix is not None or rates.variance is not None:
        raise ValueError("draw samples per-pixel rates, not a weighted aperture")
    rate, bad = _floored_total(rates)
    return eqx.error_if(
        rate,
        bad,
        f"a band rate is negative by more than {RATE_FLOOR_RTOL:g} of the frame "
        "peak: the band rule under-resolves the scene. Declare a break at the "
        "edge with breaks_nm, or raise order.",
    )


class AbstractDetector(eqx.Module):
    """Focal-plane hardware; exposure timing belongs to a separate FramePlan.

    Noise parameters are per pixel, QE and survival are fractions, and
    ``excess_noise_factor_sq`` is a variance multiplier, at least one.
    Pixel scale, shape and read overlap are static PyTree geometry fields;
    ``grid`` is the PixelGrid they define. Pixel scale and shape are static
    because the detector grid, every resampling operator onto it and every
    support mask are built from them on the host, so a fit cannot free them. An
    astrometric plate-scale uncertainty is a scale on the sky-to-detector map,
    not a free pixel scale. Every other measured field is an array leaf.
    ``photon_counting`` is a static flag that selects the Poisson form of
    ``log_prob``; it requires read noise 0 and excess factor 1.
    Subclasses supply QE; statistics and sampling share the electron contract.

    Attributes:
        pixel_scale_arcsec: Detector pixel size in arcsec.
        shape: Detector pixel shape (ny, nx).
        survival_fraction: Fraction of photons surviving the detector window, in
            [0, 1].
        dark_current_e_per_pixel_s: Dark current in electrons per pixel per second.
        clock_induced_charge_e_per_pixel_frame: Clock-induced charge in electrons
            per pixel per frame.
        read_noise_e_rms: Read noise in electrons rms per pixel per frame.
        excess_noise_factor_sq: Variance multiplier F2 of the gain, at least one.
        read_time_s: Seconds spent reading a frame.
        overlaps_read: Whether collection continues during the read.
        photon_counting: Whether ``log_prob`` takes the Poisson form.
    """

    pixel_scale_arcsec: float = eqx.field(static=True, converter=float)
    shape: tuple[int, int] = eqx.field(static=True)
    survival_fraction: Array = eqx.field(default=1.0, converter=as_float_leaf)
    dark_current_e_per_pixel_s: Array = eqx.field(default=0.0, converter=as_float_leaf)
    clock_induced_charge_e_per_pixel_frame: Array = eqx.field(
        default=0.0, converter=as_float_leaf
    )
    read_noise_e_rms: Array = eqx.field(default=0.0, converter=as_float_leaf)
    excess_noise_factor_sq: Array = eqx.field(default=1.0, converter=as_float_leaf)
    read_time_s: Array = eqx.field(default=0.0, converter=as_float_leaf)
    overlaps_read: bool = eqx.field(default=False, static=True)
    photon_counting: bool = eqx.field(default=False, static=True)

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
        if self.photon_counting:
            _check_range("read_noise_e_rms", self.read_noise_e_rms, 0.0, 0.0)
            _check_range(
                "excess_noise_factor_sq", self.excess_noise_factor_sq, 1.0, 1.0
            )

    @property
    def grid(self) -> PixelGrid:
        """Centered detector pixel grid in arcsec, without a sky orientation."""
        return PixelGrid.regular(self.shape, self.pixel_scale_arcsec, "arcsec")

    @abc.abstractmethod
    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return QE in [0, 1], with the same shape as ``wavelength_nm``.

        Args:
            wavelength_nm: Wavelengths in nm.

        Returns:
            Quantum efficiency in [0, 1], shaped like ``wavelength_nm``.
        """
        ...

    @abc.abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the detector's quantum efficiency [nm].

        Returns:
            Wavelengths in nm where the quantum efficiency has a kink or a jump.
        """

    def electrons(self, rates: Rates) -> Rates:
        """Integrate survival * QE * photon/s/nm into electron/s per field.

        This is the only place detector yield is applied anywhere in the
        library. Inputs have shape ``(W, Q, *spatial)``; outputs have shape
        ``(W, *spatial)``, including scalar-zero absent contributions.
        QE is evaluated at every node before dimensional integration.
        An aperture's ``n_pix`` and both support flags are kept and its
        ``variance`` rates are converted the same way.

        Args:
            rates: Spectral Rates in photon/s/nm.

        Returns:
            Band-integrated Rates in electron/s.
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
            supported=rates.supported,
            static_supported=rates.static_supported,
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

        Args:
            rates: Band-integrated Rates in electron/s.
            plan: Frame time and frame count per band.
            sum_bands: Whether to sum over the band axis.

        Returns:
            Mean and variance of the electron count, each of shape ``(W, *spatial)``
            (``spatial`` only when ``sum_bands`` is True).
        """
        _check_electrons(rates)

        def poisson_counts(rate, pixels):
            live_time, frames = plan._for_bands(rates.bands, rate.ndim)
            counts = (rate + pixels * self.dark_current_e_per_pixel_s) * live_time + (
                pixels * self.clock_induced_charge_e_per_pixel_frame * frames
            )
            return counts, frames

        total = rates.total()
        n_pix = _per_band_pixels(rates.n_pix, total.ndim)
        mean, frames = poisson_counts(total, n_pix)
        variance_pixels = n_pix
        shot_variance = mean
        if rates.variance is not None:
            shot_total = rates.variance.total()
            if rates.variance.n_pix is not None:
                variance_pixels = _per_band_pixels(
                    rates.variance.n_pix, shot_total.ndim
                )
            shot_variance, frames = poisson_counts(shot_total, variance_pixels)
        variance = self.excess_noise_factor_sq * shot_variance + (
            variance_pixels * self.read_noise_e_rms**2 * frames
        )
        if sum_bands:
            return jnp.sum(mean, axis=0), jnp.sum(variance, axis=0)
        return mean, variance

    def log_prob(self, observed: ArrayLike, rates: Rates, plan: FramePlan) -> Array:
        """Return the elementwise log density of an observed frame given electron rates.

        ``observed`` is in electrons and broadcasts against the mean of
        ``mean_and_variance(rates, plan)``; the result has the broadcast shape.
        Nothing is masked or summed: the caller masks the pixels it uses and
        sums them, in the precision it needs. Pixels and bands are independent,
        as in ``mean_and_variance``, and every term is normalized, constants
        included. The form is a static fact of the detector, ``photon_counting``,
        so one form is traced and the density is continuous in every traced
        hardware parameter.

        * ``photon_counting`` true (always for ``IdealDetector``): the Poisson
          pmf of the mean ``lambda``, ``y log(lambda) - lambda - log(y!)``. The
          observation must be integer counts; a non-integer value is outside the
          contract and is scored through ``gammaln`` without notice. A zero mean
          gives ``-inf`` for ``y > 0`` and 0 for ``y = 0``, with a zero gradient
          (a wall, not a penalty).
        * Otherwise, for every read noise including zero: the Gaussian density
          with the mean and variance of ``mean_and_variance``,
          ``-(y - m)^2 / (2 v) - log(2 pi v) / 2``. This is the moment-matched
          approximation of the Poisson plus gain plus read-noise draw (the first
          two moments of ``draw`` are exact) and is not exact at low counts.
          A zero variance gives 0 where ``y`` equals the mean and ``-inf``
          elsewhere.

        The total rate passes through ``floor_rate``, as in ``draw``: a negative
        total within ``RATE_FLOOR_RTOL`` of the frame peak is scored as a rate
        of zero, and a more negative one gives ``-inf`` in both forms, so a
        sampler rejects such a proposal instead of stopping.

        Args:
            observed: Observed frame in electrons, broadcast against the mean.
            rates: Band-integrated Rates in electron/s.
            plan: Frame time and frame count per band.

        Returns:
            Log density of ``observed``, with the broadcast shape.
        """
        floored, bad = _floored(rates)
        mean, variance = self.mean_and_variance(floored, plan)
        observed = jnp.asarray(observed, dtype=mean.dtype)
        if self.photon_counting:
            # Safe values keep the gradient finite where the mean is zero.
            positive = mean > 0
            log_mean = jnp.log(jnp.where(positive, mean, 1.0))
            count_term = jnp.where(
                positive,
                observed * log_mean,
                jnp.where(observed == 0, 0.0, -jnp.inf),
            )
            density = count_term - mean - jsp.gammaln(observed + 1.0)
            return jnp.where(bad | (mean < 0), -jnp.inf, density)
        positive = variance > 0
        safe_variance = jnp.where(positive, variance, 1.0)
        gauss = -0.5 * (observed - mean) ** 2 / safe_variance - 0.5 * jnp.log(
            2.0 * jnp.pi * safe_variance
        )
        density = jnp.where(positive, gauss, jnp.where(observed == mean, 0.0, -jnp.inf))
        return jnp.where(bad, -jnp.inf, density)

    def fisher_root(self, rates: Rates, plan: FramePlan) -> Array:
        """Return a vector ``r`` whose Jacobian gives the Fisher information.

        For parameters ``theta`` that enter through ``rates`` (or the plan),
        ``J^T J`` with ``J = dr/dtheta`` is the expected information of
        ``log_prob``, summed over independent pixels and bands. The root is in
        the shape of the law's parts, not flattened, so a caller may mask pixels
        before it flattens.

        * ``photon_counting``: ``2 sqrt(mean)``, shape of the mean. The
          information of a Poisson mean is ``dm^2 / m``. It is zero (and has zero
          derivative, no NaN) where the mean is not positive, the wall of
          ``log_prob``.
        * Otherwise, the moment-matched Gaussian of ``log_prob``: the stack
          ``(mean / sqrt(v0), ln(variance) / sqrt(2))`` with ``v0 =
          stop_gradient(variance)``, shape ``(2, *mean.shape)``. Its information
          is ``dm^2 / v + dv^2 / (2 v^2)``, covering the mean-dependent variance.
          Both parts are zero where the variance is not positive.

        A subclass that overrides ``log_prob`` has a different law and must
        override this too; otherwise it raises ``NotImplementedError``.

        Args:
            rates: Band-integrated Rates in electron/s.
            plan: Frame time and frame count per band.

        Returns:
            The root vector, in the shape of the law's parts described above.
        """
        cls = type(self)
        if (
            cls.log_prob is not AbstractDetector.log_prob
            and cls.fisher_root is AbstractDetector.fisher_root
        ):
            raise NotImplementedError(
                f"{cls.__name__} overrides log_prob; its law has no closed-form "
                "Fisher root here. Override fisher_root with the root of that law."
            )
        mean, variance = self.mean_and_variance(rates, plan)
        if self.photon_counting:
            positive = mean > 0
            root = 2.0 * jnp.sqrt(jnp.where(positive, mean, 1.0))
            return jnp.where(positive, root, 0.0)
        positive = variance > 0
        safe_variance = jnp.where(positive, variance, 1.0)
        mean_part = mean / jnp.sqrt(jax.lax.stop_gradient(safe_variance))
        variance_part = jnp.log(safe_variance) / jnp.sqrt(2.0)
        return jnp.where(positive, jnp.stack([mean_part, variance_part]), 0.0)

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

        The total source rate passes through ``floor_rate``: a negative total
        within ``RATE_FLOOR_RTOL`` of the frame peak is drawn as zero, and a
        more negative one raises, also under ``jit``, ``vmap`` and ``grad``.

        Args:
            key: JAX PRNG key.
            rates: Per-pixel Rates in electron/s.
            plan: Frame time and frame count per band.

        Returns:
            Sampled electron counts, shape ``(W, *spatial)``.
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
        """Create ideal hardware; survival and read overhead remain configurable.

        Args:
            pixel_scale_arcsec: Detector pixel size in arcsec.
            shape: Detector pixel shape (ny, nx).
            survival_fraction: Fraction of photons surviving the detector window.
            read_time_s: Seconds spent reading a frame.
            overlaps_read: Whether collection continues during the read.
        """
        super().__init__(
            pixel_scale_arcsec=pixel_scale_arcsec,
            shape=shape,
            survival_fraction=survival_fraction,
            read_time_s=read_time_s,
            overlaps_read=overlaps_read,
            photon_counting=True,
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Unit QE has no wavelength breaks.

        Returns:
            An empty array.
        """
        return np.empty(0)

    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return ones with the input wavelength shape.

        Args:
            wavelength_nm: Wavelengths in nm.

        Returns:
            Ones, shaped like ``wavelength_nm``.
        """
        return jnp.ones_like(jnp.asarray(wavelength_nm), dtype=jnp.result_type(float))

    def draw(self, key: Array, rates: Rates, plan: FramePlan) -> Array:
        """Draw only Poisson source electrons of shape ``(W, *spatial)``.

        The total rate is floored as in ``AbstractDetector.draw``, and a total
        below the floor raises.

        Args:
            key: JAX PRNG key.
            rates: Per-pixel Rates in electron/s.
            plan: Frame time and frame count per band.

        Returns:
            Poisson electron counts, shape ``(W, *spatial)``.
        """
        rate = _per_pixel_rate(rates)
        live_time, _ = plan._for_bands(rates.bands, rate.ndim)
        return jax.random.poisson(key, rate * live_time)

    def __repr__(self) -> str:
        """Summarize detector geometry and ideal yield.

        Returns:
            One-line summary string.
        """
        ny, nx = self.shape
        return (
            f"IdealDetector({ny}x{nx} @ {self.pixel_scale_arcsec:.3g} arcsec/px, "
            f"QE=1.00, survival={self.survival_fraction:.3g})"
        )


@final
class Detector(AbstractDetector):
    """Constant or tabulated QE with Poisson, gamma gain, and Gaussian noise."""

    _qe: Array
    _qe_knots: Knots | None = eqx.field(static=True)
    _qe_values: Array | None

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
        photon_counting: bool = False,
    ):
        """Create hardware with constant ``qe`` or a measured ``qe_curve``.

        ``qe_curve=(wavelengths_nm, efficiencies)`` overrides ``qe``.
        Interpolation is linear and QE is zero outside the measured range.
        Concrete QE and survival outside [0, 1] raise at construction;
        traced values must satisfy the same contract.

        Args:
            pixel_scale_arcsec: Detector pixel size in arcsec.
            shape: Detector pixel shape (ny, nx).
            qe: Constant quantum efficiency in [0, 1].
            qe_curve: Measured QE as (wavelengths in nm, efficiencies), overriding
                ``qe``.
            survival_fraction: Fraction of photons surviving the detector window.
            dark_current_e_per_pixel_s: Dark current in electrons per pixel per second.
            clock_induced_charge_e_per_pixel_frame: Clock-induced charge in electrons
                per pixel per frame.
            read_noise_e_rms: Read noise in electrons rms per pixel per frame.
            excess_noise_factor_sq: Variance multiplier F2 of the gain, at least one.
            read_time_s: Seconds spent reading a frame.
            overlaps_read: Whether collection continues during the read.
            photon_counting: Whether ``log_prob`` takes the Poisson form; requires
                zero read noise and F2 of one.
        """
        self._qe = as_float_leaf(qe)
        if qe_curve is None:
            self._qe_knots, self._qe_values = None, None
        else:
            wavelengths_nm, efficiencies = qe_curve
            self._qe_knots = as_knots(wavelengths_nm)
            self._qe_values = as_float_leaf(efficiencies)
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
            photon_counting=photon_counting,
        )

    def __check_init__(self):
        """Validate concrete QE values and measured wavelength samples."""
        _check_range("QE", self._qe, 0.0, 1.0)
        if self._qe.ndim != 0:
            raise ValueError("constant QE must be scalar")
        if self._qe_knots is not None:
            qe = self._qe_values
            if qe.shape != (len(self._qe_knots),):
                raise ValueError(
                    "QE curve requires matching 1D arrays with >=2 samples"
                )
            _check_range("QE", qe, 0.0, 1.0)

    def quantum_efficiency(self, wavelength_nm: ArrayLike) -> Array:
        """Return constant or linearly interpolated QE at every input wavelength.

        Args:
            wavelength_nm: Wavelengths in nm.

        Returns:
            Quantum efficiency in [0, 1], shaped like ``wavelength_nm``.
        """
        wavelength_nm = jnp.asarray(wavelength_nm)
        if self._qe_knots is None:
            return jnp.broadcast_to(self._qe, wavelength_nm.shape)
        return jnp.interp(
            wavelength_nm,
            self._qe_knots.array(),
            self._qe_values,
            left=0.0,
            right=0.0,
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Tabulated QE wavelengths [nm]; empty for a constant QE.

        The linear interpolant has a kink at every table point, and the zero
        extension outside the table has a jump at both ends.

        Returns:
            Tabulated QE wavelengths in nm, or an empty array for a constant QE.
        """
        if self._qe_knots is None:
            return np.empty(0)
        return self._qe_knots.nm

    def __repr__(self) -> str:
        """Summarize geometry, yield, and noise parameters.

        Returns:
            One-line summary string.
        """
        ny, nx = self.shape
        qe_str = "spectral" if self._qe_knots is not None else f"{self._qe:.2f}"
        return (
            f"Detector({ny}x{nx} @ {self.pixel_scale_arcsec:.3g} arcsec/px, "
            f"QE={qe_str}, survival={self.survival_fraction:.3g}, "
            f"dark={self.dark_current_e_per_pixel_s:.2g} e-/s/px, "
            f"RN={self.read_noise_e_rms:.2g} e-/read, "
            f"CIC={self.clock_induced_charge_e_per_pixel_frame:.2g} e-/frame, "
            f"F2={self.excess_noise_factor_sq:.3g})"
        )
