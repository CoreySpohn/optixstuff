# Detector models

All detector calculations follow one chain:
`PhotonRates -> detector.electrons -> ElectronRates -> moments or draw`.
Readers supply the six source terms in photons/s/nm at quadrature nodes,
with shape `(W, Q)` for apertures or `(W, Q, ny, nx)` for images.
Absent terms may be scalar zero. The terms are `planet`, `star_leakage`,
`local_zodi`, `exozodi`, `thermal`, and `stellar_companions`.

`PhotonRates.integrate(quadrature)` returns `IntegratedPhotonRates`, a
record in photons/s with the same fields and shape `(W, *spatial)`.
It contains no detector effects. `total()` sums the fields with broadcasting.
`PhotonRates.zeros((W, Q, ny, nx))` allocates six zero integrands.

## Hardware and yield

`IdealDetector` has QE one, zero dark current, zero clock-induced charge
(CIC), zero read noise, and excess noise factor squared one. Its draw is
only a Poisson draw of the supplied electron rate times live time.
`Detector` supports constant `qe` or a measured `qe_curve`, interpolated
linearly and zero outside its wavelength range. Both constructors are
keyword-only. `quantum_efficiency(wavelength_nm)` returns an array with
the same shape as the input, including for constant QE.

```python
import jax
import jax.numpy as jnp
from hwoutils.spectral import Band, Quadrature
from optixstuff import Detector, PhotonRates, frames_for_time

quadrature = Quadrature.build([Band(500.0, 600.0)], knots_nm=None, order=3)
rates = PhotonRates(planet=jnp.ones((1, 3, 2, 2)))
detector = Detector(
    pixel_scale_arcsec=0.01,
    shape=(2, 2),
    qe_curve=([400.0, 600.0, 800.0], [0.2, 0.8, 0.4]),
    survival_fraction=0.9,
    dark_current_e_per_pixel_s=1e-4,
    clock_induced_charge_e_per_pixel_frame=1e-3,
    read_noise_e_rms=3.0,
    excess_noise_factor_sq=1.0,
    read_time_s=0.05,
    overlaps_read=False,
)
electrons = detector.electrons(rates, quadrature)
plan = frames_for_time(time_s=600.0, frame_time_s=300.0)
mean, variance = detector.mean_and_variance(electrons, plan)
image = detector.draw(jax.random.key(0), electrons, plan)
```

`electrons` is the only place detector yield is applied: it integrates
`survival_fraction * quantum_efficiency(nodes_nm) * rates` with the
quadrature's dimensional weights in nm. Yield is applied before integration,
so a band-mean QE cannot replace the spectral calculation. The result has
shape `(W, *spatial)` in electrons/s. Survival is the fraction of
photoelectrons that survive to be counted; its neutral value is one.
Concrete survival or QE outside `[0, 1]` raises at construction.

## Frames and observatory time

Frame duration belongs to the observation's `FramePlan(frame_time_s, n_frames)`.
Each field is scalar or a vector with one entry per band. Vectors align with
axis 0 of the electron rates, including when the rates are images. Use
`jax.vmap` to batch independent plans and their rates.
`frames_for_time` rounds requested live time up to whole frames. Ratios
within relative tolerance `max(1e-12, 4 * eps)` of a positive integer count
as exact multiples. Every positive sub-frame request takes one frame;
zero time takes zero. Counts remain floats under JIT and have zero derivative.
The final frame collects for its entire duration.

`plan.live_time_s = n_frames * frame_time_s` is collecting time.
`plan.occupied_time_s(detector)` additionally pays one `read_time_s` per
frame unless `overlaps_read=True`, for a sensor that collects during readout.

## Moments and sampling

For total electron rate E, effective aperture size n_pix, live time t,
frame count N, dark current d, CIC c, read RMS sigma_r, and excess noise F2:

```text
mean = (E + n_pix*d)*t + n_pix*c*N
variance = F2*mean + n_pix*sigma_r^2*N
```

`mean_and_variance(electrons, plan, n_pix=1.0, sum_bands=False)` returns
two arrays of shape `(W, *spatial)`. With `sum_bands=True`, it sums only
axis zero and returns shape `spatial`, scalar for an aperture. Image
pixels use `n_pix=1`; aperture readers supply their effective pixel count.
Independent bands pay their own detector noise.

`draw(key, electrons, plan)` samples per pixel, using independent subkeys
for Poisson source plus dark, Poisson CIC, gain, and Gaussian read noise.
With F2=1 it adds the two Poisson counts and read noise. With F2>1, let
K be the sum of the Poisson counts and theta=F2-1. The gain output is zero
when K=0, otherwise Gamma(shape=K/theta, scale=theta). Its conditional
mean is K and variance is K*theta. Thus E[G]=lambda and
Var[G]=lambda + lambda*theta=F2*lambda. Read variance is added after gain.
This is an analog gain model for independent electrons; it omits
thresholding, saturation, gain-register geometry, and correlated noise.

Combine all source contributions before drawing once, so the detector
noise is paid once for each band and pixel. QE and survival never act
again on `ElectronRates`.
