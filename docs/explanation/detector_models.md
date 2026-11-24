# Detector models

All detector calculations follow one chain:
`Rates` (photon) `-> detector.electrons -> Rates` (electron) `-> moments or draw`.
Readers supply the six source terms in photons/s/nm at quadrature nodes,
with shape `(W, Q)` for apertures or `(W, Q, ny, nx)` for images.
Absent terms may be scalar zero. The terms are `planet`, `star_leakage`,
`local_zodi`, `exozodi`, `thermal`, and `stellar_companions`.

One `Rates` type serves the whole chain. Its static `carrier` is `"photon"`
or `"electron"`. Its `quadrature` is set while the fields are per-nm
integrands at the nodes and is None once they are integrated over the bands.
Every record carries the tuple of bands in `.bands`.

| Rate sampling | Leading image axis names | Component shape | Units |
| --- | --- | --- | --- |
| Spectral density at quadrature nodes | `wavelength`, `quadrature` | `(n_band, n_node, *spatial)` | photon/s/nm or electron/s/nm |
| Integrated over each band | `band` | `(n_band, *spatial)` | photon/s or electron/s |

Each row of quadrature nodes samples one band. `rates.integrate()` weights and
reduces the `quadrature` axis and renames the surviving `wavelength` axis to
`band`. It preserves the carrier and all spatial axes.

`total()` sums the six sources. A scalar applies to every sample. All other
components must include every leading and spatial axis. Leading lengths must
match the sampling exactly. Each spatial length must be one or the full
length for that axis. To migrate a per-band component of an integrated image,
reshape it from `(n_band,)` to `(n_band, 1, 1)` so it repeats across rows and
columns. `Rates.zeros(quadrature, "photon", (ny, nx))` allocates six full-sized
zero integrands.

Per-pixel records leave `n_pix` as None. A weighted aperture sets `n_pix` to
the sum of its weights in detector pixels, and may carry a second record in
`variance`, summed with the squared weights, whose own `n_pix` is the sum of
squared weights.

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
from optixstuff.spectral import Band, Quadrature
from optixstuff import Detector, Rates, frames_for_time

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
quadrature = Quadrature.build([Band(500.0, 600.0)], detector, order=3)
rates = Rates(quadrature, "photon", planet=jnp.ones((1, 3, 2, 2)))
electrons = detector.electrons(rates)
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
Frame duration must be finite and strictly positive. A frame count must
be a nonnegative whole number or NaN. A positive finite time request
takes the smallest whole number of frames whose live time covers it
in the working dtype. Zero seconds takes zero frames. A non-finite or negative
request gives a NaN count and live time, as for an unreachable exposure target.
Counts remain floats under JIT and have zero derivative. The final frame
collects for its entire duration.

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

`mean_and_variance(electrons, plan, sum_bands=False)` returns
two arrays of shape `(W, *spatial)`. With `sum_bands=True`, it sums only
axis zero and returns shape `spatial`, scalar for an aperture. Image
pixels use `n_pix=1`; aperture readers supply their effective pixel count
in `electrons.n_pix`.
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
again on electron rates.

A band rate is a sum of node values, and integration or reconstruction error
can leave a total slightly below zero where the true rate is near zero.
`optixstuff.rates.floor_rate` treats a negative total within
`RATE_FLOOR_RTOL = 1e-4` of the frame peak (the largest summed absolute
component rate over the band's pixels) as that error and draws it as zero. A
more negative total means the band rule under-resolves the scene, and `draw`
raises, also under `jit`, `vmap` and `grad`, because a Poisson draw of a
negative mean returns -1 without notice. The remedy is a break at the edge
(`breaks_nm`) or a higher `order`.

## Likelihood and Fisher information

`log_prob(observed, electrons, plan)` is the elementwise log density of an
observed frame. A detector with `photon_counting=True` scores the Poisson pmf
of the mean. Every other detector scores the Gaussian with the mean and
variance of `mean_and_variance`, the moment-matched approximation of `draw`.
The total rate passes through the same floor as `draw`: a total within the
tolerance below zero is scored as a rate of zero, and a more negative one is
`-inf` in both forms, so a sampler rejects that proposal rather than stopping.
Where the mean is zero the Poisson form is `-inf` for a positive count (with a
zero gradient), and the Gaussian form is `-inf` where the variance is not
positive and the observation differs from the mean. A fit that walks a rate
negative meets a wall, not a penalty.

`fisher_root(electrons, plan)` returns a vector `r` whose Jacobian gives the
expected information of that likelihood. For parameters `theta` that enter
through the rates, `J = dr/dtheta` and the information is `J^T J`, summed over
independent pixels and bands.

```text
Poisson:   r = 2 sqrt(mean)                      J^T J = sum dm^2 / m
Gaussian:  r = (mean / sqrt(v0), ln(v) / sqrt 2) J^T J = sum dm^2 / v + dv^2 / (2 v^2)
           v0 = stop_gradient(v)
```

The Gaussian root covers the mean-dependent variance. The root has the shape of
the mean for the Poisson law and `(2, *mean.shape)` for the Gaussian law, so a
caller can mask pixels before it flattens. It is zero, and so is its
derivative, where the law has no support (a mean that is not positive for the
Poisson law, a variance that is not positive for the Gaussian law); it is never
NaN. A subclass that overrides `log_prob` has a different law, and
`fisher_root` raises `NotImplementedError` naming the class until the subclass
supplies its own root.

The Gaussian is the assumed law of `log_prob`, not the law of `draw`: the gamma
gain of `draw` has the same first two moments and a different score. The
information above is that of the assumed law.
