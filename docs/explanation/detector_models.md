# Detector models

optixstuff ships two concrete detector implementations and three
noise-generator primitives. This page explains when to reach for
each.

## The two detectors

### `IdealDetector`

```python
from optixstuff import IdealDetector

detector = IdealDetector(
    pixel_scale_arcsec=0.01,         # arcsec/pixel
    shape=(512, 512),         # detector grid
    quantum_efficiency=1.0,   # constant QE (default)
    dark_current_rate_e_per_s=0.0,    # no dark current
    read_noise_e=0.0, # no read noise
    clock_induced_charge_rate_e_per_frame=0.0,             # no clock-induced charge
)
```

**Use for**: broadband imaging studies where wavelength-dependent QE
variation and detector noise contributions are negligible. Sandbox /
debugging / first-pass forward modeling.

All noise sources default to zero -- `IdealDetector` models a pure
photon counter.

### `Detector`

```python
from optixstuff import Detector

detector = Detector(
    pixel_scale_arcsec=0.01,
    shape=(512, 512),
    quantum_efficiency=0.9,
    dark_current_rate_e_per_s=1e-4,       # e/s/pixel
    read_noise_e=3.0,     # e RMS per read
    clock_induced_charge_rate_e_per_frame=1e-3,                # e per frame per pixel
    frame_time_s=300.0,             # seconds per readout frame
    read_time_s=0.05,
    dqe=0.0,                      # detector quantum efficiency factor
)
```

**Use for**: realistic noise budgets, mission yield calculations,
end-to-end performance simulations.

`Detector` uses **all** of its noise-source fields. Dark current and
CIC enter as Poisson processes; read noise is Gaussian. Compose with
the detector's `readout` method or with the standalone noise
primitives (below).

### Same schema, different behavior

`IdealDetector` and `Detector` accept the same field signature. The
difference is in the methods: `IdealDetector` uses constant QE
(ignores wavelength) and skips noise contributions when rates are
zero. `Detector` uses every field and returns the full physical noise
budget. Use `Detector` whenever you want realistic noise.

## The noise primitives

Three standalone functions for direct noise simulation:

```python
from optixstuff import (
    dark_current,
    clock_induced_charge,
    read_noise,
)

# Dark current: Poisson with mean = rate * exposure_time_s
dark_e = dark_current(
    dark_current_rate_e_per_s=1e-4,    # e/s/pixel
    exposure_time_s=3600.0,      # seconds
    shape=(512, 512),
    prng_key=jax.random.PRNGKey(0),
)

# Clock-induced charge: Poisson per-frame, scaled by num_frames
cic_e = clock_induced_charge(
    clock_induced_charge_rate_e_per_frame=1e-3,             # e/frame/pixel
    num_frames=12.0,
    shape=(512, 512),
    prng_key=jax.random.PRNGKey(1),
)

# Read noise: Gaussian per-frame, RMS-summed across frames
read_e = read_noise(
    read_noise_e=3.0,            # e RMS per read
    num_frames=12.0,
    shape=(512, 512),
    prng_key=jax.random.PRNGKey(2),
)
```

These are **noise contributions** (additive to a source readout), not
full readouts. Each returns the per-pixel noise electrons for one
exposure, ready to be summed into a complete frame alongside the
source-readout contributions from coronagraphoto.

## Composing a realistic frame

```python
import jax
from optixstuff import dark_current, clock_induced_charge, read_noise
from coronagraphoto import star_readout, planet_readout

keys = jax.random.split(jax.random.PRNGKey(0), 6)

# Sources (Poisson realisations of the count rates)
star_e   = star_readout(star, optical_path, keys[0], ...)
planet_e = planet_readout(planet, optical_path, keys[1], ...)

# Detector noise contributions
shape = optical_path.detector.shape
dark_e = dark_current(
    optical_path.detector.dark_current_rate_e_per_s,
    exposure_time_s,
    shape,
    keys[2],
)
cic_e = clock_induced_charge(
    optical_path.detector.clock_induced_charge_rate_e_per_frame,
    num_frames,
    shape,
    keys[3],
)
read_e = read_noise(
    optical_path.detector.read_noise_e,
    num_frames,
    shape,
    keys[4],
)

# Compose
frame = star_e + planet_e + dark_e + cic_e + read_e
```

For most use cases the detector's high-level `readout` method handles
this composition for you; the primitives are exposed for advanced use
(custom noise budgets, sensitivity studies, partial-noise traces).

## See also

- [Architecture](architecture) -- where detectors fit in the
  optixstuff abstraction hierarchy
