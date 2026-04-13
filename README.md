# optixstuff

Flux-level optical system abstractions for the HWO direct imaging simulation suite.

## What optixstuff is

`optixstuff` is the **radiometric hardware layer** for HWO simulations. It operates on
**flux** — photon rates, throughput fractions, and detector-level noise — providing the
shared interfaces that simulation tools and exposure time calculators build on top of.

The same `OpticalPath` object drives both scalar ETC calculations (`jaxEDITH`) and full
2D image generation (`coronagraphoto`), ensuring that the hardware model is consistent
across all downstream science products.

## What optixstuff is *not*

`optixstuff` does not model diffraction, wavefront propagation, or E-field interference.
That level of physical optics belongs to tools like [dLux](https://github.com/LouisDesdoigts/dLux)
and [HCIPy](https://github.com/ehpor/hcipy), which generate PSFs from first principles.

`optixstuff` and these wavefront tools are **complementary**: dLux/HCIPy compute the PSFs,
which are delivered as yield input packages (YIPs). `optixstuff` consumes those PSFs as
flux patterns (via [yippy](https://github.com/CoreySpohn/yippy)) and composes them with
the rest of the observatory — throughput chain, detector QE, noise — to produce
science-level outputs.

## Architecture

Built on [JAX](https://github.com/google/jax) and
[Equinox](https://github.com/patrick-kidger/equinox), `optixstuff` provides:

- **Abstract interfaces** — `AbstractPrimary`, `AbstractOpticalElement`,
  `AbstractCoronagraph`, `AbstractDetector`
- **Concrete implementations** — `SimplePrimary`, `ConstantThroughputElement`,
  `SimpleDetector`
- **Container** — `OpticalPath`, a composable hardware configuration passed to all
  simulators

Every abstract method accepts three fidelity axes — **wavelength**, **position**, and
**time** — with defaults so that simple implementations can ignore unused axes while
future high-fidelity models (wavelength-dependent coatings, position-dependent vignetting,
time-dependent detector degradation) can use them without breaking the interface.

### Ecosystem position

```
                          ┌───────────────────────────┐
                          │  Physical optics           │
                          │  (dLux, HCIPy, PROPER)     │
                          │  E-fields → PSFs           │
                          └──────────┬────────────────┘
                                     │ YIP
                          ┌──────────▼────────────────┐
                          │  yippy                     │
                          │  PSF interpolation         │
                          └──────────┬────────────────┘
                                     │ flux patterns
              ┌──────────────────────▼────────────────────────┐
              │                  optixstuff                    │
              │  Telescope • Coronagraph • Detector • OpticalPath │
              │  Throughput chains • QE • Noise rates          │
              └────────┬──────────────────────┬───────────────┘
                       │                      │
            ┌──────────▼──────────┐ ┌────────▼───────────────┐
            │  jaxEDITH           │ │  coronagraphoto         │
            │  Scalar count rates │ │  2D image simulation    │
            │  Exposure times     │ │  Multi-epoch scenes     │
            └─────────────────────┘ └────────────────────────┘
```

## Installation

```bash
pip install optixstuff
```

## Status

This package is in early development (pre-v0.1.0).
