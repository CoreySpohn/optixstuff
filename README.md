# optixstuff

Fidelity-agnostic hardware abstractions for the HWO direct imaging simulation suite.

`optixstuff` defines the shared hardware interfaces — primary mirror, optical elements,
coronagraph, and detector — that are consumed by simulation tools (`coronagraphoto`),
exposure time calculators (`jaxEDITH`), and IFS simulators (`coronachrome`).

Built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox)
using the Abstract/Final pattern for type-safe, JIT-compatible hardware descriptions.

## Overview

This package is currently in early development.

## Installation

```bash
pip install optixstuff
```

## Design

`optixstuff` separates *hardware description* from *simulation*. It provides:

- **Abstract interfaces** — `AbstractPrimary`, `AbstractOpticalElement`, `AbstractCoronagraph`, `AbstractDetector`
- **Concrete implementations** — `SimplePrimary`, `ConstantThroughputElement`, `SimpleDetector`
- **Container** — `OpticalPath`, the universal hardware configuration passed to all simulators

The same `OpticalPath` object can be passed to an ETC for scalar flux calculations
or to a simulator for full 2D image generation, without modification.
