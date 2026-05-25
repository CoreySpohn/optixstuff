"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.coronagraph import AbstractCoronagraph, AbstractScalarCoronagraph
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    IdealDetector,
    simulate_cic,
    simulate_dark_current,
    simulate_read_noise,
)
from optixstuff.exposure import ExposureConfig
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughput,
    LinearThroughput,
    OpticalFilter,
)
from optixstuff.optical_path import OpticalPath
from optixstuff.primary import AbstractPrimary, SimplePrimary
from optixstuff.yippy_coronagraph import YippyCoronagraph

__all__ = [
    "AbstractCoronagraph",
    "AbstractDetector",
    "AbstractOpticalElement",
    "AbstractPrimary",
    "AbstractScalarCoronagraph",
    "AbstractUniformElement",
    "ConstantThroughput",
    "Detector",
    "ExposureConfig",
    "LinearThroughput",
    "OpticalFilter",
    "OpticalPath",
    "IdealDetector",
    "SimplePrimary",
    "YippyCoronagraph",
    "__version__",
    "simulate_cic",
    "simulate_dark_current",
    "simulate_read_noise",
]
