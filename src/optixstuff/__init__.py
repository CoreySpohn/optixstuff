"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.coronagraph import AbstractCoronagraph, AbstractScalarCoronagraph
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    IdealDetector,
    clock_induced_charge,
    dark_current,
    read_noise,
)
from optixstuff.exposure import ExposureConfig
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughput,
    SpectralThroughput,
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
    "IdealDetector",
    "OpticalPath",
    "SimplePrimary",
    "SpectralThroughput",
    "YippyCoronagraph",
    "__version__",
    "clock_induced_charge",
    "dark_current",
    "read_noise",
]
