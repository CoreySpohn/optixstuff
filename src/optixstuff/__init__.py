"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.coronagraph import AbstractCoronagraph, AbstractScalarOnlyCoronagraph
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    SimpleDetector,
    simulate_cic,
    simulate_dark_current,
    simulate_read_noise,
)
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughputElement,
    LinearThroughputElement,
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
    "AbstractScalarOnlyCoronagraph",
    "AbstractUniformElement",
    "ConstantThroughputElement",
    "Detector",
    "LinearThroughputElement",
    "OpticalFilter",
    "OpticalPath",
    "SimpleDetector",
    "SimplePrimary",
    "YippyCoronagraph",
    "__version__",
    "simulate_cic",
    "simulate_dark_current",
    "simulate_read_noise",
]
