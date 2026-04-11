"""optixstuff — Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.optical_path import OpticalPath
from optixstuff.primary import AbstractPrimary, SimplePrimary
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughputElement,
)
from optixstuff.coronagraph import AbstractCoronagraph, AbstractScalarOnlyCoronagraph
from optixstuff.detector import AbstractDetector, SimpleDetector

__all__ = [
    "__version__",
    "OpticalPath",
    "AbstractPrimary",
    "SimplePrimary",
    "AbstractOpticalElement",
    "AbstractUniformElement",
    "ConstantThroughputElement",
    "AbstractCoronagraph",
    "AbstractScalarOnlyCoronagraph",
    "AbstractDetector",
    "SimpleDetector",
]
