"""optixstuff — Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.coronagraph import AbstractCoronagraph, AbstractScalarOnlyCoronagraph
from optixstuff.detector import AbstractDetector, SimpleDetector
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughputElement,
)
from optixstuff.optical_path import OpticalPath
from optixstuff.primary import AbstractPrimary, SimplePrimary

__all__ = [
    "AbstractCoronagraph",
    "AbstractDetector",
    "AbstractOpticalElement",
    "AbstractPrimary",
    "AbstractScalarOnlyCoronagraph",
    "AbstractUniformElement",
    "ConstantThroughputElement",
    "OpticalPath",
    "SimpleDetector",
    "SimplePrimary",
    "__version__",
]
