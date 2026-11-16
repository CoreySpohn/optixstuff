"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.coronagraph import (
    AbstractCoronagraph,
    AbstractScalarCoronagraph,
    AbstractTableCoronagraph,
    MultiBandCoronagraph,
)
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    IdealDetector,
)
from optixstuff.disperser import AbstractDisperser, LensletDisperser
from optixstuff.exposure import ExposureConfig, FramePlan, frames_for_time
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughput,
    SpectralThroughput,
)
from optixstuff.optical_path import OpticalPath
from optixstuff.primary import AbstractPrimary, SegmentedPrimary, SimplePrimary
from optixstuff.rates import ElectronRates, IntegratedPhotonRates, PhotonRates
from optixstuff.speckle import AbstractSpeckleField
from optixstuff.yippy_coronagraph import YippyCoronagraph

__all__ = [
    "AbstractCoronagraph",
    "AbstractDetector",
    "AbstractDisperser",
    "AbstractOpticalElement",
    "AbstractPrimary",
    "AbstractScalarCoronagraph",
    "AbstractSpeckleField",
    "AbstractTableCoronagraph",
    "AbstractUniformElement",
    "ConstantThroughput",
    "Detector",
    "ElectronRates",
    "ExposureConfig",
    "FramePlan",
    "IdealDetector",
    "IntegratedPhotonRates",
    "LensletDisperser",
    "MultiBandCoronagraph",
    "OpticalPath",
    "PhotonRates",
    "SegmentedPrimary",
    "SimplePrimary",
    "SpectralThroughput",
    "YippyCoronagraph",
    "__version__",
    "frames_for_time",
]
