"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.aperture_binding import (
    MatchedStamps,
    PreparedAperture,
    matched_moments,
    prepare_aperture,
)
from optixstuff.apertures import (
    DEFAULT_APERTURE_RULE,
    ApertureRule,
    Circular,
    HalfMaximum,
    Matched,
)
from optixstuff.coronagraph import (
    AbstractCoronagraph,
    AbstractScalarCoronagraph,
    AbstractTableCoronagraph,
    MultiBandCoronagraph,
    ScalarCoronagraph,
)
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    IdealDetector,
)
from optixstuff.disperser import AbstractDisperser, LensletDisperser
from optixstuff.exposure import FramePlan, frames_for_time
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughput,
    TransmissiveElement,
)
from optixstuff.optical_path import OpticalPath
from optixstuff.performance import PerformanceCurves, SampledProfile
from optixstuff.primary import AbstractPrimary, SegmentedPrimary, SimplePrimary
from optixstuff.rates import ElectronRates, IntegratedPhotonRates, PhotonRates
from optixstuff.speckle import AbstractSpeckleField, FocalResponse, LinearSpeckleField

__all__ = [
    "DEFAULT_APERTURE_RULE",
    "AbstractCoronagraph",
    "AbstractDetector",
    "AbstractDisperser",
    "AbstractOpticalElement",
    "AbstractPrimary",
    "AbstractScalarCoronagraph",
    "AbstractSpeckleField",
    "AbstractTableCoronagraph",
    "AbstractUniformElement",
    "ApertureRule",
    "Circular",
    "ConstantThroughput",
    "Detector",
    "ElectronRates",
    "FocalResponse",
    "FramePlan",
    "HalfMaximum",
    "IdealDetector",
    "IntegratedPhotonRates",
    "LensletDisperser",
    "LinearSpeckleField",
    "Matched",
    "MatchedStamps",
    "MultiBandCoronagraph",
    "OpticalPath",
    "PerformanceCurves",
    "PhotonRates",
    "PreparedAperture",
    "SampledProfile",
    "ScalarCoronagraph",
    "SegmentedPrimary",
    "SimplePrimary",
    "TransmissiveElement",
    "__version__",
    "frames_for_time",
    "matched_moments",
    "prepare_aperture",
]
