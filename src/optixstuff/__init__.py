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
    AbstractImagingCoronagraph,
    AbstractSceneCoronagraph,
    AbstractTableCoronagraph,
    MultiBandCoronagraph,
    ProductGrids,
    ScalarCoronagraph,
    SceneRender,
    SourcePSFs,
)
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    IdealDetector,
)
from optixstuff.disperser import AbstractDisperser, LensletDisperser
from optixstuff.exposure import Exposure, FramePlan, frames_for_time
from optixstuff.instrument import Instrument
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughput,
    TransmissiveElement,
)
from optixstuff.performance import PerformanceCurves, SampledProfile
from optixstuff.primary import AbstractPrimary, SegmentedPrimary, SimplePrimary
from optixstuff.psflet_pack import PsfletPack, load_psflet_pack, save_psflet_pack
from optixstuff.rates import ElectronRates, IntegratedPhotonRates, PhotonRates
from optixstuff.speckle import AbstractSpeckleField, FocalResponse, LinearSpeckleField

__all__ = [
    "DEFAULT_APERTURE_RULE",
    "AbstractCoronagraph",
    "AbstractDetector",
    "AbstractDisperser",
    "AbstractImagingCoronagraph",
    "AbstractOpticalElement",
    "AbstractPrimary",
    "AbstractSceneCoronagraph",
    "AbstractSpeckleField",
    "AbstractTableCoronagraph",
    "AbstractUniformElement",
    "ApertureRule",
    "Circular",
    "ConstantThroughput",
    "Detector",
    "ElectronRates",
    "Exposure",
    "FocalResponse",
    "FramePlan",
    "HalfMaximum",
    "IdealDetector",
    "Instrument",
    "IntegratedPhotonRates",
    "LensletDisperser",
    "LinearSpeckleField",
    "Matched",
    "MatchedStamps",
    "MultiBandCoronagraph",
    "PerformanceCurves",
    "PhotonRates",
    "PreparedAperture",
    "ProductGrids",
    "PsfletPack",
    "SampledProfile",
    "ScalarCoronagraph",
    "SceneRender",
    "SegmentedPrimary",
    "SimplePrimary",
    "SourcePSFs",
    "TransmissiveElement",
    "__version__",
    "frames_for_time",
    "load_psflet_pack",
    "matched_moments",
    "prepare_aperture",
    "save_psflet_pack",
]
