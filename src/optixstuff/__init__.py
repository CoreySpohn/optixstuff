"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.apertures import (
    DEFAULT_APERTURE_RULE,
    Circular,
    HalfMaximum,
)
from optixstuff.coronagraph import (
    AbstractCoronagraph,
    AbstractImagingCoronagraph,
    AbstractSceneCoronagraph,
    AbstractTableCoronagraph,
    MultiBandCoronagraph,
    ProductGrids,
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
from optixstuff.grids import Image, Measure, PixelGrid
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
from optixstuff.rates import Rates
from optixstuff.speckle import AbstractSpeckleField, FocalResponse, LinearSpeckleField
from optixstuff.spectral import Band, Quadrature

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
    "Band",
    "Circular",
    "ConstantThroughput",
    "Detector",
    "Exposure",
    "FocalResponse",
    "FramePlan",
    "HalfMaximum",
    "IdealDetector",
    "Image",
    "Instrument",
    "LensletDisperser",
    "LinearSpeckleField",
    "Measure",
    "MultiBandCoronagraph",
    "PerformanceCurves",
    "PixelGrid",
    "ProductGrids",
    "PsfletPack",
    "Quadrature",
    "Rates",
    "SampledProfile",
    "SceneRender",
    "SegmentedPrimary",
    "SimplePrimary",
    "SourcePSFs",
    "TransmissiveElement",
    "__version__",
    "frames_for_time",
    "load_psflet_pack",
    "save_psflet_pack",
]
