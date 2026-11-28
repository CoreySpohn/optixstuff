"""optixstuff -- Hardware abstractions for the HWO simulation suite."""

from optixstuff._version import __version__
from optixstuff.apertures import (
    DEFAULT_APERTURE_RULE,
    Circular,
    HalfMaximum,
)
from optixstuff.calculator import ExposureTime, PostProcessing, exposure_time, snr
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
from optixstuff.data import (
    ELECTRON_COUNT_QUANTITIES,
    ExposureData,
    HostRecords,
    stack_exposure_data,
)
from optixstuff.detector import (
    AbstractDetector,
    Detector,
    IdealDetector,
)
from optixstuff.disperser import AbstractDisperser, LensletDisperser
from optixstuff.exposure import Exposure, FramePlan, Pointing, frames_for_time
from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.instrument import Instrument, WarmEmission
from optixstuff.observatory import Observatory, SelectedInstrument, Telescope
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    AbstractUniformElement,
    ConstantThroughput,
    TransmissiveElement,
)
from optixstuff.performance import (
    PerformanceCurves,
    SampledProfile,
    performance_curves,
)
from optixstuff.psflet_pack import (
    TEMPLATE_ORIGIN,
    PsfletPack,
    load_psflet_pack,
    save_psflet_pack,
)
from optixstuff.pupil import (
    AbstractPupil,
    SegmentedPupil,
    SimplePupil,
    check_pupil_matches_frame,
)
from optixstuff.rates import Rates
from optixstuff.sampling import SpectralSampling
from optixstuff.speckle import AbstractSpeckleField, FocalResponse, LinearSpeckleField
from optixstuff.spectral import Band, Quadrature, Reduction, Resolution

# Keep the PSFlet origin constant next to the pack class.
__all__ = [  # noqa: RUF022
    "DEFAULT_APERTURE_RULE",
    "AbstractCoronagraph",
    "AbstractDetector",
    "AbstractDisperser",
    "AbstractImagingCoronagraph",
    "AbstractOpticalElement",
    "AbstractPupil",
    "AbstractSceneCoronagraph",
    "AbstractSpeckleField",
    "AbstractTableCoronagraph",
    "AbstractUniformElement",
    "Band",
    "Circular",
    "ConstantThroughput",
    "ELECTRON_COUNT_QUANTITIES",
    "Detector",
    "Exposure",
    "ExposureData",
    "ExposureTime",
    "FocalResponse",
    "FramePlan",
    "HalfMaximum",
    "HostRecords",
    "IdealDetector",
    "Image",
    "Instrument",
    "LensletDisperser",
    "LinearSpeckleField",
    "Measure",
    "MultiBandCoronagraph",
    "Observatory",
    "PerformanceCurves",
    "PixelGrid",
    "Pointing",
    "PostProcessing",
    "ProductGrids",
    "PsfletPack",
    "TEMPLATE_ORIGIN",
    "Quadrature",
    "Rates",
    "Reduction",
    "Resolution",
    "SampledProfile",
    "SceneRender",
    "SegmentedPupil",
    "SelectedInstrument",
    "SimplePupil",
    "SourcePSFs",
    "SpectralSampling",
    "Telescope",
    "TransmissiveElement",
    "WarmEmission",
    "__version__",
    "check_pupil_matches_frame",
    "exposure_time",
    "frames_for_time",
    "load_psflet_pack",
    "performance_curves",
    "save_psflet_pack",
    "snr",
    "stack_exposure_data",
]
