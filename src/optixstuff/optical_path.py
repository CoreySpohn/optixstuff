"""OpticalPath container -- the universal hardware configuration."""

import functools

import equinox as eqx

from optixstuff.coronagraph import AbstractCoronagraph
from optixstuff.detector import AbstractDetector
from optixstuff.optical_elements import AbstractOpticalElement
from optixstuff.primary import AbstractPrimary


class OpticalPath(eqx.Module):
    """Universal hardware container for a coronagraphic telescope.

    Bundles a primary mirror, ordered chain of attenuating elements,
    a coronagraph, and a detector into a single configuration object.
    This is the interface passed to simulators (coronagraphoto),
    exposure time calculators (jaxEDITH), and IFS instruments (coronachrome).

    Args:
        primary: Primary mirror description.
        attenuating_elements: Ordered tuple of throughput elements
            between the primary and coronagraph (mirrors, filters, etc.).
        coronagraph: Coronagraph performance model.
        detector: Focal-plane detector model.
        n_channels: Number of parallel identical optical-path copies
            (AYO shorthand, multiplicative factor on count rates;
            not a spectral channel count). Default 1.0.
        npix_multiplier: IFS signal-spread multiplier on detector pixel
            counts. Default 1.0.
    """

    primary: AbstractPrimary
    attenuating_elements: tuple[AbstractOpticalElement, ...]
    coronagraph: AbstractCoronagraph
    detector: AbstractDetector
    n_channels: float = 1.0
    npix_multiplier: float = 1.0

    def system_throughput(self, wavelength_nm: float) -> float:
        """Total throughput of all attenuating elements.

        Args:
            wavelength_nm: Wavelength in nanometres.

        Returns:
            Combined fractional throughput in [0, 1].
        """
        return functools.reduce(
            lambda acc, el: acc * el.get_throughput(wavelength_nm),
            self.attenuating_elements,
            1.0,
        )
