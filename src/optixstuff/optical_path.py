"""OpticalPath container -- the universal hardware configuration."""

from __future__ import annotations

import functools
from pathlib import Path

import equinox as eqx

from optixstuff._repr import indent
from optixstuff.coronagraph import AbstractCoronagraph
from optixstuff.detector import AbstractDetector, IdealDetector
from optixstuff.disperser import AbstractDisperser
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    ConstantThroughput,
)
from optixstuff.primary import AbstractPrimary, SimplePrimary


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
        disperser: Optional IFS disperser descriptor; None for imaging mode.
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
    disperser: AbstractDisperser | None = None
    n_channels: float = 1.0
    npix_multiplier: float = 1.0

    @classmethod
    def from_default_setup(
        cls,
        coronagraph: AbstractCoronagraph | str | Path,
        *,
        diameter_m: float = 6.0,
        obscuration: float = 0.0,
        attenuating_throughput: float = 1.0,
        detector_shape: tuple[int, int] = (512, 512),
        pixel_scale_arcsec: float = 0.01,
        quantum_efficiency: float = 0.9,
        dark_current_rate_e_per_s: float = 0.0,
        n_channels: float = 1.0,
        npix_multiplier: float = 1.0,
    ) -> OpticalPath:
        """Build an OpticalPath with reasonable HWO-like defaults.

        Convenience for notebook / dev-script work: spin up a working
        ``OpticalPath`` by specifying only the coronagraph. All other
        parameters get sensible defaults that can be overridden.

        Args:
            coronagraph: One of:
                - an :class:`AbstractCoronagraph` instance (used as-is),
                - a YIP path (str or :class:`pathlib.Path`, wrapped with
                  :class:`YippyCoronagraph`),
                - a ``yippy.EqxCoronagraph`` instance (wrapped via
                  ``YippyCoronagraph(backend=...)`` so callers can keep
                  using existing yippy code without rebuilding).
            diameter_m: Primary mirror diameter [m]. Default ``6.0`` (HWO
                EAC1 baseline).
            obscuration: Linear central-obscuration fraction. Default 0.
            attenuating_throughput: Combined throughput of the optical
                chain (one :class:`ConstantThroughput`). Default
                ``1.0`` -- a perfect path; override for realistic studies.
            detector_shape: Detector ``(ny, nx)`` in pixels. Default
                ``(512, 512)``.
            pixel_scale_arcsec: Detector plate scale [arcsec/px]. Default
                ``0.01``.
            quantum_efficiency: Default ``0.9``.
            dark_current_rate_e_per_s: Default ``0.0`` e-/s/px (perfect detector;
                callers add realistic noise when needed).
            n_channels: AYO parallel-path multiplier. Default ``1.0``.
            npix_multiplier: IFS signal-spread multiplier. Default ``1.0``.

        Returns:
            A ready-to-use :class:`OpticalPath`.
        """
        if isinstance(coronagraph, AbstractCoronagraph):
            coro = coronagraph
        elif isinstance(coronagraph, (str, Path)):
            from optixstuff.yippy_coronagraph import YippyCoronagraph

            coro = YippyCoronagraph(str(coronagraph))
        else:
            # Anything else -- expected to be a ``yippy.EqxCoronagraph``
            # or compatible backend. Defer the import so optixstuff stays
            # decoupled from yippy at the type-check level.
            from optixstuff.yippy_coronagraph import YippyCoronagraph

            coro = YippyCoronagraph(backend=coronagraph)

        return cls(
            primary=SimplePrimary(diameter_m=diameter_m, obscuration=obscuration),
            attenuating_elements=(
                ConstantThroughput(throughput=attenuating_throughput, name="optics"),
            ),
            coronagraph=coro,
            detector=IdealDetector(
                pixel_scale_arcsec=pixel_scale_arcsec,
                shape=detector_shape,
                quantum_efficiency=quantum_efficiency,
                dark_current_rate_e_per_s=dark_current_rate_e_per_s,
            ),
            n_channels=n_channels,
            npix_multiplier=npix_multiplier,
        )

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

    def __repr__(self) -> str:
        """Tree-shaped summary of every component."""
        lines = [
            (
                f"OpticalPath(n_channels={self.n_channels:.3g}, "
                f"npix_multiplier={self.npix_multiplier:.3g})"
            ),
            indent("primary: " + repr(self.primary)),
        ]
        if self.attenuating_elements:
            lines.append("  attenuating_elements:")
            for i, el in enumerate(self.attenuating_elements):
                lines.append(indent(f"[{i}] {el!r}", prefix="    "))
        else:
            lines.append("  attenuating_elements: ()")
        lines.append(indent("coronagraph: " + repr(self.coronagraph)))
        lines.append(indent("detector: " + repr(self.detector)))
        return "\n".join(lines)
