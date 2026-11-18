"""OpticalPath container -- the universal hardware configuration."""

from __future__ import annotations

import functools
import math
from math import pi

import equinox as eqx
import jax

from optixstuff._repr import indent
from optixstuff.coronagraph import AbstractCoronagraph
from optixstuff.detector import AbstractDetector, Detector
from optixstuff.disperser import AbstractDisperser
from optixstuff.optical_elements import (
    AbstractOpticalElement,
    ConstantThroughput,
)
from optixstuff.primary import AbstractPrimary, SimplePrimary
from optixstuff.speckle import AbstractSpeckleField


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
        speckle: Optional time-varying speckle field (the stochastic
            wavefront-error residual on top of the deterministic
            coronagraph leakage floor); None for a perfect-wavefront path.
    """

    primary: AbstractPrimary
    attenuating_elements: tuple[AbstractOpticalElement, ...]
    coronagraph: AbstractCoronagraph | None
    detector: AbstractDetector
    disperser: AbstractDisperser | None = None
    n_channels: float = 1.0
    npix_multiplier: float = 1.0
    speckle: AbstractSpeckleField | None = None

    def __check_init__(self):
        """Reject a primary whose diameter disagrees with the coronagraph's frame.

        Coronagraph tables are computed for one telescope: their lambda/D frame
        carries its diameter. Pairing them with a primary of another diameter
        would take the angular scale from one telescope and the collecting area
        from another. Rescaling a coronagraph to a new telescope is a modelling
        choice the caller makes explicitly, not a silent default.
        """
        if self.coronagraph is None:
            return
        coronagraphs = getattr(self.coronagraph, "bands", None) or (self.coronagraph,)
        for coronagraph in coronagraphs:
            frame = coronagraph.frame
            supplied = self.primary.diameter(frame.diameter_convention)
            if isinstance(supplied, jax.core.Tracer):
                continue
            if not math.isclose(float(supplied), frame.diameter_m, rel_tol=1e-6):
                raise ValueError(
                    f"primary {frame.diameter_convention} diameter {float(supplied)} m "
                    f"differs from the coronagraph's lambda/D frame "
                    f"({frame.diameter_m} m); build the primary with the "
                    "coronagraph's diameters"
                )

    @classmethod
    def from_default_setup(
        cls,
        coronagraph: AbstractCoronagraph,
        *,
        circumscribed_diameter_m: float | None = None,
        obscuration: float = 0.0,
        inscribed_diameter_m: float | None = None,
        attenuating_throughput: float = 1.0,
        detector_shape: tuple[int, int] = (512, 512),
        pixel_scale_arcsec: float = 0.01,
        qe: float = 0.9,
        dark_current_e_per_pixel_s: float = 0.0,
        n_channels: float = 1.0,
        npix_multiplier: float = 1.0,
        speckle: AbstractSpeckleField | None = None,
    ) -> OpticalPath:
        """Build an OpticalPath with reasonable HWO-like defaults.

        Convenience for notebook / dev-script work: spin up a working
        ``OpticalPath`` by specifying only the coronagraph. All other
        parameters get sensible defaults that can be overridden.

        Args:
            coronagraph: An AbstractCoronagraph instance.
            circumscribed_diameter_m: Primary mirror diameter [m]. Default: the
                coronagraph frame's diameter (when that frame is circumscribed),
                so the tables and the primary describe one telescope.
            inscribed_diameter_m: Explicit inscribed diameter; defaults to the
                circular outer diameter.
            obscuration: Linear central-obscuration fraction. Default 0.
            attenuating_throughput: Combined throughput of the optical
                chain (one :class:`ConstantThroughput`). Default
                ``1.0`` -- a perfect path; override for realistic studies.
            detector_shape: Detector ``(ny, nx)`` in pixels. Default
                ``(512, 512)``.
            pixel_scale_arcsec: Detector plate scale [arcsec/px]. Default
                ``0.01``.
            qe: Default ``0.9``.
            dark_current_e_per_pixel_s: Default ``0.0`` e-/s/px (perfect detector;
                callers add realistic noise when needed).
            n_channels: AYO parallel-path multiplier. Default ``1.0``.
            npix_multiplier: IFS signal-spread multiplier. Default ``1.0``.
            speckle: Optional speckle field attached to the path. Default
                ``None`` -- a perfect-wavefront path.

        Returns:
            A ready-to-use :class:`OpticalPath`.
        """
        coro = coronagraph

        if circumscribed_diameter_m is None:
            frame = coro.frame
            if frame.diameter_convention != "circumscribed":
                raise ValueError(
                    "the coronagraph frame is inscribed: pass circumscribed_diameter_m"
                )
            circumscribed_diameter_m = frame.diameter_m
        return cls(
            primary=SimplePrimary(
                circumscribed_diameter_m=circumscribed_diameter_m,
                inscribed_diameter_m=inscribed_diameter_m,
                collecting_area_m2=pi
                * circumscribed_diameter_m**2
                / 4
                * (1 - obscuration**2),
            ),
            attenuating_elements=(
                ConstantThroughput(throughput=attenuating_throughput, name="optics"),
            ),
            coronagraph=coro,
            detector=Detector(
                pixel_scale_arcsec=pixel_scale_arcsec,
                shape=detector_shape,
                qe=qe,
                dark_current_e_per_pixel_s=dark_current_e_per_pixel_s,
            ),
            n_channels=n_channels,
            npix_multiplier=npix_multiplier,
            speckle=speckle,
        )

    def system_throughput(self, wavelength_nm: float) -> float:
        """Total throughput of all attenuating elements.

        Args:
            wavelength_nm: Wavelength in nanometers.

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
        if self.speckle is not None:
            lines.append(indent("speckle: " + repr(self.speckle)))
        return "\n".join(lines)
