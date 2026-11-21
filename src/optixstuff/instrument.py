"""Instrument container -- the universal hardware configuration."""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import numpy as np

from optixstuff._repr import indent
from optixstuff.coronagraph import AbstractCoronagraph
from optixstuff.detector import AbstractDetector
from optixstuff.disperser import AbstractDisperser
from optixstuff.optical_elements import (
    AbstractOpticalElement,
)
from optixstuff.primary import AbstractPrimary
from optixstuff.speckle import AbstractSpeckleField
from optixstuff.spectral import spectral_breaks


class Instrument(eqx.Module):
    """Universal hardware container for a coronagraphic telescope.

    Bundles a primary mirror, ordered chain of attenuating elements,
    a coronagraph, and a detector into a single configuration object.
    This is the interface passed to simulators (coronagraphoto) and IFS
    instruments (coronachrome).

    Args:
        primary: Primary mirror description.
        attenuating_elements: Ordered tuple of throughput elements
            between the primary and coronagraph (mirrors, filters, etc.).
        coronagraph: Coronagraph slot (the image capability).
        detector: Focal-plane detector model.
        disperser: Optional IFS disperser descriptor; None for imaging mode.
        speckle: Optional time-varying speckle field (the stochastic
            wavefront-error residual on top of the deterministic
            coronagraph leakage floor); None for a perfect-wavefront path.
    """

    primary: AbstractPrimary
    attenuating_elements: tuple[AbstractOpticalElement, ...]
    coronagraph: AbstractCoronagraph | None
    detector: AbstractDetector
    disperser: AbstractDisperser | None = None
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

    def spectral_breaks_nm(self) -> np.ndarray:
        """Combine the breaks of the optical train and detector yield."""
        optional = (self.coronagraph, self.disperser, self.speckle)
        return spectral_breaks(
            *self.attenuating_elements,
            self.detector,
            *(model for model in optional if model is not None),
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
            "Instrument",
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
