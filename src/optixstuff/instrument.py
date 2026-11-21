"""Instrument container -- the universal hardware configuration."""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

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


def check_primary_matches_frame(primary, frame, *, source="coronagraph"):
    """Raise when a lambda/D frame was measured for another telescope diameter.

    The frame carries the diameter its angular scale was computed for, in the
    convention it names; a primary of another diameter would take the angular
    scale from one telescope and the collecting area from another. Traced
    diameters are skipped.

    Args:
        primary: The primary whose diameter is checked.
        frame: The ``Frame`` of the coronagraph tables or curves.
        source: What carries the frame, for the error message.

    Raises:
        ValueError: If the primary diameter differs from ``frame.diameter_m``.
    """
    supplied = primary.diameter(frame.diameter_convention)
    if isinstance(supplied, jax.core.Tracer):
        return
    if not math.isclose(float(supplied), frame.diameter_m, rel_tol=1e-6):
        raise ValueError(
            f"primary {frame.diameter_convention} diameter {float(supplied)} m "
            f"differs from the {source}'s lambda/D frame ({frame.diameter_m} m); "
            f"build the primary with the {source}'s diameters"
        )


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
        flux_scale: Dimensionless factor on every photon rate the instrument
            collects, default one: the absolute flux calibration. It acts in
            :meth:`effective_area_m2`, the one place the collecting area and
            throughput are combined. It is linear, so a fit that frees it
            scales a rate record and need not rebuild one.
    """

    primary: AbstractPrimary
    attenuating_elements: tuple[AbstractOpticalElement, ...]
    coronagraph: AbstractCoronagraph | None
    detector: AbstractDetector
    disperser: AbstractDisperser | None = None
    speckle: AbstractSpeckleField | None = None
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self):
        """Reject a primary whose diameter disagrees with the coronagraph's frame.

        Coronagraph tables are computed for one telescope: their lambda/D frame
        carries its diameter. Pairing them with a primary of another diameter
        would take the angular scale from one telescope and the collecting area
        from another. Rescaling a coronagraph to a new telescope is a modelling
        choice the caller makes explicitly, not a silent default.

        The flux scale must be finite and positive when concrete.
        """
        scale = self.flux_scale
        if not isinstance(scale, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(scale) & (scale > 0))
        ):
            raise ValueError("flux_scale must be finite and positive")
        if self.coronagraph is None:
            return
        coronagraphs = getattr(self.coronagraph, "bands", None) or (self.coronagraph,)
        for coronagraph in coronagraphs:
            check_primary_matches_frame(self.primary, coronagraph.frame)

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

    def effective_area_m2(self, wavelength_nm: float) -> float:
        """Collecting area times system throughput times the flux scale.

        The photon rate collected from a source of unit flux density is this
        area times that flux density. Image and aperture rates both take it
        from here, so the flux scale is applied once.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            Effective area in square meters, shaped like ``wavelength_nm``.
        """
        return (
            self.primary.collecting_area_m2
            * self.system_throughput(wavelength_nm)
            * self.flux_scale
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
        if not isinstance(self.flux_scale, jax.core.Tracer) and bool(
            jnp.any(self.flux_scale != 1.0)
        ):
            lines.append(indent(f"flux_scale: {self.flux_scale}"))
        return "\n".join(lines)
