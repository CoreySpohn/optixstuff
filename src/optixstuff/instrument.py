"""Instrument container -- the universal hardware configuration."""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.planck import planck_photon_radiance
from jaxtyping import Array

from optixstuff._repr import indent
from optixstuff.coronagraph import AbstractCoronagraph
from optixstuff.detector import AbstractDetector
from optixstuff.disperser import AbstractDisperser
from optixstuff.grids import Frame
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

    Returns:
        None.

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


class WarmEmission(eqx.Module):
    """Thermal emission of the warm telescope, one lumped blackbody.

    The warm surfaces (primary, secondary and any other optics at telescope
    temperature) radiate as one blackbody at ``temperature_K`` with an
    effective emissivity; the cold optics behind them transmit that light
    to the focal plane. This is the lumped description the analytic
    exposure-time calculators use, so ``effective_emissivity`` is the
    emissivity of the warm surfaces times the transmission of the cold
    optics behind them, not a property of any single element.

    Attributes:
        temperature_K: Temperature of the warm telescope [K], finite and
            positive.
        effective_emissivity: Warm emissivity times cold transmission,
            dimensionless in [0, 1].
    """

    temperature_K: Array = eqx.field(converter=jnp.asarray)
    effective_emissivity: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Reject a non-positive temperature or an emissivity outside [0, 1].

        Returns:
            None.
        """
        temperature, emissivity = self.temperature_K, self.effective_emissivity
        if isinstance(temperature, jax.core.Tracer) or isinstance(
            emissivity, jax.core.Tracer
        ):
            return
        if not bool(jnp.all(jnp.isfinite(temperature) & (temperature > 0))):
            raise ValueError("temperature_K must be finite and positive")
        if not bool(jnp.all((emissivity >= 0) & (emissivity <= 1))):
            raise ValueError("effective_emissivity must lie in [0, 1]")

    def photon_radiance(self, wavelength_nm):
        """Return the emitted radiance in photons/s/m^2/nm/sr, emissivity applied.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            Planck photon radiance at ``temperature_K`` times
            ``effective_emissivity``, shaped like ``wavelength_nm``.
        """
        return self.effective_emissivity * planck_photon_radiance(
            wavelength_nm, self.temperature_K
        )


class Instrument(eqx.Module):
    """Universal hardware container for a coronagraphic telescope.

    Bundles a primary mirror, ordered chain of attenuating elements,
    a coronagraph, and a detector into a single configuration object.
    This is the interface passed to simulators (coronagraphoto) and IFS
    instruments (coronachrome).

    Attributes:
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
            collects from the sky, default one: the absolute flux calibration.
            It acts in :meth:`effective_area_m2`, the one place the collecting
            area and throughput are combined. Thermal emission does not carry
            it. It is linear, so a fit that frees it scales the sky fields of a
            rate record and need not rebuild one.
        warm_emission: Optional :class:`WarmEmission` of the warm telescope,
            owned here because it describes the assembled warm optical train
            and not the primary or any one element (swap the primary and the
            same train still radiates). None, the default, is no thermal
            emission.
    """

    primary: AbstractPrimary
    attenuating_elements: tuple[AbstractOpticalElement, ...]
    coronagraph: AbstractCoronagraph | None
    detector: AbstractDetector
    disperser: AbstractDisperser | None = None
    speckle: AbstractSpeckleField | None = None
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray)
    warm_emission: WarmEmission | None = None

    def __check_init__(self):
        """Reject a primary whose diameter disagrees with the coronagraph's frame.

        Coronagraph tables are computed for one telescope: their lambda/D frame
        carries its diameter. Pairing them with a primary of another diameter
        would take the angular scale from one telescope and the collecting area
        from another. Rescaling a coronagraph to a new telescope is a modelling
        choice the caller makes explicitly, not a silent default.

        The flux scale must be finite and positive when concrete.

        Returns:
            None.
        """
        scale = self.flux_scale
        if not isinstance(scale, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(scale) & (scale > 0))
        ):
            raise ValueError("flux_scale must be finite and positive")
        if self.coronagraph is None:
            return
        coronagraphs = getattr(self.coronagraph, "coronagraphs", None) or (
            self.coronagraph,
        )
        for coronagraph in coronagraphs:
            check_primary_matches_frame(self.primary, coronagraph.frame)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Combine the breaks of the optical train and detector yield.

        Returns:
            Sorted union of the break wavelengths in nm of the optical train, detector,
            coronagraph, disperser and speckle field.
        """
        optional = (self.coronagraph, self.disperser, self.speckle)
        return spectral_breaks(
            *self.attenuating_elements,
            self.detector,
            *(model for model in optional if model is not None),
        )

    def spectral_field_lod(self, band) -> float:
        """Field radius the detector spans, in lambda/D at the center of ``band``.

        The half-diagonal of the detector: the largest separation from the
        detector center at which any pixel reads the diffraction pattern. A
        pattern at that separation slides across a band by more chromatic cycles
        than one nearer the center, so ``Quadrature.build`` gives the band more
        nodes the larger this is. The detector is fixed in angle, so the value
        is converted at the band center; it is larger for a bluer band. An
        instrument that reads a smaller field overrides this method, or the
        caller passes ``field_lod`` to ``Quadrature.build``. The separation is
        measured from the axis. For a smooth response, a caller who wants the
        rings of a source at separation ``q`` resolved everywhere passes the
        half-diagonal plus ``q``; on a tabulated product the extra nodes do not
        remove the floor set by how the pixels read the table.

        Args:
            band: The wavelength band whose center sets the lambda/D scale.

        Returns:
            The detector half-diagonal in lambda/D at the band center.
        """
        ny, nx = self.detector.shape
        half_diagonal_arcsec = (
            0.5 * math.hypot(ny, nx) * self.detector.pixel_scale_arcsec
        )
        if self.coronagraph is None:
            diameter_m = float(self.primary.diameter("circumscribed"))
            frame = Frame(band.center_nm, diameter_m, "circumscribed")
        else:
            frame = self.coronagraph.frame
        return half_diagonal_arcsec / float(frame.lod_to_arcsec(band.center_nm))

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

    def thermal_photon_rate_per_sr(self, wavelength_nm: float) -> float:
        """Photon rate of the warm telescope's own emission per steradian.

        The radiance of :class:`WarmEmission` times the geometric collecting
        area of the primary, in photons/s/nm/sr. It does not carry the system
        throughput, because the effective emissivity already includes the
        cold transmission, nor the flux scale, which calibrates sky photons.
        Multiply by the aperture solid angle for the rate in an aperture. Zero
        when ``warm_emission`` is None.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            Photons/s/nm/sr, shaped like ``wavelength_nm``.
        """
        if self.warm_emission is None:
            return jnp.zeros_like(jnp.asarray(wavelength_nm, dtype=float))
        return (
            self.warm_emission.photon_radiance(wavelength_nm)
            * self.primary.collecting_area_m2
        )

    def __repr__(self) -> str:
        """Tree-shaped summary of every component.

        Returns:
            Multi-line summary string.
        """
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
        if self.warm_emission is not None:
            lines.append(indent("warm_emission: " + repr(self.warm_emission)))
        if not isinstance(self.flux_scale, jax.core.Tracer) and bool(
            jnp.any(self.flux_scale != 1.0)
        ):
            lines.append(indent(f"flux_scale: {self.flux_scale}"))
        return "\n".join(lines)
