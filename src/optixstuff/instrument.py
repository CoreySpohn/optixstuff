"""Instrument -- one named hardware configuration behind a shared telescope."""

from __future__ import annotations

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
    check_element_names,
    check_name,
)
from optixstuff.speckle import AbstractSpeckleField
from optixstuff.spectral import spectral_breaks


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
        if isinstance(wavelength_nm, (tuple, list)):
            wavelength_nm = jnp.asarray(wavelength_nm)
        dtype = jnp.result_type(
            wavelength_nm, self.temperature_K, self.effective_emissivity, 1.0
        )
        return self.effective_emissivity * planck_photon_radiance(
            jnp.asarray(wavelength_nm, dtype=dtype), self.temperature_K
        )


def train_throughput(elements, wavelength_nm):
    """Product of the throughputs of one train of elements.

    Args:
        elements: Tuple of optical elements.
        wavelength_nm: Wavelength in nanometers, scalar or an array.

    Returns:
        The product, shaped like ``wavelength_nm``. The empty train gives ones
        of the wavelength's dtype, weak when the wavelength is a Python scalar.
    """
    if not elements:
        return jnp.ones_like(jnp.asarray(wavelength_nm))
    result = 1.0
    for element in elements:
        result = result * element.get_throughput(wavelength_nm)
    return result


class Instrument(eqx.Module):
    """One named hardware configuration behind a shared telescope.

    A configuration holds what one observing mode adds to the telescope: its
    own train of attenuating elements (filters, relay optics), a coronagraph,
    a detector and the optional disperser, speckle field and warm emission. It
    does not know the pupil; the :class:`~optixstuff.Observatory` pairs it with
    the telescope, and the reads that need the pupil take ``telescope=``. Read a
    configuration through ``observatory.select(name)``, which returns the
    :class:`~optixstuff.SelectedInstrument` view with every read in one place.

    Attributes:
        name: Name of the configuration, unique in its observatory: a letter or
            underscore followed by letters, digits, underscores and hyphens.
        attenuating_elements: Ordered tuple of throughput elements of this
            configuration (filters, relay optics), after the telescope's own.
        coronagraph: Coronagraph slot (the image capability), or None for a
            curve-only configuration.
        detector: Focal-plane detector model.
        disperser: Optional IFS disperser descriptor; None for imaging mode.
        speckle: Optional time-varying speckle field (the stochastic
            wavefront-error residual on top of the deterministic
            coronagraph leakage floor); None for a perfect-wavefront path.
        flux_scale: Dimensionless factor on every photon rate the configuration
            collects from the sky, default one: the absolute flux calibration.
            It acts in :meth:`effective_area_m2`, the one place the collecting
            area and throughput are combined. Thermal emission does not carry
            it. It is linear, so a fit that frees it scales the sky fields of a
            rate record and need not rebuild one.
        warm_emission: Optional :class:`WarmEmission` of the warm telescope,
            owned here because it describes the assembled warm optical train
            and not the pupil or any one element. None, the default, is no
            thermal emission.
    """

    name: str = eqx.field(static=True)
    attenuating_elements: tuple[AbstractOpticalElement, ...] = eqx.field(
        default=(), converter=tuple, kw_only=True
    )
    coronagraph: AbstractCoronagraph | None = eqx.field(default=None, kw_only=True)
    detector: AbstractDetector = eqx.field(kw_only=True)
    disperser: AbstractDisperser | None = eqx.field(default=None, kw_only=True)
    speckle: AbstractSpeckleField | None = eqx.field(default=None, kw_only=True)
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)
    warm_emission: WarmEmission | None = eqx.field(default=None, kw_only=True)

    def __check_init__(self):
        """Validate the name, the element names and the flux scale.

        The frame check against the pupil belongs to the observatory, which
        knows the pupil. The flux scale must be finite and positive when
        concrete.
        """
        check_name(self.name, "instrument configuration")
        check_element_names(self.attenuating_elements)
        scale = self.flux_scale
        if not isinstance(scale, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(scale) & (scale > 0))
        ):
            raise ValueError("flux_scale must be finite and positive")

    def check_bands(self, bands, *, allow_wide_bands=False) -> None:
        """Check the observing bands against the installed coronagraph product.

        Args:
            bands: Concrete observing bands.
            allow_wide_bands: Accept the product's blur for wider observing bands.
        """
        if self.coronagraph is not None:
            self.coronagraph.check_bands(bands, allow_wide_bands=allow_wide_bands)

    def spectral_breaks_nm(self, *, telescope, detector) -> np.ndarray:
        """Combine the breaks of this configuration's train and detector yield.

        The telescope's own breaks are added by the selected view, not here.

        Args:
            telescope: The ``Telescope`` of the observatory; unused by this
                default, accepted so every configuration hook has one signature.
            detector: The detector to read, as resolved by the view.

        Returns:
            Sorted union of the break wavelengths in nm of the optical train,
            detector, coronagraph, disperser and speckle field.
        """
        optional = (self.coronagraph, self.disperser, self.speckle)
        return spectral_breaks(
            *self.attenuating_elements,
            detector,
            *(model for model in optional if model is not None),
        )

    def spectral_field_lod(self, band, *, telescope, detector) -> float:
        """Field radius the detector spans, in lambda/D at the center of ``band``.

        The half-diagonal of the detector: the largest separation from the
        detector center at which any pixel reads the diffraction pattern. A
        pattern at that separation slides across a band by more chromatic cycles
        than one nearer the center, so ``Quadrature.build`` gives the band more
        nodes the larger this is. The detector is fixed in angle, so the value
        is converted at the band center; it is larger for a bluer band. A
        configuration that reads a smaller field overrides this method, or the
        caller passes ``field_lod`` to ``Quadrature.build``. The separation is
        measured from the axis. For a smooth response, a caller who wants the
        rings of a source at separation ``q`` resolved everywhere passes the
        half-diagonal plus ``q``; on a tabulated product the extra nodes do not
        remove the floor set by how the pixels read the table.

        Args:
            band: The wavelength band whose center sets the lambda/D scale.
            telescope: The ``Telescope`` whose pupil sets the lambda/D scale
                when there is no coronagraph.
            detector: The detector whose half-diagonal is read.

        Returns:
            The detector half-diagonal in lambda/D at the band center.
        """
        ny, nx = detector.shape
        half_diagonal_arcsec = 0.5 * math.hypot(ny, nx) * detector.pixel_scale_arcsec
        if self.coronagraph is None:
            diameter_m = float(telescope.pupil.diameter("circumscribed"))
            frame = Frame(band.center_nm, diameter_m, "circumscribed")
        else:
            frame = self.coronagraph.frame
        return half_diagonal_arcsec / float(frame.lod_to_arcsec(band.center_nm))

    def throughput(self, wavelength_nm: float) -> float:
        """Throughput of this configuration's own attenuating elements.

        The telescope's train is not included; the selected view multiplies both.

        Args:
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Combined fractional throughput in [0, 1].
        """
        return train_throughput(self.attenuating_elements, wavelength_nm)

    def effective_area_m2(self, wavelength_nm: float, *, telescope) -> float:
        """Collecting area times both throughput trains times the flux scale.

        The photon rate collected from a source of unit flux density is this
        area times that flux density. Image and aperture rates both take it
        from here, so the flux scale is applied once.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.
            telescope: The ``Telescope`` with the pupil and its own train.

        Returns:
            Effective area in square meters, shaped like ``wavelength_nm``.
        """
        return (
            telescope.pupil.collecting_area_m2
            * telescope.throughput(wavelength_nm)
            * self.throughput(wavelength_nm)
            * self.flux_scale
        )

    def thermal_photon_rate_per_sr(self, wavelength_nm: float, *, telescope) -> float:
        """Photon rate of the warm telescope's own emission per steradian.

        The radiance of :class:`WarmEmission` times the geometric collecting
        area of the pupil, in photons/s/nm/sr. It does not carry either
        throughput train, because the effective emissivity already includes the
        cold transmission, nor the flux scale, which calibrates sky photons.
        Multiply by the aperture solid angle for the rate in an aperture. Zero
        when ``warm_emission`` is None.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.
            telescope: The ``Telescope`` whose pupil sets the area.

        Returns:
            Photons/s/nm/sr, shaped like ``wavelength_nm``.
        """
        if isinstance(wavelength_nm, (tuple, list)):
            wavelength_nm = jnp.asarray(wavelength_nm)
        area = telescope.pupil.collecting_area_m2
        if self.warm_emission is None:
            dtype = jnp.result_type(wavelength_nm, area, self.flux_scale, 1.0)
            return jnp.zeros(jnp.shape(wavelength_nm), dtype=dtype)
        dtype = jnp.result_type(
            wavelength_nm,
            area,
            self.warm_emission.temperature_K,
            self.warm_emission.effective_emissivity,
            1.0,
        )
        wavelength = jnp.asarray(wavelength_nm, dtype=dtype)
        return self.warm_emission.photon_radiance(wavelength) * area

    def __repr__(self) -> str:
        """Tree-shaped summary of every component.

        Returns:
            Multi-line summary string.
        """
        lines = [f"Instrument {self.name!r}"]
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
