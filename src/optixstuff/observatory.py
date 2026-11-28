"""Observatory: one telescope, named instrument configurations, selected view."""

from __future__ import annotations

import equinox as eqx
import numpy as np
from hwoutils.provenance import build_stamps, compare_stamps
from jaxtyping import Array

from optixstuff.coronagraph import AbstractCoronagraph
from optixstuff.detector import AbstractDetector
from optixstuff.disperser import AbstractDisperser
from optixstuff.instrument import Instrument, WarmEmission, train_throughput
from optixstuff.optical_elements import AbstractOpticalElement, check_element_names
from optixstuff.pupil import AbstractPupil, check_pupil_matches_frame
from optixstuff.speckle import AbstractSpeckleField

__all__ = [
    "Observatory",
    "SelectedInstrument",
    "Telescope",
    "require_selected",
]


def require_selected(selected) -> None:
    """Refuse an object that is not the view returned by ``Observatory.select``.

    A wrong object is a caller error, not an unknown fact, so every producer
    and reader that takes the view uses this one refusal.

    Args:
        selected: The candidate view.

    Raises:
        TypeError: If ``selected`` is not a ``SelectedInstrument``.
    """
    if not isinstance(selected, SelectedInstrument):
        raise TypeError(
            "expected the view returned by Observatory.select(name), got "
            f"{type(selected).__name__}; pass observatory.select(name)"
        )


class Telescope(eqx.Module):
    """The collecting geometry and the throughput every configuration shares.

    The pupil holds the one copy of the collecting area and the diameters in
    the model, and the identifier of the prescription it declares, so replacing
    the pupil replaces the identity in the same write. The attenuating elements
    are the throughput all configurations share, such as mirror coatings.

    Attributes:
        pupil: The collecting geometry.
        attenuating_elements: Ordered throughput elements shared by every
            configuration.
    """

    pupil: AbstractPupil
    attenuating_elements: tuple[AbstractOpticalElement, ...] = eqx.field(
        default=(), converter=tuple, kw_only=True
    )

    def __check_init__(self):
        """Require a pupil and unique valid element names."""
        if not isinstance(self.pupil, AbstractPupil):
            raise TypeError("pupil must be an AbstractPupil")
        check_element_names(self.attenuating_elements)

    def throughput(self, wavelength_nm):
        """Product of the shared elements' throughputs.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            The product, shaped like ``wavelength_nm``; ones of the wavelength's
            dtype for an empty train.
        """
        return train_throughput(self.attenuating_elements, wavelength_nm)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Sorted unique union of the shared elements' break wavelengths.

        Returns:
            Break wavelengths in nm, a float64 array (empty for no breaks).
        """
        breaks = [
            np.asarray(element.spectral_breaks_nm(), dtype=np.float64).ravel()
            for element in self.attenuating_elements
        ]
        if not breaks:
            return np.empty(0, dtype=np.float64)
        return np.unique(np.concatenate(breaks))


def _coronagraph_children(coronagraph):
    if coronagraph is None:
        return ()
    return tuple(getattr(coronagraph, "coronagraphs", None) or (coronagraph,))


def _bind(instrument, telescope):
    """Bind the coronagraph to the telescope, naming the owner on a refusal."""
    if instrument.coronagraph is None:
        return None
    try:
        return instrument.coronagraph.for_telescope(telescope)
    except ValueError as error:
        owner = f"instruments[{instrument.name}].coronagraph"
        raise type(error)(f"{owner}: {error}") from error


def _check_configuration(instrument, telescope):
    for child in _coronagraph_children(instrument.coronagraph):
        check_pupil_matches_frame(
            telescope.pupil,
            child.frame,
            source=f"instrument '{instrument.name}' coronagraph",
        )
    _bind(instrument, telescope)


def _instruments(value):
    if isinstance(value, Instrument):
        raise TypeError(
            "instruments must be a tuple of Instrument; pass (instrument,) "
            "for a single configuration"
        )
    return tuple(value)


class Observatory(eqx.Module):
    """One telescope and the named instrument configurations behind it.

    Every configuration shares the telescope's pupil and its throughput train,
    so a leaf of the pupil exists once and a gradient with respect to the
    observatory reaches it from every configuration. Two configurations built
    from one detector or coronagraph object hold independent copies of its
    leaves. Select a configuration with :meth:`select`; a raw assignment into
    the tree is not revalidated until the next ``select``.

    Attributes:
        telescope: The shared telescope.
        instruments: The configurations, with unique names.
    """

    telescope: Telescope
    instruments: tuple[Instrument, ...] = eqx.field(converter=_instruments)

    def __check_init__(self):
        """Require types, a configuration, unique names and a matching pupil.

        Every coronagraph (every child of a multiband product) must carry a
        lambda/D frame of the telescope's diameter, and two known, different
        pupil identifiers refuse the pairing here, at construction.
        """
        if not isinstance(self.telescope, Telescope):
            raise TypeError("telescope must be a Telescope")
        if not self.instruments:
            raise ValueError("an Observatory needs at least one instrument")
        for instrument in self.instruments:
            if not isinstance(instrument, Instrument):
                raise TypeError("instruments must hold Instrument configurations")
        names = [instrument.name for instrument in self.instruments]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"instrument names repeat: {repeated}")
        for instrument in self.instruments:
            _check_configuration(instrument, self.telescope)

    @property
    def names(self) -> tuple[str, ...]:
        """The configuration names, in stored order."""
        return tuple(instrument.name for instrument in self.instruments)

    def select(self, name: str) -> SelectedInstrument:
        """Return the derived view of one configuration on this telescope.

        The lookup is on static names, so it is legal under ``jit``, ``vmap``
        and ``grad``. The checks of construction run again, so an edit made by
        ``eqx.tree_at`` is validated here. The view is new on every call and
        owns no leaf.

        Args:
            name: The configuration name.

        Returns:
            The :class:`SelectedInstrument` for ``name``.

        Raises:
            ValueError: If ``name`` is not a configuration, or the coronagraph
                cannot serve the telescope.
        """
        for instrument in self.instruments:
            if instrument.name == name:
                _check_configuration(instrument, self.telescope)
                return SelectedInstrument(self.telescope, instrument)
        raise ValueError(f"no instrument named {name!r}; names are {list(self.names)}")


class SelectedInstrument(eqx.Module):
    """A derived view of one configuration on the shared telescope.

    The view holds exactly two references, so every leaf appears once and a
    gradient with respect to the observatory reaches the owner. It has the read
    signatures of a single instrument and is the object a simulation or an
    exposure-time calculation takes. It is built on use by
    :meth:`Observatory.select` and is not stored in a model.

    Attributes:
        telescope: The shared telescope.
        instrument: The selected configuration.
    """

    telescope: Telescope
    instrument: Instrument

    def __check_init__(self):
        """Require the two field types."""
        if not isinstance(self.telescope, Telescope):
            raise TypeError("telescope must be a Telescope")
        if not isinstance(self.instrument, Instrument):
            raise TypeError("instrument must be an Instrument")

    @property
    def name(self) -> str:
        """Name of the selected configuration."""
        return self.instrument.name

    @property
    def pupil(self) -> AbstractPupil:
        """The telescope pupil."""
        return self.telescope.pupil

    @property
    def coronagraph(self) -> AbstractCoronagraph | None:
        """The coronagraph bound to the telescope, or None; bound on each access."""
        return _bind(self.instrument, self.telescope)

    @property
    def detector(self) -> AbstractDetector:
        """The detector of the configuration, the one route to it in library code."""
        return self.instrument.detector

    @property
    def disperser(self) -> AbstractDisperser | None:
        """The disperser of the configuration, or None."""
        return self.instrument.disperser

    @property
    def speckle(self) -> AbstractSpeckleField | None:
        """The speckle field of the configuration, or None."""
        return self.instrument.speckle

    @property
    def warm_emission(self) -> WarmEmission | None:
        """The warm emission of the configuration, or None."""
        return self.instrument.warm_emission

    @property
    def flux_scale(self) -> Array:
        """The flux scale of the configuration."""
        return self.instrument.flux_scale

    @property
    def pupil_match(self) -> str:
        """``"matched"`` or ``"unverified"``, a display verdict only.

        ``"matched"`` when the pupil identifier is known and every coronagraph
        child declares the same identifier; ``"unverified"`` otherwise,
        including no coronagraph. Two unknown identifiers are never a match.
        """
        children = _coronagraph_children(self.instrument.coronagraph)
        if not children:
            return "unverified"
        pupil = build_stamps(pupil_id=self.telescope.pupil.pupil_id)
        for child in children:
            result = compare_stamps(build_stamps(pupil_id=child.pupil_id), pupil)
            if result.matched != ("pupil_id",):
                return "unverified"
        return "matched"

    def check_bands(self, bands, *, allow_wide_bands=False) -> None:
        """Check the observing bands against the coronagraph product.

        Args:
            bands: Concrete observing bands.
            allow_wide_bands: Accept the product's blur for wider observing bands.
        """
        self.instrument.check_bands(bands, allow_wide_bands=allow_wide_bands)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Union of the telescope's breaks and the configuration's.

        Returns:
            Sorted unique break wavelengths in nm.
        """
        own = self.instrument.spectral_breaks_nm(
            telescope=self.telescope, detector=self.detector
        )
        shared = np.asarray(self.telescope.spectral_breaks_nm(), dtype=float)
        return np.unique(np.concatenate([shared, np.asarray(own, dtype=float)]))

    def spectral_field_lod(self, band) -> float:
        """Field radius the detector spans, in lambda/D at the center of ``band``.

        Args:
            band: The wavelength band whose center sets the lambda/D scale.

        Returns:
            The value of the configuration's own hook for this telescope.
        """
        return self.instrument.spectral_field_lod(
            band, telescope=self.telescope, detector=self.detector
        )

    def system_throughput(self, wavelength_nm):
        """Total throughput: the telescope's train times the configuration's.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            Fractional throughput, shaped like ``wavelength_nm``.
        """
        return self.telescope.throughput(wavelength_nm) * self.instrument.throughput(
            wavelength_nm
        )

    def effective_area_m2(self, wavelength_nm):
        """Collecting area times both trains times the flux scale.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            Effective area in square meters, shaped like ``wavelength_nm``.
        """
        return self.instrument.effective_area_m2(
            wavelength_nm, telescope=self.telescope
        )

    def thermal_photon_rate_per_sr(self, wavelength_nm):
        """Photon rate of the warm telescope's emission per steradian.

        Args:
            wavelength_nm: Wavelength in nanometers, scalar or an array.

        Returns:
            Photons/s/nm/sr, shaped like ``wavelength_nm``.
        """
        return self.instrument.thermal_photon_rate_per_sr(
            wavelength_nm, telescope=self.telescope
        )

    def owner_path(self, attribute: str = "") -> str:
        """Path from the observatory to what an accessor reads, in name-key form.

        Args:
            attribute: ``""`` for the configuration, or one of ``"detector"``,
                ``"coronagraph"``, ``"disperser"``, ``"speckle"``,
                ``"warm_emission"``, ``"flux_scale"``, ``"pupil"`` or
                ``"telescope"``.

        Returns:
            The path, for example ``instruments[imager].detector``.

        Raises:
            ValueError: If ``attribute`` is not one of those.
        """
        if attribute == "telescope":
            return "telescope"
        if attribute == "pupil":
            return "telescope.pupil"
        base = f"instruments[{self.instrument.name}]"
        if attribute == "":
            return base
        if attribute in _CONFIGURATION_ATTRIBUTES:
            return f"{base}.{attribute}"
        raise ValueError(f"unknown attribute {attribute!r} for owner_path")


_CONFIGURATION_ATTRIBUTES = (
    "detector",
    "coronagraph",
    "disperser",
    "speckle",
    "warm_emission",
    "flux_scale",
)
