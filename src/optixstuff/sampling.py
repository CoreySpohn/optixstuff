"""The spectral sampling: what a build produced for one commanded event."""

import re

import equinox as eqx
import numpy as np
from hwoutils.provenance import (
    Stamps,
    build_stamps,
    merge_stamps,
    require_stamps,
    stamp_value,
)

from optixstuff.exposure import Exposure
from optixstuff.spectral import Band, Quadrature, Resolution

_CHECKABLE = frozenset(
    (
        "telescope_pupil_id",
        "coronagraph_pupil_id",
        "breaks_nm",
        "field_lod",
        "scene_layout",
    )
)
_READ = re.compile(r"[A-Za-z_]\w*(?:[.\[].*)?")


def _host(read):
    """Return ``read()`` or None when its inputs are traced or unavailable."""
    try:
        return read()
    except (TypeError, AttributeError):
        return None


def _hardware_facts(selected, bands) -> dict:
    """Recompute the build facts that a selected instrument view determines."""
    facts = {}
    pupil = getattr(selected, "pupil", None)
    facts["telescope_pupil_id"] = getattr(pupil, "pupil_id", None)
    coronagraph = getattr(selected, "coronagraph", None)
    children = getattr(coronagraph, "coronagraphs", None)
    if children:
        for i, child in enumerate(children):
            facts[f"child{i}.coronagraph_pupil_id"] = getattr(child, "pupil_id", None)
    else:
        facts["coronagraph_pupil_id"] = getattr(coronagraph, "pupil_id", None)
    facts["breaks_nm"] = _host(
        lambda: tuple(float(x) for x in np.asarray(selected.spectral_breaks_nm()))
    )
    facts["field_lod"] = _host(
        lambda: tuple(float(selected.spectral_field_lod(band)) for band in bands)
    )
    return facts


def _scene_layout(scene):
    """Digest facts of the tabulated sources a scene hands to ``Resolution.build``."""
    tables = getattr(scene, "spectral_tables", None)
    if tables is None:
        return None
    return _host(
        lambda: tuple((t.knot_digest, int(t.spectral_degree)) for t in tables())
    )


def build_facts(
    bands, *, instrument, selected=None, scene=None, field_lod=None, **extra
) -> Stamps:
    """Encode the host facts that determine a sampling, as build stamps.

    A builder passes the result (merged with its own ``order``, ``layout``,
    ``allow_wide_bands``, ``precision`` and ``producer`` facts) as ``built_at``;
    ``SpectralSampling.check_event`` recomputes the same facts to compare.

    Args:
        bands: The commanded bands, in order.
        instrument: Name of the configuration.
        selected: The selected instrument view, or None when unavailable. It is
            read for ``pupil.pupil_id``, ``coronagraph`` (and its
            ``coronagraphs`` children) ``pupil_id``, ``spectral_breaks_nm()`` and
            ``spectral_field_lod(band)``.
        scene: The scene whose ``spectral_tables()`` the resolution read, or None.
        field_lod: An explicit per-band field radius the build used. It is
            stamped with ``field_lod_source`` set to explicit; None records the
            value the view derives.
        **extra: Further facts, such as ``order``.

    Returns:
        Sorted build stamps; a fact that is unavailable or traced is unknown.
    """
    facts = {"instrument": instrument, **extra}
    if selected is not None:
        facts |= _hardware_facts(selected, tuple(bands))
    if field_lod is not None:
        facts["field_lod"] = tuple(
            float(x) for x in np.broadcast_to(np.asarray(field_lod, float), len(bands))
        )
        facts["field_lod_source"] = "explicit"
    if scene is not None:
        facts["scene_layout"] = _scene_layout(scene)
    return build_stamps(**facts)


class SpectralSampling(eqx.Module):
    """What a build produced for one commanded event, passed beside it.

    The quadrature nodes and weights are JAX leaves, so a rule hand-built from
    tracers is a legal sampling and node gradients and priors keep working;
    ``SpectralSampling(quadrature)`` is the hand-built form. ``resolution``
    holds the operators that turn each tabulated source of the scene into
    node values on ``quadrature``; it is empty when the scene has none.

    ``reads`` names, as owner-qualified paths (``scene.<path>`` or
    ``observatory.<path>``), the fitted values the build froze. ``built_at``
    records the facts that determined the sampling as typed stamps (see
    ``build_facts``); a hand-built sampling has none.

    Attributes:
        quadrature: Per-band wavelength nodes and weights.
        resolution: Operators of the scene's tabulated sources on
            ``quadrature``; empty when the scene has none.
        reads: Owner-qualified paths of the values the build cached; static.
        built_at: Typed build facts, sorted by label; static.
    """

    quadrature: Quadrature
    resolution: Resolution = eqx.field(default_factory=Resolution)
    reads: tuple[str, ...] = eqx.field(
        default=(), static=True, converter=tuple, kw_only=True
    )
    built_at: Stamps = eqx.field(default=(), static=True, converter=tuple, kw_only=True)

    def __check_init__(self):
        """Require a resolution built for this quadrature and well-formed records."""
        built = self.resolution.quadratures
        if built and built[0].layout_digest != self.quadrature.layout_digest:
            raise ValueError("the resolution was built for another quadrature")
        for read in self.reads:
            if not isinstance(read, str) or _READ.fullmatch(read) is None:
                raise ValueError(f"reads must be owner-qualified paths, got {read!r}")
        if self.built_at != merge_stamps(self.built_at):
            raise ValueError("built_at must be sorted stamps with unique labels")

    @property
    def bands(self) -> tuple[Band, ...]:
        """The bands of the quadrature, in order."""
        return self.quadrature.bands

    def check_event(
        self, exposure: Exposure, *, selected=None, scene=None
    ) -> tuple[str, ...]:
        """Check that this sampling serves a commanded event.

        A host check on static facts, legal under trace. The ordered bands and
        the configuration name are always compared. With ``selected`` the pupil
        identifiers, spectral breaks and (when the build derived it) the
        per-band field radius are recomputed from the view and compared; with
        ``scene`` the scene layout is. Unknown on either side is unverified; two
        known and different values raise. A label whose inputs are traced is
        unverified. A hand-built sampling is admitted with every label
        unverified.

        Args:
            exposure: The commanded event.
            selected: The selected instrument view for the event, or None.
            scene: The scene the resolution was built for, or None.

        Returns:
            The labels that could not be verified, sorted.

        Raises:
            ValueError: If the event bands differ from the sampling bands.
            StampMismatchError: If a recorded fact differs from the current one.
        """
        if exposure.bands != self.bands:
            raise ValueError(
                "the event bands differ from the sampling bands: "
                f"{exposure.bands!r} vs {self.bands!r}"
            )
        recorded = dict(self.built_at)
        facts = _hardware_facts(selected, self.bands) if selected is not None else {}
        explicit = recorded.get("field_lod_source") == stamp_value("explicit")
        if explicit:
            facts.pop("field_lod", None)
        facts["instrument"] = exposure.instrument
        if scene is not None:
            facts["scene_layout"] = _scene_layout(scene)
        current = build_stamps(**facts)
        compared = [label for label, _ in current]
        unverified = set(
            require_stamps(
                self.built_at, current, what="spectral sampling", labels=compared
            )
        )
        unverified |= {
            label
            for label in recorded
            if label.split(".")[-1] in _CHECKABLE and label not in compared
        }
        return tuple(sorted(unverified))
