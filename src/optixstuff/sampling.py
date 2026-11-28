"""The spectral sampling: what a build produced for one commanded event."""

import re

import equinox as eqx
import numpy as np
from hwoutils.provenance import (
    Digest,
    Stamps,
    build_stamps,
    merge_stamps,
    require_stamps,
    stamp_value,
)

from optixstuff.coronagraph import MultiBandCoronagraph
from optixstuff.exposure import Exposure
from optixstuff.observatory import SelectedInstrument, require_selected
from optixstuff.spectral import Band, Quadrature, Resolution

_CHECKABLE = frozenset(
    ("pupil_id", "children", "breaks_nm", "field_lod", "scene_layout", "layout")
)
_READ = re.compile(r"[A-Za-z_]\w*(?:[.\[].*)?")


def _host(read):
    """Return ``read()`` or None when its inputs are traced or unavailable."""
    try:
        return read()
    except (TypeError, AttributeError):
        return None


def hardware_facts(selected: SelectedInstrument) -> Stamps:
    """Stamp the hardware identity a selected view determines.

    The labels are ``instrument`` (the configuration name),
    ``telescope.pupil_id``, ``coronagraph.children`` (the number of children of
    a multiband product, ``int:0`` for a single product or none) and, for a
    single product or none, ``coronagraph.pupil_id`` or, for a multiband
    product, ``coronagraph.child{i}.pupil_id`` for every child. A pupil
    identifier that is not declared is unknown. The stamps are computed from the
    view in hand, never copied from another product.

    Args:
        selected: The view returned by ``Observatory.select(name)``.

    Returns:
        Sorted build stamps.

    Raises:
        TypeError: If ``selected`` is not a ``SelectedInstrument``.
    """
    require_selected(selected)
    coronagraph = selected.coronagraph
    facts = {
        "instrument": selected.name,
        "telescope.pupil_id": selected.pupil.pupil_id,
    }
    if isinstance(coronagraph, MultiBandCoronagraph):
        facts["coronagraph.children"] = len(coronagraph.coronagraphs)
        for i, child in enumerate(coronagraph.coronagraphs):
            facts[f"coronagraph.child{i}.pupil_id"] = child.pupil_id
    else:
        facts["coronagraph.children"] = 0
        facts["coronagraph.pupil_id"] = (
            None if coronagraph is None else coronagraph.pupil_id
        )
    return build_stamps(**facts)


def quadrature_facts(quadrature: Quadrature, *, values: bool = False) -> Stamps:
    """Stamp the identity of a quadrature.

    ``layout`` is ``Quadrature.layout_digest`` (bands and pieces), static and
    readable under trace; it is unknown for a hand-built rule. With
    ``values=True`` the result also holds ``quadrature``, ``Quadrature.digest``
    (adds nodes and weights), which is unknown when the leaves are traced. A
    product that holds its quadrature stamps ``layout`` only.

    Args:
        quadrature: The rule.
        values: Also stamp the value identity.

    Returns:
        Sorted build stamps with 64-hex SHA-256 digests.
    """
    layout = quadrature.layout_digest
    facts = {"layout": None if layout is None else Digest(layout)}
    if values:
        value = _host(lambda: quadrature.digest)
        facts["quadrature"] = None if value is None else Digest(value)
    return build_stamps(**facts)


def _view_facts(selected, bands) -> Stamps:
    """Stamp the breaks and derived field radius of a view over ``bands``."""
    return build_stamps(
        breaks_nm=_host(
            lambda: tuple(float(x) for x in np.asarray(selected.spectral_breaks_nm()))
        ),
        field_lod=_host(
            lambda: tuple(float(selected.spectral_field_lod(band)) for band in bands)
        ),
    )


def _scene_layout(scene):
    """Digest facts of the tabulated sources a scene hands to ``Resolution.build``."""
    tables = getattr(scene, "spectral_tables", None)
    if tables is None:
        return None
    return _host(
        lambda: tuple((t.knot_digest, int(t.spectral_degree)) for t in tables())
    )


def build_facts(
    quadrature, /, *, instrument, selected=None, scene=None, field_lod=None, **extra
) -> Stamps:
    """Encode the host facts that determine a sampling, as build stamps.

    A builder passes the result (merged with its own ``order``,
    ``allow_wide_bands``, ``precision`` and ``producer`` facts) as ``built_at``;
    ``SpectralSampling.check_event`` recomputes the same facts to compare. The
    ``layout`` stamp is written here, from the quadrature in hand.

    Args:
        quadrature: The rule the build produced; its bands are the commanded
            bands, in order.
        instrument: Name of the configuration.
        selected: The view returned by ``Observatory.select(name)``, or None
            when unavailable. Its ``hardware_facts`` are stamped, with its
            ``spectral_breaks_nm()`` and ``spectral_field_lod(band)``.
        scene: The scene whose ``spectral_tables()`` the resolution read, or None.
        field_lod: An explicit per-band field radius the build used. It is
            stamped with ``field_lod_source`` set to explicit; None records the
            value the view derives.
        **extra: Further facts, such as ``order``.

    Returns:
        Sorted build stamps; a fact that is unavailable or traced is unknown.

    Raises:
        TypeError: If ``selected`` is neither None nor a ``SelectedInstrument``.
        ValueError: If ``layout`` or ``quadrature`` is in ``extra``, or the
            instrument name differs from the view's.
    """
    clash = {"layout", "quadrature"} & extra.keys()
    if clash:
        raise ValueError(
            f"{sorted(clash)} are stamped by build_facts from the quadrature; "
            "do not pass them"
        )
    bands = quadrature.bands
    facts = {"instrument": instrument, **extra}
    if field_lod is not None:
        facts["field_lod"] = tuple(
            float(x) for x in np.broadcast_to(np.asarray(field_lod, float), len(bands))
        )
        facts["field_lod_source"] = "explicit"
    if scene is not None:
        facts["scene_layout"] = _scene_layout(scene)
    parts = [build_stamps(**facts), quadrature_facts(quadrature)]
    if selected is not None:
        hardware = hardware_facts(selected)
        view = _view_facts(selected, bands)
        if field_lod is not None:
            view = tuple(item for item in view if item[0] != "field_lod")
        parts += [hardware, view]
    return merge_stamps(*parts)


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

        A host check on static facts, legal under trace. The ordered bands, the
        configuration name and the ``layout`` recomputed from the quadrature
        are always compared. With ``selected`` the ``hardware_facts``, the
        spectral breaks and (when the build derived it) the per-band field
        radius are recomputed from the view and compared; with ``scene`` the
        scene layout is. Unknown on either side is unverified; two known and
        different values raise. A label whose inputs are traced is unverified.
        A hand-built sampling is admitted with every label unverified.

        Args:
            exposure: The commanded event.
            selected: The view returned by ``Observatory.select(name)``, or
                None.
            scene: The scene the resolution was built for, or None.

        Returns:
            The labels that could not be verified, sorted.

        Raises:
            TypeError: If ``selected`` is neither None nor a
                ``SelectedInstrument``.
            ValueError: If the event bands differ from the sampling bands.
            StampMismatchError: If a recorded fact differs from the current one.
        """
        if exposure.bands != self.bands:
            raise ValueError(
                "the event bands differ from the sampling bands: "
                f"{exposure.bands!r} vs {self.bands!r}"
            )
        recorded = dict(self.built_at)
        facts = dict(quadrature_facts(self.quadrature))
        if selected is not None:
            facts |= dict(hardware_facts(selected)) | dict(
                _view_facts(selected, self.bands)
            )
        if recorded.get("field_lod_source") == stamp_value("explicit"):
            facts.pop("field_lod", None)
        facts["instrument"] = stamp_value(exposure.instrument)
        if scene is not None:
            facts["scene_layout"] = stamp_value(_scene_layout(scene))
        current = tuple(sorted(facts.items()))
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
