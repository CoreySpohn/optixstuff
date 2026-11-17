"""Bind a fixed angular photometric aperture to spectral quadrature nodes."""

from collections.abc import Iterable
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates
from hwoutils.spectral import Band, Quadrature

from optixstuff.apertures import (
    DEFAULT_APERTURE_RULE,
    ApertureRule,
    Circular,
    HalfMaximum,
    Matched,
)
from optixstuff.coronagraph import AbstractCoronagraph, AbstractScalarCoronagraph
from optixstuff.optical_path import OpticalPath
from optixstuff.performance import PerformanceCurves


class _ApertureNode(eqx.Module):
    """Prepared photometry and angular scales at one quadrature wavelength."""

    coronagraph: AbstractCoronagraph
    record: PerformanceCurves
    wavelength_nm: jax.Array
    arcsec_per_lod: float | np.floating
    stellar_arcsec_per_lod: float | np.floating


class _ApertureReference(eqx.Module):
    """Reference geometry and quadrature nodes in one band."""

    wavelength_nm: float = eqx.field(static=True)
    arcsec_per_lod: float | np.floating
    record: PerformanceCurves
    nodes: tuple[_ApertureNode, ...]


class PreparedAperture(eqx.Module):
    """Host-prepared geometry and records for one fixed angular aperture.

    Rebuild when the instrument, bands, quadrature or reference rule changes.
    Each node retains the model that can evaluate its aperture and the
    stellar weights for that rule. Separation, time and stellar diameter
    remain runtime inputs to response.
    """

    quadrature: Quadrature
    references: tuple[_ApertureReference, ...]

    def response(self, node: _ApertureNode, separation_lod, time_d, stellar_diam_lod):
        """Return traceable throughput, aperture leakage and sky transmission."""
        coro = node.coronagraph
        wave = node.wavelength_nm
        rule = node.record.rule
        if isinstance(coro, AbstractScalarCoronagraph):
            tau = coro.throughput(separation_lod, wave, time_d=time_d, rule=rule)
            sky = coro.occulter_transmission(separation_lod, wave, time_d=time_d)
        else:
            tau = node.record.evaluate("throughput", separation_lod)
            px, py = coro.grid.unit_to_pixel(separation_lod, 0.0)
            coordinates = jnp.array([[py], [px]])
            sky = map_coordinates(
                coro.background_transmission(wave, grid=coro.grid).data,
                coordinates,
                order=1,
                mode="constant",
                cval=jnp.nan,
            )[0]
        leak = coro.leakage_fraction(
            separation_lod,
            wave,
            time_d=time_d,
            rule=rule,
            stellar_diam_lod=stellar_diam_lod,
        )
        return jnp.stack((tau, leak, sky))


def prepare_aperture(
    optical_path: OpticalPath,
    bands: Band | Iterable[Band],
    quadrature: Quadrature,
    *,
    rule: ApertureRule = DEFAULT_APERTURE_RULE,
    reference_wavelength_nm: float | None = None,
) -> PreparedAperture:
    """Prepare one fixed angular aperture per band before tracing its response.

    The band centre or supplied static reference wavelength sets the angular
    radius. A circular table aperture at a node has radius
    r_ref * lambda_ref / lambda_q in node lambda/D. The solid angle comes
    from the reference area and its angular scale.

    Scalar models describe calibrated fixed angular apertures and retain their
    rule at every node. Polychromatic HalfMaximum and Matched table rules
    require dilation of the frozen reference aperture or template.
    """
    bands = (bands,) if isinstance(bands, Band) else tuple(bands)
    waves = np.asarray(quadrature.nodes_nm)
    if waves.shape[0] != len(bands):
        raise ValueError("quadrature must have one row per band")
    coro = optical_path.coronagraph
    primary = optical_path.primary
    references = []
    for band, row, node_waves in zip(bands, waves, quadrature.nodes_nm, strict=True):
        ref = (
            band.center_nm
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )
        models = tuple(coro._performance_band(wave) for wave in row)
        if isinstance(rule, (HalfMaximum, Matched)) and any(
            w != ref and not isinstance(model, AbstractScalarCoronagraph)
            for w, model in zip(row, models, strict=True)
        ):
            raise NotImplementedError(
                f"{type(rule).__name__} needs support for dilating a frozen "
                "reference aperture/template at each separation"
            )
        _, reference = coro._performance_band(ref).prepared(rule)
        frame = primary.frame(ref, reference.frame.diameter_convention)
        nodes = []
        for w, wave, model in zip(row, node_waves, models, strict=True):
            if isinstance(model, AbstractScalarCoronagraph) or w == ref:
                scaled = rule
            elif isinstance(rule, Circular):
                scaled = replace(rule, radius_lod=rule.radius_lod * ref / float(w))
            else:
                raise TypeError("rule must be Circular, HalfMaximum or Matched")
            prepared_model, record = model.prepared(scaled)
            node_frame = primary.frame(w, record.frame.diameter_convention)
            stellar_frame = prepared_model.stellar_frame(w)
            nodes.append(
                _ApertureNode(
                    prepared_model,
                    record,
                    wave,
                    node_frame.lod_to_arcsec(w),
                    stellar_frame.lod_to_arcsec(w),
                )
            )
        references.append(
            _ApertureReference(
                float(ref), frame.lod_to_arcsec(ref), reference, tuple(nodes)
            )
        )
    return PreparedAperture(quadrature, tuple(references))
