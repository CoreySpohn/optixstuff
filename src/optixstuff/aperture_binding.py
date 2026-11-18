"""Bind a fixed angular photometric aperture to spectral quadrature nodes."""

from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_rad
from hwoutils.spectral import Quadrature

from optixstuff.apertures import (
    DEFAULT_APERTURE_RULE,
    ApertureRule,
    Circular,
    Matched,
)
from optixstuff.coronagraph import (
    AbstractCoronagraph,
    AbstractImagingCoronagraph,
)
from optixstuff.optical_path import OpticalPath
from optixstuff.performance import PerformanceCurves, SampledProfile


class MatchedStamps(eqx.Module):
    """Native planet fractions (node,knot,y,x) and stellar/sky maps (node,y,x).

    Pixel solid angle is in steradians. The maps span the entire off-axis
    field, so there is no aperture truncation or special stellar-core mask.
    """

    planet: jax.Array
    stellar: jax.Array
    sky: jax.Array
    pixel_solid_angle_sr: float | jax.Array


class MatchedMoments(NamedTuple):
    """Node-dependent flux/solid-angle pairs and band-wide native pixel counts."""

    tau1: jax.Array
    tau2: jax.Array
    L1: jax.Array
    L2: jax.Array
    Omega1: jax.Array
    Omega2: jax.Array
    n_pix1: jax.Array
    n_pix2: jax.Array


def matched_moments(stamps, A, B, D, *, planet_spectrum=None):
    """Measure both moments on every knot with one inverse-variance band profile.

    A (incident star rate), B (incident sky rate per sr) and planet_spectrum
    include quadrature and spectral response at each node. The spectrum sets
    the shape of the band-average planet fraction. A flat spectrum is used
    when omitted or when the source is dark. D is the integrated uniform
    variance rate per native pixel. Use the same count units for A, B and D.

    The first six outputs have shape (node,knot), and the last two (knot,).
    Omega includes the local sky transmission. A noiseless pixel carrying
    signal takes precedence in the zero-background limit.
    """
    p, s, k = stamps.planet, stamps.stellar, stamps.sky
    a = stamps.pixel_solid_angle_sr
    spectrum = (
        jnp.ones(p.shape[0])
        if planet_spectrum is None
        else jnp.broadcast_to(planet_spectrum, (p.shape[0],))
    )
    spectrum = jnp.where(jnp.all(spectrum == 0), jnp.ones_like(spectrum), spectrum)
    profile = jnp.einsum("q,qkyx->kyx", spectrum / jnp.sum(spectrum), p)
    v = (
        jnp.sum(jnp.asarray(A)[..., None, None] * s, axis=0)
        + jnp.sum(jnp.asarray(B)[..., None, None] * k, axis=0) * a
        + D
    )
    noiseless = (v == 0) & (profile > 0)
    w = jnp.where(
        jnp.any(noiseless, axis=(-2, -1), keepdims=True),
        jnp.where(noiseless, profile, 0),
        profile / jnp.where(v == 0, 1, v),
    )
    w2 = w**2
    return MatchedMoments(
        jnp.einsum("kyx,qkyx->qk", w, p),
        jnp.einsum("kyx,qkyx->qk", w2, p),
        jnp.einsum("kyx,qyx->qk", w, s),
        jnp.einsum("kyx,qyx->qk", w2, s),
        jnp.einsum("kyx,qyx->qk", w, k) * a,
        jnp.einsum("kyx,qyx->qk", w2, k) * a,
        w.sum(axis=(-2, -1)),
        w2.sum(axis=(-2, -1)),
    )


def _stellar_pixels(model, wave, diameter):
    """Read native fractions directly, resampling only across distinct grids."""
    products = model.at_wavelength(wave)
    target = (
        None if products.stellar_grid == products.source_grid else products.source_grid
    )
    return model.stellar_map(wave, diameter, grid=target).data


def _sky_area_profile(model, wave, record):
    """Integrate the sky map with the calibrated native pixel coverage at each knot.

    Circular and HalfMaximum describe geometric regions. The subpixel indicator
    squared is itself, so the same covered area measures both Poisson moments.
    """
    grid = model.at_wavelength(wave).source_grid
    sky = model.uniform_sky_map(wave, grid=grid).data
    samples = model.aperture_samples(wave, record.separation_lod)
    dtype = jnp.result_type(sky, 1.0)
    with jax.enable_x64():
        areas = jnp.stack(
            [
                jnp.sum(record.rule.weights(psf, position) * sky) * grid.pixel_area
                for psf, position in samples
            ]
        ).astype(dtype)
    return SampledProfile(
        record.separation_lod,
        areas,
        quantity="transmitted_area_lod2",
        frame=record.frame,
        interpolation="linear",
    )


class _ApertureNode(eqx.Module):
    """Prepared photometry and angular scales at one quadrature wavelength."""

    coronagraph: AbstractCoronagraph
    record: PerformanceCurves
    wavelength_nm: jax.Array
    arcsec_per_lod: float | np.floating
    stellar_arcsec_per_lod: float | np.floating
    sky_area: SampledProfile | None


def _prepare_matched_stamps(nodes, record):
    # The table rule admits only nodes at the reference wavelength.
    first = nodes[0]
    model, wave = first.coronagraph, first.wavelength_nm
    grid = model.at_wavelength(wave).source_grid
    p = jnp.stack(
        [image.data for image, _ in model.aperture_samples(wave, record.separation_lod)]
    )
    s = _stellar_pixels(model, wave, record.stellar_diameter or 0.0)
    sky = model.uniform_sky_map(wave, grid=grid)

    def at_nodes(data):
        return jnp.broadcast_to(data, (len(nodes), *data.shape))

    return MatchedStamps(
        at_nodes(p),
        at_nodes(s),
        at_nodes(sky.data),
        grid.pixel_area * arcsec_to_rad(first.arcsec_per_lod) ** 2,
    )


class _ApertureReference(eqx.Module):
    """Reference geometry and quadrature nodes in one band."""

    wavelength_nm: float = eqx.field(static=True)
    arcsec_per_lod: float | np.floating
    record: PerformanceCurves
    nodes: tuple[_ApertureNode, ...]
    stamps: MatchedStamps | None = None

    def matched_response(
        self, separation_lod, stellar_diam_lod, A, B, D, planet_spectrum
    ):
        """Interpolate the scene-dependent moments using the record's conventions."""
        if self.stamps is None or A is None or B is None or D is None:
            raise ValueError("Matched band response requires stamps and A, B, D")
        stamps = self.stamps
        # Table nodes share the reference wavelength and the native field.
        if isinstance(stellar_diam_lod, jax.core.Tracer) or stellar_diam_lod != 0:
            diameter = jnp.where(
                stellar_diam_lod == 0,
                self.record.stellar_diameter or 0.0,
                stellar_diam_lod,
            )
            first = self.nodes[0]
            star = _stellar_pixels(first.coronagraph, first.wavelength_nm, diameter)
            stamps = replace(
                stamps, stellar=jnp.broadcast_to(star, stamps.stellar.shape)
            )
        moments = matched_moments(stamps, A, B, D, planet_spectrum=planet_spectrum)
        names = (
            "throughput",
            "throughput_variance",
            "leakage_fraction",
            "leakage_fraction_variance",
            "core_area_lod2",
            "core_area_variance_lod2",
            "core_area_lod2",
            "core_area_variance_lod2",
        )

        def interpolate(name, values):
            def evaluate(y):
                return self.record.evaluate_samples(name, separation_lod, y)

            return jax.vmap(evaluate)(values) if values.ndim == 2 else evaluate(values)

        return MatchedMoments(
            *(
                interpolate(name, values)
                for name, values in zip(names, moments, strict=True)
            )
        )


class PreparedAperture(eqx.Module):
    """Host-prepared geometry and records for one fixed angular aperture.

    Rebuild when the instrument, bands, quadrature or reference rule changes.
    Each node retains the model that can evaluate its aperture and the
    stellar weights for that rule. Separation, time and stellar diameter
    remain runtime inputs to response.
    """

    quadrature: Quadrature
    references: tuple[_ApertureReference, ...]

    def response(
        self,
        node,
        separation_lod,
        time_d,
        stellar_diam_lod,
        *,
        A=None,
        B=None,
        D=None,
        planet_spectrum=None,
    ):
        """Stack throughput, leakage, sky transmission and both variance fractions.

        The first three entries are planet, stellar and aperture-averaged sky
        fractions. The next three use the variance weights. For geometric
        apertures, both moments integrate the same indicator function.

        For Matched, pass the band reference and quadrature-weighted scene
        rates. The result is a MatchedMoments at the requested separation.
        Stellar diameter is in the native stellar frame at the reference.
        """
        if isinstance(node, _ApertureReference):
            return node.matched_response(
                separation_lod, stellar_diam_lod, A, B, D, planet_spectrum
            )
        coro = node.coronagraph
        wave = node.wavelength_nm
        rule = node.record.rule
        tau = coro.throughput(separation_lod, wave, time_d=time_d, rule=rule)
        tau_variance = coro.throughput(
            separation_lod, wave, time_d=time_d, rule=rule, variance=True
        )
        if node.sky_area is None:
            sky = coro.uniform_sky_response(separation_lod, wave, time_d=time_d)
            sky_variance = sky
        else:
            area = node.sky_area.evaluate(separation_lod)
            mean_area = node.record.evaluate("core_area_lod2", separation_lod)
            variance_area = node.record.evaluate(
                "core_area_variance_lod2", separation_lod
            )
            sky = jnp.where(
                mean_area > 0, area / jnp.where(mean_area > 0, mean_area, 1), 0
            )
            sky_variance = jnp.where(
                variance_area > 0,
                area / jnp.where(variance_area > 0, variance_area, 1),
                0,
            )
        leakage_args = dict(
            time_d=time_d,
            rule=rule,
            stellar_diam_lod=stellar_diam_lod,
        )
        leak = coro.leakage_fraction(separation_lod, wave, **leakage_args)
        leak_variance = coro.leakage_fraction(
            separation_lod, wave, variance=True, **leakage_args
        )
        return jnp.stack((tau, leak, sky, tau_variance, leak_variance, sky_variance))


def prepare_aperture(
    optical_path: OpticalPath,
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
    waves = np.asarray(quadrature.nodes_nm)
    coro = optical_path.coronagraph
    primary = optical_path.primary
    references = []
    for band, row, node_waves in zip(
        quadrature.bands, waves, quadrature.nodes_nm, strict=True
    ):
        ref = (
            band.center_nm
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )
        is_matched = isinstance(rule, Matched)
        record_rule = DEFAULT_APERTURE_RULE if is_matched else rule
        nodes = []
        for w, wave in zip(row, node_waves, strict=True):
            model = coro.band(w)
            imaging = isinstance(model, AbstractImagingCoronagraph)
            if is_matched and not imaging:
                raise ValueError(
                    "Matched requires native planet, stellar and sky images"
                )
            scaled = record_rule
            if imaging and w != ref:
                if not isinstance(rule, Circular):
                    raise NotImplementedError(
                        f"{type(rule).__name__} needs support for dilating a frozen "
                        "reference aperture/template at each separation"
                    )
                scaled = replace(rule, radius_lod=rule.radius_lod * ref / float(w))
            prepared_model, record = model.prepared(scaled)
            node_frame = primary.frame(w, record.frame.diameter_convention)
            stellar_frame = (
                prepared_model.stellar_frame(w) if imaging else prepared_model.frame
            )
            sky_area = (
                _sky_area_profile(prepared_model, w, record)
                if imaging and not is_matched
                else None
            )
            nodes.append(
                _ApertureNode(
                    prepared_model,
                    record,
                    wave,
                    node_frame.lod_to_arcsec(w),
                    stellar_frame.lod_to_arcsec(w),
                    sky_area,
                )
            )
        _, reference = coro.band(ref).prepared(record_rule)
        frame = primary.frame(ref, reference.frame.diameter_convention)
        stamps = _prepare_matched_stamps(nodes, reference) if is_matched else None
        references.append(
            _ApertureReference(
                float(ref), frame.lod_to_arcsec(ref), reference, tuple(nodes), stamps
            )
        )
    return PreparedAperture(quadrature, tuple(references))
