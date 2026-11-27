"""Mixed precision checks for the optics and detector forwards."""

import re
from dataclasses import replace
from functools import partial
from inspect import unwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_instrument, make_linear_speckle, synthetic_coronagraph
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from test_disperser import _lenslet

import optixstuff as ox
from optixstuff.coronagraph import on_grid
from optixstuff.grids import (
    Frame,
    SkyOrientation,
    coverage,
    footprint_overlap,
    integrate_footprints,
    resample,
)
from optixstuff.performance import SampledProfile, _stellar_samples
from optixstuff.radial import radial_profile
from optixstuff.rates import floor_rate, rate_below_floor
from optixstuff.spectral import Knots, Transmission, rebin_bands, trapezoid_weights

pytestmark = pytest.mark.float64_only(
    reason="Traces float32 forwards with x64 enabled."
)


class _UpstreamPrecisionFailure(Exception):
    """Only a reproduced dependency failure qualifies for strict xfail."""


@pytest.mark.parametrize("same_grid", [True, False])
def test_stellar_samples_precision_closed(same_grid):
    """Fails if stellar interpolation widens a float32 image with x64 enabled."""
    frame = Frame(500, 6, "circumscribed")
    stellar = ox.PixelGrid.focal(6, 0.25, frame)
    planet = stellar if same_grid else ox.PixelGrid.focal(6, 0.2, frame)
    samples = partial(
        _stellar_samples,
        sg=stellar,
        planet_grid=planet,
        sy=1,
        sx=2,
        shape=(4, 4),
        factor=2,
    )
    assert_precision_closed(samples, jnp.ones(stellar.shape, dtype=jnp.float32))


def _deterministic_poisson(key, lam, shape=None):
    """Return integer fixture means while exercising the caller's mean arithmetic."""
    values = lam if shape is None else jnp.broadcast_to(lam, shape)
    return values.astype(jnp.int32)


def _spectral_cases():
    band = ox.Band(480, 520)
    rule = ox.Quadrature.build((band,), order=3)
    wave = jnp.array([488.0, 512.0])
    table = Transmission([480.0, 500.0, 520.0], [0.5, 0.75, 1.0])
    values = jnp.array([[0.75, 1.0, 1.25]])
    return {
        "band": (band.contains_wavelength, (wave,), 0),
        "rebin": (
            lambda x: rebin_bands(x, (band,), band.channels(2)),
            (values[:, 0],),
            4,
        ),
        "trapezoid": (trapezoid_weights, (wave,), 3),
        "transmission": (Transmission.__call__, (table, wave), 8),
        "transmission_mean": (lambda t: t.mean_nm, (table,), 24),
        "transmission_integral": (lambda t: t.integrate(band), (table,), 24),
        "quadrature_integral": (ox.Quadrature.integrate, (rule, values), 6),
        "quadrature_mean": (ox.Quadrature.mean, (rule, values), 8),
        "quadrature_band": (lambda r: r.band(0).weights_nm, (rule,), 1),
        "quadrature_active": (lambda r: r.active, (rule,), 0),
        "quadrature_weights": (ox.Quadrature.weights, (rule,), 1),
    }


def _reduction_case(kind):
    band = ox.Band(480, 528)
    rule = ox.Quadrature.build((band,), order=3)
    knots = Knots([470, 540] if kind == "identity" else [470, 510, 540])
    interpolant = (
        (lambda values, w: jnp.interp(w, knots.array(), values))
        if (kind == "table")
        else None
    )
    reduction = ox.Reduction.build(rule, knots, degree=1, interpolant=interpolant)
    assert reduction.kind == kind
    values = {
        "identity": lambda: rule.nodes_nm / 500,
        "table": lambda: jnp.array([0.875, 1.0, 1.125]),
        "fine": lambda: reduction.fine_nm / 500,
    }[kind]()
    return ox.Reduction.apply, (reduction, values), 16


def _grid_case(entry):
    frame = ox.SimplePrimary(
        circumscribed_diameter_m=6.0, collecting_area_m2=28.0
    ).frame(500.0)
    source = ox.PixelGrid.focal(5, 0.5, frame)
    target = ox.PixelGrid.regular((3, 4), 0.4, "lod", frame=frame)
    data = 1 + jnp.arange(25, dtype=jnp.float64).reshape(5, 5) / 32
    image = ox.Image(data, source, (), ox.Measure.density("grid"), "flux")
    wave = jnp.array(512.0)
    x = jnp.array([1.125, 2.375])
    bounds = {
        "nearest": 0,
        "linear": 24,
        "cubic": 96,
        "cubic_spline": 96,
        "band_limited": 112,
        "band_limited_integral": 128,
        "linear_integral": 48,
    }
    if entry.startswith("resample_"):
        method = entry.removeprefix("resample_")
        integral = method.endswith("_integral")
        method = method.removesuffix("_integral")
        if method == "band_limited":
            image = ox.Image(data + 0.25j * data, source, (), image.measure, "field")
        return (
            lambda im, w: (
                resample(
                    replace(
                        im,
                        grid=im.grid.to_arcsec(w),
                        measure=ox.Measure.integral() if integral else im.measure,
                    ),
                    target.to_arcsec(500.0),
                    reconstruction=method,
                ).data
            ),
            (image, wave),
            bounds[entry.removeprefix("resample_")],
        )
    if entry.startswith("on_grid_") and entry != "on_grid_cells":
        method = entry.removeprefix("on_grid_")
        footprint_bounds = {
            "nearest": 72,
            "linear": 84,
            "cubic": 104,
            "cubic_spline": 126,
            "band_limited": bounds["band_limited_integral"],
        }
        return (
            lambda im, w: (
                on_grid(
                    ox.Image(
                        im.data,
                        im.grid.to_arcsec(w),
                        (),
                        im.measure,
                        im.quantity,
                        method,
                    ),
                    target.to_arcsec(500.0),
                ).data
            ),
            (image, wave),
            footprint_bounds[method],
        )
    if entry.startswith("radial_"):
        return (
            partial(radial_profile, nbins=2, stat=entry.removeprefix("radial_")),
            (image,),
            16,
        )
    return {
        "coordinates": (lambda w: source.to_arcsec(w).coordinates(), (wave,), 4),
        "radius": (lambda w: source.to_arcsec(w).radius(), (wave,), 6),
        "to_lod": (
            lambda w: target.to_arcsec(500.0).to_lod(w, frame).coordinates(),
            (wave,),
            6,
        ),
        "sky_to_detector": (SkyOrientation(30).sky_to_detector, (x, x), 4),
        "detector_to_sky": (SkyOrientation(30).detector_to_sky, (x, x), 4),
        "pixel_to_unit": (source.pixel_to_unit, (x, x), 4),
        "unit_to_pixel": (source.unit_to_pixel, (x, x), 4),
        "coverage": (
            lambda w: coverage(source.to_arcsec(w), target.to_arcsec(500.0)),
            (wave,),
            0,
        ),
        "coverage_integral": (
            lambda w: coverage(
                source.to_arcsec(w),
                target.to_arcsec(500.0),
                reconstruction="linear",
                measure=ox.Measure.integral(),
            ),
            (wave,),
            0,
        ),
        "coverage_mask": (
            lambda valid: coverage(
                source,
                target,
                reconstruction="linear",
                measure=image.measure,
                valid=valid,
            ),
            (jnp.ones(source.shape, dtype=bool),),
            0,
        ),
        "image_integral": (lambda im: im.integrated().data, (image,), 2),
        "image_density": (
            lambda im: im.density("sr").data,
            (ox.Image(data, source.to_arcsec(500.0), (), image.measure, "flux"),),
            4,
        ),
        "footprints": (
            lambda im, overlap: integrate_footprints(im, overlap).data,
            (image, footprint_overlap(source, target)),
            32,
        ),
        "on_grid_cells": (
            lambda im: on_grid(im, target).data,
            (ox.Image(data, source, (), ox.Measure.integral(), "flux", "nearest"),),
            32,
        ),
    }[entry]


def _detector_case(entry):
    detector = ox.Detector(
        shape=(2, 2),
        pixel_scale_arcsec=0.01,
        qe_curve=([480.0, 520.0], [0.7, 0.8]),
        read_noise_e_rms=1.5,
        dark_current_e_per_pixel_s=0.125,
        clock_induced_charge_e_per_pixel_frame=0.25,
        excess_noise_factor_sq=1.5,
    )
    ideal = ox.IdealDetector(shape=(2, 2), pixel_scale_arcsec=0.01)
    rule = ox.Quadrature.build((ox.Band(480, 520),), order=3)
    plan = ox.FramePlan(jnp.array(2.0), jnp.array(3.0))
    poisson_detector = ox.Detector(
        shape=(2, 2),
        pixel_scale_arcsec=0.01,
        read_noise_e_rms=0,
        excess_noise_factor_sq=1,
    )
    photons = ox.Rates(rule, "photon", planet=jnp.array([[1.0, 1.25, 1.5]]))
    electrons = ox.Rates(rule.bands, "electron", planet=jnp.array([2.0]))
    wave = rule.nodes_nm
    rate, scale = jnp.array([-0.125, -1e-6, 0.5]), jnp.array(1.0)
    key = jax.random.key(0)
    post = ox.PostProcessing(background_variance_factor=1.25)
    return {
        "rate_total": (ox.Rates.total, (electrons,), 6),
        "rate_construct": (
            lambda x: ox.Rates(rule.bands, "electron", planet=x).total(),
            (electrons.planet,),
            6,
        ),
        "rate_zeros": (
            lambda q: ox.Rates.zeros(q, "photon").total(),
            (rule,),
            0,
        ),
        "rate_integrate": (ox.Rates.integrate, (photons,), 6),
        "rate_below_floor": (rate_below_floor, (rate, scale), 0),
        "floor_rate": (floor_rate, (rate, scale), 1),
        "frame_plan": (
            lambda t, dt: ox.frames_for_time(t, dt).live_time_s,
            (jnp.array(5.0), plan.frame_time_s),
            2,
        ),
        "frame_plan_float_duration": (
            lambda t: ox.FramePlan(t, 3).n_frames,
            (plan.frame_time_s,),
            0,
        ),
        "occupied_time": (ox.FramePlan.occupied_time_s, (plan, detector), 2),
        "qe": (ox.Detector.quantum_efficiency, (detector, wave), 8),
        "ideal_qe": (ox.IdealDetector.quantum_efficiency, (ideal, wave), 0),
        "electrons": (ox.AbstractDetector.electrons, (detector, photons), 16),
        "ideal_electrons": (ox.AbstractDetector.electrons, (ideal, photons), 8),
        "counts": (
            ox.AbstractDetector.mean_and_variance,
            (detector, electrons, plan),
            16,
        ),
        "counts_aperture": (
            lambda d, r, p: d.mean_and_variance(
                eqx.tree_at(lambda x: x.n_pix, r, jnp.asarray(3, dtype=np.int64)), p
            ),
            (
                detector,
                ox.Rates(rule.bands, "electron", planet=electrons.planet, n_pix=3),
                plan,
            ),
            16,
        ),
        "snr": (
            lambda r, d, p, t: ox.snr(r, d, t, frame_time_s=t / 2, post_processing=p),
            (electrons, detector, post, plan.frame_time_s),
            32,
        ),
        "exposure_time": (
            lambda r, d, p, t: ox.exposure_time(
                r, d, t, frame_time_s=t / 2, post_processing=p
            ),
            (electrons, detector, post, plan.frame_time_s),
            32,
        ),
        "draw": (ox.AbstractDetector.draw, (poisson_detector, key, electrons, plan), 0),
        "ideal_draw": (ox.IdealDetector.draw, (ideal, key, electrons, plan), 0),
    }[entry]


def _coronagraph_case(entry):
    model = synthetic_coronagraph(n=8)
    wavelength = jnp.array(500.0)
    position = jnp.array([0.125, 0.25])
    diameter = jnp.array(0.125)
    target = ox.PixelGrid.focal(4, 0.5, model.frame)
    if entry == "source_psfs":

        def forward(coro, w, x):
            result = coro.source_psfs(w, x, x, grid=target)
            return result.image.data, result.valid

        return forward, (model, wavelength, position), 80
    if entry == "stellar_map":
        return (
            lambda c, w, d: c.stellar_map(w, d, grid=target).data,
            (model, wavelength, diameter),
            64,
        )
    if entry == "on_axis_psf":
        return lambda c, w: c.on_axis_psf(w, target).data, (model, wavelength), 64
    if entry == "off_axis_psf":
        return (
            lambda c, w, x: c.off_axis_psf(w, x, target).data,
            (model, wavelength, position[0]),
            80,
        )
    if entry == "uniform_sky_map":
        return (
            lambda c, w: c.uniform_sky_map(w, grid=target).data,
            (model, wavelength),
            24,
        )
    curves = model.curves()
    separation = jnp.array([0.5, 1.375])
    if entry == "performance_diameter":
        curves = replace(
            curves,
            stellar_diameters_lod=[0.1, 0.2, 0.3],
            **{
                name: curve * np.array([[1.0], [1.25], [1.5]])
                for name in (
                    "leakage_fraction",
                    "leakage_density_lod2",
                    "raw_contrast",
                    "leakage_fraction_variance",
                    "leakage_density_variance_lod2",
                )
                if (curve := getattr(curves, name)) is not None
            },
        )
        return (
            lambda q, d: curves.evaluate("leakage_fraction", q, d),
            (separation, diameter),
            48,
        )
    # Use host-built records directly, as well as model arguments in the table tests.
    return {
        "sampled_profile": (model.uniform_sky_profile.evaluate, (separation,), 16),
        "sampled_profile_dynamic": (
            lambda y, q: SampledProfile(
                [0.0, 1.0, 2.0],
                y,
                quantity="flux",
                frame=model.frame,
                interpolation="log",
            ).evaluate(q),
            (jnp.array([0.5, 0.75, 1.0]), separation),
            48,
        ),
        "sampled_support": (model.uniform_sky_profile.supported, (separation,), 0),
        "performance": (lambda q: curves.evaluate("throughput", q), (separation,), 16),
        "performance_leakage": (
            lambda q: curves.evaluate("leakage_fraction", q),
            (separation,),
            32,
        ),
        "performance_support": (curves.supported, (separation,), 0),
    }[entry]


def _optics_case(entry):
    wavelength = jnp.array([600.0, 700.0])
    table = Transmission([550.0, 800.0], [0.5, 1.0])
    element = ox.TransmissiveElement(table)
    lenslet = _lenslet(dispersion_coeffs=jnp.array([2.0, 100.0, 0.0]))
    model = make_instrument(synthetic_coronagraph(n=8))
    warm = ox.WarmEmission(jnp.array(290.0), jnp.array(0.125))
    warm_model = make_instrument(model.coronagraph, warm_emission=warm)
    speckle = make_linear_speckle()
    elapsed = jnp.array(0.5)
    quadrature = ox.Quadrature.build((ox.Band(550.0, 800.0),), order=3)
    return {
        "dispersion": (ox.LensletDisperser.dispersion_px, (lenslet, wavelength), 16),
        "resolution": (
            ox.LensletDisperser.spectral_resolution,
            (lenslet, wavelength),
            8,
        ),
        "resolution_constant": (
            lambda d: d.spectral_resolution(600.0),
            (_lenslet(dispersion_coeffs=jnp.array([2.0])),),
            0,
        ),
        "spread": (ox.LensletDisperser.n_pix_spread, (lenslet, *wavelength), 16),
        "disperser_throughput": (
            ox.LensletDisperser.throughput,
            (lenslet, wavelength),
            2,
        ),
        "optical_apply": (
            ox.TransmissiveElement.apply,
            (element, wavelength / 500, wavelength),
            8,
        ),
        "optical_throughput": (
            ox.TransmissiveElement.get_throughput,
            (element, wavelength),
            8,
        ),
        "constant_throughput": (
            ox.ConstantThroughput.get_throughput,
            (ox.ConstantThroughput(jnp.array(0.75)), wavelength),
            0,
        ),
        "system_throughput": (ox.Instrument.system_throughput, (model, wavelength), 2),
        "effective_area": (ox.Instrument.effective_area_m2, (model, wavelength), 3),
        "thermal_none": (
            ox.Instrument.thermal_photon_rate_per_sr,
            (model, wavelength),
            0,
        ),
        "thermal_none_scalar": (
            lambda m: m.thermal_photon_rate_per_sr(500.0),
            (model,),
            0,
        ),
        "thermal_warm": (
            lambda m: m.thermal_photon_rate_per_sr(7000.0),
            (warm_model,),
            32,
        ),
        "photon_radiance": (
            ox.WarmEmission.photon_radiance,
            (warm, jnp.array([7000.0, 9000.0])),
            32,
        ),
        "focal_per_pixel": (ox.FocalResponse.per_pixel, (speckle.response,), 4),
        "elapsed": (ox.AbstractSpeckleField.elapsed_s, (speckle, elapsed), 1),
        "elapsed_scalar": (lambda f: f.elapsed_s(0.5), (speckle,), 1),
        "speckle_average_scalars": (
            lambda f: f.realize_average(wavelength_nm=600.0, exposure_s=2.0).data,
            (speckle,),
            64,
        ),
        "speckle": (
            lambda f, w, t: f.realize(wavelength_nm=w, elapsed_s=t).data,
            (speckle, wavelength, elapsed),
            48,
        ),
        "speckle_average": (
            lambda f, w, t: f.realize_average(wavelength_nm=w, exposure_s=t).data,
            (speckle, wavelength, elapsed),
            64,
        ),
        "speckle_broadened": (
            lambda f, q, t: f.broadened(quadrature=q, elapsed_s=t).data,
            (speckle, quadrature, elapsed),
            64,
        ),
    }[entry]


@pytest.mark.parametrize(
    ("group", "entry"),
    [
        *(
            ("spectral", name)
            for name in (
                "band",
                "rebin",
                "trapezoid",
                "transmission",
                "transmission_mean",
                "transmission_integral",
                "quadrature_integral",
                "quadrature_mean",
                "quadrature_band",
                "quadrature_active",
                "quadrature_weights",
            )
        ),
        *(("reduction", kind) for kind in ("identity", "table", "fine")),
        *(
            ("grid", name)
            for name in (
                "coordinates",
                "radius",
                "to_lod",
                "sky_to_detector",
                "detector_to_sky",
                "pixel_to_unit",
                "unit_to_pixel",
                "coverage",
                "coverage_mask",
                "coverage_integral",
                "image_integral",
                "image_density",
                "footprints",
                "on_grid_cells",
                "radial_mean",
                "radial_max",
                "radial_median",
                "resample_nearest",
                "resample_linear",
                "resample_cubic",
                "resample_cubic_spline",
                "resample_band_limited",
                "resample_band_limited_integral",
                "resample_linear_integral",
                "on_grid_nearest",
                "on_grid_linear",
                "on_grid_cubic",
                "on_grid_cubic_spline",
                "on_grid_band_limited",
            )
        ),
        *(
            ("detector", name)
            for name in (
                "rate_total",
                "rate_construct",
                "rate_zeros",
                "rate_integrate",
                "rate_below_floor",
                "floor_rate",
                "frame_plan",
                "frame_plan_float_duration",
                "occupied_time",
                "qe",
                "ideal_qe",
                "electrons",
                "ideal_electrons",
                "counts",
                "counts_aperture",
                "snr",
                "exposure_time",
                "draw",
                "ideal_draw",
            )
        ),
        *(
            ("coronagraph", name)
            for name in (
                "source_psfs",
                "stellar_map",
                "uniform_sky_map",
                "on_axis_psf",
                "off_axis_psf",
                "sampled_profile",
                "sampled_support",
                "performance",
                "performance_leakage",
                "performance_diameter",
                "performance_support",
            )
        ),
        pytest.param(
            "coronagraph",
            "sampled_profile_dynamic",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_UpstreamPrecisionFailure,
                reason="Interpax PCHIP endpoint branches differ in precision.",
            ),
        ),
        *(
            ("optics", name)
            for name in (
                "dispersion",
                "resolution",
                "resolution_constant",
                "spread",
                "disperser_throughput",
                "optical_apply",
                "optical_throughput",
                "constant_throughput",
                "system_throughput",
                "effective_area",
                "thermal_none",
                "thermal_none_scalar",
                "thermal_warm",
                "photon_radiance",
                "focal_per_pixel",
                "elapsed",
                "elapsed_scalar",
                "speckle_average_scalars",
                "speckle",
                "speckle_average",
                "speckle_broadened",
            )
        ),
    ],
)
def test_forward_precision_closed(group, entry, monkeypatch):
    """Trace each forward and compare to float64 at the peak of each leaf.

    Masks and dyadic frame counts agree exactly. The two exact Poisson draws
    use the same key and a positive mean of twelve electrons. The closure
    check isolates Poisson, whose trace is tested directly below.

    Allow up to 24 float32 epsilons for short spectral and count sums and
    32 for two footprint products or the ratio of count statistics. Allow
    16 for radial sums and three-node reduction dot products, and eight for
    linear interpolation fractions. The wavelengths are exactly representable,
    and local subtraction is exact for these nearby knots (Sterbenz's lemma).
    One division, one response difference and one multiply-add plus staging
    keep the response within eight epsilons at each leaf's peak.

    The grid fixture is a diagonal 25/32 map on a positive 5x5 ramp. The
    map has at most 12 roundings per axis (wavelength conversions, inversion
    and offset), and absolute position error is scaled in pixels. Nearest
    center samples are at least 0.1 pixel from a tie and agree exactly. Linear uses
    12 more roundings for four weights, products and sums, with total ramp
    slope 6/32 per pixel (conditioning below one). Its integral case allows
    twice that budget for the transpose accumulation and area ratio.
    Keys cubic uses up to 36 further roundings for two sets of piecewise
    cubic weights and contractions. Cubic splines use up to 36 for the
    compact cubic, four differences and five-term contractions. Allow a
    factor of two for signed weights and edge differences in both.
    Band-limited evaluation uses up to 44 further roundings for phase,
    exponentials and the two short complex contractions. The padded box
    integral adds up to eight for sinc and clipping. A factor of two covers
    phase and cancellation at the peak of the positive ramp.

    The on_grid density cases integrate over a footprint. Apart from the
    band-limited box, allow 12 for the map, four for endpoints and subtraction,
    18 for two five-term contractions, and two for area scaling. Beyond the
    nearest box clamp, the hat adds six arithmetic roundings for its endpoint
    polynomials, Keys adds 16, and the spline adds 27 for its quartic and
    nine-term prefilter. Double those counts for width differences at both
    ends of a 25/32-pixel footprint and signed weights on the positive ramp.
    Both lanes compile the grid comparisons.

    Coronagraph reads allow 64 to 80 for two DFT sums and analytic PSF
    evaluation, and 24 for the intensive sky lookup. Profile reads allow
    16 for a cubic, 32 with log-space interpolation, and 48 with the diameter
    spline. Lenslet reads allow up to 16 for log, Horner and differencing.
    Planck allows 32 for exponent sensitivity with x near six. Speckle
    intensities allow 48 for the complex two-mode dot and cross terms,
    or 64 with eight time samples or a three-node spectral mean.
    """
    if group == "spectral":
        fn, args, n_ops = _spectral_cases()[entry]
    else:
        build = {
            "reduction": _reduction_case,
            "grid": _grid_case,
            "detector": _detector_case,
            "coronagraph": _coronagraph_case,
            "optics": _optics_case,
        }[group]
        fn, args, n_ops = build(entry)
    compare = eqx.filter_jit(fn) if group == "grid" else fn
    bound = float(ulps(n_ops, np.float32(1)))
    draw = entry in {"draw", "ideal_draw"}
    if draw:
        precision_pair(compare, args, bound=bound)
        detector, key, rates, plan = args
        source_key = jax.random.split(key, 4)[0] if entry == "draw" else key
        expected = jax.random.poisson(source_key, jnp.array([12.0]))
        assert np.all(np.asarray(expected) > 0)
        np.testing.assert_array_equal(compare(detector, key, rates, plan), expected)
        monkeypatch.setattr(jax.random, "poisson", _deterministic_poisson)
    try:
        assert_precision_closed(fn, *canonical(args, precision="float32"))
    except TypeError as error:
        if (
            entry == "sampled_profile_dynamic"
            and re.search(
                r"true_fun is slope_zero at .*/interpax/_fd_derivs\.py:\d+\n",
                str(error),
            )
            and re.search(
                r"false_fun is slope_nonzero at .*/interpax/_fd_derivs\.py:\d+\n",
                str(error),
            )
            and str(error).count(
                "type float32[1,1] but the corresponding output of false_fun "
                "has type float64[1,1]"
            )
            == 2
        ):
            raise _UpstreamPrecisionFailure(str(error)) from error
        raise
    if not draw:
        precision_pair(compare, args, bound=bound)


@pytest.mark.xfail(
    jax.__version_info__ < (0, 12, 1),
    strict=True,
    raises=_UpstreamPrecisionFailure,
    reason="JAX exact Poisson traces float64 uniform bounds under x64.",
)
def test_exact_poisson_precision_closed():
    """The exact sampler alone must reproduce the upstream location and dtype."""
    try:
        assert_precision_closed(jax.random.poisson, jax.random.key(0), jnp.float32(12))
    except AssertionError as error:
        # The helper reports the sampler definition for this nested bound,
        # which has no individual source location in the JAX trace.
        source = unwrap(jax.random.poisson).__code__
        location = f"{source.co_filename}:{source.co_firstlineno}"
        expected = f"{location}: precision widened to float64"
        if str(error) == expected:
            raise _UpstreamPrecisionFailure(str(error)) from error
        raise


def test_noisy_draw_after_poisson(monkeypatch):
    """Check the nonzero gamma gain and Gaussian read term with separate keys."""
    monkeypatch.setattr(jax.random, "poisson", _deterministic_poisson)
    detector = ox.Detector(
        shape=(2, 2),
        pixel_scale_arcsec=0.01,
        dark_current_e_per_pixel_s=0.5,
        clock_induced_charge_e_per_pixel_frame=1.0,
        excess_noise_factor_sq=1.5,
        read_noise_e_rms=1.5,
    )
    rates = ox.Rates((ox.Band(480, 520),), "electron", planet=jnp.array([2.0]))
    plan = ox.FramePlan(jnp.array(2.0), 3)
    key = jax.random.key(0)
    fn = ox.AbstractDetector.draw
    narrow = canonical((detector, key, rates, plan), precision="float32")
    assert_precision_closed(fn, *narrow)
    _, _, gain_key, read_key = jax.random.split(key, 4)
    for dtype, args in [
        (np.float32, narrow),
        (np.float64, (detector, key, rates, plan)),
    ]:
        # Six seconds give 15 source electrons plus 3 CIC electrons.
        # The gain scale is half an electron and the read variance is 3*1.5**2.
        gain = 0.5 * jax.random.gamma(gain_key, jnp.array([36], dtype), dtype=dtype)
        read = dtype(1.5 * np.sqrt(3)) * jax.random.normal(read_key, (1,), dtype=dtype)
        assert np.all(np.asarray(gain) > 0) and np.all(np.asarray(read) != 0)
        np.testing.assert_allclose(
            eqx.filter_jit(fn)(*args),
            gain + read,
            rtol=0,
            atol=ulps(4, jnp.max(jnp.abs(gain) + jnp.abs(read))),
        )


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
def test_resample_differences_static_centers_before_narrowing(compile_forward):
    """A 3/4-pixel phase shift of a linear ramp gives exact dyadic samples."""
    source = ox.PixelGrid.regular((2, 3), 1, "arcsec", center=(2**24, 0))
    target = ox.PixelGrid.regular((2, 2), 1, "arcsec", center=(2**24 - 0.75, 0))
    image = ox.Image(
        jnp.asarray([[0, 4, 8]] * 2, dtype=jnp.float32),
        source,
        (),
        ox.Measure.intensive(),
        "response",
    )

    def forward(im):
        return resample(im, target, reconstruction="linear").data

    run = eqx.filter_jit(forward) if compile_forward else forward
    np.testing.assert_array_equal(run(image), [[3, 7], [3, 7]])
    assert_precision_closed(forward, image)


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("entry", ["transmission", "qe", "profile", "mean"])
def test_close_host_knots(compile_forward, entry):
    """Read edges and short intervals against interpolation in host coordinates."""
    knots = np.asarray([500.000014, 500.000138])
    query = np.asarray([500, 500.00006103515625, 500.000152587890625], np.float32)
    if entry == "transmission":
        fn, model = Transmission.__call__, Transmission(knots, [0.5, 1])
    elif entry == "qe":
        fn = ox.Detector.quantum_efficiency
        model = ox.Detector(
            shape=(1, 1), pixel_scale_arcsec=0.1, qe_curve=(knots, [0.5, 1])
        )
    elif entry == "profile":
        model = SampledProfile(
            [1, 1 + 2**-26, 1 + 2**-25],
            [0.25, 1, 0.5],
            quantity="flux",
            frame=Frame(500, 6, "circumscribed"),
            interpolation="linear",
        )
        # The next float32 query above the last knot is six tail intervals away.
        args = (jnp.asarray([1, 1 + 2**-23], np.float32),)

        def evaluate(radius):
            return model.evaluate(radius)

        run = jax.jit(evaluate) if compile_forward else evaluate
        actual = run(*args)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, [0.25, -2.5])
        assert_precision_closed(evaluate, *args)
        return
    else:
        knots = np.array([499, 500.000014, 500.000138, 500.000262, 900, 901, 902])
        response = np.array([0, 0, 1, 0, 0, 0.000124, 0])
        # The equal-area triangles have centroids at their middle knots.
        ref = np.float32((knots[2] + knots[-2]) / 2)
        model = canonical(Transmission(knots, response), precision="float32")

        def fn(table):
            return table.mean_nm

        run = eqx.filter_jit(fn) if compile_forward else fn
        # Widths, two endpoint moments and two short sums; moderate cancellation
        # in the centered first moment is absent because the origin is the first knot.
        np.testing.assert_allclose(run(model), ref, rtol=0, atol=ulps(8, ref))
        assert_precision_closed(fn, model)
        return
    model = canonical(model, precision="float32")
    args = model, jnp.asarray(query)
    run = eqx.filter_jit(fn) if compile_forward else fn
    expected = np.interp(query, knots, [0.5, 1], left=0, right=0).astype(np.float32)
    actual = run(*args)
    np.testing.assert_array_equal(np.asarray(actual)[[0, 2]], expected[[0, 2]])
    # Query minus rounded anchor is exact here. The correction, interval
    # division, response multiply/add and host casts fit within six epsilons.
    np.testing.assert_allclose(
        actual[1], expected[1], rtol=0, atol=ulps(6, expected[1])
    )
    assert_precision_closed(fn, *args)


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
def test_speckle_midpoints_include_field_precision(compile_forward):
    """Binary64 midpoint times survive a narrow wavelength or narrow field."""
    field = make_linear_speckle()
    start, duration = 2**24 + 0.125, 0.5
    wave = jnp.float32(600)

    def average(f, w, t):
        return f.realize_average(
            wavelength_nm=w, exposure_s=duration, start_elapsed_s=t
        ).data

    run = eqx.filter_jit(average) if compile_forward else average
    expected = (
        sum(
            field.realize(wavelength_nm=wave, elapsed_s=start + (i + 0.5) / 16).data
            for i in range(8)
        )
        / 8
    )
    # The same eight dyadic midpoints and a short sum.
    np.testing.assert_allclose(
        run(field, wave, start), expected, rtol=0, atol=ulps(8, expected.max())
    )
    narrow = canonical(field, precision="float32")
    explicit = jnp.asarray(start, dtype=np.float64)
    expected = (
        sum(
            narrow.realize(wavelength_nm=wave, elapsed_s=explicit + (i + 0.5) / 16).data
            for i in range(8)
        )
        / 8
    )
    np.testing.assert_allclose(
        run(narrow, wave, explicit),
        expected,
        rtol=0,
        atol=ulps(8, jnp.max(jnp.abs(expected))),
    )


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("aperture", [False, True], ids=["frames", "pixels"])
def test_detector_preserves_integer_counts_until_floating_use(
    compile_forward, aperture
):
    """Half an electron of dark current per count has an exact binary64 mean."""
    detector = ox.Detector(
        shape=(1, 1),
        pixel_scale_arcsec=0.01,
        dark_current_e_per_pixel_s=0.5,
        read_noise_e_rms=1,
    )
    band = ox.Band(500, 510)
    zero = jnp.zeros(1, np.float32)
    count = 2**24 + 1
    rates = ox.Rates(
        (band,), "electron", planet=zero, n_pix=np.int64(count) if aperture else None
    )
    if aperture:
        plan = ox.FramePlan(1.0, 1)
    else:
        plan = ox.FramePlan(jnp.float32(1), count)
        assert np.asarray(plan.n_frames).item() == count
    fn = ox.AbstractDetector.mean_and_variance
    run = eqx.filter_jit(fn) if compile_forward else fn
    mean, var = run(detector, rates, plan)
    np.testing.assert_array_equal(mean, [count / 2])
    np.testing.assert_array_equal(var, [count * 1.5])


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
def test_float64_wavelength_list(compile_forward):
    """A binary64 wavelength list returns two binary64 zeros."""
    model = make_instrument(synthetic_coronagraph(n=4))
    fn = ox.Instrument.thermal_photon_rate_per_sr
    run = eqx.filter_jit(fn) if compile_forward else fn
    assert run(model, [np.float64(500), np.float64(501)]).dtype == np.float64
    np.testing.assert_array_equal(
        run(model, [np.float64(500), np.float64(501)]), [0, 0]
    )


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
def test_float64_timing_sequences(compile_forward):
    """JIT can form a timing vector from traced scalars in lists and tuples."""

    def plan(dt, count):
        return ox.FramePlan([dt], (count,)).n_frames

    run = jax.jit(plan) if compile_forward else plan
    np.testing.assert_array_equal(run(jnp.float64(0.1), jnp.float64(3)), [3.0])


@pytest.mark.parametrize("compile_forward", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("sequence", [list, tuple])
def test_rate_array_like_components_set_precision(compile_forward, sequence):
    """The total includes both nonzero values, which underflow in float32."""
    band = ox.Band(500, 510)

    def forward(leakage):
        return ox.Rates(
            (band,),
            "electron",
            planet=sequence([1e-50]),
            star_leakage=leakage,
            thermal=1e-50,
        ).total()

    run = jax.jit(forward) if compile_forward else forward
    np.testing.assert_array_equal(run(jnp.zeros(1, np.float32)), [2e-50])


def test_constant_resolution_uses_polynomial_precision():
    """An identically zero derivative retains the explicitly wide coefficient dtype."""
    model = canonical(_lenslet(dispersion_coeffs=jnp.array([2.0])), precision="float32")
    model = eqx.tree_at(
        lambda d: d.dispersion_coeffs, model, jnp.array([2.0], dtype=np.float64)
    )
    result = eqx.filter_jit(ox.LensletDisperser.spectral_resolution)(
        model, jnp.float32(600)
    )
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, 0)
