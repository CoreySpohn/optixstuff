"""Absolute photometry and the shared curve record."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import eps
from hwoutils.testing import ulps
from scipy.interpolate import CubicSpline

from optixstuff import SampledProfile
from optixstuff.apertures import Circular, HalfMaximum
from optixstuff.grids import Frame, Image, Measure, PixelGrid
from optixstuff.performance import (
    DIAMETER_QUANTITIES,
    PerformanceCurves,
    performance_curves,
)


def test_circular_centre_offset_gaussian(displaced_gaussian_source):
    """Fails if source photometry searches for the displaced PSF peak.

    The offset-circle integral is a 1-D polar quadrature using I0; peak
    photometry is T*(1-exp(-R**2/(2*sigma**2))). The Gaussian's alias/edge
    errors are below 1e-18. A 1e-4 absolute bound covers the retained midpoint
    intensity quadrature at h=0.02 lod; the two estimands differ by >0.0005.
    Basis: mutation (swapping the centre fails), quadrature checked below.
    """
    from scipy.integrate import quad
    from scipy.special import i0

    total, sigma, radius = 0.31, 0.6, 0.7
    displacement = np.hypot(0.08, 0.06)
    offset, error = quad(
        lambda r: (
            total
            * r
            / sigma**2
            * np.exp(-(r * r + displacement**2) / (2 * sigma**2))
            * i0(r * displacement / sigma**2)
        ),
        0,
        radius,
        epsabs=1e-14,
        epsrel=1e-14,
    )
    assert error < 1e-14
    centred = total * (1 - np.exp(-(radius**2) / (2 * sigma**2)))
    assert centred - offset > 0.0005
    source, star = displaced_gaussian_source(0.2)
    for centre, expected in (("source", offset), ("peak", centred)):
        rule = Circular(radius, centre=centre)
        record = performance_curves(source, star, rule, separations_lod=[0.375])
        assert record.rule == rule
        np.testing.assert_allclose(record.throughput, expected, atol=1e-4, rtol=0)
    assert Circular(radius).centre == "source"
    assert len({Circular(radius), Circular(radius, centre="peak")}) == 2
    for invalid in ("centroid", "SOURCE", None, 1):
        with pytest.raises(ValueError, match="centre"):
            Circular(radius, centre=invalid)


def test_source_centre_exact_coverage_and_quadrature(displaced_gaussian_source):
    """Fails if pixelization or oversampling moves the source-centred disk.

    Basis: floating-point, 256 operations for exact native coverage and
    pi R**2, including segment integration and subpixel and pixel sums in
    the host measurement dtype.
    Equal physical subpixel pitch (0.02 lod) also preserves throughput to
    256 lane operations across two detector pitches for the sampled Gaussian
    intensity, Fourier reconstruction and weighted sums; alias error <1e-18.
    Fixed detector pitch with factors 5,10,20 retains a finite quadrature
    residual: require refinement toward the independent offset integral,
    rather than falsely calling midpoint intensity samples exact integrals.
    """
    from scipy.integrate import quad
    from scipy.special import i0

    from optixstuff.performance import _performance_curves

    radius, sigma, total = 0.7, 0.6, 0.31
    d2 = 0.08**2 + 0.06**2
    expected = quad(
        lambda r: (
            total
            * r
            / sigma**2
            * np.exp(-(r * r + d2) / (2 * sigma**2))
            * i0(r * np.sqrt(d2) / sigma**2)
        ),
        0,
        radius,
        epsabs=1e-14,
    )[0]
    fluxes = []
    rule = Circular(radius)
    for pitch, factor in ((0.2, 10), (0.1, 5)):
        source, star = displaced_gaussian_source(pitch)
        psf = source(0.375)
        native = np.asarray(rule.weights(psf, (0.375, 0.0), oversample=1))
        coverage_tol = 256 * np.finfo(native.dtype).eps
        for oversample in (factor, 20):
            weights = np.asarray(rule.weights(psf, (0.375, 0.0), oversample=oversample))
            np.testing.assert_allclose(weights, native, atol=coverage_tol, rtol=0)
            np.testing.assert_allclose(
                weights.sum() * pitch**2, np.pi * radius**2, rtol=coverage_tol, atol=0
            )
        record = _performance_curves(
            source, star, rule, separations_lod=[0.375], oversample=factor
        )
        fluxes.append(record.throughput[0])
    np.testing.assert_allclose(fluxes[0], fluxes[1], rtol=float(ulps(256, 1.0)), atol=0)
    source, star = displaced_gaussian_source(0.2)
    errors = [
        abs(
            _performance_curves(
                source, star, rule, separations_lod=[0.375], oversample=f
            ).throughput[0]
            - expected
        )
        for f in (5, 10, 20)
    ]
    assert errors[2] < errors[1] < errors[0] < 1e-4


def test_record_interpolation_spaces():
    """Fails if contrast is interpolated linearly or throughput logarithmically.

    The two-point midpoint is exactly an arithmetic or geometric mean.
    Allow 48 operations for the log-space polynomial, powers of ten and
    their analytic tangents at the host or device read dtype. Compare the
    tiny radial tangent of the log floor to its correctly rounded lane value.
    """
    rtol = float(ulps(48, 1.0))
    record = PerformanceCurves(
        separation_lod=np.array([0.0, 2.0]),
        frame=Frame(500, 6, "circumscribed"),
        rule=Circular(0.7),
        throughput=np.array([0.1, 0.9]),
        throughput_variance=np.array([0.05, 0.45]),
        core_area_variance_lod2=np.array([0.5, 1.0]),
        leakage_fraction_variance=np.array([1e-10, 1e-6]),
        leakage_density_variance_lod2=np.array([1e-10, 1e-6]),
        core_area_lod2=np.ones(2),
        leakage_fraction=np.array([1e-8, 1e-4]),
        leakage_density_lod2=np.array([1e-8, 1e-4]),
        raw_contrast=np.array([1e-8, 1e-4]),
        iwa_lod=1.0,
        data_extent_lod=4.0,
    )
    midpoints = {
        "leakage_fraction": 1e-6,
        "leakage_density_lod2": 1e-6,
        "raw_contrast": 1e-6,
        "throughput": 0.5,
        "leakage_fraction_variance": 1e-8,
        "leakage_density_variance_lod2": 1e-8,
        "throughput_variance": 0.25,
        "core_area_variance_lod2": 0.75,
    }
    for name, midpoint in midpoints.items():
        for radius in (np.float64(1.0), jnp.asarray(1.0)):
            value = record.evaluate(name, radius)
            assert value.dtype == radius.dtype
            np.testing.assert_allclose(
                value, midpoint, rtol=48 * eps(value.dtype), atol=0
            )
    slope = np.log(1e4) / 2
    for point in (0.0, 1.0, 2.0):
        derivative = jax.grad(lambda q: record.evaluate("raw_contrast", q))(
            jnp.asarray(point)
        )
        np.testing.assert_allclose(
            derivative, slope * 1e-8 * np.exp(slope * point), rtol=rtol, atol=0
        )
    assert np.isposinf(record.evaluate("raw_contrast", jnp.asarray(10000.0)))

    # A traced amplitude changes both knots and tails. These two-node
    # profiles have analytic derivatives even at their endpoints.
    axis = np.array([0.0, 2.0])
    q = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    for space, knots in (("log", [0.03, 0.12]), ("linear", [0.0, 0.12])):

        def evaluate(amplitude, radius, space=space, knots=knots):
            return SampledProfile(
                axis,
                amplitude * jnp.asarray(knots),
                quantity="flux_fraction",
                frame=record.frame,
                interpolation=space,
            ).evaluate(radius)

        predicted = knots[0] * 2.0**q if space == "log" else 0.06 * q
        slope = predicted * np.log(2) if space == "log" else np.full(5, 0.06)
        values = jax.jit(evaluate)(1.0, q)
        np.testing.assert_array_equal(
            np.asarray(values)[[1, 3]], np.asarray(knots, dtype=values.dtype)
        )
        np.testing.assert_allclose(values, predicted, rtol=rtol, atol=0)
        d_amp, d_radius = jax.vmap(jax.grad(evaluate, (0, 1)), (None, 0))(1.0, q)
        np.testing.assert_allclose(d_amp, predicted, rtol=rtol, atol=0)
        np.testing.assert_allclose(d_radius, slope, rtol=rtol, atol=0)

    # Even a zero ordinate is returned verbatim at its knot. Its sample
    # derivative is one, while the radial derivative uses the log fit.
    def log_profile(samples, radius):
        return SampledProfile(
            axis,
            samples,
            quantity="flux_fraction",
            frame=record.frame,
            interpolation="log",
        ).evaluate(radius)

    samples = jnp.array([0.0, 1.0])
    for jacobian in (jax.jacfwd, jax.jacrev):
        d_samples, d_radius = jacobian(log_profile, (0, 1))(samples, 0.0)
        np.testing.assert_array_equal(d_samples, [1.0, 0.0])
        tiny_slope = np.dtype(d_radius.dtype).type(1e-300 * 150 * np.log(10))
        np.testing.assert_allclose(d_radius, tiny_slope, rtol=rtol, atol=0)


def _short_record(rule, last, extent):
    """A two-sample record ending at ``last`` in an array of half-width ``extent``."""
    x = np.array([1.0, last])
    ones = np.ones(2)
    return PerformanceCurves(
        x,
        Frame(500, 6, "circumscribed"),
        rule,
        0.5 * ones,
        ones,
        1e-9 * ones,
        1e-9 * ones,
        1e-9 * ones,
        1.0,
        extent,
        throughput_variance=0.25 * ones,
        core_area_variance_lod2=ones,
        leakage_fraction_variance=1e-18 * ones,
        leakage_density_variance_lod2=1e-18 * ones,
    )


@pytest.mark.parametrize(
    ("separation", "expected"),
    [
        (0.5, False),  # inside the first sample
        (1.0, True),  # first sample
        (3.0, True),  # between samples
        (4.0, True),  # last sample, aperture 0.7 fits in 6
        (4.0 + 1e-6, False),  # just past the last sample, inside the extent
        (5.0, False),  # between the last sample and the extent
        (5.4, False),  # aperture fits, samples do not reach
        (7.0, False),  # past the extent
    ],
)
def test_record_supported_ends_at_the_last_sample(separation, expected):
    """Fails if a read past the samples but inside the array extent counts as data.

    Basis: exact comparisons against the first sample, the last sample and
    ``data_extent_lod - radius``; no numerical tolerance is involved.
    """
    record = _short_record(Circular(0.7), last=4.0, extent=6.0)
    assert bool(record.supported(separation)) is expected
    assert bool(record.supported(jnp.asarray(separation))) is expected


def test_record_supported_also_needs_the_aperture_inside_the_extent():
    """Fails if the extent test is dropped when the samples run to the edge.

    The samples end at 4.0, the extent is 4.5 and the radius is 0.7.
    The aperture fits up to 3.8 at float32 coordinate resolution in either
    lane. One float32 spacing past that boundary is outside the extent.
    """
    record = _short_record(Circular(0.7), last=4.0, extent=4.5)
    beyond = np.nextafter(np.float32(3.8), np.float32(np.inf))
    q = np.array([3.8, np.nextafter(3.8, np.inf), beyond, 4.0])
    compiled = jax.jit(lambda s: record.supported(s))(jnp.asarray(q))
    for flags in (record.supported(q), compiled):
        np.testing.assert_array_equal(flags, [True, True, False, False])


def test_profile_supported_spans_the_samples():
    """Fails if a profile reports its end-segment continuation as data.

    Basis: exact comparisons at the end samples; ``evaluate`` itself still
    continues the segment linearly (rtol 1e-14, float64 arithmetic).
    """
    profile = SampledProfile(
        np.array([1.0, 3.0]),
        np.array([2.0, 4.0]),
        quantity="uniform_sky_response",
        frame=Frame(500, 6, "circumscribed"),
        interpolation="linear",
    )
    np.testing.assert_array_equal(
        profile.supported(np.array([0.9, 1.0, 3.0, 3.1])), [False, True, True, False]
    )
    np.testing.assert_allclose(profile.evaluate(4.0), 5.0, rtol=1e-14)


def test_exact_binary_moments():
    """Fails if binary variance differs from integrated full-field flux.

    The three-point row interpolates to a sinusoid plus a constant. A source-centred
    circle enclosing this integer-centred grid covers every pixel completely.
    """
    grid = PixelGrid.focal(3, 0.5, Frame(500, 6, "circumscribed"))
    planet = np.zeros(grid.shape, dtype=np.float32)
    planet[1, 1:] = [0.5, 0.25]
    stellar = np.zeros(grid.shape)
    stellar[1, 1:] = [0.125, 0.25]

    def measure(data, star, rule):
        psf = Image(data, star.grid, (), Measure.integral(), "stellar_flux_fraction")
        return performance_curves(lambda s: psf, star, rule, separations_lod=[0.0])

    binary_grid = PixelGrid.focal(3, 0.3, grid.frame)
    binary_star = Image(
        stellar, binary_grid, (), Measure.integral(), "stellar_flux_fraction"
    )
    # The full-field DFT integral preserves flux to summation roundoff.
    binary = measure(planet, binary_star, Circular(2.0))
    for mean, variance, expected in (
        ("throughput", "throughput_variance", 0.75),
        ("core_area_lod2", "core_area_variance_lod2", 0.81),
        ("leakage_fraction", "leakage_fraction_variance", 0.375),
        ("leakage_density_lod2", "leakage_density_variance_lod2", 0.375 / 0.81),
    ):
        np.testing.assert_allclose(
            np.ravel(getattr(binary, mean)),
            [expected],
            rtol=32 * np.finfo(float).eps,
            atol=0,
        )
        np.testing.assert_array_equal(getattr(binary, variance), getattr(binary, mean))


def _gaussian_source():
    """Small analytic source; the nearest border is over eight sigma away."""
    grid = PixelGrid.focal(49, 0.25, Frame(500, 6, "circumscribed"))
    x, y = grid.coordinates()

    def source(separation):
        p = (
            0.31
            * np.exp(-((x - separation) ** 2 + y * y) / (2 * 0.4**2))
            / (2 * np.pi * 0.4**2)
            * 0.25**2
        )
        return Image(p, grid, (), Measure.integral(), "stellar_flux_fraction")

    return source, x, y, grid


@pytest.mark.medium(reason="measures and independently rereads both aperture rules")
def test_geometric_rules_preserve_primitive_curves():
    """Fails if the record moves a core or renormalizes its planet flux.

    Compare the host record with retained low-level measurements. Allow
    128 lane operations for the coordinate map, Fourier sampling or cubic
    weights, and pairwise sums of the subpixels. The ring mean uses the
    parent's spatial reader divided by pixel area.
    """
    from optixstuff.performance import (
        _rule_samples,
        azimuthal_mean_profile,
        circular_aperture,
        truncation_core,
    )

    source, x, y, grid = _gaussian_source()
    star = Image(
        1e-8 * (2 + 0.03 * (x * x + y * y) + 0.2 * x) * 0.25**2,
        grid,
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    seps = np.array([1.0, 2.0])
    for rule in (Circular(0.7, centre="peak"), HalfMaximum(0.5)):
        record = performance_curves(
            source,
            star,
            rule,
            separations_lod=seps,
            leakage_rule="azimuthal_ring_mean",
        )
        for i, sep in enumerate(seps):
            nominal = (24.0, 24.0 + sep / 0.25)
            if isinstance(rule, Circular):
                tau, area, _ = circular_aperture(
                    source(sep).data,
                    0.7 / 0.25,
                    0.25,
                    oversample=10,
                    half_width_pix=8,
                    nominal_center=nominal,
                )
            else:
                tau, area = truncation_core(
                    source(sep).data,
                    0.5,
                    0.25,
                    oversample=10,
                    half_width_pix=12,
                    nominal_center=nominal,
                )
            rtol = float(ulps(128, 1.0))
            np.testing.assert_allclose(record.throughput[i], tau, rtol=rtol, atol=0)
            np.testing.assert_allclose(
                record.core_area_lod2[i], area, rtol=rtol, atol=0
            )
            # This primitive is the host measurement path (binary64).
            psf = source(sep)
            with jax.enable_x64():
                fine, coverage, _, _, factor = _rule_samples(rule, psf, (sep, 0.0))
                assert record.throughput[i] == float(
                    jnp.sum(fine * coverage) / factor**2
                )
                assert record.core_area_lod2[i] == (
                    float(jnp.sum(coverage)) * grid.pixel_area / factor**2
                )
        for mean, variance in (
            ("throughput", "throughput_variance"),
            ("core_area_lod2", "core_area_variance_lod2"),
            ("leakage_fraction", "leakage_fraction_variance"),
            ("leakage_density_lod2", "leakage_density_variance_lod2"),
        ):
            np.testing.assert_array_equal(
                getattr(record, mean), getattr(record, variance)
            )
        parent_mean = azimuthal_mean_profile(star.data, (24.0, 24.0), seps / 0.25)
        np.testing.assert_allclose(
            record.leakage_density_lod2[0],
            parent_mean / 0.25**2,
            rtol=float(ulps(128, 1.0)),
            atol=0,
        )


def test_absolute_leakage_sampling():
    """Fails if leakage is density, contrast, or normalized by PSF throughput.

    A star with one eighth the local planet intensity gives one eighth its
    throughput, including partially covered boundary cells.
    """
    from optixstuff.performance import _performance_curves

    source, _, _, grid = _gaussian_source()
    rule, sep = Circular(0.7), 1.0
    star = Image(
        source(sep).data / 8, grid, (), Measure.integral(), "stellar_flux_fraction"
    )
    for factor in (5, 10):
        record = _performance_curves(
            source, star, rule, separations_lod=[sep], oversample=factor
        )
        np.testing.assert_allclose(
            record.leakage_fraction[0], record.throughput / 8, rtol=2e-14
        )
    small_grid = PixelGrid.focal(5, 0.25, grid.frame)
    small_star = Image(
        np.full(small_grid.shape, 3e-9),
        small_grid,
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    outside = performance_curves(source, small_star, rule, separations_lod=[2.0])
    # The disk's nearest edge is 1.3 lod, beyond the 0.625 lod stellar edge.
    # Zero extension must not wrap the opposite side of the stellar image.
    assert outside.leakage_fraction[0, 0] == 0.0


def test_leakage_mean_conventions_and_disconnected_core():
    """Fails if a ring mean replaces a local mean or islands join the peak core.

    Constant density is symmetric and agrees at the 2e-12 DFT floor. An
    azimuthal cosine modulation has exactly zero ring mean but nonzero local
    mean. The 1 percent gap is far below its 10 percent imposed modulation.
    """
    source, x, _, grid = _gaussian_source()
    for asymmetric in (False, True):
        density = (
            1e-8 * (1 + 0.1 * np.sin(x)) if asymmetric else np.full(grid.shape, 1e-8)
        )
        star = Image(
            density * 0.25**2, grid, (), Measure.integral(), "stellar_flux_fraction"
        )
        local = performance_curves(
            source, star, Circular(0.7), separations_lod=[1.0, 2.0]
        )
        ring = performance_curves(
            source,
            star,
            Circular(0.7),
            separations_lod=[1.0, 2.0],
            leakage_rule="azimuthal_ring_mean",
        )
        assert local.leakage_rule == "aperture_mean"
        assert ring.leakage_rule == "azimuthal_ring_mean"
        np.testing.assert_allclose(
            local.leakage_fraction, ring.leakage_fraction, rtol=2e-12
        )
        if asymmetric:
            assert np.all(
                abs(local.leakage_density_lod2 / ring.leakage_density_lod2 - 1) > 0.01
            )
        else:
            np.testing.assert_allclose(
                local.leakage_density_lod2, ring.leakage_density_lod2, rtol=2e-12
            )
    # A brighter, disconnected second core inside the local window
    # must not become part of the nominal source's aperture.
    p = source(1.0).data + source(3.0).data * 1.1
    image = Image(p, grid, (), Measure.integral(), "stellar_flux_fraction")
    weights = np.asarray(HalfMaximum().weights(image, (1.0, 0.0)))
    assert weights[24, 28] > 0.99
    assert weights[24, 36] == 0.0


MOD_AMPLITUDE = 0.8
MOD_SCALE_LOD = 4.0
MOD_LEVEL = 1e-9


def _i0(r_lod):
    return MOD_LEVEL * np.exp(-r_lod / MOD_SCALE_LOD)


def _modulated_star():
    """Stellar map I0(r) (1 + m cos 2 phi) on the _gaussian_source grid."""
    source, x, y, grid = _gaussian_source()
    r2 = np.maximum(x**2 + y**2, 1e-30)
    density = _i0(np.hypot(x, y)) * (1 + MOD_AMPLITUDE * (x**2 - y**2) / r2)
    star = Image(
        density * grid.pixel_area,
        grid,
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    return source, star, grid


def _disk_integral(f, center_x, radius=0.7, n_r=48, n_phi=96):
    """Integral of f(x, y) over a disk by midpoint polar quadrature."""
    r = (np.arange(n_r) + 0.5) / n_r * radius
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    area = rr * (radius / n_r) * (2 * np.pi / n_phi)
    return float(np.sum(f(center_x + rr * np.cos(pp), rr * np.sin(pp)) * area))


@pytest.mark.parametrize("separation", (2.0, 3.0))
def test_ring_rule_on_a_modulated_map(separation):
    """aperture_mean is the +x integral; aperture_ring_mean is I0 times the disk.

    Fails if the ring rule keeps any of the cos 2 phi modulation (the +x reading
    would pass it), averages a point instead of the aperture, or reads the
    band-limited samples of the symmetrized map, which ripple on its ring edges.
    """
    source, star, _ = _modulated_star()
    rule = Circular(0.7)
    kwargs = {"separations_lod": [separation]}
    along_x = performance_curves(source, star, rule, **kwargs)
    ring = performance_curves(
        source, star, rule, leakage_rule="aperture_ring_mean", **kwargs
    )
    x_axis = _disk_integral(
        lambda x, y: (
            _i0(np.hypot(x, y)) * (1 + MOD_AMPLITUDE * (x**2 - y**2) / (x**2 + y**2))
        ),
        separation,
    )
    ring_ref = _disk_integral(lambda x, y: _i0(np.hypot(x, y)), separation)
    # The disk integrals are densities (per lod^2); fractions are per pixel.
    np.testing.assert_allclose(along_x.leakage_fraction[0], x_axis, rtol=1e-3)
    np.testing.assert_allclose(ring.leakage_fraction[0], ring_ref, rtol=1e-3)
    assert abs(along_x.leakage_fraction[0] / ring.leakage_fraction[0] - 1) > 0.05
    assert ring.leakage_rule == "aperture_ring_mean"
    np.testing.assert_allclose(
        ring.leakage_density_lod2, ring.leakage_fraction / ring.core_area_lod2
    )
    np.testing.assert_allclose(ring.throughput, along_x.throughput)


def test_ring_symmetrization_converges_in_azimuth_count():
    """A cos 2 phi map is averaged exactly by any rung above four angles.

    Fails if the ring mean depends on the angle count once it exceeds the map's
    highest harmonic.
    """
    from optixstuff.performance import ring_azimuth_samples, ring_symmetrized

    _, star, grid = _modulated_star()
    default = np.asarray(ring_symmetrized(star).data)
    center = (grid.center_y, grid.center_x)
    assert ring_azimuth_samples(grid.shape, center) > 64
    for n_theta in (8, 16, 32, 64):
        rung = np.asarray(ring_symmetrized(star, n_theta=n_theta).data)
        np.testing.assert_allclose(rung, default, rtol=1e-3, atol=1e-3 * default.max())


def test_azimuthal_ring_mean_is_deprecated_but_works():
    """Fails if the old rule stops warning or stops producing a record."""
    source, star, _ = _modulated_star()
    with pytest.warns(DeprecationWarning, match="azimuthal_ring_mean"):
        record = performance_curves(
            source,
            star,
            Circular(0.7),
            separations_lod=[2.0],
            leakage_rule="azimuthal_ring_mean",
        )
    assert record.leakage_rule == "azimuthal_ring_mean"
    with pytest.raises(ValueError, match="unknown leakage_rule"):
        performance_curves(
            source, star, Circular(0.7), separations_lod=[2.0], leakage_rule="x"
        )


# A star's leakage depends on its diameter: the record tabulates it over a
# diameter axis and reads between the tabulated diameters.
DIAMETERS = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0])


def _diameter_record(diameters=DIAMETERS, **kwargs):
    """Uniform stellar maps whose level grows with the diameter, on a test source.

    Plane ``k`` holds ``1e-8 (1 + 40 d_k**2)`` in every pixel, so the leakage
    of an aperture of uniform coverage is that level times the aperture's
    pixel count.
    """
    source, _, _, grid = _gaussian_source()
    levels = 1e-8 * (1 + 40 * np.asarray(diameters) ** 2)
    stars = Image(
        levels[:, None, None] * np.ones(grid.shape),
        grid,
        ("diameter",),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    return levels, performance_curves(
        source,
        stars,
        Circular(0.7),
        separations_lod=[0.5, 1.0, 1.5, 2.0],
        stellar_diameters_lod=diameters,
        **kwargs,
    )


def test_record_uniform_sky_uses_the_record_frame():
    """Exact radial-map samples catch a lost or misframed profile."""
    source, _, _, grid = _gaussian_source()
    stellar = Image(
        np.full(grid.shape, 1e-8), grid, (), Measure.integral(), "stellar_flux_fraction"
    )
    sky = Image(
        np.linspace(0.0, 1.0, np.prod(grid.shape)).reshape(grid.shape),
        grid,
        (),
        Measure.intensive(),
        "uniform_sky_response",
    )
    profile = SampledProfile.from_uniform_sky(sky)
    record = performance_curves(
        source, stellar, Circular(0.7), separations_lod=[1.0], uniform_sky=sky
    )
    assert record.uniform_sky.frame == grid.frame
    np.testing.assert_array_equal(
        record.uniform_sky.separation_lod, profile.separation_lod
    )
    np.testing.assert_array_equal(record.uniform_sky.values, profile.values)
    assert replace(record, uniform_sky=profile).uniform_sky is profile
    with pytest.raises(ValueError, match="record's lambda/D frame"):
        replace(
            record, uniform_sky=replace(profile, frame=Frame(400, 6, "circumscribed"))
        )
    with pytest.raises(TypeError, match=r"uniform_sky.*SampledProfile"):
        replace(record, uniform_sky=sky)


def test_diameter_knots_read_the_measured_rows_exactly():
    """Single-row reads give the exact-algebra oracle and catch log-floor leakage.

    At the zero-diameter knot the selected row is flat, so its separation
    slope is exactly zero. The spline still supplies the diameter tangent.
    The independent float64 solution on the first interval is analytic.
    Allow 48 operations for both row log fits, the diameter weights, exp,
    and the tangent. At a positive knot the analytic radial slope is nonzero.
    Between mixed-zero rows, the cubic weights and three-term dot product
    allow 12 rounding operations at the largest input log magnitude. The
    float64 reference uses the read dtype's positive floor.
    """
    diameters = np.array([0.0, 0.2, 0.5])
    q = np.array([1.0, 1.25, 2.0])
    base = _short_record(Circular(0.7), last=2.0, extent=4.0)
    rows = np.array([[0.0, 0.0], [1e-8, 2e-8], [0.0, 4e-8]])
    record = replace(
        base,
        stellar_diameters_lod=diameters,
        **dict.fromkeys(DIAMETER_QUANTITIES, rows),
    )
    singles = [replace(base, **dict.fromkeys(DIAMETER_QUANTITIES, row)) for row in rows]

    # Every leakage quantity shares this read; two of them suffice.
    names = ("leakage_fraction", "raw_contrast")

    def read(radius, diameter):
        return tuple(record.evaluate(name, radius, diameter) for name in names)

    def read_rows(radius):
        return tuple(
            tuple(single.evaluate(name, radius) for single in singles) for name in names
        )

    for evaluate, expected, axis, separations in (
        (read, read_rows, diameters, q),
        (jax.jit(read), jax.jit(read_rows), jnp.asarray(diameters), jnp.asarray(q)),
    ):
        actual = evaluate(separations + np.zeros((3, 1)), axis[:, None])
        np.testing.assert_array_equal(actual, expected(separations))
        np.testing.assert_array_equal(np.asarray(actual)[:, 0, [0, 2]], 0.0)
    assert np.all(np.isfinite(jax.grad(lambda d: read(1.25, d)[0])(0.0)))

    # The last row has a measured zero at radius 1. A diameter just below
    # 0.5 rounds onto that row; a resolved offset still reads the spline.
    for evaluate, axis in (
        (read, np.asarray),
        (read, jnp.asarray),
        (jax.jit(read), jnp.asarray),
    ):
        near, between = evaluate(axis(1.0), axis([0.5 - 1e-8, 0.4999998]))[0]
        assert near == 0.0
        assert between > 0.0

    samples = np.asarray(read_rows(q)[0])
    for evaluate, separations in ((read, q), (jax.jit(read), jnp.asarray(q))):
        between = np.asarray(evaluate(separations, 0.1)[0])
        logs = np.log(np.maximum(samples, np.finfo(between.dtype).tiny))
        expected_log = CubicSpline(diameters, logs)(0.1)
        assert np.all(np.isfinite(between) & (between > 0))
        np.testing.assert_allclose(
            np.log(between.astype(np.float64)),
            expected_log,
            atol=12 * eps(between.dtype) * np.max(np.abs(logs)),
            rtol=0,
        )
    # At zero diameter the first row has no separation slope. The second
    # row varies with separation and contributes only to the diameter tangent.
    positive = replace(
        base,
        stellar_diameters_lod=np.array([0.0, 0.001]),
        **dict.fromkeys(DIAMETER_QUANTITIES, np.array([[1e-8, 1e-8], [1e-8, 1e-6]])),
    )

    def leakage(s, d):
        return positive.evaluate("leakage_fraction", s, d)

    radial, diameter = jax.grad(leakage, (0, 1))(1.5, 0.0)
    assert radial == 0.0
    log_slope = np.log(10.0) / 0.001
    tolerance = float(ulps(48, 1.0))
    np.testing.assert_allclose(diameter, 1e-8 * log_slope, rtol=tolerance, atol=0)
    radial_at_positive = jax.grad(leakage, 0)
    for gradient in (radial_at_positive, jax.jit(radial_at_positive)):
        np.testing.assert_allclose(
            gradient(1.5, 0.001), 1e-7 * np.log(100.0), rtol=tolerance, atol=0
        )
    d = 1e-10
    expected = 1e-8 * np.exp(d * log_slope)
    np.testing.assert_allclose(
        jax.jit(leakage)(1.5, d), expected, rtol=tolerance, atol=0
    )


def test_record_tabulates_the_leakage_at_each_diameter():
    """Fails if a leakage profile is read from the wrong diameter plane.

    Each row of the record is the aperture sum of its own plane; the aperture
    is covered entirely by the uniform map, so the sum is the level times the
    aperture's pixel count. Allow 256 operations for native map conversion,
    Fourier interpolation and the weighted subpixel sum. Each row
    is returned exactly at the measured separation and diameter knots.
    """
    levels, record = _diameter_record()
    assert record.leakage_fraction.shape == (len(DIAMETERS), 4)
    np.testing.assert_array_equal(record.stellar_diameters_lod, DIAMETERS)
    pixels = record.core_area_lod2 / 0.25**2
    np.testing.assert_allclose(
        record.leakage_fraction,
        levels[:, None] * pixels,
        rtol=float(ulps(256, 1.0)),
        atol=0,
    )
    for k, diameter in enumerate(DIAMETERS):
        np.testing.assert_array_equal(
            record.evaluate("leakage_fraction", record.separation_lod, diameter),
            record.leakage_fraction[k],
        )


def test_record_reads_between_diameters_on_the_measured_axis():
    """Fails if diameter interpolation shifts a knot or uses the wrong spline.

    The reference applies SciPy's float64 log spline directly to the
    tabulated rows. Allow 128 operations for the diameter weights, log and
    exp at the NumPy or JAX read dtype.
    A linear rule in place of the spline differs by more than 1e-3 at 0.2,
    which the check requires, so a switch to it fails.
    """
    _, record = _diameter_record()
    spline = CubicSpline(DIAMETERS, np.log(record.leakage_density_lod2))

    def leakage(separation, diameter):
        return record.evaluate("leakage_density_lod2", separation, diameter)

    diameters = np.array([0.002, 0.02, 0.05, 0.2, 0.7])
    for evaluate, axis in ((leakage, np.asarray), (jax.jit(leakage), jnp.asarray)):
        queries = axis(diameters)
        read = evaluate(axis(record.separation_lod), queries[:, None])
        assert read.dtype == queries.dtype
        expected = np.exp(spline(np.asarray(queries, dtype=np.float64)))
        np.testing.assert_allclose(read, expected, rtol=128 * eps(read.dtype), atol=0)
    linear = np.array(
        [np.interp(0.2, DIAMETERS, record.leakage_density_lod2[:, s]) for s in range(4)]
    )
    read = record.evaluate("leakage_density_lod2", record.separation_lod, 0.2)
    assert np.all(abs(linear / read - 1) > 1e-3)


def test_record_reads_every_leakage_quantity_at_a_diameter():
    """Fails if a diameter-dependent quantity is left at the first diameter."""
    _, record = _diameter_record()
    for name in (
        "leakage_fraction",
        "leakage_fraction_variance",
        "leakage_density_lod2",
        "leakage_density_variance_lod2",
        "raw_contrast",
    ):
        wide = record.evaluate(name, 1.0, 0.7)
        point = record.evaluate(name, 1.0, 0.0)
        assert wide > 1.5 * point, name
    for name in ("throughput", "core_area_lod2"):
        with pytest.raises(ValueError, match="does not depend"):
            record.evaluate(name, 1.0, 0.1)


def test_record_over_diameter_refuses_a_silent_read():
    """Fails if a read without, or outside, the tabulated diameters returns a value."""
    _, record = _diameter_record([0.0, 1.0])
    with pytest.raises(ValueError, match="needs stellar_diameter_lod"):
        record.evaluate("leakage_density_lod2", 1.0)
    for outside in (-0.01, 1.01):
        with pytest.raises(ValueError, match="outside the tabulated range"):
            record.evaluate("leakage_density_lod2", 1.0, outside)

    def read(diameter):
        return record.evaluate("leakage_density_lod2", 1.0, diameter)

    # The query rounds to the last admitted float32 coordinate. The next
    # float32 coordinate is beyond the table's relative slack.
    edge = 1.00000101
    outside = float(np.nextafter(np.float32(edge), np.float32(np.inf)))
    for evaluate, asarray, error in (
        (read, np.asarray, ValueError),
        (read, jnp.asarray, ValueError),
        (eqx.filter_jit(read), jnp.asarray, eqx.EquinoxRuntimeError),
    ):
        assert np.isfinite(evaluate(asarray(edge)))
        with pytest.raises(error, match="outside the tabulated range"):
            np.asarray(evaluate(asarray(outside)))


def test_record_of_one_profile_serves_only_its_own_star():
    """Fails if a profile of one diameter is read for a star of another.

    The record serves a star within 1e-3 of its diameter (relative; the
    excess leakage grows as the square of the diameter, so 2e-3 of the excess).
    """
    source, _, _, grid = _gaussian_source()
    star = Image(
        np.full(grid.shape, 1e-8),
        grid,
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    point = performance_curves(
        source, star, Circular(0.7), separations_lod=[1.0], stellar_diameter_lod=0.0
    )
    sized = performance_curves(
        source, star, Circular(0.7), separations_lod=[1.0], stellar_diameter_lod=0.3
    )
    unknown = performance_curves(source, star, Circular(0.7), separations_lod=[1.0])
    assert point.leakage_fraction.shape == (1, 1)
    value = sized.evaluate("leakage_fraction", 1.0, 0.3 * (1 + 5e-4))
    np.testing.assert_array_equal(value, sized.leakage_fraction[0])
    for record, wrong in ((point, 0.02), (sized, 0.0), (sized, 0.31)):
        with pytest.raises(ValueError, match="one profile"):
            record.evaluate("leakage_fraction", 1.0, wrong)
    # No diameter given, or a profile of unknown diameter: the caller vouches.
    np.testing.assert_array_equal(
        sized.evaluate("leakage_fraction", 1.0), sized.leakage_fraction[0]
    )
    np.testing.assert_array_equal(
        unknown.evaluate("leakage_fraction", 1.0, 0.5), unknown.leakage_fraction[0]
    )
    assert unknown.stellar_diameters_lod is None

    def read(diameter):
        return sized.evaluate("leakage_fraction", 1.0, diameter)

    edge = 0.3003010001
    outside = float(np.nextafter(np.float32(edge), np.float32(np.inf)))
    for evaluate, asarray, error in (
        (read, np.asarray, ValueError),
        (read, jnp.asarray, ValueError),
        (eqx.filter_jit(read), jnp.asarray, eqx.EquinoxRuntimeError),
    ):
        assert np.isfinite(evaluate(asarray(edge)))
        with pytest.raises(error, match="one profile"):
            np.asarray(evaluate(asarray(outside)))


@pytest.mark.float64_only(reason="finite-difference gradient reference")
def test_record_gradient_in_the_stellar_diameter():
    """Fails if the diameter read is not differentiable, or its slope is wrong.

    Central differences at h = 1e-6 of a float64 read bound the slope to 1e-6
    relative; the slope is nonzero (the leakage grows with the diameter).
    """
    _, record = _diameter_record()

    def read(d):
        return record.evaluate("leakage_density_lod2", 1.0, d)

    for d in (0.004, 0.07, 0.5):
        slope = jax.grad(read)(d)
        h = 1e-6
        numeric = (read(d + h) - read(d - h)) / (2 * h)
        assert np.isfinite(slope) and slope > 0
        np.testing.assert_allclose(slope, numeric, rtol=1e-6)
    batch = jax.vmap(read)(jnp.array([0.0, 0.05, 0.9]))
    assert batch.shape == (3,) and bool(jnp.all(jnp.diff(batch) > 0))


def test_record_diameters_are_in_the_planet_frame():
    """Fails if the diameter axis keeps the stellar frame's lambda/D.

    The stellar map's frame has half the planet frame's aperture (3 m against
    6 m), so its lambda/D is twice as large in arcsec and a diameter of one
    tabulated unit there spans two of the planet frame's units, to float
    rounding.
    """
    source, *_ = _gaussian_source()
    stellar_grid = PixelGrid.focal(49, 0.25, Frame(500, 3, "circumscribed"))
    star = Image(
        np.ones((2, *stellar_grid.shape)) * 1e-8,
        stellar_grid,
        ("diameter",),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    record = performance_curves(
        source,
        star,
        Circular(0.7),
        separations_lod=[1.0],
        stellar_diameters_lod=[0.0, 1.0],
    )
    np.testing.assert_allclose(record.stellar_diameters_lod, [0.0, 2.0], rtol=1e-14)


def test_record_diameter_axis_validation():
    """Fails if a malformed diameter axis or map stack is accepted."""
    source, _, _, grid = _gaussian_source()
    plane = np.full(grid.shape, 1e-8)
    two = Image(
        np.stack([plane, plane]),
        grid,
        ("diameter",),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    one = Image(plane, grid, (), Measure.integral(), "stellar_flux_fraction")
    base = dict(separations_lod=[1.0])
    with pytest.raises(ValueError, match="one map per entry"):
        performance_curves(
            source, two, Circular(0.7), stellar_diameters_lod=[0.0], **base
        )
    with pytest.raises(ValueError, match="one map per entry"):
        performance_curves(
            source, one, Circular(0.7), stellar_diameters_lod=[0.0, 1.0], **base
        )
    with pytest.raises(ValueError, match="not both"):
        performance_curves(
            source,
            two,
            Circular(0.7),
            stellar_diameters_lod=[0.0, 1.0],
            stellar_diameter_lod=0.0,
            **base,
        )
    for bad in ([1.0, 0.0], [-0.1, 1.0], [0.0, np.nan], [1.0, 1.0 + 1e-9]):
        with pytest.raises(ValueError, match="increasing"):
            performance_curves(
                source, two, Circular(0.7), stellar_diameters_lod=bad, **base
            )
