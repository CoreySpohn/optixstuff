"""Grid, fold, aperture-record and JAX contracts on synthetic optics."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    GaussianTableCoronagraph,
    concrete_subclasses,
    synthetic_coronagraph,
)
from hwoutils.grids import Image, PixelGrid, resample
from hwoutils.spectral import Band, Quadrature

import optixstuff as ox
from optixstuff.apertures import Circular
from optixstuff.coronagraph import _convolve_quadrants


@pytest.fixture(scope="module")
def table():
    return synthetic_coronagraph(rules=(Circular(0.7), Circular(1.0)))


def scalar(grid, *, bandpass=None):
    return ox.ScalarCoronagraph(
        grid=grid,
        bandpass=bandpass,
        throughput_fraction=0.4,
        aperture_area_lod2=1.5,
        stellar_leakage_density_lod2=2e-9,
    )


@pytest.fixture(params=concrete_subclasses(ox.AbstractCoronagraph))
def coro(request, table):
    cls = request.param
    if cls is GaussianTableCoronagraph:
        return table
    if cls is ox.ScalarCoronagraph:
        return scalar(table.grid)
    if cls is ox.MultiBandCoronagraph:
        return cls([500.0, 800.0], [table, scalar(table.grid)])
    pytest.fail(f"Add a synthetic fixture for {cls.__name__}")


def test_image_contract(coro):
    """Exact metadata and 8 float32 eps for native-grid identity operations."""
    g = coro.grid
    scene = Image(
        jnp.zeros(g.shape).at[13, 14].set(1.0), g, (), "per_pixel", "photons/s"
    )
    targets = [
        g,
        PixelGrid.regular((30, 32), 0.4, "lod", frame=g.frame, center=(15.2, 14.1)),
    ]
    for target in targets:
        results = [
            coro.on_axis_psf(500.0, target),
            coro.off_axis_psf(500.0, 1.0, target),
            coro.stellar_map(500.0, 0.1, grid=target),
            coro.source_psfs(500.0, jnp.array([1.0]), jnp.array([0.5]), grid=target),
            coro.background_transmission(500.0, grid=target),
            coro.extended_scene(scene, 500.0, grid=target),
        ]
        for image in results:
            assert isinstance(image, Image)
            assert image.grid == target
        assert results[4].measure == "per_unit_area"
        assert results[4].quantity == "transmission"
        assert results[5].quantity == "photons/s"
        assert all(im.measure == "per_pixel" for im in results[:4])
    if isinstance(coro, GaussianTableCoronagraph):
        for native, expected_native in [
            (
                coro.stellar_map(500.0, 0.1, grid=None),
                coro._native_stellar_map(0.1),
            ),
            (
                coro.source_psfs(500.0, jnp.array([1.0]), jnp.array([0.0]), grid=None),
                coro._native_source_psfs(jnp.array([1.0]), jnp.array([0.0])),
            ),
        ]:
            assert native.grid == expected_native.grid
            assert native.leading_axes == expected_native.leading_axes
            assert native.reconstruction == expected_native.reconstruction
            np.testing.assert_array_equal(native.data, expected_native.data)
        # On the native grid, targets are box integrals of the declared
        # reconstruction, exactly as at any other pitch (no verbatim shortcut).
        np.testing.assert_array_equal(
            coro.on_axis_psf(500.0, g).data,
            resample(coro.stellar_map(500.0, 0.0, grid=None), g).data,
        )
        np.testing.assert_array_equal(
            coro.off_axis_psf(500.0, 1.0, g).data,
            resample(coro._native_source_psfs(1.0, 0.0), g).data[0],
        )
        np.testing.assert_array_equal(
            coro.background_transmission(500.0, grid=g).data,
            resample(coro.sky_trans, g).data,
        )


@pytest.mark.parametrize("n", [24, 25])
def test_fold_and_direction(n):
    """32 float32 eps for fold sums; centroid 0.02 px rejects the old 0.5 px shift."""
    c = synthetic_coronagraph(n)
    g = c.grid
    x, y = g.coordinates()
    full = c.source_psfs(500.0, x.ravel(), y.ravel(), grid=None).data.reshape(
        n, n, n, n
    )
    flux = jnp.asarray(np.random.default_rng(0).uniform(size=(n, n)))
    folded = _convolve_quadrants(flux, c.psf_datacube, g)
    direct = jnp.einsum("ij,ijyx->yx", flux, full)
    np.testing.assert_allclose(
        folded,
        direct,
        rtol=32 * np.finfo(np.float32).eps,
        atol=32 * np.finfo(np.float32).eps * float(direct.max()),
    )

    def centroid(data):
        yy, xx = np.indices(data.shape)
        return np.array([(data * xx).sum(), (data * yy).sum()]) / data.sum()

    for row, col in [(n // 2 + 1, n // 2 + 2), (n // 2 - 2, n // 2 - 1)]:
        px, py = g.pixel_to_unit(col, row)
        planet = c.source_psfs(500.0, jnp.array([px]), jnp.array([py]), grid=g).data[0]
        scene = Image(
            jnp.zeros(g.shape).at[row, col].set(1.0), g, (), "per_pixel", "flux"
        )
        extended = c.extended_scene(scene, 500.0, grid=g).data
        np.testing.assert_allclose(
            centroid(planet), centroid(extended), atol=0.02, rtol=0
        )
        np.testing.assert_allclose(centroid(planet), [col, row], atol=0.02, rtol=0)
    center = np.array([g.center_x, g.center_y])
    for px, py, axis in [(1.0, 0.0, 0), (0.0, 1.0, 1)]:
        im = c.source_psfs(500.0, jnp.array([px]), jnp.array([py]), grid=g).data[0]
        assert centroid(im)[axis] > center[axis] + 1.9
        assert abs(centroid(im)[1 - axis] - center[1 - axis]) < 0.02


@pytest.mark.parametrize("ratio", [0.5, 1.7])
def test_flux_resampling(table, ratio):
    """Integer refinement / deposition conserve flux to 16 float32 eps with coverage."""
    g = table.grid
    target = PixelGrid.regular((64, 64), g.step * ratio, "lod", frame=g.frame)
    native = table.off_axis_psf(500.0, 1.0, g)
    fine = table.off_axis_psf(500.0, 1.0, target)
    np.testing.assert_allclose(
        fine.data.sum(), native.data.sum(), rtol=16 * np.finfo(np.float32).eps
    )


def test_curve_contract(coro):
    """Exact delegation; ratio identity uses 16 eps and floor arithmetic uses 4 eps."""

    class UnitBearingSeparation:
        unit = "lod"

    for name in (
        "throughput",
        "core_area",
        "leakage_fraction",
        "leakage_density_lod2",
        "raw_contrast",
        "occulter_transmission",
    ):
        with pytest.raises(TypeError, match="dimensionless"):
            getattr(coro, name)(UnitBearingSeparation())
    rule = Circular(0.7)
    record = coro.curves(rule)
    model, prepared_record = coro.prepared(rule)
    assert model is (
        coro.band(500.0) if isinstance(coro, ox.MultiBandCoronagraph) else coro
    )
    assert prepared_record is record
    s = jnp.asarray(record.separation_lod)
    np.testing.assert_array_equal(
        coro.throughput(s, stellar_diam_lod=0.1), record.evaluate("throughput", s)
    )
    np.testing.assert_array_equal(
        coro.core_area(s, stellar_diam_lod=0.1), record.evaluate("core_area_lod2", s)
    )
    for accessor, field in (
        (coro.throughput, "throughput_variance"),
        (coro.core_area, "core_area_variance_lod2"),
        (coro.leakage_fraction, "leakage_fraction_variance"),
        (coro.leakage_density_lod2, "leakage_density_variance_lod2"),
    ):
        np.testing.assert_array_equal(
            accessor(s, variance=True), record.evaluate(field, s)
        )
    expected = coro.leakage_density_lod2(s) * coro.core_area(s) / coro.throughput(s)
    np.testing.assert_allclose(
        expected, coro.raw_contrast(s), rtol=16 * np.finfo(np.float32).eps
    )
    # Post-processing uses max(abs(raw contrast), engineering floor) / ppf.
    # Maps are inherited once and project the same contract records on the grid.
    radius = jnp.hypot(*coro.grid.coordinates())
    np.testing.assert_array_equal(coro.separation_map(), radius)
    for method, field, variance_field in [
        (coro.throughput_map, "throughput", "throughput_variance"),
        (coro.core_area_map, "core_area_lod2", "core_area_variance_lod2"),
        (
            coro.leakage_density_map,
            "leakage_density_lod2",
            "leakage_density_variance_lod2",
        ),
    ]:
        for variance, name in ((False, field), (True, variance_field)):
            expected_map = record.evaluate(name, radius)
            np.testing.assert_array_equal(method(variance=variance), expected_map)
            np.testing.assert_array_equal(
                method([rule, rule], variance=variance),
                jnp.stack([expected_map, expected_map], axis=-1),
            )
    floor, ppf = 1e-7, 25.0
    expected_noise = np.maximum(np.abs(np.asarray(record.raw_contrast)), floor) / ppf
    np.testing.assert_allclose(
        jnp.maximum(jnp.abs(coro.raw_contrast(s)), floor) / ppf,
        expected_noise,
        rtol=4 * np.finfo(np.float32).eps,
    )
    for name in (
        "core_mean_intensity",
        "stellar_intens",
        "create_psfs",
        "noise_floor_ayo",
        "noise_floor_exosims",
        "IWA",
        "OWA",
        "diameter_m",
        "area_m2",
        "pixel_scale_lod",
    ):
        assert not hasattr(coro, name)


def test_rule_records_and_constants(table):
    """Exact band bounds/record identity; constants allow two float64 rounding eps.

    Fails if multiband coverage clips a child edge or invents unknown coverage.
    """
    a, b = table.curves(Circular(0.7)), table.curves(Circular(1.0))
    assert a is table.curves(Circular(0.7))
    assert b is table.curves(Circular(1.0))
    assert np.all(np.asarray(b.throughput) > np.asarray(a.throughput))
    prepared = synthetic_coronagraph(rules=(Circular(0.9),))
    extra = prepared.curves(Circular(0.9))
    assert extra is prepared.curves(Circular(0.9))
    assert float(prepared.throughput(2.0, rule=Circular(0.9))) > float(
        table.throughput(2.0)
    )
    with pytest.raises(ValueError, match="prepared"):
        table.curves(Circular(0.9))
    c = scalar(table.grid, bandpass=Band(750.0, 850.0))
    np.testing.assert_array_equal(c.curves().throughput, [0.4])
    np.testing.assert_array_equal(c.curves().core_area_lod2, [1.5])
    np.testing.assert_allclose(
        c.curves().leakage_fraction, [3e-9], rtol=2 * np.finfo(float).eps
    )
    mb = ox.MultiBandCoronagraph([500.0, 800.0], [table, c])
    assert mb.curves() is a
    assert mb.bandpass == Band(450.0, 850.0)
    assert (
        ox.MultiBandCoronagraph([500.0, 800.0], [table, scalar(table.grid)]).bandpass
        is None
    )
    np.testing.assert_array_equal(mb.throughput(2.0, 800.0), c.throughput(2.0))


def test_stellar_frame(coro):
    """Frame metadata agrees with the native stellar product in each band."""
    if isinstance(coro, ox.MultiBandCoronagraph):
        other_grid = replace(
            coro.bands[1].grid,
            frame=replace(coro.bands[1].grid.frame, reference_wavelength_nm=800.0),
        )
        coro = ox.MultiBandCoronagraph(
            (500.0, 800.0), (coro.bands[0], scalar(other_grid))
        )
        assert coro.stellar_frame(800.0) == other_grid.frame
        assert coro.stellar_frame(500.0) != other_grid.frame
    assert coro.stellar_frame(500.0) == coro.stellar_map(500.0, 0.0).grid.frame


def test_prepare_fixed_angle_preserves_centre_and_reference():
    """The exact radii at 480, 600 and 720 nm keep the requested peak centre."""
    rules = tuple(Circular(radius, centre="peak") for radius in (0.75, 0.6, 0.5))
    coro = synthetic_coronagraph(n=8, rules=rules)
    path = ox.OpticalPath.from_default_setup(coro, detector_shape=(4, 4))
    band = Band(450.0, 750.0)
    q = Quadrature(jnp.array([[480.0, 600.0, 720.0]]), jnp.ones((1, 3)))

    aperture = ox.prepare_aperture(path, band, q, rule=rules[1])
    (reference,) = aperture.references
    assert aperture.quadrature is q
    assert reference.wavelength_nm == 600.0
    assert reference.record is coro.curves(rules[1])
    assert tuple(node.record.rule for node in reference.nodes) == rules
    assert reference.nodes[1].record is reference.record


@pytest.mark.parametrize("rule", [ox.HalfMaximum(), ox.Matched()])
def test_polychromatic_preparation_requires_dilation_for_tables(table, rule):
    """A shifted reference requires template dilation only for table models."""
    path = ox.OpticalPath.from_default_setup(table, detector_shape=(4, 4))
    band = Band(499.0, 501.0)
    q = Quadrature(jnp.array([[500.0]]), jnp.ones((1, 1)))
    with pytest.raises(NotImplementedError, match="frozen reference"):
        ox.prepare_aperture(path, band, q, rule=rule, reference_wavelength_nm=490.0)

    if isinstance(rule, ox.Matched):
        with pytest.raises(ValueError, match=r"native.*images"):
            ox.prepare_aperture(
                eqx.tree_at(lambda p: p.coronagraph, path, scalar(table.grid)),
                band,
                q,
                rule=rule,
            )
        return

    coro = ox.ScalarCoronagraph(
        grid=table.grid,
        throughput_fraction=0.4,
        aperture_area_lod2=1.5,
        stellar_leakage_density_lod2=2e-9,
        rule=rule,
    )
    path = eqx.tree_at(lambda p: p.coronagraph, path, coro)
    aperture = ox.prepare_aperture(
        path, band, q, rule=rule, reference_wavelength_nm=490.0
    )
    (reference,) = aperture.references
    (node,) = reference.nodes
    assert reference.wavelength_nm == 490.0
    assert node.record.rule == reference.record.rule == rule


def test_jit_and_gradient(table):
    """32 float32 eps JIT roundoff; h=.002 central difference agrees within 0.2%."""

    def image(x):
        return table.source_psfs(
            500.0, jnp.array([x]), jnp.array([0.5]), grid=table.grid
        ).data

    np.testing.assert_allclose(
        eqx.filter_jit(image)(jnp.array(1.2)),
        image(1.2),
        rtol=32 * np.finfo(np.float32).eps,
        atol=1e-8,
    )

    def f(s):
        return table.throughput(s)

    s, h = jnp.array(1.4), 0.002
    np.testing.assert_allclose(jax.jit(f)(s), f(s), rtol=8 * np.finfo(np.float32).eps)
    derivative = jax.grad(f)(s)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, (f(s + h) - f(s - h)) / (2 * h), rtol=0.002)

    # In this table the stellar map is spatially uniform and proportional
    # to (1 + diameter**2). With 2 uniform counts/s the entire inverse
    # variance profile, including its interpolated throughput, scales as
    # 1/(3 + diameter**2).
    path = ox.OpticalPath.from_default_setup(table, detector_shape=(4, 4))
    band = Band(499.5, 500.5)
    aperture = ox.prepare_aperture(
        path, band, Quadrature.build((band,), None, 1), rule=ox.Matched()
    )
    reference = aperture.references[0]

    def tau(diameter):
        return aperture.response(reference, s, 0.0, diameter, A=1e8, B=0.0, D=2.0).tau1[
            0
        ]

    value, slope = jax.jit(jax.value_and_grad(tau))(0.5)
    np.testing.assert_allclose(slope, -value / 3.25, rtol=2e-5)
    np.testing.assert_allclose(value, tau(0.0) * 3.0 / 3.25, rtol=2e-5)
