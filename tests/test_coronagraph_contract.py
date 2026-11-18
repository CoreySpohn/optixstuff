"""Grid, fold, aperture-record and JAX contracts on synthetic optics."""

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
from hwoutils.grids import PixelGrid, resample
from hwoutils.spectral import Band, Quadrature

import optixstuff as ox
from optixstuff.apertures import Circular


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
        return cls([table, scalar(table.grid, bandpass=Band(750.0, 850.0))])
    pytest.fail(f"Add a synthetic fixture for {cls.__name__}")


def test_image_contract(table):
    """Native metadata and explicit resampling agree with the shared grid operator."""
    g = table.at_wavelength(500.0).source_grid
    for target in (
        g,
        PixelGrid.regular((30, 32), 0.4, "lod", frame=g.frame, center=(15.2, 14.1)),
    ):
        source = table.source_psfs(
            500.0, jnp.array([1.0]), jnp.array([0.5]), grid=target
        )
        native = table.source_psfs(500.0, jnp.array([1.0]), jnp.array([0.5]))
        np.testing.assert_array_equal(source.valid, [True])
        for actual, expected in (
            (source.image, resample(native.image, target)),
            (
                table.on_axis_psf(500.0, target),
                resample(table.stellar_map(500.0, 0.0), target),
            ),
            (
                table.uniform_sky_map(500.0, grid=target),
                resample(table.sky_trans, target),
            ),
        ):
            assert actual.grid == target
            assert actual.measure == expected.measure
            assert actual.quantity == expected.quantity
            np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_allclose(
        table.point_energy_fraction(1.0, 0.5), native.image.data.sum(), rtol=1e-6
    )


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
        "uniform_sky_response",
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


def test_native_performance_maps(table):
    rule = Circular(0.7)
    record = table.curves(rule)
    # Maps are inherited once and project the same contract records on the grid.
    radius = jnp.hypot(*table.grid.coordinates())
    np.testing.assert_array_equal(table.separation_map(), radius)
    for method, field, variance_field in [
        (table.throughput_map, "throughput", "throughput_variance"),
        (table.core_area_map, "core_area_lod2", "core_area_variance_lod2"),
        (
            table.leakage_density_map,
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
    mb = ox.MultiBandCoronagraph([table, c])
    assert mb.curves() is a
    assert mb.bandpass == Band(450.0, 850.0)
    np.testing.assert_array_equal(mb.spectral_breaks_nm(), [450, 550, 750, 850])
    with pytest.raises(ValueError, match="declare its bandpass"):
        ox.MultiBandCoronagraph([table, scalar(table.grid)])
    with pytest.raises(ValueError, match="covers"):
        mb.band(650)
    np.testing.assert_array_equal(mb.throughput(2.0, 800.0), c.throughput(2.0))


def test_capability_selection(table):
    performance = scalar(table.grid, bandpass=Band(550, 850))
    model = ox.MultiBandCoronagraph((table, performance))
    assert model.band(565.0) is performance
    assert model.band(800.0).frame == performance.frame
    assert (
        model.band(500.0).stellar_frame(500.0)
        == table.at_wavelength(500.0).stellar_grid.frame
    )
    assert isinstance(table, ox.AbstractImagingCoronagraph)
    for item in (performance, model):
        assert not isinstance(item, ox.AbstractImagingCoronagraph)
        assert not hasattr(item, "source_psfs")
    assert not hasattr(model, "grid")


def test_prepare_fixed_angle_preserves_centre_and_reference():
    """The exact radii at 480, 600 and 720 nm keep the requested peak centre."""
    rules = tuple(Circular(radius, centre="peak") for radius in (0.75, 0.6, 0.5))
    coro = synthetic_coronagraph(n=8, rules=rules)
    path = ox.OpticalPath.from_default_setup(coro, detector_shape=(4, 4))
    band = Band(450.0, 750.0)
    q = Quadrature((band,), jnp.array([[480.0, 600.0, 720.0]]), jnp.ones((1, 3)))

    aperture = ox.prepare_aperture(path, q, rule=rules[1])
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
    q = Quadrature((band,), jnp.array([[500.0]]), jnp.ones((1, 1)))
    with pytest.raises(NotImplementedError, match="frozen reference"):
        ox.prepare_aperture(path, q, rule=rule, reference_wavelength_nm=490.0)

    if isinstance(rule, ox.Matched):
        with pytest.raises(ValueError, match=r"native.*images"):
            ox.prepare_aperture(
                eqx.tree_at(lambda p: p.coronagraph, path, scalar(table.grid)),
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
    aperture = ox.prepare_aperture(path, q, rule=rule, reference_wavelength_nm=490.0)
    (reference,) = aperture.references
    (node,) = reference.nodes
    assert reference.wavelength_nm == 490.0
    assert node.record.rule == reference.record.rule == rule


def test_jit_and_gradient(table):
    """32 float32 eps JIT roundoff; h=.002 central difference agrees within 0.2%."""

    def image(x):
        return table.source_psfs(
            500.0, jnp.array([x]), jnp.array([0.5]), grid=table.grid
        ).image.data

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
        path, Quadrature.build((band,), path, order=1), rule=ox.Matched()
    )
    reference = aperture.references[0]

    def tau(diameter):
        return aperture.response(reference, s, 0.0, diameter, A=1e8, B=0.0, D=2.0).tau1[
            0
        ]

    value, slope = jax.jit(jax.value_and_grad(tau))(0.5)
    np.testing.assert_allclose(slope, -value / 3.25, rtol=2e-5)
    np.testing.assert_allclose(value, tau(0.0) * 3.0 / 3.25, rtol=2e-5)
