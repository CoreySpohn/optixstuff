"""Grid, fold, aperture-record and JAX contracts on synthetic optics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import concrete_subclasses, synthetic_coronagraph
from hwoutils.grids import Image, PixelGrid, resample
from yippy import Circular

import optixstuff as ox
from optixstuff.coronagraph import _convolve_quadrants


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return synthetic_coronagraph(
        tmp_path_factory.mktemp("grid_yip"), rules=(Circular(0.7), Circular(1.0))
    )


def scalar(grid):
    return ox.ScalarCoronagraph(
        grid=grid,
        throughput_fraction=0.4,
        aperture_area_lod2=1.5,
        stellar_leakage_density_lod2=2e-9,
    )


@pytest.fixture(params=concrete_subclasses(ox.AbstractCoronagraph))
def coro(request, table):
    cls = request.param
    if cls is ox.YippyCoronagraph:
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
    if isinstance(coro, ox.YippyCoronagraph):
        # On the native grid, targets are box integrals of the declared
        # reconstruction, exactly as at any other pitch (no verbatim shortcut).
        np.testing.assert_array_equal(
            coro.on_axis_psf(500.0, g).data, resample(coro.stellar_intens(0.0), g).data
        )
        np.testing.assert_array_equal(
            coro.off_axis_psf(500.0, 1.0, g).data,
            resample(coro._backend.create_psf(1.0, 0.0), g).data,
        )
        np.testing.assert_array_equal(
            coro.background_transmission(500.0, grid=g).data,
            resample(coro.sky_trans, g).data,
        )


@pytest.mark.parametrize("n", [24, 25])
def test_fold_and_direction(n, tmp_path):
    """32 float32 eps for fold sums; centroid 0.02 px rejects the old 0.5 px shift."""
    c = synthetic_coronagraph(tmp_path, n)
    g = c.grid
    x, y = g.coordinates()
    full = c.create_psfs(x.ravel(), y.ravel()).data.reshape(n, n, n, n)
    flux = jnp.asarray(np.random.default_rng(7).uniform(size=(n, n)))
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
    """Exact evaluator delegation; ratio identity at measured knots uses 16 eps."""
    rule = Circular(0.7)
    record = coro.curves(rule)
    s = jnp.asarray(record.separation_lod)
    np.testing.assert_array_equal(coro.throughput(s), record.evaluate("throughput", s))
    expected = coro.leakage_density_lod2(s) * coro.core_area(s) / coro.throughput(s)
    np.testing.assert_allclose(
        expected, coro.raw_contrast(s), rtol=16 * np.finfo(np.float32).eps
    )
    for name in (
        "core_mean_intensity",
        "IWA",
        "OWA",
        "diameter_m",
        "area_m2",
        "pixel_scale_lod",
    ):
        assert not hasattr(coro, name)


def test_rule_records_and_constants(table):
    """Exact record identity; derived constants allow two float64 rounding eps."""
    a, b = table.curves(Circular(0.7)), table.curves(Circular(1.0))
    assert a is table.curves(Circular(0.7))
    assert b is table.curves(Circular(1.0))
    assert np.all(np.asarray(b.throughput) > np.asarray(a.throughput))
    prepared = ox.YippyCoronagraph(backend=table._backend, rules=(Circular(0.9),))
    extra = prepared.curves(Circular(0.9))
    assert extra is prepared.curves(Circular(0.9))
    assert float(prepared.throughput(2.0, rule=Circular(0.9))) > float(
        table.throughput(2.0)
    )
    with pytest.raises(ValueError, match="prepared"):
        table.curves(Circular(0.9))
    c = scalar(table.grid)
    np.testing.assert_array_equal(c.curves().throughput, [0.4])
    np.testing.assert_array_equal(c.curves().core_area_lod2, [1.5])
    np.testing.assert_allclose(
        c.curves().leakage_fraction, [3e-9], rtol=2 * np.finfo(float).eps
    )
    mb = ox.MultiBandCoronagraph([500.0, 800.0], [table, c])
    assert mb.curves() is a
    np.testing.assert_array_equal(mb.throughput(2.0, 800.0), c.throughput(2.0))


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
