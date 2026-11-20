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

import optixstuff as ox
from optixstuff.apertures import Circular
from optixstuff.grids import PixelGrid, resample
from optixstuff.spectral import Band


@pytest.fixture(scope="module")
def table():
    return synthetic_coronagraph(rules=(Circular(0.7), Circular(1.0)))


def second_band():
    return GaussianTableCoronagraph(bandpass=Band(750.0, 850.0))


@pytest.fixture(params=concrete_subclasses(ox.AbstractCoronagraph))
def coro(request, table):
    cls = request.param
    if cls is GaussianTableCoronagraph:
        return table
    if cls is ox.MultiBandCoronagraph:
        return cls([table, second_band()])
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


def test_slot_is_the_image_capability(coro):
    """The slot carries a frame and a bandpass; no curve query is part of it."""
    assert isinstance(coro.frame, ox.grids.Frame)
    for cls in (
        ox.AbstractCoronagraph,
        ox.AbstractImagingCoronagraph,
        ox.AbstractSceneCoronagraph,
        ox.AbstractTableCoronagraph,
        ox.MultiBandCoronagraph,
    ):
        for name in (
            "curves",
            "prepared",
            "throughput",
            "core_area",
            "leakage_fraction",
            "leakage_density_lod2",
            "raw_contrast",
            "uniform_sky_response",
            "outer_working_angle_lod",
            "throughput_map",
            "core_area_map",
            "leakage_density_map",
        ):
            assert not hasattr(cls, name)
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


def test_separation_map(table):
    """The native separation map is the radius of every pixel centre."""
    radius = jnp.hypot(*table.grid.coordinates())
    np.testing.assert_array_equal(table.separation_map(), radius)


def test_rule_records_and_bands(table):
    """Exact band bounds and record identity.

    Fails if multiband coverage clips a child edge or invents unknown coverage.
    """
    a, b = table.curves(Circular(0.7)), table.curves(Circular(1.0))
    assert a is table.curves(Circular(0.7))
    assert b is table.curves(Circular(1.0))
    assert np.all(np.asarray(b.throughput) > np.asarray(a.throughput))
    prepared = synthetic_coronagraph(rules=(Circular(0.9),))
    extra = prepared.curves(Circular(0.9))
    assert extra is prepared.curves(Circular(0.9))
    assert float(extra.evaluate("throughput", 2.0)) > float(
        a.evaluate("throughput", 2.0)
    )
    with pytest.raises(ValueError, match="prepared"):
        table.curves(Circular(0.9))
    c = second_band()
    mb = ox.MultiBandCoronagraph([table, c])
    assert mb.band(None) is table
    assert mb.bandpass == Band(450.0, 850.0)
    np.testing.assert_array_equal(mb.spectral_breaks_nm(), [450, 550, 750, 850])
    assert mb.band(800.0) is c
    with pytest.raises(ValueError, match="declare its bandpass"):
        ox.MultiBandCoronagraph([table, GaussianTableCoronagraph(bandpass=None)])
    with pytest.raises(ValueError, match="covers"):
        mb.band(650)


def test_capability_selection(table):
    performance = GaussianTableCoronagraph(bandpass=Band(550, 850))
    model = ox.MultiBandCoronagraph((table, performance))
    assert model.band(565.0) is performance
    assert model.band(800.0).frame == performance.frame
    assert (
        model.band(500.0).stellar_frame(500.0)
        == table.at_wavelength(500.0).stellar_grid.frame
    )
    assert isinstance(table, ox.AbstractImagingCoronagraph)
    assert not isinstance(model, ox.AbstractImagingCoronagraph)
    assert not hasattr(model, "source_psfs")
    assert not hasattr(model, "grid")


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
        return table.curves(Circular(0.7)).evaluate("throughput", s)

    s, h = jnp.array(1.4), 0.002
    np.testing.assert_allclose(jax.jit(f)(s), f(s), rtol=8 * np.finfo(np.float32).eps)
    derivative = jax.grad(f)(s)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, (f(s + h) - f(s - h)) / (2 * h), rtol=0.002)
