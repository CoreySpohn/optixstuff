"""Grid, fold, aperture-record and JAX contracts on synthetic optics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    GaussianTableCoronagraph,
    concrete_subclasses,
    make_instrument,
    synthetic_coronagraph,
)

import optixstuff as ox
from optixstuff.apertures import Circular
from optixstuff.grids import PixelGrid, resample
from optixstuff.spectral import Band, Quadrature


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


def test_single_wavelength_scale_follows_node(coro):
    """Single-wavelength fixtures select the node argument exactly, even traced."""
    node, center = jnp.array(490.0), jnp.array(505.0)
    assert coro.band_averaged is False
    assert coro.scale_wavelength_nm(node, center) is node
    assert eqx.filter_jit(coro.scale_wavelength_nm)(node, center) == node


def test_band_average_has_one_native_map_and_scale():
    """A band table stores one stellar map and selects its exact observing center.

    The native table is a single array, so the two wavelength reads agree bitwise.
    """
    model = GaussianTableCoronagraph(band_averaged=True)
    first = model.stellar_map(475.0, 0.15).data
    second = model.stellar_map(510.0, 0.15).data
    np.testing.assert_array_equal(first, second)
    node, center = jnp.array(490.0), jnp.array(505.0)
    for coro in (model, ox.MultiBandCoronagraph([model])):
        assert coro.band_averaged is True
        assert coro.scale_wavelength_nm(node, center) is center
        assert eqx.filter_jit(coro.scale_wavelength_nm)(node, center) == center


@pytest.mark.parametrize(
    ("single_band", "interval"), [(Band(450, 550), "450-550 nm"), (None, "None")]
)
def test_mixed_band_declarations_name_children(single_band, interval):
    """Distinct product declarations are mutually exclusive, an exact type check."""
    band_product = GaussianTableCoronagraph(bandpass=Band(550, 650), band_averaged=True)
    single = GaussianTableCoronagraph(bandpass=single_band)
    with pytest.raises(ValueError, match="Cannot mix") as exc:
        ox.MultiBandCoronagraph([single, band_product])
    assert f"{interval}, band_averaged=False" in str(exc.value)
    assert "550-650 nm, band_averaged=True" in str(exc.value)


def test_wide_observing_bands_require_tiles_or_opt_in():
    """A 10% product admits 12%, refuses 13%, and two log tiles each fit.

    The geometric mean of the observing edges is the single shared split.
    Widths, thresholds and rounded error text have exact expected values.
    A 382-418 nm product and a 377.5-422.5 nm band have widths 0.09 and
    0.1125, a ratio of exactly 1.25 that float64 rounds 1.4e-17 above the
    limit; the band is admitted.
    """
    model = GaussianTableCoronagraph(bandpass=Band(475, 525), band_averaged=True)
    instrument = make_instrument(model).select("imager")
    wide = Band(467.5, 532.5)
    with pytest.raises(ValueError) as exc:
        Quadrature.build([wide], instrument)
    split = np.sqrt(467.5 * 532.5)
    for expected in (
        "467.5-532.5 nm",
        "fractional width 0.13.",
        "fractional width is 0.1.",
        f"467.5-{split:.12g} nm",
        f"{split:.12g}-532.5 nm",
    ):
        assert expected in str(exc.value)
    assert Quadrature.build([wide], instrument, allow_wide_bands=True).bands == (wide,)
    narrow = Band(470, 530)
    assert Quadrature.build([narrow], instrument).bands == (narrow,)
    edge = GaussianTableCoronagraph(bandpass=Band(382, 418), band_averaged=True)
    limit = Band(377.5, 422.5)
    assert limit.width_nm / limit.center_nm > 1.25 * (36 / 400)
    assert Quadrature.build([limit], make_instrument(edge).select("imager")).bands == (
        limit,
    )


def test_unresolved_product_has_no_positive_width_tiles(monkeypatch):
    """Adjacent binary64 edges at 600 nm admit only a zero-width observation."""
    product = Band(np.nextafter(600.0, 0.0), 600.0)
    model = GaussianTableCoronagraph(bandpass=product, band_averaged=True)

    def fail_if_tiled(self, n, spacing):
        pytest.fail(f"Unresolved product requested {n} tiles")

    monkeypatch.setattr(Band, "channels", fail_if_tiled)
    model.check_bands((Band(600.0, 600.0),))
    for band in (Band(599.99999, 600.00001), product):
        with pytest.raises(ValueError, match="No positive-width tile fits"):
            model.check_bands((band,))


def test_band_average_requires_one_covering_child():
    """Coverage is inclusive at declared edges and exact across the 550 nm split."""
    blue = GaussianTableCoronagraph(bandpass=Band(450, 550), band_averaged=True)
    red = GaussianTableCoronagraph(bandpass=Band(550, 650), band_averaged=True)
    coro = ox.MultiBandCoronagraph([blue, red])
    bands = (Band(450, 550), Band(550, 650))
    assert Quadrature.build(bands, coro).bands == bands
    for allow in (False, True):
        with pytest.raises(ValueError, match=r"split.*550 nm"):
            Quadrature.build([Band(510, 590)], coro, allow_wide_bands=allow)
    with pytest.raises(ValueError, match="covers"):
        Quadrature.build([Band(700, 720)], coro)
    # Exact selection at a float64 conversion edge and a stored float32 edge.
    for edge, below in ((600.0, np.nextafter(600.0, 0.0)), (565.1, np.float32(565.1))):
        converted = GaussianTableCoronagraph(
            bandpass=Band(500, below), band_averaged=True
        )
        later = GaussianTableCoronagraph(bandpass=Band(edge, 700), band_averaged=True)
        split = ox.MultiBandCoronagraph([converted, later])
        assert split.coronagraph_for_band(Band(550, edge)) is converted
        assert split.coronagraph_at(edge) is converted
        assert split.coronagraph_at(jnp.asarray(edge)) is converted
        assert split.coronagraph_at(edge + 0.01) is later


def test_dispatch_at_the_wavelength_merge_threshold():
    """Three float32 spacings merge, four do not, for points and both band edges."""
    spacing = float(np.spacing(np.float32(600.0)))
    for edge in (600.0, 600.0 + 0.25 * spacing):
        blue = GaussianTableCoronagraph(bandpass=Band(500, edge), band_averaged=True)
        red = GaussianTableCoronagraph(bandpass=Band(edge, 700), band_averaged=True)
        blue_first = ox.MultiBandCoronagraph([blue, red])
        red_first = ox.MultiBandCoronagraph([red, blue])
        for steps, above, below in ((3.25, blue, red), (3.75, red, blue)):
            for scalar in (float, np.float32, jnp.asarray):
                high = scalar(600.0 + steps * spacing)
                low = scalar(600.0 - steps * spacing)
                assert blue_first.coronagraph_at(high) is above
                assert red_first.coronagraph_at(low) is below
                upper = Band(scalar(599.9999), high)
                lower = Band(low, scalar(600.0001))
                assert blue_first.coronagraph_for_band(upper) is above
                assert red_first.coronagraph_for_band(lower) is below

    blue = GaussianTableCoronagraph(bandpass=Band(500, 600))
    red = GaussianTableCoronagraph(bandpass=Band(600, 700))
    mono = ox.MultiBandCoronagraph([blue, red])
    # Rounded edges at two and five spacings give a midpoint at 3.5, rounding to four.
    for low_steps, child in ((1.4, blue), (1.6, red)):
        for scalar in (float, np.float32, jnp.asarray):
            band = Band(scalar(600 + low_steps * spacing), scalar(600 + 5.2 * spacing))
            assert mono.coronagraph_for_band(band) is child


def test_overlapping_band_products_use_first_full_coverage():
    """The interval 490-540 lies entirely in the second child only, exactly."""
    blue = GaussianTableCoronagraph(bandpass=Band(450, 525), band_averaged=True)
    red = GaussianTableCoronagraph(bandpass=Band(475, 550), band_averaged=True)
    coro = ox.MultiBandCoronagraph([blue, red])
    band = Band(490, 540)
    assert coro.coronagraph_at(band.center_nm) is blue
    assert coro.coronagraph_for_band(band) is red
    assert Quadrature.build((band,), coro).bands == (band,)


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
    assert mb.coronagraph_at(None) is table
    assert mb.coronagraphs == (table, c)
    assert mb.bandpass == Band(450.0, 850.0)
    np.testing.assert_array_equal(mb.spectral_breaks_nm(), [450, 550, 750, 850])
    assert mb.coronagraph_at(800.0) is c
    with pytest.raises(ValueError, match="declare its bandpass"):
        ox.MultiBandCoronagraph([table, GaussianTableCoronagraph(bandpass=None)])
    with pytest.raises(ValueError, match="covers"):
        mb.coronagraph_at(650)


def test_capability_selection(table):
    performance = GaussianTableCoronagraph(bandpass=Band(550, 850))
    model = ox.MultiBandCoronagraph((table, performance))
    assert model.coronagraph_at(565.0) is performance
    assert model.coronagraph_at(800.0).frame == performance.frame
    assert (
        model.coronagraph_at(500.0).at_wavelength(500.0).stellar_grid.frame
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


def test_for_telescope_returns_the_same_product_or_refuses():
    """Fails if the default binding copies a product or lets a mismatch through.

    Contract, exact. A product of the pupil's diameter and an unknown identifier
    comes back as the same object; another diameter raises on the frame, and
    two known, different identifiers raise a stamp mismatch naming both.
    """
    from hwoutils.provenance import StampMismatchError

    model = GaussianTableCoronagraph()
    telescope = ox.Telescope(
        ox.SimplePupil(circumscribed_diameter_m=6.0, collecting_area_m2=28.0)
    )
    assert model.pupil_id is None
    assert model.for_telescope(telescope) is model
    wide = ox.Telescope(
        ox.SimplePupil(circumscribed_diameter_m=6.5, collecting_area_m2=28.0)
    )
    with pytest.raises(ValueError, match="GaussianTableCoronagraph"):
        model.for_telescope(wide)

    class Declared(GaussianTableCoronagraph):
        @property
        def pupil_id(self):
            return "old"

    other = ox.Telescope(
        ox.SimplePupil(
            circumscribed_diameter_m=6.0, collecting_area_m2=28.0, pupil_id="new"
        )
    )
    with pytest.raises(StampMismatchError, match=r"old.*new"):
        Declared().for_telescope(other)
