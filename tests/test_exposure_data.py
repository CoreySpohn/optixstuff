"""Synthetic measured images and their exposure geometry."""

from dataclasses import fields, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.provenance import build_stamps
from hwoutils.testing import ulps

from optixstuff import (
    Band,
    Exposure,
    ExposureData,
    FramePlan,
    HostRecords,
    Image,
    Measure,
    PixelGrid,
    Pointing,
    Quadrature,
    stack_exposure_data,
)
from optixstuff.grids import SkyOrientation


@pytest.fixture(scope="module")
def measurement(precision):
    """Two synthetic bands on a four-by-four detector with 125 mas pixels."""
    grid = PixelGrid.regular((4, 4), 0.125, "arcsec")
    image = Image(
        jnp.ones((2, 4, 4), dtype=precision),
        grid,
        ("band",),
        Measure.integral(),
        "electrons",
    )
    rule = Quadrature.build((Band(500, 550), Band(550, 600)), order=2)
    exposure = Exposure(1.0, Pointing(0.0), rule, FramePlan(2.0, 3))
    return ExposureData(image, exposure)


def test_exposure_data_construction(measurement):
    """The array, axis and orientation contracts are checked exactly."""
    image, exposure = measurement.image, measurement.exposure
    mask = np.eye(4, dtype=bool)
    variance = np.full(image.data.shape, 2, dtype=np.int16)
    data = ExposureData(image, exposure, valid=mask, variance=variance)
    assert data.image.data is image.data
    assert measurement.variance is None
    assert measurement.flags is None
    np.testing.assert_array_equal(measurement.valid, np.ones(image.data.shape, bool))
    assert data.valid.dtype == jnp.bool_
    np.testing.assert_array_equal(data.valid, np.broadcast_to(mask, image.data.shape))
    assert data.variance.dtype == image.data.dtype
    np.testing.assert_array_equal(data.variance, variance)
    counts = replace(image, data=jnp.zeros(image.data.shape, dtype=jnp.int32))
    integer_data = ExposureData(counts, exposure)
    assert integer_data.image.data is counts.data
    np.testing.assert_array_equal(integer_data.target_px(), measurement.target_px())
    with pytest.raises(ValueError, match=r"image\.data"):
        ExposureData(replace(counts, quantity="photons"), exposure)
    with pytest.raises(ValueError, match="variance"):
        ExposureData(counts, exposure, variance=variance)
    with pytest.raises(ValueError, match="valid"):
        ExposureData(image, exposure, valid=np.ones((4, 4)))
    with pytest.raises(ValueError, match="flags"):
        ExposureData(image, exposure, flags=np.ones(image.data.shape, dtype=np.int8))
    with pytest.raises(ValueError, match="orientation"):
        grid = replace(image.grid, orientation=SkyOrientation(0))
        ExposureData(replace(image, grid=grid), exposure)
    with pytest.raises(ValueError, match="band"):
        ExposureData(replace(image, data=image.data[:1]), exposure)
    with pytest.raises(ValueError, match="leading_axes"):
        ExposureData(replace(image, leading_axes=("wavelength",)), exposure)


def test_invalid_flag_bits_define_valid(measurement):
    """A hand-built NumPy bit mask gives exact truth for each flag combination."""
    image, exposure = measurement.image, measurement.exposure
    flags = np.zeros(image.data.shape, dtype=np.uint8)
    flags[0, 0, :] = [0, 1, 2, 3]
    expected = (flags & 2) == 0
    kwargs = dict(flags=flags, flag_names=("warning", "bad"), invalid_flags=("bad",))
    data = ExposureData(image, exposure, **kwargs)
    assert data.flags.dtype == flags.dtype
    np.testing.assert_array_equal(data.flags, flags)
    np.testing.assert_array_equal(data.valid, expected)
    np.testing.assert_array_equal(
        ExposureData(image, exposure, valid=expected, **kwargs).valid, expected
    )
    with pytest.raises(ValueError, match=r"valid.*flags"):
        ExposureData(image, exposure, valid=np.ones((4, 4), bool), **kwargs)


def test_commanded_target_position(measurement):
    """Two 125 mas pixels shift x by two at any roll, within four ULPs."""
    image, exposure = measurement.image, measurement.exposure
    cx, cy = image.grid.center_x, image.grid.center_y
    center = jnp.asarray([cx, cy], dtype=image.data.dtype)
    assert jnp.all(jnp.abs(measurement.target_px() - center) <= ulps(4, center))
    shifted = eqx.tree_at(lambda x: x.pointing, exposure, Pointing(90.0, [250.0, 0.0]))
    expected = jnp.asarray([cx + 2, cy], dtype=image.data.dtype)
    pixels = ExposureData(image, shifted).target_px()
    assert jnp.all(jnp.abs(pixels - expected) <= ulps(4, expected))


def test_absolute_sky_orientation_and_round_trip(measurement):
    """Pointing's parity -1 formula anchors signs. Inversion allows eight ULPs."""
    image, exposure = measurement.image, measurement.exposure
    cx, cy = image.grid.center_x, image.grid.center_y
    assert measurement.sky_to_pixel(0.125, 0.0) == (cx - 1, cy)
    assert measurement.sky_to_pixel(0.0, 0.125) == (cx, cy + 1)
    rotated = eqx.tree_at(
        lambda x: x.pointing, exposure, Pointing(90.0, [125.0, -125.0])
    )
    data = ExposureData(image, rotated)
    # x = -east*cos + north*sin; y = east*sin + north*cos.
    x, _ = data.sky_to_pixel(0.0, 0.125)
    assert x > cx + 1
    sky = jnp.asarray([0.125, -0.25], dtype=image.data.dtype)
    recovered = jnp.stack(data.pixel_to_sky(*data.sky_to_pixel(*sky)))
    assert jnp.all(jnp.abs(recovered - sky) <= ulps(8, sky))


def test_roll_is_traced_without_recompiling(measurement):
    """Two rolls share one trace and match eager values within four ULPs."""
    traces = []

    def locate(roll):
        traces.append(None)
        exposure = eqx.tree_at(
            lambda x: x.pointing.roll_deg, measurement.exposure, roll
        )
        return ExposureData(measurement.image, exposure).sky_to_pixel(0.125, -0.125)

    locate_jit = jax.jit(locate)
    for roll in (0.0, 90.0):
        value = jnp.asarray(roll, dtype=measurement.image.data.dtype)
        expected = jnp.stack(locate(value))
        assert jnp.all(
            jnp.abs(jnp.stack(locate_jit(value)) - expected) <= ulps(4, expected)
        )
    assert len(traces) == 3  # Two eager calls and one trace.


@pytest.fixture(scope="module")
def events(measurement):
    """Two events whose masks, rolls and times differ on the same detector."""
    exposure = eqx.tree_at(
        lambda x: (x.pointing, x.time_d),
        measurement.exposure,
        (Pointing(90.0), measurement.exposure.time_d + 1),
    )
    second = ExposureData(measurement.image, exposure, valid=np.eye(4, dtype=bool))
    return measurement, second


def test_stack_events_and_take(events):
    """Each input array and exposure leaf is recovered exactly by taking its event."""
    stacked = stack_exposure_data(events)
    assert stacked.image.leading_axes == ("event", "band")
    assert stacked.image.data.shape == (2, 2, 4, 4)
    rolls = jnp.stack([data.exposure.pointing.roll_deg for data in events])
    times = jnp.stack([data.exposure.time_d for data in events])
    np.testing.assert_array_equal(stacked.exposure.pointing.roll_deg, rolls)
    np.testing.assert_array_equal(stacked.exposure.time_d, times)
    np.testing.assert_array_equal(
        stacked.target_px(), [data.target_px() for data in events]
    )
    with pytest.raises(ValueError, match="take"):
        stacked.sky_to_pixel(0.0, 0.0)
    with pytest.raises(ValueError, match="take"):
        stacked.pixel_to_sky(0.0, 0.0)
    for index, data in enumerate(events):
        assert eqx.tree_equal(stacked.take(index), data)
    draws = stack_exposure_data((stacked, stacked), axis="draw")
    assert draws.image.leading_axes == ("draw", "event", "band")
    assert eqx.tree_equal(draws.take(-1, axis="draw"), stacked)
    assert eqx.tree_equal(draws.take(1).take(0, axis="draw"), events[1])


@pytest.fixture(scope="module")
def mismatches(measurement):
    """Pairs with one incompatible field for the homogeneous-stack checks."""
    image, exposure = measurement.image, measurement.exposure
    rule = exposure.quadrature
    flags = np.zeros(image.data.shape, np.uint8)
    flag_policy = dict(
        flags=flags, flag_names=("warning", "bad"), invalid_flags=("bad",)
    )
    flagged = ExposureData(image, exposure, **flag_policy)
    by_image = {
        "grid": dict(grid=PixelGrid.regular(image.grid.shape, 0.25, "arcsec")),
        "quantity": dict(quantity="photons"),
        "measure": dict(measure=Measure.intensive()),
        "reconstruction": dict(reconstruction="nearest"),
        "dtype": dict(data=image.data.astype(jnp.float16)),
    }
    by_exposure = {
        "bands": dict(quadrature=Quadrature.build(rule.bands[::-1], order=2)),
        "layout_digest": dict(quadrature=Quadrature.build(rule.bands, order=3)),
        "plan": dict(plan=None),
        "offset_mas": dict(pointing=Pointing(0, [0, 0])),
        "exposure leaf shape": dict(plan=FramePlan([2.0, 2.0], 3)),
        "exposure leaf dtype": dict(plan=FramePlan(2.0, 3.0)),
        "exposure.*structure": dict(
            resolution=replace(exposure.resolution, quadratures=(rule,))
        ),
    }
    by_data = {
        "variance": dict(variance=image.data),
        "flags": dict(flags=flags),
        "built_at": dict(built_at=build_stamps(producer="different")),
    }
    by_flag = {
        "flag_names": dict(flag_names=("bad", "warning")),
        "invalid_flags": dict(invalid_flags=("warning",)),
        "flags dtype": dict(flags=flags.astype(np.uint16)),
    }
    return {
        **{
            field: (measurement, ExposureData(replace(image, **change), exposure))
            for field, change in by_image.items()
        },
        **{
            field: (measurement, ExposureData(image, replace(exposure, **change)))
            for field, change in by_exposure.items()
        },
        **{
            field: (measurement, ExposureData(image, exposure, **change))
            for field, change in by_data.items()
        },
        **{
            field: (flagged, ExposureData(image, exposure, **(flag_policy | change)))
            for field, change in by_flag.items()
        },
    }


@pytest.mark.parametrize(
    "field",
    [
        "grid",
        "quantity",
        "measure",
        "reconstruction",
        "dtype",
        "bands",
        "layout_digest",
        "plan",
        "offset_mas",
        "exposure leaf shape",
        "exposure leaf dtype",
        "exposure.*structure",
        "variance",
        "flags",
        "flag_names",
        "invalid_flags",
        "flags dtype",
        "built_at",
    ],
)
def test_stack_refuses_heterogeneous_items(mismatches, field):
    """One differing field must raise and identify item 1 and that field exactly."""
    with pytest.raises(ValueError, match=rf"item 1.*{field}"):
        stack_exposure_data(mismatches[field])


@pytest.mark.parametrize("field", ["shape", "leading_axes"])
def test_draw_stack_refuses_different_event_images(events, field):
    """The field in the error identifies a mismatched image shape or axes."""
    first = stack_exposure_data(events[:1])
    second = stack_exposure_data(events) if field == "shape" else events[0]
    image_field = "data shape" if field == "shape" else field
    with pytest.raises(ValueError, match=rf"item 1: image\.{image_field}"):
        stack_exposure_data((first, second), axis="draw")


def test_stack_refuses_wrong_axis_and_unequal_draw_exposures(events):
    """The fixed axis order and draw exposure equality are structural rules."""
    with pytest.raises(ValueError, match=r"item 1.*exposure"):
        stack_exposure_data(events, axis="draw")
    draw = stack_exposure_data((events[0],), axis="draw")
    with pytest.raises(ValueError, match=r"item 0.*leading_axes"):
        stack_exposure_data((draw,), axis="event")
    with pytest.raises(ValueError, match=r"item 0.*leading_axes"):
        stack_exposure_data((draw,), axis="draw")
    with pytest.raises(ValueError, match=r"item 0.*axis"):
        stack_exposure_data(events, axis="band")
    with pytest.raises(ValueError, match="items"):
        stack_exposure_data(())


def test_host_records_are_aligned_and_separate(events):
    """The original file names and event count give exact host alignment."""
    first = HostRecords("event", {"file": ("a.fits",)})
    second = HostRecords("event", {"file": ("b.fits",)})
    combined = HostRecords.concatenate((first, second))
    stacked = stack_exposure_data(events)
    first.require_aligned(events[0])
    combined.require_aligned(stacked)
    assert len(combined) == 2
    assert combined.row(0) == {"file": "a.fits"}
    assert combined.row(1) == {"file": "b.fits"}
    with pytest.raises(ValueError, match="event"):
        HostRecords("event", {"file": ("a", "b", "c")}).require_aligned(stacked)
    with pytest.raises(ValueError, match="columns"):
        HostRecords.concatenate((first, HostRecords("event", {"target": ("b",)})))
    assert {field.name for field in fields(ExposureData)} == {
        "image",
        "variance",
        "valid",
        "flags",
        "exposure",
        "flag_names",
        "invalid_flags",
        "built_at",
    }
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(stacked))
