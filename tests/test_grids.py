"""Coordinate, handedness, PyTree, and image resampling contracts."""

from dataclasses import FrozenInstanceError, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import arcsec2rad
from hwoutils.numerics import eps
from hwoutils.precision import complex_of
from hwoutils.testing import precision_pair, ulps

from optixstuff.grids import (
    Frame,
    GridArray,
    Image,
    Measure,
    PixelGrid,
    SkyOrientation,
    footprint_overlap,
    integrate_footprints,
    resample,
)


@pytest.fixture(autouse=True, scope="module")
def _clear_caches():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def frame():
    return Frame(550, 7.2, "circumscribed")


def test_worked_pixel_and_corner_count(frame):
    """Fails on a row/column swap or a half-pixel center convention error.

    Binary-exact quarter-pixel spacing needs exact equality, no tolerance.
    """
    grid = PixelGrid.focal(256, 0.25, frame)
    coords = grid.coordinates()
    assert coords.x.shape == coords.y.shape == (256, 256)
    assert (coords.x[127, 128], coords.y[127, 128]) == (0.125, -0.125)
    assert (grid.center_x, grid.center_y) == (127.5, 127.5)
    header = PixelGrid.from_corner_count((256, 256), 128, 128, grid.basis, "lod", frame)
    assert header == grid
    point = grid.unit_to_pixel(0.125, -0.125)
    assert (point.x, point.y) == (128, 127)
    pupil = PixelGrid.pupil(256)
    assert pupil.step == 1 / 256
    assert pupil.unit == "pupil_diameter"


@pytest.mark.parametrize(
    ("shape", "center", "basis", "extent"),
    [
        ((4, 4), (1.5, 1.5), ((1, 0), (0, 1)), (-2, 2, -2, 2)),
        ((3, 5), (2, 1), ((2, 0), (0, 1)), (-5, 5, -1.5, 1.5)),
        ((3, 4), (0.25, 2), ((2, 0), (0, -1)), (-1.5, 6.5, 2.5, -0.5)),
    ],
)
def test_raster_corner_and_extent(shape, center, basis, extent):
    """Fails if raster edges drop a half pixel, axis length, center or sign."""
    grid = PixelGrid(shape, *center, basis, "arcsec")
    x0, x1, y0, y1 = extent
    assert grid.extent() == extent
    np.testing.assert_array_equal(
        grid.corners(), [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    )


@pytest.mark.parametrize(
    ("basis", "corners", "x", "y"),
    [
        (
            ((2, 1), (0, 3)),
            [(-5.5, -1.5), (0.5, -1.5), (2.5, 4.5), (-3.5, 4.5)],
            [[-4, -2, 0], [-3, -1, 1]],
            [[0, 0, 0], [3, 3, 3]],
        ),
        (
            ((0, -1), (2, 0)),
            [(0.5, -5), (0.5, 1), (-1.5, 1), (-1.5, -5)],
            [[0, 0, 0], [-1, -1, -1]],
            [[-4, -2, 0], [-4, -2, 0]],
        ),
    ],
)
def test_affine_grid_corners_and_radius(basis, corners, x, y):
    """Fails if an axis is swapped or shear or rotation is silently lost.

    Floating-point basis. The integer affine coordinates are exact; allow four
    operations for squaring, adding and taking the norm.
    """
    grid = PixelGrid((2, 3), 2, 0, basis, "arcsec")
    with pytest.raises(ValueError, match="rotated or sheared"):
        grid.extent()
    np.testing.assert_array_equal(grid.corners(), corners)
    np.testing.assert_allclose(
        grid.radius(), np.hypot(x, y), rtol=float(ulps(4, 1.0)), atol=0
    )


def test_frame_conversion_and_native_geometry(frame):
    """Fails if lambda/D uses wrong wavelength units or a duplicated constant.

    Primitives: 1 nm=1e-9 m, pi rad=180 degrees, 3600 arcsec/degree.

    Floating-point basis. Allow six operations for converting wavelength,
    diameter and radians into arcseconds and for the inverse conversion.
    The traced scale uses the input dtype. Concrete grid geometry and the
    round trip use host binary64.
    """
    expected = 550e-9 / 7.2 * (180 / np.pi) * 3600
    actual = jax.jit(frame.lod_to_arcsec)(jnp.array(550.0))
    np.testing.assert_allclose(actual, expected, rtol=float(ulps(6, 1.0)), atol=0)
    if actual.dtype == np.float64:
        precision_pair(
            frame.lod_to_arcsec,
            (jnp.asarray(550.0),),
            bound=float(ulps(6, np.float32(1.0))),
        )
    grid = PixelGrid.focal(8, 0.25, frame)
    angular = grid.to_arcsec(550)
    assert angular.unit == "arcsec" and angular.frame == frame
    np.testing.assert_allclose(
        angular.step, expected * 0.25, rtol=6 * eps(np.float64), atol=0
    )
    offaxis = PixelGrid(
        (3, 4),
        0.25,
        1.5,
        ((0.25, 0.125), (0, 0.5)),
        "lod",
        frame,
        SkyOrientation(12, -1),
    )
    restored = offaxis.to_arcsec(600).to_lod(600, frame)
    assert replace(restored, basis=offaxis.basis) == offaxis
    np.testing.assert_allclose(
        restored.basis, offaxis.basis, rtol=6 * eps(np.float64), atol=0
    )
    assert grid.is_native(550) and not grid.is_native(600)
    assert not angular.is_native(550)
    with pytest.raises(ValueError, match="convention"):
        Frame(550, 7.2, "unknown")
    with pytest.raises(ValueError, match="positive"):
        Frame(550, 0, "inscribed")
    assert frame.coordinate_model == "paraxial_entrance_tilt"
    assert frame == Frame(550, 7.2, "circumscribed", "paraxial_entrance_tilt")
    with pytest.raises(ValueError, match="coordinate_model"):
        replace(frame, coordinate_model="tangent")


def test_sky_orientation_parity_anchor_and_roundtrip():
    """Fails on an east sign error, reversed PA, or use of inverse as forward.

    North-up east-left anchors use exact zeros/ones.

    Floating-point basis. Allow 12 operations at vector scale for two
    rotations, their trigonometric values, products and pairwise sums. The
    rotation products run in the lane dtype on host constants.
    """
    orientation = SkyOrientation(0, -1)
    assert orientation.sky_to_detector(1, 0) == (-1, 0)
    assert orientation.sky_to_detector(0, 1) == (0, 1)
    for parity in (-1, 1):
        orientation = SkyOrientation(31, parity)
        east = orientation.sky_to_detector(1, 0)
        north = orientation.sky_to_detector(0, 1)
        determinant = east.x * north.y - north.x * east.y
        np.testing.assert_allclose(
            determinant, parity, rtol=0, atol=float(ulps(12, 1.0))
        )
        detector = orientation.sky_to_detector(2.3, -0.7)
        sky = orientation.detector_to_sky(*detector)
        np.testing.assert_allclose(
            [sky.east, sky.north], [2.3, -0.7], rtol=0, atol=float(ulps(12, 2.3))
        )
        aligned = orientation.sky_to_detector(
            np.sin(np.deg2rad(31)), np.cos(np.deg2rad(31))
        )
        np.testing.assert_allclose(aligned, [0, 1], rtol=0, atol=float(ulps(12, 1.0)))


def test_equal_grids_share_trace_and_arrays_transform(frame):
    """Fails if grid identity enters the trace key or data becomes static.

    Two independently allocated equal grids must cause exactly one trace.
    All data arithmetic here is binary exact; no numeric tolerance is used.
    This test intentionally builds its own jit and vmap programs to count traces.
    """
    grid = PixelGrid.regular((4, 5), 0.25, "lod", frame=frame)
    equal = PixelGrid([4, 5], np.float64(2), 1.5, np.array(grid.basis), "lod", frame)
    assert grid == equal and hash(grid) == hash(equal)
    assert grid != replace(grid, center_x=2.5)
    with pytest.raises(FrozenInstanceError):
        grid.center_x = 0
    traces = []

    @jax.jit
    def twice(image):
        traces.append(None)
        return jax.tree.map(lambda a: 2 * a, image)

    data = jnp.arange(20.0).reshape(4, 5)
    for geometry in (grid, equal):
        image = Image(data, geometry, (), Measure.integral(), "phot/s")
        result = twice(image)
        np.testing.assert_array_equal(result.data, 2 * data)
        assert result.grid == geometry
    assert len(traces) == 1
    stacked = Image(
        jnp.stack((data, 2 * data)), grid, ("wavelength",), Measure.integral(), "phot/s"
    )
    mapped = jax.vmap(lambda image: jax.tree.map(lambda a: a + 1, image))(stacked)
    np.testing.assert_array_equal(mapped.data, stacked.data + 1)
    assert mapped.grid == grid and mapped.leading_axes == ("wavelength",)
    sampled = jax.vmap(lambda image: resample(image, grid))(stacked)
    np.testing.assert_array_equal(sampled.data, stacked.data)
    assert sampled.leading_axes == stacked.leading_axes
    raw = GridArray(data, grid, ())
    np.testing.assert_array_equal(jax.jit(lambda x: x)(raw).data, data)


def test_validation_rejects_inconsistent_shapes_and_geometry(frame):
    """Fails if bad geometry or metadata reaches traced kernels; no tolerance."""
    grid = PixelGrid.focal(4, 0.25, frame)
    with pytest.raises(ValueError, match="spatial shape"):
        GridArray(jnp.zeros((4, 3)), grid, ())
    with pytest.raises(ValueError, match="leading_axes count"):
        Image(jnp.zeros((2, 4, 4)), grid, (), Measure.integral(), "phot/s")
    with pytest.raises(ValueError, match="measure"):
        Image(jnp.zeros((4, 4)), grid, (), "invalid", "phot/s")
    with pytest.raises(TypeError, match="outside"):
        Image(jnp.zeros((4, 4)), grid, (), Measure.integral(), "phot/s", outside="zero")
    with pytest.raises(ValueError, match="requires a frame"):
        PixelGrid.regular((4, 4), 1, "lod")
    with pytest.raises(ValueError, match="nonsingular"):
        replace(grid, basis=((1, 2), (2, 4)))
    with pytest.raises(ValueError, match="step"):
        _ = replace(grid, basis=((1, 0.1), (0, 1))).step
    with pytest.raises(ValueError, match="step"):
        _ = replace(grid, basis=((1, 0), (0, 2))).step


def test_cell_solid_angle_and_measures():
    """Fails if radiance integrates over square arcseconds or scales twice.

    Floating-point basis. The host angular constant is checked in float64.
    Allow eight lane operations for the determinant, angular scaling and
    density conversion, and 12 for the additional footprint sum.
    """
    # One full turn is 2*pi radians, 360 degrees, or 360*3600 arcseconds.
    radians = 2 * np.pi / (360 * 3600)
    np.testing.assert_allclose(arcsec2rad, radians, rtol=3e-16)
    grid = PixelGrid((2, 4), 1.5, 0.5, ((0.5, 0.25), (0, 2)), "arcsec")
    radiance = Image(
        jnp.full(grid.shape, 8.0), grid, (), Measure.density("sr"), "ph s^-1 m^-2 nm^-1"
    )
    expected = 8 * radians**2
    flux = radiance.integrated()
    np.testing.assert_allclose(flux.data, expected, rtol=float(ulps(8, 1.0)), atol=0)
    np.testing.assert_allclose(
        flux.density("sr").data, 8.0, rtol=float(ulps(8, 1.0)), atol=0
    )
    np.testing.assert_allclose(
        radiance.density("grid").data, expected, rtol=float(ulps(8, 1.0)), atol=0
    )
    result = integrate_footprints(radiance, footprint_overlap(grid, grid))
    np.testing.assert_allclose(
        result.data.sum(), 8 * expected, rtol=float(ulps(12, 1.0)), atol=0
    )
    assert result.quantity == radiance.quantity
    with pytest.raises(ValueError, match="arcsec"):
        PixelGrid.pupil(2).cell_area("sr")
    with pytest.raises(ValueError, match="numerator"):
        replace(radiance, quantity="ph s^-1 m^-2 nm^-1 sr^-1")

    source = PixelGrid.regular((4, 4), 1.0, "pixel")
    target = PixelGrid.regular((2, 2), 2.0, "pixel")
    transmission = Image(
        jnp.full(source.shape, 0.375), source, (), Measure.intensive(), "transmission"
    )
    np.testing.assert_array_equal(resample(transmission, target).data, 0.375)
    with pytest.raises(ValueError, match="intensive"):
        transmission.integrated()
    with pytest.raises(ValueError, match="intensive"):
        integrate_footprints(transmission, footprint_overlap(source, target))


def test_resample_identity_with_leading_axes():
    """Fails if identity shifts by half a pixel or loses image metadata.

    rtol=0, atol=8e-15 allows the float64 floor at values below 20.
    The three local reconstructions each need a distinct stencil program.
    """
    grid = PixelGrid.regular((4, 5), 0.25, "arcsec")
    data = jnp.arange(40.0).reshape(2, 4, 5) / 2
    image = Image(data, grid, ("time",), Measure.integral(), "phot/s")
    for reconstruction in ("nearest", "linear", "cubic"):
        actual = jax.jit(
            lambda im, mode=reconstruction: resample(im, grid, reconstruction=mode)
        )(image)
        np.testing.assert_allclose(actual.data, data, rtol=0, atol=8e-15)
        assert actual.grid == grid
        assert actual.measure == image.measure and actual.quantity == image.quantity
        assert actual.leading_axes == image.leading_axes


def test_resample_interior_flux_on_integer_regrid():
    """Fails if regridding drops or duplicates integrated source flux.

    A compact source must retain its sum on integer reductions/refinements.
    Per-unit-area constants retain amplitude.

    Floating-point basis. Allow 24 operations for linear weights, area scaling
    and pairwise sums across the source and target pixels.
    """
    source = PixelGrid.regular((12, 12), 1, "arcsec")
    coarse = PixelGrid.regular((6, 6), 2, "arcsec")
    coarser = PixelGrid.regular((4, 4), 3, "arcsec")
    fine = PixelGrid.regular((24, 24), 0.5, "arcsec")
    data = jnp.zeros(source.shape).at[4:8, 4:8].set(jnp.arange(16.0).reshape(4, 4))
    image = Image(data, source, (), Measure.integral(), "phot/s")
    for target in (coarse, coarser, fine):
        actual = resample(image, target)
        np.testing.assert_allclose(
            actual.data.sum(), data.sum(), rtol=float(ulps(24, 1.0)), atol=0
        )
    # Refinement samples a density, so a flat patch stays flat with no gaps.
    flat = Image(jnp.ones(source.shape) * 4, source, (), Measure.integral(), "phot/s")
    np.testing.assert_allclose(
        resample(flat, fine).data[4:20, 4:20], 1, rtol=float(ulps(24, 1.0)), atol=0
    )
    uniform = Image(
        jnp.ones(source.shape) * 7, source, (), Measure.density("grid"), "Jy"
    )
    np.testing.assert_allclose(
        resample(uniform, coarse).data, 7, rtol=float(ulps(24, 1.0)), atol=0
    )


def test_resample_half_pixel_center_shift_matches_linear_ramp():
    """Fails if center displacement has the wrong sign or uses corner counts.

    Linear interpolation reproduces an affine ramp exactly away from zero
    padding; atol=2e-14 covers float64 multiply/add in values up to 40.
    """
    grid = PixelGrid.regular((8, 9), 1, "arcsec")
    x, y = grid.coordinates()
    image = Image(10 + 2 * x - 3 * y, grid, (), Measure.density("grid"), "Jy")
    shifted = replace(grid, center_x=grid.center_x + 0.5, center_y=grid.center_y - 0.5)
    tx, ty = shifted.coordinates()
    for measure in (Measure.integral(), Measure.density("grid")):
        actual = resample(replace(image, measure=measure), shifted)
        np.testing.assert_allclose(
            actual.data[1:-1, 1:-1],
            (10 + 2 * tx - 3 * ty)[1:-1, 1:-1],
            rtol=0,
            atol=2e-14,
        )


def test_resample_composes_anisotropic_rotation_and_orientation():
    """Fails if basis columns, inverse mapping, or orientation order are swapped.

    An affine sky ramp is exact under linear interpolation.

    Floating-point basis. Allow 48 operations at the ramp scale for both
    affine maps, orientation, matrix inversion and the weighted read. The area
    uses three host binary64 determinant operations. Compact deposition uses
    48 operations including polygon clipping and reduction.
    """
    source = PixelGrid.regular((12, 14), 1, "arcsec", orientation=SkyOrientation(0))
    target = PixelGrid(
        (4, 5),
        1.7,
        1.4,
        ((0.4, -0.3), (0.2, 0.7)),
        "arcsec",
        orientation=SkyOrientation(27),
    )
    coords = source.coordinates()
    sky = source.orientation.detector_to_sky(*coords)
    image = Image(
        10 + 2 * sky.east - 3 * sky.north, source, (), Measure.density("grid"), "Jy"
    )
    target_sky = target.orientation.detector_to_sky(*target.coordinates())
    expected = 10 + 2 * target_sky.east - 3 * target_sky.north
    actual = jax.jit(resample)(image, target)
    np.testing.assert_allclose(
        actual.data, expected, rtol=0, atol=float(ulps(48, jnp.max(jnp.abs(expected))))
    )
    np.testing.assert_allclose(
        target.pixel_area, 0.34, rtol=0, atol=3 * eps(np.float64) * 0.34
    )
    # Refinement (area ratio 0.34): per-pixel flux is the sampled density
    # times the target pixel area, exact for the affine ramp.
    refined = resample(replace(image, measure=Measure.integral()), target)
    np.testing.assert_allclose(
        refined.data,
        0.34 * expected,
        rtol=0,
        atol=float(ulps(48, 0.34 * jnp.max(jnp.abs(expected)))),
    )
    # Coarsening (area ratio 3.06): deposition conserves a compact source.
    coarse = replace(
        target,
        shape=(9, 9),
        center_x=4.0,
        center_y=4.0,
        basis=((1.2, -0.9), (0.6, 2.1)),
    )
    compact = jnp.zeros(source.shape).at[5:7, 6:8].set(1.0)
    flux = Image(compact, source, (), Measure.integral(), "phot/s")
    np.testing.assert_allclose(
        resample(flux, coarse).data.sum(), 4, rtol=0, atol=float(ulps(48, 4.0))
    )


def test_band_limited_periodic_exactness(precision):
    """Fails on phase, axis, pixel-area, or sinc normalization errors.

    Basis: exact sinusoidal point values and analytic rectangular integrals;
    200 epsilon allows DFT phase/reduction rounding in float32 and float64.
    Rectangular odd/even axes, anisotropic spacing, and parity reversal cover
    distinct geometry failures using the same small transform structure.
    """
    source = PixelGrid(
        (7, 8),
        3.2,
        2.6,
        ((0.4, 0), (0, 0.6)),
        "arcsec",
        orientation=SkyOrientation(13, -1),
    )
    target = PixelGrid(
        (11, 12),
        5.7,
        4.5,
        ((0.23, 0), (0, 0.37)),
        "arcsec",
        orientation=SkyOrientation(13, 1),
    )

    def field(grid, width_x=0, width_y=0):
        # Coordinates in the SOURCE detector frame: target parity flips x.
        x, y = map(np.asarray, grid.coordinates())
        if grid.orientation.parity == 1:
            x = -x
        fx, fy = 2 / (8 * 0.4), 1 / (7 * 0.6)
        return (
            2
            + 0.3
            * np.sin(2 * np.pi * (fx * x + fy * y) + 0.2)
            * np.sinc(fx * width_x)
            * np.sinc(fy * width_y)
            + 0.4 * np.cos(2 * np.pi * fy * y - 0.7) * np.sinc(fy * width_y)
        )

    dtype = precision
    for measure in (Measure.density("grid"), Measure.integral()):
        source_area = 0.4 * 0.6 if measure == Measure.integral() else 1
        image = Image(
            jnp.asarray(field(source) * source_area, dtype=dtype),
            source,
            (),
            measure,
            "flux",
            "band_limited",
        )
        # Keep complete boxes inside the footprint for this periodic oracle.
        shifted = replace(source, shape=(5, 6), center_x=2.57, center_y=1.31)
        for grid in (target, shifted):
            expected = field(grid)
            if measure == Measure.integral():
                dx, dy = grid.basis[0][0], grid.basis[1][1]
                expected = field(grid, dx, dy) * dx * dy
            actual = resample(image, grid, pad_factor=1)
            assert actual.data.dtype == dtype
            np.testing.assert_allclose(
                actual.data, expected, rtol=200 * np.finfo(dtype).eps, atol=0
            )


def test_band_limited_metadata_and_overrides():
    """Fails if reconstruction metadata disappears or overrides are ignored.

    Metadata and local-kernel parity are exact. The Fourier path refuses
    rotation and shear, including rotation introduced by orientation with
    diagonal grid bases.

    Floating-point basis. Allow 64 operations at the peak image scale for the
    FFT and Fourier synthesis on the unchanged grid.
    """
    source = PixelGrid.regular((4, 5), 1, "arcsec")
    image = Image(
        jnp.arange(20.0).reshape(4, 5),
        source,
        (),
        Measure.density("grid"),
        "flux",
        "band_limited",
    )
    for transformed in (
        jax.jit(lambda im: im)(image),
        jax.tree.map(lambda d: d * 2, image),
        replace(image, quantity="density"),
    ):
        assert transformed.reconstruction == "band_limited"
    with pytest.raises(ValueError, match="reconstruction"):
        replace(image, reconstruction="unknown")
    with pytest.raises(ValueError, match="reconstruction"):
        resample(image, source, reconstruction="unknown")
    for factor in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="pad_factor"):
            resample(image, source, pad_factor=factor)
    target = replace(source, center_x=2.3)
    for mode in ("nearest", "linear", "cubic"):
        actual = resample(image, target, reconstruction=mode)
        expected = resample(replace(image, reconstruction=mode), target)
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.reconstruction == "band_limited"
    overridden = resample(
        replace(image, reconstruction="linear"), target, reconstruction="band_limited"
    )
    np.testing.assert_array_equal(overridden.data, resample(image, target).data)
    assert overridden.reconstruction == "linear"
    np.testing.assert_allclose(
        resample(
            replace(image, reconstruction="linear"),
            source,
            reconstruction="band_limited",
        ).data,
        image.data,
        rtol=0,
        atol=float(ulps(64, jnp.max(jnp.abs(image.data)))),
    )
    assert resample(image, source).reconstruction == "band_limited"
    # Area decreases, but x coarsens: an area-only rule would be wrong.
    coarse = replace(source, basis=((1.2, 0), (0, 0.5)))
    assert resample(image, coarse).reconstruction == "linear"
    for basis in (((0.8, -0.6), (0.6, 0.8)), ((1, 0.2), (0, 1))):
        rotated = replace(target, basis=basis)
        with pytest.raises(NotImplementedError, match="rotated or sheared"):
            resample(image, rotated)
        for mode in ("nearest", "linear", "cubic"):
            resample(image, rotated, reconstruction=mode)
    oriented = replace(image, grid=replace(source, orientation=SkyOrientation(0)))
    with pytest.raises(NotImplementedError, match="composed"):
        resample(oriented, replace(source, orientation=SkyOrientation(27)))


def test_band_limited_period_flux_and_nyquist(precision):
    """Fails if box integration loses DC flux or mixes real Nyquist axes.

    Basis: with pad_factor=1 the footprint is exactly one Fourier period, whose
    integral is the sample sum to 100 eps. The target contains that footprint.
    Even coarsening conserves this one integral, although
    modes above the new Nyquist alias and must lose the band_limited label.
    Nyquist cosines on an even grid must remain separable, not cos(x+y).
    """
    source = PixelGrid.regular((6, 8), 0.5, "arcsec")
    rng = np.random.default_rng(0)
    dtype = precision
    image = Image(
        jnp.asarray(rng.uniform(size=(2, 6, 8)), dtype=dtype),
        source,
        ("time",),
        Measure.integral(),
        "flux",
        "band_limited",
    )
    for shape, step in (((18, 24), 1 / 3), ((6, 8), 1)):
        target = PixelGrid.regular(shape, step, "arcsec")
        actual = resample(image, target, pad_factor=1)
        np.testing.assert_allclose(
            actual.data.sum(axis=(-2, -1)),
            image.data.sum(axis=(-2, -1)),
            rtol=100 * np.finfo(dtype).eps,
            atol=0,
        )
        assert actual.reconstruction == ("band_limited" if step < 0.5 else "linear")
    grid = PixelGrid.regular((4, 4), 1, "arcsec", center=(0, 0))
    rows, cols = np.indices((4, 4))
    checker = Image(
        (-1.0) ** (rows + cols),
        grid,
        (),
        Measure.density("grid"),
        "flux",
        "band_limited",
    )
    shifted = replace(grid, center_x=-0.2, center_y=-0.3)
    expected = np.cos(np.pi * (cols + 0.2)) * np.cos(np.pi * (rows + 0.3))
    np.testing.assert_allclose(
        resample(checker, shifted, pad_factor=1).data,
        expected,
        rtol=0,
        atol=ulps(18, precision.type(1.0)),
    )


@pytest.mark.float64_only(reason="finite-difference gradient reference")
def test_band_limited_jit_and_transpose_gradient():
    """Fails if tracing changes values or the data derivative is not linear.

    Basis: transpose of independently evaluated impulse responses, 200 eps;
    finite differences additionally check both AD modes at 1e-8 in float64.
    Batch axes, real/complex data and both measures share one small geometry.
    """
    from jax.test_util import check_grads

    source = PixelGrid.regular((3, 4), 0.7, "arcsec")
    target = PixelGrid.regular((5, 6), 0.39, "arcsec", center=(2.2, 1.8))
    rng = np.random.default_rng(0)
    for dtype in (jnp.float32, jnp.float64, jnp.complex128):
        for measure in (Measure.density("grid"), Measure.integral()):
            data = jnp.asarray(rng.normal(size=(2, 3, 4)), dtype=dtype)
            if jnp.iscomplexobj(data):
                data = data + 0.2j * data[::-1]
            image = Image(data, source, ("time",), measure, "flux", "band_limited")

            def apply(values, image=image):
                return resample(replace(image, data=values), target).data

            eager = apply(data)
            tolerance = 200 * np.finfo(data.real.dtype).eps
            np.testing.assert_allclose(
                jax.jit(apply)(data), eager, rtol=tolerance, atol=tolerance
            )
            weights = jnp.asarray(rng.normal(size=eager.shape), dtype=dtype)
            gradient = jax.grad(
                lambda d, weights=weights: jnp.real(jnp.sum(apply(d) * weights))
            )(data)
            impulses = jnp.eye(data.size, dtype=dtype).reshape((-1, *data.shape))
            responses = jax.vmap(apply)(impulses)
            expected = jnp.einsum("ibyx,byx->i", responses, weights).reshape(data.shape)
            np.testing.assert_allclose(
                gradient, expected, rtol=tolerance, atol=tolerance
            )
            if dtype == jnp.float64 and measure == Measure.integral():
                check_grads(
                    apply,
                    (data,),
                    order=1,
                    modes=("fwd", "rev"),
                    eps=1e-4,
                    atol=1e-8,
                    rtol=1e-8,
                )


def _circle_square_areas(grid, radius):
    """Exact circle-square intersections from a signed quadrant primitive."""
    ny, nx = grid.shape
    x = (np.arange(nx + 1) - grid.center_x - 0.5) * grid.step
    y = (np.arange(ny + 1) - grid.center_y - 0.5) * grid.step

    def primitive(x, y):
        sign = np.sign(x) * np.sign(y)
        x, y = np.minimum(abs(x), radius), np.minimum(abs(y), radius)
        split = np.minimum(x, np.sqrt(np.maximum(radius**2 - y**2, 0)))

        def segment(t):
            return (
                t * np.sqrt(np.maximum(radius**2 - t**2, 0))
                + radius**2 * np.arcsin(t / radius)
            ) / 2

        return sign * (y * split + segment(x) - segment(split))

    # Neighboring pixels share exactly the same host corner primitive.
    return np.diff(np.diff(primitive(x[None, :], y[:, None]), axis=0), axis=1)


def test_band_limited_airy_aperture(frame):
    """Fails if Nyquist PSF photometry uses a local interpolation kernel.

    Basis: analytic Airy encircled energy (code verification), plus a bound
    derived from discarded wings and periodic replicas, not observed errors.
    Set u=sqrt(x)*J1(x), q=1-3/(4*x*x). The Bessel ODE gives
    (u'^2+q*u^2)'=q'*u^2>=0 and limiting energy 2/pi, hence
    I(r)<=C/r^3 for r>=1, C=4/pi^3 (a conservative envelope).
    On a padded square of side L, replica sup norm <=88*C/L^3: shell m
    has 8m copies at distance >=L*(m-1/2), and sum m/(m-1/2)^3<=11.
    Parseval and Cauchy-Schwarz bound the aperture error by
    sqrt(A)*h*||discarded samples||_2 + (sqrt(A)*L+A)*replica_bound.

    Each detector covers one full padded period. Its declared reconstruction
    is evaluated on a fine 0.005 lambda/D aperture grid (no further padding),
    with exact circle-square areas. Thus coarse aperture-edge quadrature is
    excluded. The extra midpoint bound is O(h_fine^2): interior Hessian and
    boundary-strip gradient terms, using the reconstructed Fourier coefficient
    absolute sum to bound derivatives (each axis has bandwidth <=1).
    This tests density photometry; exact per-pixel boxes are tested separately.

    The independent host-area reference stays in float64. Allow 32 float64
    operations at aperture area scale for the segment primitive and the
    pairwise pixel sum. The Airy bounds apply to the lane's reconstructed flux.
    """
    from scipy.special import j0, j1

    def airy(x, y):
        z = np.pi * np.hypot(x, y)
        amplitude = np.where(z == 0, 1.0, 2 * j1(z) / np.where(z == 0, 1, z))
        return np.pi / 4 * amplitude**2

    radius = 0.7
    source = PixelGrid.focal(32, 0.5, frame)
    image = Image(
        airy(*map(np.asarray, source.coordinates())),
        source,
        (),
        Measure.density("grid"),
        "fraction",
        "band_limited",
    )
    expected = 1 - j0(np.pi * radius) ** 2 - j1(np.pi * radius) ** 2
    aperture = PixelGrid.focal(282, 0.005, frame)
    weights = _circle_square_areas(aperture, radius)
    area = np.pi * radius**2
    area_tolerance = 32 * np.finfo(weights.dtype).eps * area
    np.testing.assert_allclose(weights.sum(), area, rtol=0, atol=area_tolerance)
    padded = PixelGrid.focal(64, 0.5, frame)
    x, y = map(np.asarray, padded.coordinates())
    wings = np.where((abs(x) >= 8) | (abs(y) >= 8), airy(x, y), 0)
    period = 32
    replicas = 88 * (4 / np.pi**3) / period**3
    truncation_bound = (
        np.sqrt(area) * 0.5 * np.linalg.norm(wings)
        + (np.sqrt(area) * period + area) * replicas
    )
    coefficient_sum = np.abs(np.fft.fft2(np.pad(np.asarray(image.data), 16))).sum()
    coefficient_sum /= 64**2
    quadrature_bound = (
        aperture.step**2
        * coefficient_sum
        * (area * (2 * np.pi) ** 2 / 6 + 8 * np.pi * radius * (2 * np.pi))
    )
    fluxes = []
    for pitch in (0.2, 0.1, 0.05):
        detector = PixelGrid.focal(round(period / pitch), pitch, frame)
        row = []
        for mode in ("band_limited", "linear"):
            sampled = resample(replace(image, reconstruction=mode), detector)
            fine = resample(sampled, aperture, pad_factor=1)
            row.append(float(np.sum(np.asarray(fine.data) * weights)))
        band_error, linear_error = (abs(value - expected) for value in row)
        assert band_error <= truncation_bound + quadrature_bound
        assert band_error * 20 < linear_error
        fluxes.append(row)
    assert np.ptp(np.asarray(fluxes)[:, 0]) <= truncation_bound
    # Report the measured comparison without freezing it as a golden value.
    print(
        f"Airy analytic={expected:.12f}; bound={truncation_bound:.6g}"
        f"+{quadrature_bound:.6g}; pitches/fluxes={fluxes}"
    )


def test_local_kernels_zero_outside_footprint():
    """Fails if local kernels extend a unit constant past the source array.

    Basis: zero-extension stencils have finite support and cannot produce
    distant copies. Linear/cubic can spill just past a half-sample edge; at
    x=-0.75 a unit constant has values 1/4 and 13/64, respectively, not zero.
    The cubic spline is restricted to the footprint and reproduces the
    constant inside it. Binary-exact arithmetic, except the spline interior
    (a dense prefilter solve, 1e-14).
    """
    source = PixelGrid.regular((4, 4), 0.5, "arcsec", center=(0, 0))
    for mode in ("nearest", "linear", "cubic", "cubic_spline"):
        zero = Image(
            jnp.ones(source.shape), source, (), Measure.density("grid"), "flux", mode
        )
        exterior = PixelGrid.regular((12, 12), 0.5, "arcsec", center=(4, 4))
        actual = np.asarray(resample(zero, exterior).data)
        expected = np.zeros((12, 12))
        expected[4:8, 4:8] = 1
        assert np.all(actual[expected == 0] == 0)
        if mode == "cubic_spline":
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        else:
            np.testing.assert_array_equal(actual, expected)
        edge = PixelGrid.regular((1, 1), 0.5, "arcsec", center=(0.75, -1))
        np.testing.assert_array_equal(
            resample(zero, edge).data,
            {"nearest": 0, "linear": 1 / 4, "cubic": 13 / 64, "cubic_spline": 0}[mode],
        )


def test_band_limited_zero_wide_gaussian(frame, precision):
    """Fails if exterior target positions see periodic copies of a PSF.

    Basis: sigma=0.8 lambda/D at (1,0), sampled at 0.5 on 24x24, has its
    nearest edge 6.25 sigma away. The omitted analytic mass is <2.1e-10;
    Fourier truncation adds <1e-7 relative (float64 spike). The requested
    16 float32 eps allowance covers reconstruction and rounding. Both target
    pitches partition the entire source footprint, so their exact box sums
    must agree. Point samples outside and boxes wholly outside are exactly
    zero; outside-centered straddlers must still integrate their overlap.
    Two target shapes exercise refinement and coarsening; 64x64 at 0.85
    spans over two padded periods and used to integrate about nine copies.
    """
    source = PixelGrid.focal(24, 0.5, frame)
    x, y = map(np.asarray, source.coordinates())
    density = np.exp(-((x - 1) ** 2 + y**2) / (2 * 0.8**2))
    dtype = precision
    image = Image(
        jnp.asarray(density * 0.5**2, dtype=dtype),
        source,
        (),
        Measure.integral(),
        "fraction",
        "band_limited",
    )
    for n, pitch in ((64, 0.25), (40, 0.85), (64, 0.85)):
        target = PixelGrid.focal(n, pitch, frame)
        result = np.asarray(resample(image, target).data)
        source_sum = np.asarray(image.data).sum(dtype=np.float64)
        np.testing.assert_allclose(
            result.sum(dtype=np.float64),
            source_sum,
            rtol=16 * np.finfo(np.float32).eps,
            atol=0,
        )
        tx, ty = map(np.asarray, target.coordinates())
        outside_centers = (abs(tx) > 6) | (abs(ty) > 6)
        outside_boxes = (abs(tx) - pitch / 2 >= 6) | (abs(ty) - pitch / 2 >= 6)
        assert outside_boxes.any()
        np.testing.assert_array_equal(result[outside_boxes], 0)
        points = resample(
            replace(image, data=image.data / 0.5**2, measure=Measure.density("grid")),
            target,
        )
        np.testing.assert_array_equal(np.asarray(points.data)[outside_centers], 0)
        print(
            f"Gaussian {dtype} {n}x{n} pitch={pitch}: relative flux error="
            f"{result.sum(dtype=np.float64) / source_sum - 1:.9g}"
        )


def test_band_limited_zero_clipped_boxes(precision):
    """Fails if a straddling box is masked by its center or uses its full width.

    Basis: pad_factor=1 makes a constant density exactly constant inside;
    its flux is density times the rectangular overlap area, in closed form.
    A separable sine additionally checks the phase of a clipped non-DC mode
    using its cosine antiderivative. 200 eps covers small DFT matrix rounding.
    Non-unit, unequal axis widths, both edges, corners, outside centers,
    a box containing the whole source, and complex values share two shapes.
    """
    source = PixelGrid.regular((4, 4), 0.5, "arcsec")
    x, y = map(np.asarray, source.coordinates())
    for dtype in (precision, complex_of(precision)):
        amplitude = 3 + 2j if jnp.issubdtype(dtype, jnp.complexfloating) else 3
        density = jnp.full(source.shape, amplitude, dtype=dtype)
        image = Image(
            density * 0.5**2, source, (), Measure.integral(), "flux", "band_limited"
        )
        # Static geometry must not round exterior points onto an edge in float32.
        exterior = PixelGrid.regular((2, 2), 2 + 2e-9, "arcsec")
        np.testing.assert_array_equal(
            resample(
                replace(image, data=density, measure=Measure.density("grid")), exterior
            ).data,
            0,
        )
        for basis in (((0.75, 0), (0, -1.25)), ((5, 0), (0, 6))):
            target = PixelGrid((5, 5), 2.25, 1.75, basis, "arcsec")
            tx, ty = map(np.asarray, target.coordinates())
            wx, wy = abs(basis[0][0]), abs(basis[1][1])
            ax, bx = np.clip(tx - wx / 2, -1, 1), np.clip(tx + wx / 2, -1, 1)
            ay, by = np.clip(ty - wy / 2, -1, 1), np.clip(ty + wy / 2, -1, 1)
            expected = amplitude * (bx - ax) * (by - ay)
            tolerance = 200 * np.finfo(density.real.dtype).eps
            actual = np.asarray(resample(image, target, pad_factor=1).data)
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            empty = (bx == ax) | (by == ay)
            np.testing.assert_array_equal(actual[empty], 0)
            if wx < 1:
                straddlers = (abs(tx) > 1) & (bx > ax) & (by > ay)
                assert straddlers.any()
                assert np.all(abs(actual[straddlers]) > 0)
            field = amplitude * (1 + 0.2 * np.sin(np.pi * x) * np.sin(np.pi * y))
            sine_x = (np.cos(np.pi * ax) - np.cos(np.pi * bx)) / np.pi
            sine_y = (np.cos(np.pi * ay) - np.cos(np.pi * by)) / np.pi
            expected += 0.2 * amplitude * sine_x * sine_y
            sampled = resample(
                replace(image, data=jnp.asarray(field * 0.5**2, dtype=dtype)),
                target,
                pad_factor=1,
            )
            np.testing.assert_allclose(
                sampled.data, expected, rtol=tolerance, atol=tolerance
            )


@pytest.mark.float64_only(reason="finite-difference gradient reference")
@pytest.mark.parametrize(
    ("measure", "reconstruction"),
    [
        (Measure.integral(), "band_limited"),
        (Measure.intensive(), "band_limited"),
        (Measure.intensive(), "linear"),
        (Measure.integral(), "linear"),
        (Measure.intensive(), "cubic_spline"),
        (Measure.integral(), "cubic_spline"),
    ],
)
def test_traced_wavelength_resample_matches_host_and_differentiates(
    frame, measure, reconstruction
):
    """Fails if a traced wavelength changes resample values or its gradient.

    A target at lambda/D scale built from a traced wavelength must match the
    host-built grid to roundoff, compile once for any value, and have a
    wavelength derivative equal to a central difference.
    """
    source = PixelGrid.regular((17, 17), 0.3, "lod", frame=frame)
    detector = PixelGrid.regular((12, 12), 0.01, "arcsec")
    data = np.random.default_rng(3).random((17, 17))
    image = Image(data, source, (), measure, "x", reconstruction)

    def render(wavelength):
        return resample(image, detector.to_lod(wavelength, frame)).data

    for wavelength in (500.0, 640.0):
        np.testing.assert_allclose(
            jax.jit(render)(wavelength), render(wavelength), rtol=1e-12, atol=1e-14
        )

    def scalar(wavelength):
        return jnp.sum(render(wavelength) ** 2)

    h = 1e-3
    fd = (scalar(600.0 + h) - scalar(600.0 - h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(scalar)(600.0), fd, rtol=1e-6)


def test_traced_roll_orientation_matches_host_values():
    """Fails if a traced position angle disagrees with the host conversion.

    Floating-point basis. Allow 12 operations at input vector scale for angle
    conversion, two trigonometric values and the two rotations.
    """
    east, north = jnp.array([0.3, -1.2]), jnp.array([0.7, 0.4])
    for parity in (-1, 1):
        host = SkyOrientation(37.0, parity)
        traced = jax.jit(
            lambda angle, parity=parity: tuple(
                SkyOrientation(angle, parity).sky_to_detector(east, north)
            )
        )(37.0)
        np.testing.assert_allclose(
            traced,
            host.sky_to_detector(east, north),
            rtol=0,
            atol=float(ulps(12, np.max(np.abs([east, north])))),
        )
        back = jax.jit(
            lambda x, y, angle, parity=parity: tuple(
                SkyOrientation(angle, parity).detector_to_sky(x, y)
            )
        )(*host.sky_to_detector(east, north), 37.0)
        np.testing.assert_allclose(
            back,
            (east, north),
            rtol=0,
            atol=float(ulps(12, np.max(np.abs([east, north])))),
        )
    with pytest.raises(ValueError, match="finite"):
        SkyOrientation(float("nan"))


def test_coverage_requires_whole_footprint_and_margin_matches_resample():
    """Fails if a straddling pixel counts as covered or a margin is too small.

    A target shifted by a quarter pixel leaves its edge pixel straddling the
    source edge, so coverage is False there. Wherever coverage with the
    reconstruction margin is True, resampling a field of ones returns 1
    for the three local reconstructions, on both a shifted and a refined
    grid. Band-limited interpolation of a constant over a zero-padded array
    rings with an algebraically decaying tail (about 1e-2 eight pixels in), so
    no finite margin makes it exact; its margin is the reach used to mask the
    edge band, and the test asserts only that the band is excluded.

    Floating-point basis. Allow 32 operations at unit scale for the affine
    map, cubic weights in both axes and accumulation. Coverage is exact.
    """
    from optixstuff.grids import coverage, reconstruction_margin

    source = PixelGrid.regular((16, 16), 1.0, "arcsec")
    ones = Image(jnp.ones(source.shape), source, (), Measure.intensive(), "1")
    shifted = PixelGrid(
        (20, 20), 9.5 + 0.25, 9.5 - 0.25, ((1.0, 0.0), (0.0, 1.0)), "arcsec"
    )
    inside = coverage(source, shifted)
    assert inside.dtype == bool and inside.shape == shifted.shape
    # The target spans source pixel offsets [-0.25 - 10, ...]: its columns
    # 0..2 and the last straddle or leave the source footprint.
    assert not inside[:, 0].any() and not inside[0, :].any()
    assert inside[3:-4, 3:-4].all()
    assert coverage(source, source).all()
    assert not coverage(source, source, 0.5)[0].any()
    refined = PixelGrid.regular((32, 32), 0.5, "arcsec")
    for target in (shifted, refined):
        for name in ("nearest", "linear", "cubic", "cubic_spline", "band_limited"):
            mask = coverage(source, target, reconstruction_margin(name))
            assert mask.any()
            actual = resample(ones, target, reconstruction=name).data
            if name == "band_limited":
                assert not mask[0].any() and not mask[:, 0].any()
                continue
            np.testing.assert_allclose(
                actual[mask], 1.0, rtol=0, atol=float(ulps(32, 1.0))
            )
    assert [
        reconstruction_margin(n)
        for n in ("nearest", "linear", "cubic", "cubic_spline", "band_limited")
    ] == [0.0, 0.5, 2.0, 2.0, 2.0]
    with pytest.raises(ValueError):
        reconstruction_margin("sinc")


# The read margin of `coverage`: a product cropped to a centred window must read
# the same as the wide product at every pixel `coverage` calls covered, and must
# read exactly zero wholly beyond the window grown by that margin.

_READ_FRAME = Frame(500.0, 6.0, "circumscribed")
_HALF_SUPPORT = {"nearest": 0.5, "linear": 1.0, "cubic": 2.0}
_SAMPLING_MARGIN = {"nearest": 0.0, "linear": 0.5, "cubic": 2.0}


def _smooth_field(n_wide, n_crop):
    """Smooth field on a wide grid and its centred crop, at unit cells."""
    yy, xx = np.indices((n_wide, n_wide)) - (n_wide - 1) / 2
    wide = 1.0 + 0.05 * xx + 0.002 * yy**2 + 0.3 * np.sin(0.9 * xx) * np.cos(0.7 * yy)
    offset = (n_wide - n_crop) // 2
    return wide, wide[offset : offset + n_crop, offset : offset + n_crop]


def _expected_margins(rec, width, angle=0.0, deposit=True):
    """Margin per source axis from the pixel's extent along it, as derived."""
    sampling = _SAMPLING_MARGIN[rec]
    if not deposit:
        return sampling, sampling
    extent = tuple(
        w * (abs(np.cos(np.radians(angle))) + abs(np.sin(np.radians(angle))))
        if angle
        else w
        for w in width
    )
    return tuple(max(sampling, (_HALF_SUPPORT[rec] - 0.5) * g - 0.5) for g in extent)


def _read_case(
    rec, width, shift=0.0, angle=None, n_wide=160, n_crop=48, kind="integral"
):
    """Resample wide and cropped products onto one detector; return the pieces."""
    from optixstuff.grids import coverage

    orientation = None if angle is None else SkyOrientation(0.0)
    wide_grid = replace(
        PixelGrid.focal(n_wide, 1.0, _READ_FRAME), orientation=orientation
    )
    crop_grid = replace(
        PixelGrid.focal(n_crop, 1.0, _READ_FRAME), orientation=orientation
    )
    wx, wy = width
    shape = (int(np.ceil(100 / wy)), int(np.ceil(100 / wx)))
    detector = PixelGrid(
        shape,
        (shape[1] - 1) / 2 + shift,
        (shape[0] - 1) / 2 - 0.61 * shift,
        ((wx, 0.0), (0.0, wy)),
        "lod",
        _READ_FRAME,
        None if angle is None else SkyOrientation(angle),
    )
    wide, crop = _smooth_field(n_wide, n_crop)
    measure = Measure.integral() if kind == "integral" else Measure.density("grid")

    def read(data, grid):
        image = Image(jnp.asarray(data), grid, (), measure, "x", reconstruction=rec)
        return np.asarray(resample(image, detector).data)

    full, cut = read(wide, wide_grid), read(crop, crop_grid)
    covered = np.asarray(
        coverage(crop_grid, detector, reconstruction=rec, measure=measure)
    )
    from optixstuff.grids import _pixel_map

    mapping = np.asarray(_pixel_map(crop_grid, detector), float)
    rows, columns = np.indices(shape)
    dx, dy = columns - detector.center_x, rows - detector.center_y
    xs, ys = [], []
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            xs.append(
                mapping[0, 0] * (dx + a) + mapping[0, 1] * (dy + b) + crop_grid.center_x
            )
            ys.append(
                mapping[1, 0] * (dx + a) + mapping[1, 1] * (dy + b) + crop_grid.center_y
            )
    return full, cut, covered, np.array(xs), np.array(ys), crop_grid, detector


def _assert_reads_agree(rec, width, margins, **case):
    full, cut, covered, xs, ys, crop_grid, _ = _read_case(rec, width, **case)
    scale = np.abs(full).max()
    assert covered.any()
    # Same linear combination of the same samples wherever no edge is read.
    assert np.abs(cut - full)[covered].max() <= ulps(64, scale)
    n = crop_grid.shape[0]
    mx, my = margins
    outside = (
        (xs.max(0) <= -0.5 - mx)
        | (xs.min(0) >= n - 0.5 + mx)
        | (ys.max(0) <= -0.5 - my)
        | (ys.min(0) >= n - 0.5 + my)
    )
    assert outside.any()
    assert np.all(cut[outside] == 0.0)


@pytest.mark.parametrize("shift", [0.0, 0.37])
@pytest.mark.parametrize("rec", ["linear", "cubic"])
@pytest.mark.parametrize("width", [0.25, 1.0, 2.0, 2.5, 4.0])
def test_coverage_read_margin_aligned_pixels(width, rec, shift):
    """Fails if the margin is the sampling reach when a read deposits.

    Truth is the resample of the wide product, which reads no edge; the two
    reads are the same linear combination of the same samples wherever no edge
    is read, so they agree to rounding (64 eps of the field maximum, measured
    0.0 to 3e-16). With the sampling margin the disagreement reaches 3.8e-2.
    """
    margins = _expected_margins(rec, (width, width), deposit=width >= 1.0)
    _assert_reads_agree(rec, (width, width), margins, shift=shift)


@pytest.mark.parametrize("rec", ["linear", "cubic"])
def test_coverage_read_margin_sampling_branch_equals_reconstruction_margin(rec):
    """Fails if a read that samples gets any margin but the reconstruction's."""
    from optixstuff.grids import coverage, reconstruction_margin

    _, _, covered, _, _, crop_grid, detector = _read_case(rec, (0.25, 0.25))
    expected = coverage(crop_grid, detector, reconstruction_margin(rec))
    assert np.array_equal(covered, expected)


@pytest.mark.parametrize(
    ("rec", "width", "n_wide", "n_crop"),
    [
        ("linear", 2.0, 160, 48),
        ("linear", 2.05, 160, 48),
        ("linear", 2.05, 161, 49),
        ("cubic", 5 / 3, 160, 48),
        ("cubic", 1.7, 160, 48),
        ("cubic", 1.7, 161, 49),
    ],
)
def test_coverage_read_margin_thresholds(rec, width, n_wide, n_crop):
    """Fails at the pixel size where the deposit reach overtakes the sampling reach."""
    margins = _expected_margins(rec, (width, width))
    _assert_reads_agree(
        rec, (width, width), margins, shift=0.13, n_wide=n_wide, n_crop=n_crop
    )


@pytest.mark.parametrize("rec", ["linear", "cubic"])
@pytest.mark.parametrize(
    ("width", "deposit"), [((3.0, 0.5), True), ((2.5, 0.3), False)]
)
def test_coverage_read_margin_non_square_pixels(width, deposit, rec):
    """Fails if the branch or margin ignores that the pixel is not square.

    The 3 by 0.5 pixel (area 1.5) deposits and has a long axis of 3 cells; the
    2.5 by 0.3 pixel (area 0.75) samples.
    """
    margins = _expected_margins(rec, width, deposit=deposit)
    _assert_reads_agree(rec, width, margins, shift=0.37)


@pytest.mark.parametrize("rec", ["linear", "cubic"])
@pytest.mark.parametrize("width", [0.5, 2.5, 4.0])
@pytest.mark.parametrize("angle", [30.0, 45.0])
def test_coverage_read_margin_rotated_detector(angle, width, rec):
    """Fails if the margin uses one axis of a rotated pixel.

    The bounding extent of the rotated pixel along each product axis is
    ``width (|cos| + |sin|)``. The aligned formula under rotation misses up to
    9.2e-4 (linear) and 1.3e-4 (cubic) at covered pixels.
    """
    margins = _expected_margins(rec, (width, width), angle=angle, deposit=width >= 1.0)
    _assert_reads_agree(rec, (width, width), margins, shift=0.37, angle=angle)


def test_coverage_read_margin_nearest_reaches_nothing():
    """Fails if nearest, which reads the cell it falls in, gains a margin."""
    _assert_reads_agree("nearest", (2.5, 2.5), (0.0, 0.0), shift=0.13)


@pytest.mark.parametrize("rec", ["linear", "cubic"])
@pytest.mark.parametrize("width", [2.5, 4.0])
def test_coverage_read_margin_density_is_the_sampling_margin(width, rec):
    """Fails if the margin ignores the measure.

    A density is always sampled at pixel centers, so a coarse pixel keeps the
    reconstruction margin and is not shrunk by the deposit reach.
    """
    from optixstuff.grids import coverage, reconstruction_margin

    full, cut, covered, _, _, crop_grid, detector = _read_case(
        rec, (width, width), shift=0.13, kind="density"
    )
    sampling = coverage(crop_grid, detector, reconstruction_margin(rec))
    assert np.array_equal(covered, sampling)
    assert np.abs(cut - full)[covered].max() <= ulps(64, np.abs(full).max())


def test_coverage_read_margin_traced_basis_matches_host():
    """Fails if the branch is a host `if` on a traced pixel area."""
    from optixstuff.grids import coverage

    source = PixelGrid.focal(48, 1.0, _READ_FRAME)

    def detector(scale):
        return PixelGrid(
            (40, 40),
            19.5 + 0.13,
            19.5 - 0.08,
            ((scale, 0.0), (0.0, scale)),
            "lod",
            _READ_FRAME,
        )

    for rec in ("linear", "cubic", "band_limited"):
        for scale in (0.5, 2.5):
            host = coverage(
                source, detector(scale), reconstruction=rec, measure=Measure.integral()
            )
            traced = jax.jit(
                lambda s, rec=rec: coverage(
                    source, detector(s), reconstruction=rec, measure=Measure.integral()
                )
            )(scale)
            assert np.array_equal(np.asarray(traced), host)


def test_coverage_read_keywords_validate():
    """Fails unless the two keywords come together and replace the margin."""
    from optixstuff.grids import coverage

    grid = PixelGrid.focal(8, 1.0, _READ_FRAME)
    with pytest.raises(ValueError, match="together"):
        coverage(grid, grid, reconstruction="linear")
    with pytest.raises(ValueError, match="together"):
        coverage(grid, grid, measure=Measure.integral())
    with pytest.raises(ValueError, match="margin"):
        coverage(grid, grid, 0.5, reconstruction="linear", measure=Measure.integral())


def _d_mask(n):
    """A data mask with no mirror symmetry, on n unit cells.

    An off-center disk cut by a vertical chord and a horizontal one, so a mask
    read flipped along either axis disagrees with it.
    """
    yy, xx = np.indices((n, n)) - (n - 1) / 2
    disk = np.hypot(xx - 0.05 * n, yy + 0.07 * n) <= 0.42 * n
    return disk & (xx >= -0.2 * n) & (yy <= 0.25 * n)


@pytest.mark.parametrize("rec", ["linear", "cubic"])
@pytest.mark.parametrize("width", [0.25, 0.7, 2.5, 4.0])
def test_coverage_with_a_data_mask_reads_the_unmasked_field(width, rec):
    """Fails if a masked product is called covered where its read reaches no data.

    A smooth field and the same field zero-filled off a D-shaped mask, on 48
    unit cells, read onto a detector of 0.25, 0.7, 2.5 and 4 cells per pixel
    (offset 0.37 pixel), linear and cubic. Truth: the read of the unmasked
    field. Wherever ``coverage(valid=...)`` is True the two reads are equal,
    exactly: the same samples through the same read, where every masked cell
    enters with weight exactly zero (measured 0.0). Every covered pixel is also covered
    without the mask, and at least one pixel the unmasked ``coverage`` admits
    is refused.
    """
    from optixstuff.grids import coverage

    n = 48
    grid = PixelGrid.focal(n, 1.0, _READ_FRAME)
    _, field = _smooth_field(n, n)
    mask = _d_mask(n)
    shape = (int(np.ceil(60 / width)),) * 2
    detector = PixelGrid(
        shape,
        (shape[1] - 1) / 2 + 0.37,
        (shape[0] - 1) / 2 - 0.23,
        ((width, 0.0), (0.0, width)),
        "lod",
        _READ_FRAME,
    )

    def read(data):
        image = Image(jnp.asarray(data), grid, (), Measure.integral(), "x", rec)
        return np.asarray(resample(image, detector).data)

    keywords = dict(reconstruction=rec, measure=Measure.integral())
    covered = np.asarray(coverage(grid, detector, valid=mask, **keywords))
    unmasked = np.asarray(coverage(grid, detector, **keywords))
    assert covered.any() and (unmasked & ~covered).any()
    assert not (covered & ~unmasked).any()
    full, masked = read(field), read(np.where(mask, field, 0.0))
    np.testing.assert_array_equal(masked[covered], full[covered])


def test_coverage_data_mask_contract():
    """Fails if the mask changes coverage without missing data or admits a bad read.

    Truth: the stated rule, exact. ``valid=None`` and an all-True mask give the
    unmasked coverage; one missing cell refuses the detector pixel over it and
    every pixel whose grown footprint reaches it, and no other; a JAX mask
    under ``jit`` gives the host answer; ``cubic_spline`` and ``band_limited``
    with a mask, a mask without ``reconstruction``, and a mask of the wrong
    shape raise. A bare speckle field holds data in every cell.
    """
    from optixstuff import AbstractSpeckleField
    from optixstuff.grids import coverage

    n = 24
    grid = PixelGrid.focal(n, 1.0, _READ_FRAME)
    detector = PixelGrid.focal(16, 1.25, _READ_FRAME)
    keywords = dict(reconstruction="linear", measure=Measure.integral())
    plain = np.asarray(coverage(grid, detector, **keywords))
    every = np.ones((n, n), bool)
    np.testing.assert_array_equal(
        coverage(grid, detector, valid=every, **keywords), plain
    )
    np.testing.assert_array_equal(
        coverage(grid, detector, valid=None, **keywords), plain
    )

    hole = every.copy()
    hole[11, 14] = False
    holed = np.asarray(coverage(grid, detector, valid=hole, **keywords))
    # Linear deposit at 1.25 cells per pixel: margin max(0.5, 0.125) cells.
    rows, columns = np.indices(detector.shape)
    x = (columns - detector.center_x) * 1.25 + grid.center_x
    y = (rows - detector.center_y) * 1.25 + grid.center_y
    reach = 1.25 / 2 + 0.5 + 0.5
    touches = (np.abs(x - 14) < reach) & (np.abs(y - 11) < reach)
    assert touches.sum() >= 4
    np.testing.assert_array_equal(holed, plain & ~touches)

    traced = jax.jit(lambda v: coverage(grid, detector, valid=v, **keywords))(
        jnp.asarray(hole)
    )
    np.testing.assert_array_equal(np.asarray(traced), holed)

    for rec in ("cubic_spline", "band_limited"):
        with pytest.raises(ValueError, match="no finite reach"):
            coverage(
                grid,
                detector,
                valid=every,
                reconstruction=rec,
                measure=Measure.integral(),
            )
    with pytest.raises(ValueError, match="requires reconstruction"):
        coverage(grid, detector, valid=every)
    with pytest.raises(ValueError, match="shape"):
        coverage(grid, detector, valid=every[:-1], **keywords)

    native = grid

    class Bare(AbstractSpeckleField):
        grid: PixelGrid = native
        epoch_d: float = eqx.field(static=True, default=0.0)

        def spectral_breaks_nm(self):
            return np.empty(0)

        def realize(self, *, wavelength_nm, elapsed_s=0.0):
            return Image(jnp.zeros((n, n)), native, (), Measure.integral(), "x")

    assert Bare().data_mask() is None
