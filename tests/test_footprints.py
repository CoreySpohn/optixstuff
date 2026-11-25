"""Conservative footprint integration and partial-coverage regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import require_tree
from hwoutils.testing import ulps

from optixstuff.grids import (
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
)


def _integral(data, step=1.0):
    return Image(
        data,
        PixelGrid.regular(data.shape[-2:], step, "pixel"),
        (),
        Measure.integral(),
        "ph",
    )


@pytest.mark.parametrize("row,col", [(0, 0), (3, 4), (4, 4), (5, 5), (11, 11)])
def test_downsampling_integrates_every_source_pixel(row, col):
    """Fails if a 3x reduction discards an impulse or multiplies its flux by 9."""
    source = _integral(jnp.zeros((12, 12)).at[row, col].set(7.0), 0.25)
    target = PixelGrid.regular((4, 4), 0.75, "pixel")
    result = integrate_footprints(source, footprint_overlap(source.grid, target))
    expected = np.zeros((4, 4))
    expected[row // 3, col // 3] = 7
    np.testing.assert_array_equal(result.data, expected)


def test_clipped_rotated_cell_and_disjoint_grid_sequence(precision):
    """Fails if a rotated square is collected by its enclosing rectangle, or
    the sparse overlap holds NumPy or weakly typed weights.

    The two unit squares differ by 45 degrees. Four isosceles triangles
    of leg 1 - 1/sqrt(2) are outside their intersection.
    """
    source = _integral(jnp.array([[1.0]]))
    c = 2**-0.5
    diamond = PixelGrid((1, 1), 0, 0, ((c, -c), (c, c)), "pixel")
    area = 1 - 2 * (1 - c) ** 2
    one = footprint_overlap(source.grid, diamond)
    np.testing.assert_allclose(
        integrate_footprints(source, one).data,
        [[area]],
        atol=ulps(9, precision.type(1.0)),
        rtol=0,
    )
    distant = PixelGrid.regular((1, 2), 1, "pixel", center=(10, 10))
    overlap = footprint_overlap(source.grid, (source.grid, diamond, distant))
    # The sparse triples are strong leaves: indices stay int32.
    require_tree(overlap)
    assert [a.dtype for a in overlap.sparse] == [jnp.int32, jnp.int32, precision]
    got = jax.jit(integrate_footprints)(source, overlap)
    np.testing.assert_allclose(
        got, [1, area, 0, 0], atol=ulps(9, precision.type(1.0)), rtol=0
    )


def test_rebin_partial_coverage():
    """Fails if the integrator renormalizes cropped source pixels."""
    source = _integral(jnp.array([[1.0, 2.0, 3.0, 4.0]]))
    target = PixelGrid.regular((1, 2), 1.5, "pixel")
    result = integrate_footprints(source, footprint_overlap(source.grid, target))
    np.testing.assert_array_equal(result.data, [[2.5, 5.0]])


def test_rebin_noninteger_ratio_and_gradient():
    """Fails if non-integer tiling loses flux or detaches it from gradients, or
    the per-axis overlap holds NumPy weights."""
    source = _integral(jnp.arange(225.0).reshape(15, 15))
    target = PixelGrid.regular((6, 6), 2.5, "pixel")
    overlap = footprint_overlap(source.grid, target)
    require_tree(overlap)
    assert overlap.axes is not None
    result = integrate_footprints(source, overlap)
    assert result.measure == Measure.integral()
    np.testing.assert_array_equal(result.data.sum(), source.data.sum())
    grad = jax.grad(lambda im: integrate_footprints(im, overlap).data.sum())(source)
    np.testing.assert_array_equal(grad.data, 1.0)
    fine = PixelGrid.regular((30, 30), 0.5, "pixel")
    refined = integrate_footprints(source, footprint_overlap(source.grid, fine))
    expected = np.repeat(np.repeat(source.data, 2, 0), 2, 1) / 4
    np.testing.assert_array_equal(refined.data, expected)


def test_footprints_preserve_storage_precision():
    """Fails if storage dtype rounds pixel edges or high precision weights leak."""
    for dtype, width in ((jnp.float16, 4096), (jnp.bfloat16, 512), (jnp.float32, 15)):
        source = _integral(jnp.ones((1, width), dtype=dtype))
        target = PixelGrid.regular((1, 1), 1.0, "pixel")
        result = integrate_footprints(source, footprint_overlap(source.grid, target))
        np.testing.assert_array_equal(result.data, [[1.0]])
        assert result.data.dtype == dtype
