"""Segment layout and limiting hexagonal geometry."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import optixstuff as ox


def _primary(n=3):
    return ox.SegmentedPrimary(
        n_rings=n, segment_point_to_point_m=1.65, segment_gap_m=0.012
    )


class TestSegmentedPrimary:
    def test_segment_centers_count_matches_n_segments(self):
        p = _primary()
        assert p.segment_centers_m.shape == (37, 2)
        assert p.n_segments == 1 + 3 * 3 * 4

    def test_center_segment_at_origin(self):
        np.testing.assert_array_equal(_primary().segment_centers_m[0], [0.0, 0.0])

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_scalar_interface(self, n):
        """16 float64 eps: vertex radius, scalloped inradius, six triangle areas."""
        p = _primary(n)
        a, g = 0.825, 0.012
        pitch = np.sqrt(3) * a + g
        circ = 2 * np.sqrt((np.sqrt(3) * n * pitch / 2 + a) ** 2 + (n * pitch / 2) ** 2)
        insc = (
            np.sqrt(3) * a
            if n == 0
            else 2
            * np.hypot(
                np.sqrt(3) * n * pitch / 2 + a / 2,
                np.sqrt(3) * a / 2 if n % 2 == 0 else 0,
            )
        )
        np.testing.assert_allclose(
            [
                p.diameter("circumscribed"),
                p.diameter("inscribed"),
                p.collecting_area_m2,
            ],
            [circ, insc, (1 + 3 * n * (n + 1)) * 6 * np.sqrt(3) / 4 * a * a],
            rtol=16 * np.finfo(float).eps,
        )

    def test_segments_fit_within_diameter(self):
        p = _primary()
        assert (
            float(jnp.linalg.norm(p.segment_centers_m, axis=1).max())
            < p.diameter("circumscribed") / 2
        )

    def test_is_pytree_jittable(self):
        """One float32 rounding of the host-computed segment area."""
        p = _primary()
        area = eqx.filter_jit(lambda q: q.collecting_area_m2)(p)
        np.testing.assert_allclose(
            area, 37 * 6 * np.sqrt(3) / 4 * 0.825**2, rtol=2 * np.finfo(np.float32).eps
        )

    def test_non_hexagon_centers_raise(self):
        with pytest.raises(ValueError, match="hexagon"):
            ox.SegmentedPrimary(
                n_rings=2, segment_point_to_point_m=1.65, segment_shape="keystone"
            )
