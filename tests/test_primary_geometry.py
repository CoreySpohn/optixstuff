"""Tests for SegmentedPrimary segment geometry (size, pitch, centers)."""

import jax.numpy as jnp
import numpy as np
import pytest

from optixstuff import SegmentedPrimary


@pytest.fixture
def eac1_like():
    return SegmentedPrimary(
        diameter_m=7.2,
        area_m2=33.6,
        n_rings=2,
        n_segments=19,
        segment_gap_m=0.004,
        segment_point_to_point_m=1.65,
        inscribed_diameter_m=5.96,
    )


class TestSegmentGeometry:
    def test_optional_fields_default_to_none(self):
        primary = SegmentedPrimary(
            diameter_m=10.033,
            area_m2=65.16,
            n_rings=3,
            n_segments=37,
            segment_gap_m=0.012,
        )
        assert primary.segment_point_to_point_m is None
        assert primary.inscribed_diameter_m is None
        # Without a measured segment size, the legacy approximation stands.
        np.testing.assert_allclose(primary.segment_flat_to_flat_m, 10.033 / 7)

    def test_flat_to_flat_and_pitch(self, eac1_like):
        f2f = 1.65 * np.sqrt(3) / 2
        np.testing.assert_allclose(eac1_like.segment_flat_to_flat_m, f2f)
        np.testing.assert_allclose(eac1_like.segment_pitch_m, f2f + 0.004)

    def test_segment_centers_layout(self, eac1_like):
        centers = np.asarray(eac1_like.segment_centers_m)
        assert centers.shape == (19, 2)
        # Center segment first, at the origin.
        np.testing.assert_allclose(centers[0], [0.0, 0.0])
        # Flat-top lattice: six nearest neighbors at one pitch.
        radii = np.hypot(centers[:, 0], centers[:, 1])
        ring1 = np.sort(radii)[1:7]
        np.testing.assert_allclose(ring1, eac1_like.segment_pitch_m, rtol=1e-12)
        # The layout spans the circumscribed diameter to within a segment.
        assert radii.max() < eac1_like.diameter_m / 2
        # Symmetric layout: the centers are closed under point reflection.
        as_set = {(round(x, 9), round(y, 9)) for x, y in centers}
        mirrored = {(round(-x, 9), round(-y, 9)) for x, y in centers}
        assert as_set == mirrored

    def test_centers_require_segment_size(self):
        primary = SegmentedPrimary(
            diameter_m=10.033,
            area_m2=65.16,
            n_rings=3,
            n_segments=37,
            segment_gap_m=0.012,
        )
        # Without a measured segment size the centers fall back to the
        # legacy approximate pitch and still evaluate.
        assert primary.segment_centers_m.shape == (37, 2)

    def test_centers_are_a_jax_array(self, eac1_like):
        assert isinstance(eac1_like.segment_centers_m, jnp.ndarray)
