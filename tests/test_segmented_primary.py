"""SegmentedPrimary -- a geometry-carrying primary.

Carries the segment layout (rings, count, gap, shape) so a diffraction backend
can build the pupil from it; exposes diameter/area like any AbstractPrimary.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

import optixstuff as ox


def _eac5_like():
    # EAC-5 OTA geometry
    return ox.SegmentedPrimary(
        diameter_m=10.033,
        area_m2=65.16,
        n_rings=3,
        n_segments=37,
        segment_gap_m=0.012,
    )


class TestSegmentedPrimary:
    def test_is_abstract_primary(self):
        assert isinstance(_eac5_like(), ox.AbstractPrimary)

    def test_scalar_interface(self):
        p = _eac5_like()
        assert float(p.diameter_m) == pytest.approx(10.033)
        assert float(p.area_m2) == pytest.approx(65.16)

    def test_segment_centres_count_matches_n_segments(self):
        p = _eac5_like()
        centres = p.segment_centres_m
        assert centres.shape == (37, 2)

    def test_n_segments_matches_ring_formula(self):
        p = _eac5_like()
        assert p.n_segments == 1 + 3 * p.n_rings * (p.n_rings + 1)

    def test_centre_segment_at_origin(self):
        centres = _eac5_like().segment_centres_m
        assert jnp.allclose(centres[0], jnp.zeros(2))

    def test_flat_to_flat(self):
        p = _eac5_like()
        assert float(p.segment_flat_to_flat_m) == pytest.approx(10.033 / 7)

    def test_segments_fit_within_diameter(self):
        p = _eac5_like()
        r = jnp.linalg.norm(p.segment_centres_m, axis=1)
        # every segment centre is inside the circumscribing radius
        assert float(r.max()) < p.diameter_m / 2.0

    def test_is_pytree_jittable(self):
        p = _eac5_like()
        area = eqx.filter_jit(lambda q: q.area_m2)(p)
        assert float(area) == pytest.approx(65.16)

    def test_non_hexagon_centres_raise(self):
        p = ox.SegmentedPrimary(
            diameter_m=8.0,
            area_m2=40.0,
            n_rings=2,
            n_segments=19,
            segment_gap_m=0.01,
            segment_shape="keystone",
        )
        with pytest.raises(NotImplementedError):
            _ = p.segment_centres_m
