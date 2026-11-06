"""Tests for primary aperture area and segment geometry."""

from types import SimpleNamespace

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from yippy import Coronagraph, EqxCoronagraph
from yippy.header import HeaderData

from optixstuff import SegmentedPrimary, SimplePrimary, YippyCoronagraph


@pytest.fixture
def yip_header():
    return fits.Header(
        [("D", 6.0, "Diameter in m"), ("OBSCURED", 0.2), ("PIXSCALE", 0.1)]
    )


@pytest.mark.parametrize(
    "source", ["path", "str", "file", "coronagraph", "wrapper", "attributes"]
)
def test_simple_primary_from_yip(source, yip_header, tmp_path):
    """Fails if YIP area obscuration is squared again or D loses its units."""
    if source == "file":
        yip_header["OBSCURED"] = 0.0
    obscured = yip_header["OBSCURED"]
    if source in ("path", "str", "file"):
        file_path = tmp_path / "stellar_intens.fits"
        fits.PrimaryHDU(header=yip_header).writeto(file_path)
        yip = file_path if source == "file" else tmp_path
        if source == "str":
            yip = str(yip)
    else:
        header = HeaderData.from_fits_header(yip_header)
        # Exercise conversion to meters rather than taking Quantity.value.
        if source == "attributes":
            yip = SimpleNamespace(diameter_m=6.0, frac_obscured=obscured)
        else:
            yip = Coronagraph.__new__(Coronagraph)
            yip.header = SimpleNamespace(
                diameter=header.diameter.to(u.cm), obscured=obscured
            )
        if source == "wrapper":
            yip = YippyCoronagraph(backend=yip)

    primary = SimplePrimary.from_yip(yip, shape_factor=0.9)
    assert primary.diameter_m == 6.0
    assert primary.obscuration == np.sqrt(obscured)
    np.testing.assert_allclose(primary.area_m2, np.pi * 3.0**2 * (1.0 - obscured) * 0.9)
    default = SimplePrimary.from_yip(yip)
    assert default.shape_factor == 1.0
    np.testing.assert_allclose(default.area_m2, np.pi * 3.0**2 * (1.0 - obscured))


@pytest.mark.parametrize("key", ["D", "OBSCURED"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "none"])
def test_simple_primary_from_yip_requires_geometry(key, missing, yip_header, tmp_path):
    """Fails if missing or null aperture metadata silently supplies defaults."""
    if missing:
        del yip_header[key]
    else:
        yip_header[key] = None
    fits.PrimaryHDU(header=yip_header).writeto(tmp_path / "stellar_intens.fits")
    with pytest.raises(ValueError, match=key):
        SimplePrimary.from_yip(tmp_path)

    header = SimpleNamespace(diameter=6.0 * u.m, obscured=0.2)
    attr = "diameter" if key == "D" else "obscured"
    if missing:
        delattr(header, attr)
    else:
        setattr(header, attr, None)
    with pytest.raises(ValueError, match=key):
        SimplePrimary.from_yip(SimpleNamespace(header=header))


def test_simple_primary_from_yip_requires_retained_diameter():
    """Fails if an Eqx backend that discarded D supplies an invented diameter."""
    backend = EqxCoronagraph.__new__(EqxCoronagraph)
    for yip in (backend, YippyCoronagraph(backend=backend)):
        with pytest.raises(ValueError, match=r"missing D.*original YIP"):
            SimplePrimary.from_yip(yip)


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
