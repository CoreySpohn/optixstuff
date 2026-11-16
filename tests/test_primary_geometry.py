"""Primary geometry from independent dimensions and header metadata."""

import inspect
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from optixstuff import AbstractPrimary, SegmentedPrimary, SimplePrimary


def concrete_subclasses(base):
    result = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        result.extend(concrete_subclasses(cls))
    return result


@pytest.mark.parametrize("cls", concrete_subclasses(AbstractPrimary))
def test_primary_contract(cls):
    """Floating-point: 16 float64 eps for hex distances and six triangle areas."""
    if cls is SimplePrimary:
        primary = cls(
            circumscribed_diameter_m=6.0,
            inscribed_diameter_m=5.0,
            collecting_area_m2=20.0,
        )
        expected = (6.0, 5.0, 20.0)
    elif cls is SegmentedPrimary:
        n, a, gap = 2, 0.825, 0.004
        pitch = np.sqrt(3) * a + gap
        primary = cls(n_rings=n, segment_point_to_point_m=2 * a, segment_gap_m=gap)
        # Extreme outer-ring vertex and nearest reentrant perimeter vertex.
        circ = 2 * np.hypot(np.sqrt(3) * n * pitch / 2 + a, n * pitch / 2)
        insc = 2 * np.hypot(np.sqrt(3) * n * pitch / 2 + a / 2, np.sqrt(3) * a / 2)
        area = (1 + 3 * n * (n + 1)) * 6 * (np.sqrt(3) / 4) * a * a
        expected = (circ, insc, area)
    else:
        pytest.fail(f"Add a primitive fixture for {cls.__name__}")
    np.testing.assert_allclose(
        [
            primary.diameter("circumscribed"),
            primary.diameter("inscribed"),
            primary.collecting_area_m2,
        ],
        expected,
        rtol=16 * np.finfo(float).eps,
    )
    assert primary.diameter("inscribed") < primary.diameter("circumscribed")
    with pytest.raises(ValueError, match="convention"):
        primary.diameter("effective")
    frame = primary.frame(500.0, "inscribed")
    assert frame.diameter_m == primary.diameter("inscribed")
    assert frame.diameter_convention == "inscribed"
    for name in ("diameter_m", "area_m2"):
        assert not hasattr(primary, name)


@pytest.mark.parametrize("source", ["path", "header"])
def test_simple_primary_from_yip(source, tmp_path):
    """Exact header diameters; changing area alone must never change D_INSC."""

    def make(obscured):
        if source == "header":
            return SimpleNamespace(
                header=SimpleNamespace(
                    diameter=600 * u.cm,
                    diameter_inscribed=470 * u.cm,
                    obscured=obscured,
                )
            )
        fits.PrimaryHDU(
            header=fits.Header(
                {"D": 6.0, "D_INSC": 4.7, "OBSCURED": obscured, "PIXSCALE": 0.25}
            )
        ).writeto(tmp_path / "stellar_intens.fits", overwrite=True)
        return tmp_path

    for obscured in (0.2, 0.7):
        p = SimplePrimary.from_yip(make(obscured), shape_factor=0.9)
        assert p.diameter("circumscribed") == 6.0
        assert p.diameter("inscribed") == 4.7
        np.testing.assert_allclose(
            p.collecting_area_m2,
            np.pi * 9 * (1 - obscured) * 0.9,
            rtol=16 * np.finfo(float).eps,
        )


@pytest.mark.parametrize("key", ["D", "D_INSC", "OBSCURED"])
def test_simple_primary_from_yip_requires_geometry(key, tmp_path):
    """Missing geometry raises instead of deriving a diameter from collecting area."""
    h = fits.Header({"D": 6.0, "D_INSC": 5.0, "OBSCURED": 0.2})
    del h[key]
    fits.PrimaryHDU(header=h).writeto(tmp_path / "stellar_intens.fits")
    with pytest.raises(ValueError, match=key):
        SimplePrimary.from_yip(tmp_path)
