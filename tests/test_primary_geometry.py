"""Primary geometry from independent dimensions and header metadata."""

import inspect

import numpy as np
import pytest

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
