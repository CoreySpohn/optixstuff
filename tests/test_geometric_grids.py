"""Geometric grid centers and band channel edges."""

import math
from dataclasses import replace

import numpy as np


def test_geometric_grids_and_bands():
    """Fails for tolerant flip centers or arithmetic log centers.

    Center and squared-distance algebra is exact; one ULP displacement must
    fail geometric symmetry. Log edges/centers allow 2e-15 float64 rounding.
    """
    from optixstuff.grids import PixelGrid
    from optixstuff.spectral import Band

    for shape in ((3, 6), (4, 5), (1, 1)):
        cy, cx = ((shape[0] - 1) / 2, (shape[1] - 1) / 2)
        grid = PixelGrid.regular(shape, 0.1, "arcsec")
        assert grid.is_geometric
        assert not replace(grid, center_x=np.nextafter(cx, math.inf)).is_geometric
        assert not replace(grid, center_y=np.nextafter(cy, math.inf)).is_geometric
    band = Band(400.0, 1600.0)
    assert band.geometric_center_nm == 800.0
    channels = band.channels(3, "log")
    edges = [channels[0].lo_nm, *(channel.hi_nm for channel in channels)]
    np.testing.assert_allclose(edges, 400 * 4 ** (np.arange(4) / 3), rtol=2e-15)
    np.testing.assert_allclose(
        [b.geometric_center_nm for b in channels],
        400 * 4 ** ((np.arange(3) + 0.5) / 3),
        rtol=2e-15,
    )
