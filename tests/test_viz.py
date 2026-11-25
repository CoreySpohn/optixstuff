"""optixstuff.viz: radial image plot and the clean base install."""

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.viz import plot_radial_image


@pytest.fixture(autouse=True)
def headless_backend():
    """Draw on the Agg canvas so a Qt or Tk default cannot import on a headless host."""
    plt.switch_backend("Agg")


def test_base_install_imports_without_eyepiece_or_matplotlib():
    """Fails if importing optixstuff pulls in the plotting stack."""
    code = (
        "import sys; sys.modules['eyepiece'] = None; "
        "sys.modules['matplotlib'] = None; import optixstuff; "
        "import optixstuff.viz"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_radial_image_computes_off_center_profile():
    """Fails if drawing ignores the image grid, requested bins, or grid units."""
    grid = PixelGrid.regular((1, 5), 0.5, "arcsec", center=(0, 0))
    image = Image([[1, 3, 5, 7, 9]], grid, (), Measure.integral(), "test")
    res = plot_radial_image(image, nbins=2)
    line = res.artists["line"]
    np.testing.assert_allclose(line.get_xdata(), [0.5, 1.5])
    np.testing.assert_allclose(line.get_ydata(), [2, 7])
    assert res.ax.get_xlabel() == r"$r$ [arcsec]"
    res.fig.canvas.draw()
    plt.close(res.fig)


def test_plot_radial_image_forwards_plot_kwargs():
    """Fails if the image bridge loses the caller's line or axes options."""
    image = Image(
        np.ones((4, 4)),
        PixelGrid.regular((4, 4), 1, "pixel"),
        (),
        Measure.integral(),
        "test",
    )
    res = plot_radial_image(image, color="red", log=True, xlabel="custom")
    assert res.artists["line"].get_color() == "red"
    assert res.ax.get_yscale() == "log"
    assert res.ax.get_xlabel() == "custom"
    plt.close(res.fig)
