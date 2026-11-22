"""A source outside the source domain is zero and flagged, for every implementer.

The contract (see AbstractImagingCoronagraph): SourcePSFs.valid is False for a
source outside the implementer's source domain, with a finite all-zero image,
and off_axis_psf returns zeros rather than NaN. Implementers outside this
repository are exercised through their own synthetic fixtures when they can be
imported; a missing library skips its parametrization.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import GaussianTableCoronagraph

from optixstuff.grids import PixelGrid


@pytest.fixture(autouse=True, scope="module")
def precision():
    with jax.enable_x64():
        yield
    jax.clear_caches()


def _library_tests(name):
    """Directory of another library's test helpers, or skip."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        pytest.skip(f"{name} is not importable")
    tests = Path(spec.origin).resolve().parents[2] / "tests"
    if not (tests / "synthetic_yip.py").is_file():
        pytest.skip(f"{name} test fixtures are not next to its source")
    return tests


@pytest.fixture(scope="module")
def helpers_on_path():
    tests = _library_tests("yippy")
    sys.path.insert(0, str(tests))
    yield tests
    sys.path.remove(str(tests))
    for name in ("synthetic_yip", "sinc_fixture"):
        sys.modules.pop(name, None)


def _double(tmp_path_factory, helpers):
    coro = GaussianTableCoronagraph()
    return coro, (1.3 * 6.0, 0.0)


def _yippy(kind):
    def build(tmp_path_factory, helpers_on_path):
        from synthetic_yip import write_synthetic_yip
        from yippy import open_yip

        root = tmp_path_factory.mktemp(f"yip_{kind}")
        written = write_synthetic_yip(root / "package", kind=kind)
        # The package tabulates offsets out to 6 lambda/D.
        return open_yip(written.path).prepare(), (1.3 * 6.0, 0.0)

    return build


def _yippy_sinc(tmp_path_factory, helpers_on_path):
    from astropy.io import fits
    from sinc_fixture import save_manifest, write_delivery
    from yippy import open_yip

    root = tmp_path_factory.mktemp("yip_sinc")
    metadata = write_delivery(root)
    metadata["checks"] = []
    metadata["served"]["bounds_lod"] = [[-0.2, 0.2], [-0.2, 0.5]]
    save_manifest(root, metadata)
    with fits.open(root / "legacy" / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data *= 1 + 0.2 * np.cos(2 * np.pi * (np.arange(4) - 1.5) / 4)
    # The served box reaches 0.5 lambda/D along y; 1.3 times that is outside.
    return open_yip(root).prepare(interpolation="sinc"), (0.0, 1.3 * 0.5)


def _physicaloptix(tmp_path_factory, helpers_on_path):
    pytest.importorskip("physicaloptix")
    from physicaloptix.core import Field, PlaneKind
    from physicaloptix.core.grid import _grid_axis
    from physicaloptix.elements import SampledOptic
    from physicaloptix.interop import PathCoronagraph
    from physicaloptix.path import OpticalPath, Stage

    from optixstuff import SimplePrimary

    npup, diameter = 64, 6.0
    grid = PixelGrid.pupil(npup)
    x = np.asarray(_grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.5**2).astype(float)
    path = OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
                ),
            ),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    primary = SimplePrimary(
        circumscribed_diameter_m=diameter, collecting_area_m2=np.pi * diameter**2 / 4
    )
    coro = PathCoronagraph.from_path(
        path,
        field,
        primary=primary,
        reference_wavelength_nm=600.0,
        max_separation_lod=10.0,
        pixel_scale_lod=0.5,
        n_sky_screens=8,
    )
    # The pupil Nyquist radius is 0.5 / step = npup / 2 lambda/D.
    return coro, (1.3 * npup / 2, 0.0)


IMPLEMENTERS = {
    "optixstuff_double": _double,
    "yippy_1d": _yippy("1d"),
    "yippy_2d": _yippy("2dq"),
    "yippy_sinc": _yippy_sinc,
    "physicaloptix_path": _physicaloptix,
}
# Implementers that do not yet meet the contract; each mark names the card whose
# landing makes the test pass, and strict=True turns the mark red then.
FAILING = {
    "yippy_1d": "Y2: valid stays True beyond the node hull",
    "yippy_2d": "Y2: valid stays True beyond the node hull",
    "yippy_sinc": "Y2: the rejected source image is NaN, not zero",
    "physicaloptix_path": "PO1x: a rejected source holds the on-axis image, not zero",
}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=FAILING[name])]
                if name in FAILING
                else []
            ),
        )
        for name in IMPLEMENTERS
    ],
)
def implementer(request, tmp_path_factory):
    needs_helpers = request.param != "optixstuff_double"
    helpers = request.getfixturevalue("helpers_on_path") if needs_helpers else None
    coro, outside = IMPLEMENTERS[request.param](tmp_path_factory, helpers)
    return coro, outside


def test_source_outside_the_domain_is_zero_and_flagged(implementer):
    """Fails if a source beyond the data gives valid True, NaN or nonzero light.

    A source at 1.3 times the source domain has valid False for that source
    alone, a finite image of exact zeros, and an off-axis PSF of zeros, not
    NaN. A second, inside source in the same call keeps its own flag.
    """
    coro, (x_out, y_out) = implementer
    wavelength = coro.frame.reference_wavelength_nm
    grid = PixelGrid.focal(16, 0.5, coro.frame)
    result = coro.source_psfs(
        wavelength,
        jnp.array([0.0, x_out]),
        jnp.array([0.0, y_out]),
        grid=grid,
    )
    np.testing.assert_array_equal(np.asarray(result.valid), [True, False])
    image = np.asarray(result.image.data)
    assert np.isfinite(image).all()
    assert (image[1] == 0.0).all()
    separation = float(np.hypot(x_out, y_out))
    off_axis = np.asarray(coro.off_axis_psf(wavelength, separation, grid).data)
    assert np.isfinite(off_axis).all() and (off_axis == 0.0).all()
