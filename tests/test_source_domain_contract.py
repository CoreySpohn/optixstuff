"""A source outside the source domain is zero and flagged.

The contract (see AbstractImagingCoronagraph): SourcePSFs.valid is False for a
source outside the implementer's source domain, with a finite all-zero image,
and off_axis_psf returns zeros rather than NaN. This file checks the optixstuff
test double; each implementing library (yippy, physicaloptix) runs the same
check on its own coronagraphs in its own suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import GaussianTableCoronagraph

from optixstuff.grids import PixelGrid


@pytest.fixture(autouse=True, scope="module")
def _clear_caches():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def implementer():
    # The table tabulates offsets out to 6 lambda/D.
    return GaussianTableCoronagraph(), (1.3 * 6.0, 0.0)


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
