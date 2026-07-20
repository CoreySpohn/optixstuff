"""Tests for the sampling-explicit image contract and its default bridges.

A synthetic table-backed coronagraph (Gaussian PSFs on a native grid)
stands in for a YIP so the bridge implementations -- native-grid member
plus flux-conserving resample -- can be checked against analytic
expectations without fetched data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optixstuff.coronagraph import (
    AbstractScalarCoronagraph,
    AbstractTableCoronagraph,
    MultiBandCoronagraph,
    _convolve_quadrants,
)

_SIZE = 41
_SCALE = 0.5  # native lambda/D per pixel


class _TableCoronagraph(AbstractTableCoronagraph):
    """Transparent table coronagraph: unit-flux Gaussian PSFs everywhere."""

    pixel_scale_lod: float = _SCALE
    IWA: float = 0.0
    OWA: float = _SIZE * _SCALE / 2.0
    psf_shape: tuple = (_SIZE, _SIZE)
    sky_trans: jnp.ndarray = None
    psf_datacube: object = None

    def __init__(self, sky_value=1.0, datacube=None):
        self.sky_trans = jnp.full((_SIZE, _SIZE), sky_value)
        self.psf_datacube = datacube

    def throughput(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_area(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(self, sep, wl, *, time_s=0.0):
        return 0.0

    def occulter_transmission(self, sep, wl, *, time_s=0.0):
        return 1.0

    def create_psfs(self, x_lod, y_lod):
        center = (_SIZE - 1) / 2.0
        x_pix = center + jnp.atleast_1d(jnp.asarray(x_lod)) / self.pixel_scale_lod
        y_pix = center + jnp.atleast_1d(jnp.asarray(y_lod)) / self.pixel_scale_lod
        yy, xx = jnp.mgrid[:_SIZE, :_SIZE]

        def gauss(xc, yc):
            g = jnp.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * 1.5**2))
            return g / jnp.sum(g)

        return jax.vmap(gauss)(x_pix, y_pix)

    def stellar_intens(self, stellar_diam_lod):
        return 1e-6 * self.create_psfs(0.0, 0.0)[0]


class TestBridgeDefaults:
    """The default bridges over the native-grid members."""

    WL = 550.0

    def test_stellar_map_conserves_flux(self):
        c = _TableCoronagraph()
        native_total = float(jnp.sum(c.stellar_intens(0.0)))
        # target grid fully covers the native FOV
        served = c.stellar_map(self.WL, 0.0, pixel_scale_lod=0.25, shape=(101, 101))
        np.testing.assert_allclose(float(jnp.sum(served)), native_total, rtol=0.01)

    def test_source_psfs_peak_positions_and_flux(self):
        c = _TableCoronagraph()
        scale, npix = 0.25, 101
        xs, ys = jnp.asarray([3.0, -2.0]), jnp.asarray([0.0, 4.0])
        psfs = np.asarray(
            c.source_psfs(self.WL, xs, ys, pixel_scale_lod=scale, shape=(npix, npix))
        )
        assert psfs.shape == (2, npix, npix)
        coords = (np.arange(npix) - npix / 2 + 0.5) * scale
        for k in range(2):
            iy, ix = np.unravel_index(np.argmax(psfs[k]), psfs[k].shape)
            assert abs(coords[ix] - float(xs[k])) <= scale
            assert abs(coords[iy] - float(ys[k])) <= scale
            np.testing.assert_allclose(psfs[k].sum(), 1.0, rtol=0.01)

    def test_background_transmission_preserves_values(self):
        """Transmission is value-like, not flux-like: resampling a flat
        0.7 map at any target sampling must return ~0.7 everywhere."""
        c = _TableCoronagraph(sky_value=0.7)
        for scale in (0.25, 1.0):
            served = np.asarray(
                c.background_transmission(
                    self.WL, pixel_scale_lod=scale, shape=(31, 31)
                )
            )
            interior = served[5:-5, 5:-5]  # avoid edge-kernel effects
            np.testing.assert_allclose(interior, 0.7, rtol=0.01)

    def test_extended_scene_full_datacube_matches_direct_sum(self):
        """With every PSF identical, the scene render is total x PSF."""
        c0 = _TableCoronagraph()
        psf = c0.create_psfs(0.0, 0.0)[0]
        datacube = jnp.broadcast_to(psf, (_SIZE, _SIZE, _SIZE, _SIZE))
        c = _TableCoronagraph(datacube=datacube)
        key = jax.random.PRNGKey(0)
        scene = jax.random.uniform(key, (_SIZE, _SIZE))
        served = c.extended_scene(
            scene,
            _SCALE,
            self.WL,
            pixel_scale_lod=_SCALE,
            shape=(_SIZE, _SIZE),
        )
        expected = float(jnp.sum(scene)) * np.asarray(psf)
        np.testing.assert_allclose(np.asarray(served), expected, rtol=1e-4)

    def test_extended_scene_raises_on_missing_datacube(self):
        c = _TableCoronagraph(datacube=None)
        with pytest.raises(ValueError, match="psf_datacube"):
            c.extended_scene(
                jnp.ones((11, 11)),
                _SCALE,
                self.WL,
                pixel_scale_lod=_SCALE,
                shape=(_SIZE, _SIZE),
            )

    def test_scalar_only_models_serve_zero_maps(self):
        class _ScalarOnly(AbstractScalarCoronagraph):
            pixel_scale_lod: float = 0.5
            IWA: float = 0.0
            OWA: float = 10.0

            def throughput(self, sep, wl, *, time_s=0.0):
                return 1.0

            def core_area(self, sep, wl, *, time_s=0.0):
                return 1.0

            def core_mean_intensity(self, sep, wl, *, time_s=0.0):
                return 0.0

            def occulter_transmission(self, sep, wl, *, time_s=0.0):
                return 1.0

        c = _ScalarOnly()
        assert not jnp.any(
            c.stellar_map(550.0, 0.0, pixel_scale_lod=0.5, shape=(11, 11))
        )
        assert c.source_psfs(
            550.0,
            jnp.asarray([1.0, 2.0]),
            jnp.zeros(2),
            pixel_scale_lod=0.5,
            shape=(11, 11),
        ).shape == (2, 11, 11)


class TestMultiBand:
    """Wavelength dispatch across a per-band stack."""

    def _stack(self):
        blue = _TableCoronagraph(sky_value=0.4)
        red = _TableCoronagraph(sky_value=0.8)
        return blue, red, MultiBandCoronagraph([450.0, 900.0], [blue, red])

    def test_nearest_band_dispatch(self):
        blue, red, mb = self._stack()
        assert mb.band(500.0) is blue
        assert mb.band(700.0) is red  # nearer 900 than 450
        assert mb.band(880.0) is red

    def test_image_contract_dispatches(self):
        _, _, mb = self._stack()
        low = mb.background_transmission(451.0, pixel_scale_lod=0.5, shape=(21, 21))
        high = mb.background_transmission(899.0, pixel_scale_lod=0.5, shape=(21, 21))
        np.testing.assert_allclose(float(low[10, 10]), 0.4, rtol=0.01)
        np.testing.assert_allclose(float(high[10, 10]), 0.8, rtol=0.01)

    def test_scalar_metadata_from_reference_band(self):
        blue, _, mb = self._stack()
        assert mb.pixel_scale_lod == blue.pixel_scale_lod
        mb2 = MultiBandCoronagraph(
            [450.0, 900.0], [blue, _TableCoronagraph()], reference_wavelength_nm=900.0
        )
        assert mb2.IWA == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="band"):
            MultiBandCoronagraph([450.0], [_TableCoronagraph(), _TableCoronagraph()])


class TestConvolveQuadrants:
    """The quarter-symmetric datacube fold (moved here from coronagraphoto)."""

    def test_delta_at_center_returns_center_psf(self):
        size = 51
        center = size // 2
        qsize = center + 1
        key = jax.random.PRNGKey(1)
        psf = jax.random.uniform(key, (size, size))
        cube = jnp.broadcast_to(psf, (qsize, qsize, size, size))
        flux = jnp.zeros((size, size)).at[center, center].set(2.0)
        out = _convolve_quadrants(flux, cube)
        np.testing.assert_allclose(np.asarray(out), 2.0 * np.asarray(psf), rtol=1e-6)

    def test_sum_preservation_with_unit_psfs(self):
        """Total flux is preserved when every PSF sums to 1."""
        size = 51
        center = size // 2
        qsize = center + 1
        flux = jnp.zeros((size, size))
        flux = flux.at[center - 2 : center + 3, center - 2 : center + 3].set(4.0)
        psf_cube = jnp.zeros((qsize, qsize, size, size))
        psf_cube = psf_cube.at[:, :, center, center].set(1.0)
        out = _convolve_quadrants(flux, psf_cube)
        np.testing.assert_allclose(float(jnp.sum(out)), float(jnp.sum(flux)), rtol=1e-6)
