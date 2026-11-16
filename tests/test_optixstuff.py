"""Tests for optixstuff hardware abstractions."""

import jax.numpy as jnp
import pytest

import optixstuff as ox


class TestSimplePrimary:
    def test_diameter(self, simple_primary):
        assert simple_primary.diameter_m == 6.0

    def test_area_no_obscuration(self):
        p = ox.SimplePrimary(diameter_m=2.0, obscuration=0.0)
        expected = jnp.pi * 1.0**2
        assert jnp.isclose(p.area_m2, expected)

    def test_area_with_obscuration(self, simple_primary):
        expected = jnp.pi * 3.0**2 * (1.0 - 0.14**2)
        assert jnp.isclose(simple_primary.area_m2, expected)

    def test_area_with_shape_factor(self):
        p = ox.SimplePrimary(diameter_m=6.0, obscuration=0.0, shape_factor=0.9)
        expected = jnp.pi * 3.0**2 * 0.9
        assert jnp.isclose(p.area_m2, expected)

    def test_is_equinox_module(self, simple_primary):
        import equinox as eqx

        assert isinstance(simple_primary, eqx.Module)


class TestConstantThroughputElement:
    def test_get_throughput(self, throughput_element):
        assert throughput_element.get_throughput(500.0) == pytest.approx(0.8)

    def test_wavelength_independent(self, throughput_element):
        t1 = throughput_element.get_throughput(400.0)
        t2 = throughput_element.get_throughput(900.0)
        assert t1 == t2

    def test_apply(self, throughput_element):
        arr = jnp.ones((10, 10))
        result = throughput_element.apply(arr, 500.0)
        assert jnp.allclose(result, 0.8 * jnp.ones((10, 10)))


class TestLinearThroughputElement:
    def test_interpolation(self):
        wls = jnp.array([400.0, 600.0, 800.0])
        tps = jnp.array([0.5, 0.9, 0.7])
        el = ox.SpectralThroughput(wavelengths_nm=wls, throughputs=tps)
        # At 600 nm should be exactly 0.9
        assert el.get_throughput(600.0) == pytest.approx(0.9, abs=1e-5)

    def test_extrapolation_returns_zero(self):
        wls = jnp.array([400.0, 600.0, 800.0])
        tps = jnp.array([0.5, 0.9, 0.7])
        el = ox.SpectralThroughput(wavelengths_nm=wls, throughputs=tps)
        # Outside range should be zero
        assert el.get_throughput(200.0) == pytest.approx(0.0, abs=1e-5)
        assert el.get_throughput(1000.0) == pytest.approx(0.0, abs=1e-5)


class TestOpticalFilter:
    def test_interpolation(self):
        wls = jnp.array([500.0, 550.0, 600.0])
        trans = jnp.array([0.0, 1.0, 0.0])
        f = ox.SpectralThroughput(wavelengths_nm=wls, throughputs=trans)
        # Peak at 550 should be 1.0
        assert f.get_throughput(550.0) == pytest.approx(1.0, abs=1e-5)


class TestOpticalPath:
    def test_system_throughput_single_element(self, simple_primary, simple_detector):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.7)

    def test_system_throughput_two_elements(self, simple_primary, simple_detector):
        el1 = ox.ConstantThroughput(throughput=0.8)
        el2 = ox.ConstantThroughput(throughput=0.9)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el1, el2),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.72)

    def test_default_n_channels_and_npix_multiplier(
        self, simple_primary, simple_detector
    ):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.n_channels == 1.0
        assert path.npix_multiplier == 1.0

    def test_custom_n_channels_and_npix_multiplier(
        self, simple_primary, simple_detector
    ):
        el = ox.ConstantThroughput(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
            n_channels=2.0,
            npix_multiplier=3.0,
        )
        assert path.n_channels == 2.0
        assert path.npix_multiplier == 3.0


class TestYippyCoronagraphExtras:
    """Tests for create_psfs and psf_datacube delegation through to _backend.

    These tests use a stubbed EqxCoronagraph-like object rather than
    loading a real YIP to stay fast and self-contained.
    """

    def test_create_psfs_delegates(self):
        """YippyCoronagraph.create_psfs should call self._backend.create_psfs."""
        called = {}

        class StubBackend:
            pixel_scale_lod = 0.1
            IWA = 3.0
            OWA = 10.0
            psf_shape = (51, 51)
            sky_trans = jnp.zeros((51, 51))
            psf_datacube = None

            def create_psfs(self, x, y):
                called["args"] = (x, y)
                return jnp.zeros((len(x), 51, 51))

            def throughput(self, s):
                return 0.5

            def core_area(self, s):
                return 1.0

            def core_mean_intensity(self, s):
                return 1e-8

            def occulter_transmission(self, s):
                return 0.5

            def stellar_intens(self, d):
                return jnp.zeros((51, 51))

            def create_psf(self, x, y, npix):
                return jnp.zeros((npix, npix))

            def noise_floor_ayo(self, s, ppf=30.0):
                return 1e-9

            def raw_contrast(self, s):
                return 1e-10

        coro = ox.YippyCoronagraph(backend=StubBackend())
        x = jnp.array([1.0, 2.0])
        y = jnp.array([0.0, 0.0])
        result = coro.create_psfs(x, y)
        assert result.shape == (2, 51, 51)
        assert called["args"][0] is x

    def test_psf_datacube_delegates(self):
        """YippyCoronagraph.psf_datacube returns self._backend.psf_datacube."""
        fake_cube = jnp.ones((3, 3, 51, 51))

        class StubBackend:
            pixel_scale_lod = 0.1
            IWA = 3.0
            OWA = 10.0
            psf_shape = (51, 51)
            sky_trans = jnp.zeros((51, 51))
            psf_datacube = fake_cube
            create_psfs = staticmethod(lambda x, y: None)

            def throughput(self, s):
                return 0.5

            def core_area(self, s):
                return 1.0

            def core_mean_intensity(self, s):
                return 1e-8

            def occulter_transmission(self, s):
                return 0.5

            def stellar_intens(self, d):
                return jnp.zeros((51, 51))

            def create_psf(self, x, y, npix):
                return jnp.zeros((npix, npix))

            def noise_floor_ayo(self, s, ppf=30.0):
                return 1e-9

            def raw_contrast(self, s):
                return 1e-10

        coro = ox.YippyCoronagraph(backend=StubBackend())
        assert coro.psf_datacube is fake_cube
