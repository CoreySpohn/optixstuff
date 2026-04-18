"""Tests for optixstuff hardware abstractions."""

import jax
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


class TestSimpleDetector:
    def test_quantum_efficiency(self, simple_detector):
        assert simple_detector.quantum_efficiency == pytest.approx(0.9)

    def test_dark_current_rate(self, simple_detector):
        assert simple_detector.dark_current_rate == pytest.approx(1e-4)

    def test_read_noise(self, simple_detector):
        assert simple_detector.read_noise_electrons == pytest.approx(3.0)

    def test_pixel_scale(self, simple_detector):
        assert simple_detector.pixel_scale == pytest.approx(0.010)

    def test_shape(self, simple_detector):
        assert simple_detector.shape == (100, 100)

    def test_add_noise_runs(self, simple_detector):
        """Verify add_noise produces output with correct shape."""
        image_rate = jnp.ones((100, 100)) * 10.0
        key = jax.random.PRNGKey(42)
        result = simple_detector.add_noise(image_rate, 100.0, key)
        assert result.shape == (100, 100)

    def test_add_noise_positive_rate_produces_counts(self, simple_detector):
        """With bright source, result should have nonzero counts."""
        image_rate = jnp.ones((100, 100)) * 1000.0
        key = jax.random.PRNGKey(0)
        result = simple_detector.add_noise(image_rate, 100.0, key)
        assert jnp.sum(result) > 0

    def test_add_source_electrons_runs(self, simple_detector):
        """add_source_electrons produces (ny, nx) output for a bright source."""
        image_rate = jnp.ones((100, 100)) * 100.0
        key = jax.random.PRNGKey(0)
        result = simple_detector.add_source_electrons(image_rate, 10.0, key)
        assert result.shape == (100, 100)
        assert jnp.all(result >= 0)

    def test_add_source_electrons_applies_qe(self):
        """Mean counts should equal rate * t * QE for a bright enough source."""
        det = ox.SimpleDetector(
            pixel_scale=0.010,
            shape=(32, 32),
            quantum_efficiency=0.5,
            dark_current_rate=0.0,
        )
        image_rate = jnp.ones((32, 32)) * 10000.0
        key = jax.random.PRNGKey(1)
        result = det.add_source_electrons(image_rate, 10.0, key)
        # Expected mean = 10000 * 10 * 0.5 = 50000; tolerance is loose for shot noise
        assert jnp.isclose(float(jnp.mean(result)), 50000.0, rtol=0.02)

    def test_add_noise_electrons_dark_only(self, simple_detector):
        """SimpleDetector noise is dark current only -- no CIC, no read."""
        key = jax.random.PRNGKey(2)
        result = simple_detector.add_noise_electrons(1000.0, key)
        assert result.shape == (100, 100)
        # Dark rate is 1e-4 e/s/pix -> mean ~0.1 e/pix for t=1000s
        assert float(jnp.mean(result)) < 1.0


class TestDetector:
    """Tests for the full Detector model (dark + CIC + read noise)."""

    @pytest.fixture
    def full_detector(self):
        return ox.Detector(
            pixel_scale=0.010,
            shape=(64, 64),
            quantum_efficiency=0.9,
            dark_current_rate=1e-4,
            read_noise_electrons=3.0,
            cic_rate=0.02,
            frame_time=100.0,
        )

    def test_add_source_electrons_shape(self, full_detector):
        image_rate = jnp.ones((64, 64)) * 10.0
        key = jax.random.PRNGKey(3)
        result = full_detector.add_source_electrons(image_rate, 10.0, key)
        assert result.shape == (64, 64)

    def test_add_noise_electrons_shape(self, full_detector):
        key = jax.random.PRNGKey(4)
        result = full_detector.add_noise_electrons(100.0, key)
        assert result.shape == (64, 64)

    def test_add_noise_matches_manual_split_compose(self, full_detector):
        """add_noise(image, t, k) is bit-identical to manually splitting the key
        and composing add_source_electrons + add_noise_electrons."""
        image_rate = jnp.ones((64, 64)) * 10.0
        key = jax.random.PRNGKey(5)
        combined = full_detector.add_noise(image_rate, 100.0, key)
        key_src, key_noise = jax.random.split(key, 2)
        src = full_detector.add_source_electrons(image_rate, 100.0, key_src)
        noise = full_detector.add_noise_electrons(100.0, key_noise)
        assert jnp.allclose(combined, src + noise)


class TestLinearThroughputElement:
    def test_interpolation(self):
        wls = jnp.array([400.0, 600.0, 800.0])
        tps = jnp.array([0.5, 0.9, 0.7])
        el = ox.LinearThroughputElement(wavelengths_nm=wls, throughputs=tps)
        # At 600 nm should be exactly 0.9
        assert el.get_throughput(600.0) == pytest.approx(0.9, abs=1e-5)

    def test_extrapolation_returns_zero(self):
        wls = jnp.array([400.0, 600.0, 800.0])
        tps = jnp.array([0.5, 0.9, 0.7])
        el = ox.LinearThroughputElement(wavelengths_nm=wls, throughputs=tps)
        # Outside range should be zero
        assert el.get_throughput(200.0) == pytest.approx(0.0, abs=1e-5)
        assert el.get_throughput(1000.0) == pytest.approx(0.0, abs=1e-5)


class TestOpticalFilter:
    def test_interpolation(self):
        wls = jnp.array([500.0, 550.0, 600.0])
        trans = jnp.array([0.0, 1.0, 0.0])
        f = ox.OpticalFilter(wavelengths_nm=wls, transmittances=trans)
        # Peak at 550 should be 1.0
        assert f.get_throughput(550.0) == pytest.approx(1.0, abs=1e-5)


class TestOpticalPath:
    def test_system_throughput_single_element(self, simple_primary, simple_detector):
        el = ox.ConstantThroughputElement(throughput=0.7)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el,),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.7)

    def test_system_throughput_two_elements(self, simple_primary, simple_detector):
        el1 = ox.ConstantThroughputElement(throughput=0.8)
        el2 = ox.ConstantThroughputElement(throughput=0.9)
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(el1, el2),
            coronagraph=None,
            detector=simple_detector,
        )
        assert path.system_throughput(500.0) == pytest.approx(0.72)


class TestPureNoiseFunctions:
    def test_dark_current_shape(self):
        key = jax.random.PRNGKey(0)
        dc = ox.simulate_dark_current(0.001, 100.0, (50, 50), key)
        assert dc.shape == (50, 50)

    def test_read_noise_shape(self):
        key = jax.random.PRNGKey(1)
        rn = ox.simulate_read_noise(3.0, 10.0, (50, 50), key)
        assert rn.shape == (50, 50)

    def test_cic_shape(self):
        key = jax.random.PRNGKey(2)
        cic = ox.simulate_cic(0.02, 5.0, (50, 50), key)
        assert cic.shape == (50, 50)
