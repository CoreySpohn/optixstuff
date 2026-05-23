"""Tests for ``__repr__`` methods and ``OpticalPath.from_default_setup``."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

import optixstuff as ox
from optixstuff.coronagraph import AbstractScalarOnlyCoronagraph


class _MockCoro(AbstractScalarOnlyCoronagraph):
    """Minimal scalar-only coronagraph for factory + repr tests."""

    pixel_scale_lod: float = 0.5
    IWA: float = 2.0
    OWA: float = 30.0

    def throughput(self, sep, wl, *, time_s=0.0):
        return 0.5

    def core_area(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(self, sep, wl, *, time_s=0.0):
        return 1e-10

    def occulter_transmission(self, sep, wl, *, time_s=0.0):
        return 1.0


class TestReprs:
    """Spot-checks: each leaf repr contains the class name + a key field."""

    def test_simple_primary(self, simple_primary):
        s = repr(simple_primary)
        assert "SimplePrimary" in s
        assert "D=6" in s
        assert "area=" in s

    def test_simple_detector(self, simple_detector):
        s = repr(simple_detector)
        assert "SimpleDetector" in s
        assert "100x100" in s
        assert "QE=0.90" in s

    def test_full_detector(self):
        det = ox.Detector(
            pixel_scale=0.010,
            shape=(64, 64),
            quantum_efficiency=0.9,
            dark_current_rate=1e-4,
            read_noise_electrons=3.0,
            cic_rate=0.02,
            frame_time=100.0,
        )
        s = repr(det)
        assert s.startswith("Detector(")
        assert "RN=3" in s
        assert "CIC=0.02" in s
        assert "frame_time=100" in s

    def test_constant_throughput_element(self, throughput_element):
        s = repr(throughput_element)
        assert "ConstantThroughputElement" in s
        assert "test_filter" in s
        assert "throughput=0.8" in s

    def test_linear_throughput_element(self):
        el = ox.LinearThroughputElement(
            wavelengths_nm=jnp.array([400.0, 500.0, 600.0, 700.0]),
            throughputs=jnp.array([0.7, 0.8, 0.9, 0.85]),
        )
        s = repr(el)
        assert "LinearThroughputElement" in s
        assert "400-700 nm" in s
        assert "n=4" in s

    def test_optical_filter(self):
        f = ox.OpticalFilter(
            wavelengths_nm=jnp.array([500.0, 550.0, 600.0]),
            transmittances=jnp.array([0.1, 0.95, 0.1]),
        )
        s = repr(f)
        assert "OpticalFilter" in s
        assert "500-600 nm" in s
        assert "peak T=0.95" in s

    def test_optical_path_is_tree_shaped(self, simple_primary, simple_detector):
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(
                ox.ConstantThroughputElement(throughput=0.9, name="m1"),
                ox.ConstantThroughputElement(throughput=0.8, name="m2"),
            ),
            coronagraph=_MockCoro(),
            detector=simple_detector,
        )
        s = repr(path)
        lines = s.split("\n")
        assert lines[0].startswith("OpticalPath(")
        # Each child indented.
        assert any(line.startswith("  primary:") for line in lines)
        assert any(line.startswith("  attenuating_elements:") for line in lines)
        assert any(line.startswith("    [0]") for line in lines)
        assert any(line.startswith("    [1]") for line in lines)
        assert any(line.startswith("  coronagraph:") for line in lines)
        assert any(line.startswith("  detector:") for line in lines)

    def test_optical_path_empty_attenuating_chain(self, simple_primary, simple_detector):
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(),
            coronagraph=_MockCoro(),
            detector=simple_detector,
        )
        s = repr(path)
        assert "attenuating_elements: ()" in s


class TestFromDefaultSetup:
    """Verify the convenience factory builds a usable OpticalPath."""

    def test_returns_optical_path_with_defaults(self):
        path = ox.OpticalPath.from_default_setup(_MockCoro())
        assert isinstance(path, ox.OpticalPath)
        assert isinstance(path.primary, ox.SimplePrimary)
        assert isinstance(path.detector, ox.SimpleDetector)
        assert len(path.attenuating_elements) == 1
        # Defaults applied.
        assert path.primary.diameter_m == 6.0
        assert path.detector.shape == (512, 512)
        assert path.detector.pixel_scale == 0.01
        assert path.detector.quantum_efficiency == 0.9
        assert path.n_channels == 1.0
        assert path.npix_multiplier == 1.0

    def test_kwargs_override(self):
        path = ox.OpticalPath.from_default_setup(
            _MockCoro(),
            diameter_m=8.0,
            obscuration=0.1,
            attenuating_throughput=0.5,
            detector_shape=(256, 256),
            pixel_scale_arcsec=0.02,
            quantum_efficiency=0.95,
            dark_current_rate=1e-3,
            n_channels=2.0,
            npix_multiplier=3.0,
        )
        assert path.primary.diameter_m == 8.0
        assert path.primary.obscuration == 0.1
        assert path.detector.shape == (256, 256)
        assert path.detector.pixel_scale == 0.02
        assert path.detector.quantum_efficiency == 0.95
        assert path.detector.dark_current_rate == 1e-3
        assert path.n_channels == 2.0
        assert path.npix_multiplier == 3.0
        # Single attenuating element built with the overridden throughput.
        assert path.attenuating_elements[0].throughput == 0.5

    def test_accepts_coronagraph_instance(self):
        coro = _MockCoro()
        path = ox.OpticalPath.from_default_setup(coro)
        # Same object reused, no copy/wrap.
        assert path.coronagraph is coro

    def test_path_is_a_pytree(self):
        path = ox.OpticalPath.from_default_setup(_MockCoro())
        # eqx.tree_at can walk it (sanity check that nothing is non-pytree).
        leaves = eqx.filter(path, eqx.is_array)
        assert leaves is not None

    def test_accepts_yippy_eqx_coronagraph(self):
        """A bare ``yippy.EqxCoronagraph`` is wrapped via ``YippyCoronagraph(backend=)``.

        Catches the regression where any non-``AbstractCoronagraph`` /
        non-path argument was unconditionally ``str()``-ified and fed
        to ``YippyCoronagraph`` as a path.
        """
        from optixstuff.yippy_coronagraph import YippyCoronagraph

        # Build a minimal yippy.EqxCoronagraph-shaped duck. We don't need
        # yippy installed for this -- ``YippyCoronagraph(backend=...)``
        # stores whatever it's given on ``_backend``, so any object that
        # quacks works.
        class _Duck:
            pixel_scale_lod = 0.5
            psf_shape = (32, 32)
            IWA = 2.0
            OWA = 30.0
            psf_datacube = None
            sky_trans = jnp.ones((32, 32))

            def throughput(self, *args, **kwargs):
                return 0.5

            def core_area(self, *args, **kwargs):
                return 1.0

            def core_mean_intensity(self, *args, **kwargs):
                return 1e-10

            def occulter_transmission(self, *args, **kwargs):
                return 1.0

            def stellar_intens(self, diam_lod):
                return jnp.ones(self.psf_shape)

            def create_psf(self, *args, **kwargs):
                return jnp.ones(self.psf_shape)

            def create_psfs(self, x_lod, y_lod):
                return jnp.zeros((x_lod.shape[0], *self.psf_shape))

            def noise_floor_ayo(self, *args, **kwargs):
                return 1e-10

            def raw_contrast(self, *args, **kwargs):
                return 1e-10

        path = ox.OpticalPath.from_default_setup(_Duck())
        assert isinstance(path.coronagraph, YippyCoronagraph)
        assert isinstance(path.coronagraph._backend, _Duck)
