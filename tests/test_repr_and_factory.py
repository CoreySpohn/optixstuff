"""Tests for ``__repr__`` methods and ``OpticalPath.from_default_setup``."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from conftest import scalar_coronagraph

import optixstuff as ox


class TestReprs:
    """Spot-checks: each leaf repr contains the class name + a key field."""

    def test_simple_primary(self, simple_primary):
        s = repr(simple_primary)
        assert "SimplePrimary" in s
        assert "D_circ=6" in s
        assert "area=" in s

    def test_simple_detector(self, simple_detector):
        s = repr(simple_detector)
        assert "IdealDetector" in s
        assert "100x100" in s
        assert "QE=1.00" in s

    def test_full_detector(self):
        det = ox.Detector(
            pixel_scale_arcsec=0.010,
            shape=(64, 64),
            qe=0.9,
            dark_current_e_per_pixel_s=1e-4,
            read_noise_e_rms=3.0,
            clock_induced_charge_e_per_pixel_frame=0.02,
        )
        s = repr(det)
        assert s.startswith("Detector(")
        assert "RN=3" in s
        assert "CIC=0.02" in s
        assert "F2=1" in s

    def test_constant_throughput_element(self, throughput_element):
        s = repr(throughput_element)
        assert "ConstantThroughput" in s
        assert "test_filter" in s
        assert "throughput=0.8" in s

    def test_linear_throughput_element(self):
        el = ox.SpectralThroughput(
            wavelengths_nm=jnp.array([400.0, 500.0, 600.0, 700.0]),
            throughputs=jnp.array([0.7, 0.8, 0.9, 0.85]),
        )
        s = repr(el)
        assert "SpectralThroughput" in s
        assert "400-700 nm" in s
        assert "n=4" in s

    def test_spectral_throughput_bandpass_shape(self):
        f = ox.SpectralThroughput(
            wavelengths_nm=jnp.array([500.0, 550.0, 600.0]),
            throughputs=jnp.array([0.1, 0.95, 0.1]),
        )
        s = repr(f)
        assert "SpectralThroughput" in s
        assert "500-600 nm" in s
        assert "n=3" in s
        assert "0.95" in s

    def test_optical_path_is_tree_shaped(self, simple_primary, simple_detector):
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(
                ox.ConstantThroughput(throughput=0.9, name="m1"),
                ox.ConstantThroughput(throughput=0.8, name="m2"),
            ),
            coronagraph=scalar_coronagraph(),
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

    def test_optical_path_empty_attenuating_chain(
        self, simple_primary, simple_detector
    ):
        path = ox.OpticalPath(
            primary=simple_primary,
            attenuating_elements=(),
            coronagraph=scalar_coronagraph(),
            detector=simple_detector,
        )
        s = repr(path)
        assert "attenuating_elements: ()" in s


class TestFromDefaultSetup:
    """Verify the convenience factory builds a usable OpticalPath."""

    def test_returns_optical_path_with_defaults(self):
        path = ox.OpticalPath.from_default_setup(scalar_coronagraph())
        assert isinstance(path, ox.OpticalPath)
        assert isinstance(path.primary, ox.SimplePrimary)
        assert isinstance(path.detector, ox.Detector)
        assert len(path.attenuating_elements) == 1
        # Defaults applied.
        assert path.primary.diameter("circumscribed") == 6.0
        assert path.detector.shape == (512, 512)
        assert path.detector.pixel_scale_arcsec == 0.01
        assert float(path.detector.quantum_efficiency(550.0)) == float(jnp.asarray(0.9))
        assert path.n_channels == 1.0
        assert path.npix_multiplier == 1.0

    def test_kwargs_override(self):
        path = ox.OpticalPath.from_default_setup(
            scalar_coronagraph(),
            circumscribed_diameter_m=8.0,
            obscuration=0.1,
            attenuating_throughput=0.5,
            detector_shape=(256, 256),
            pixel_scale_arcsec=0.02,
            qe=0.95,
            dark_current_e_per_pixel_s=1e-3,
            n_channels=2.0,
            npix_multiplier=3.0,
        )
        assert path.primary.diameter("circumscribed") == 8.0
        assert path.primary.collecting_area_m2 == 3.141592653589793 * 16 * 0.99
        assert path.detector.shape == (256, 256)
        assert path.detector.pixel_scale_arcsec == 0.02
        assert float(path.detector.quantum_efficiency(550.0)) == float(
            jnp.asarray(0.95)
        )
        assert path.detector.dark_current_e_per_pixel_s == float(jnp.asarray(1e-3))
        assert path.n_channels == 2.0
        assert path.npix_multiplier == 3.0
        # Single attenuating element built with the overridden throughput.
        assert path.attenuating_elements[0].throughput == 0.5

    def test_accepts_coronagraph_instance(self):
        coro = scalar_coronagraph()
        path = ox.OpticalPath.from_default_setup(coro)
        # Same object reused, no copy/wrap.
        assert path.coronagraph is coro

    def test_path_is_a_pytree(self):
        path = ox.OpticalPath.from_default_setup(scalar_coronagraph())
        # eqx.tree_at can walk it (sanity check that nothing is non-pytree).
        leaves = eqx.filter(path, eqx.is_array)
        assert leaves is not None

    def test_accepts_yippy_eqx_coronagraph(self, tmp_path):
        """A bare ``yippy.EqxCoronagraph`` is wrapped via the ``backend`` kwarg.

        Catches the regression where any non-``AbstractCoronagraph`` /
        non-path argument was unconditionally ``str()``-ified and fed
        to ``YippyCoronagraph`` as a path.
        """
        from conftest import synthetic_coronagraph

        from optixstuff.yippy_coronagraph import YippyCoronagraph

        backend = synthetic_coronagraph(tmp_path)._backend
        path = ox.OpticalPath.from_default_setup(backend)
        assert isinstance(path.coronagraph, YippyCoronagraph)
        assert path.coronagraph._backend is backend
