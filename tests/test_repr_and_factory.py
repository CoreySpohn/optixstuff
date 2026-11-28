"""Tests for ``__repr__`` methods and the Instrument pytree."""

from __future__ import annotations

import equinox as eqx
import pytest
from conftest import make_instrument, make_telescope, table_coronagraph

import optixstuff as ox


class TestReprs:
    """Spot-checks: each leaf repr contains the class name + a key field."""

    def test_simple_pupil(self, simple_pupil):
        s = repr(simple_pupil)
        assert "SimplePupil" in s
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

    def test_optical_path_is_tree_shaped(self, simple_pupil, simple_detector):
        path = ox.Instrument(
            "imager",
            attenuating_elements=(
                ox.ConstantThroughput(throughput=0.9, name="m1"),
                ox.ConstantThroughput(throughput=0.8, name="m2"),
            ),
            coronagraph=table_coronagraph(),
            detector=simple_detector,
        )
        s = repr(path)
        lines = s.split("\n")
        assert lines[0] == "Instrument 'imager'"
        # Each child indented.
        assert any(line.startswith("  attenuating_elements:") for line in lines)
        assert any(line.startswith("    [0]") for line in lines)
        assert any(line.startswith("    [1]") for line in lines)
        assert any(line.startswith("  coronagraph:") for line in lines)
        assert any(line.startswith("  detector:") for line in lines)

    def test_optical_path_empty_attenuating_chain(self, simple_pupil, simple_detector):
        path = ox.Instrument(
            "imager",
            attenuating_elements=(),
            coronagraph=table_coronagraph(),
            detector=simple_detector,
        )
        s = repr(path)
        assert "attenuating_elements: ()" in s


class TestInstrument:
    """Verify the hardware record is a usable pytree."""

    def test_path_is_a_pytree(self):
        path = make_instrument(table_coronagraph())
        # eqx.tree_at can walk it (sanity check that nothing is non-pytree).
        leaves = eqx.filter(path, eqx.is_array)
        assert leaves is not None

    def test_pupil_must_match_the_coronagraph_frame(self):
        coro = table_coronagraph()
        with pytest.raises(ValueError, match="frame"):
            ox.Observatory(
                make_telescope(8.0),
                (
                    ox.Instrument(
                        "imager",
                        coronagraph=coro,
                        detector=ox.IdealDetector(
                            pixel_scale_arcsec=0.01, shape=(8, 8)
                        ),
                    ),
                ),
            )
        assert make_instrument(coro).select("imager").coronagraph is coro

    def test_public_check_names_the_source(self):
        coro = table_coronagraph()
        pupil = ox.SimplePupil(circumscribed_diameter_m=8.0, collecting_area_m2=28.0)
        with pytest.raises(ValueError, match="differs from the curves"):
            ox.check_pupil_matches_frame(pupil, coro.frame, source="curves")
