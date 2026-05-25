"""Shared fixtures for optixstuff tests."""

import pytest

import optixstuff as ox


@pytest.fixture
def simple_primary():
    """A basic 6m primary mirror with 14% obscuration."""
    return ox.SimplePrimary(diameter_m=6.0, obscuration=0.14)


@pytest.fixture
def simple_detector():
    """A simple detector with typical HWO-like parameters."""
    return ox.IdealDetector(
        pixel_scale_arcsec=0.010,
        shape=(100, 100),
        quantum_efficiency=0.9,
        dark_current_rate_e_per_s=1e-4,
        read_noise_e=3.0,
        clock_induced_charge_rate_e_per_frame=0.02,
        frame_time_s=1000.0,
    )


@pytest.fixture
def throughput_element():
    """A constant 80% throughput optical element."""
    return ox.ConstantThroughput(throughput=0.8, name="test_filter")
