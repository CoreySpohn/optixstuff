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
    return ox.SimpleDetector(
        qe=0.9,
        dark_current_electrons_per_s=1e-4,
        read_noise_electrons=3.0,
        cic_electrons=0.02,
    )


@pytest.fixture
def throughput_element():
    """A constant 80% throughput optical element."""
    return ox.ConstantThroughputElement(throughput=0.8, name="test_filter")
