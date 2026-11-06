"""Shared fixtures for optixstuff tests."""

import importlib.util
import os
import shutil
from zipfile import ZipFile

import pytest
from hypothesis import settings
from yippy import EqxCoronagraph
from yippy.datasets import CATALOG, cache_dir

import optixstuff as ox

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


@pytest.fixture(scope="session")
def cached_diameter_coro(tmp_path_factory):
    """Use one cached public package copy across all diameter and frame tests."""
    root = cache_dir()
    scratch = tmp_path_factory.mktemp("stellar_diameter_yip")
    for name in CATALOG:
        candidates = (root / f"{name}.zip.unzip" / name, root / name)
        source = next((path for path in candidates if path.is_dir()), None)
        target = scratch / name
        if source is not None:
            shutil.copytree(source, target)
        elif (root / f"{name}.zip").is_file():
            with ZipFile(root / f"{name}.zip") as archive:
                archive.extractall(scratch)
        else:
            continue
        backend = EqxCoronagraph(target)
        if backend._core_intensity_varies_with_diameter:
            return ox.YippyCoronagraph(backend=backend)
    pytest.skip(f"No cached public YIP with a stellar-diameter axis in {root}")


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
