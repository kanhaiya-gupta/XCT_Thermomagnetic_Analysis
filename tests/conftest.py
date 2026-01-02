"""
Pytest configuration and shared fixtures for XCT Thermomagnetic Analysis tests.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set random seed for reproducibility
np.random.seed(42)


@pytest.fixture
def simple_volume():
    """Create a simple 3D test volume (50x50x50)."""
    volume = np.zeros((50, 50, 50), dtype=np.uint8)
    # Add a sphere in the center
    center = (25, 25, 25)
    radius = 10
    for i in range(50):
        for j in range(50):
            for k in range(50):
                dist = np.sqrt(
                    (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
                )
                if dist <= radius:
                    volume[i, j, k] = 255
    return volume


@pytest.fixture
def porous_volume():
    """Create a volume with known porosity (30% void fraction)."""
    volume = np.ones((100, 100, 100), dtype=np.uint8) * 255
    # Add random pores
    n_pores = 3000
    for _ in range(n_pores):
        i = np.random.randint(10, 90)
        j = np.random.randint(10, 90)
        k = np.random.randint(10, 90)
        size = np.random.randint(2, 5)
        volume[i - size : i + size, j - size : j + size, k - size : k + size] = 0
    return volume


@pytest.fixture
def filament_volume():
    """Create a volume with filaments along z-direction."""
    volume = np.zeros((100, 100, 100), dtype=np.uint8)
    # Create filaments (cylinders) along z-direction
    filament_radius = 3
    n_filaments = 5

    for f in range(n_filaments):
        center_x = 20 + f * 15
        center_y = 50
        for z in range(100):
            for i in range(100):
                for j in range(100):
                    dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    if dist <= filament_radius:
                        volume[i, j, z] = 255
    return volume


@pytest.fixture
def edge_case_volumes():
    """Create edge case volumes for testing."""
    cases = {
        "empty": np.zeros((50, 50, 50), dtype=np.uint8),
        "full": np.ones((50, 50, 50), dtype=np.uint8) * 255,
        "single_value": np.ones((50, 50, 50), dtype=np.uint8) * 128,
        "all_zeros": np.zeros((50, 50, 50), dtype=np.uint8),
        "all_ones": np.ones((50, 50, 50), dtype=np.uint8) * 255,
    }
    return cases


@pytest.fixture
def voxel_size():
    """Standard voxel size for tests (0.1 mm)."""
    return (0.1, 0.1, 0.1)


@pytest.fixture
def sample_metrics():
    """Sample metric values for testing."""
    return {
        "volume": 1000.0,  # mm³
        "surface_area": 500.0,  # mm²
        "void_fraction": 0.3,
        "relative_density": 0.7,
        "specific_surface_area": 0.5,  # mm⁻¹
    }


@pytest.fixture
def sample_distributions():
    """Sample data with known distributions for testing."""
    np.random.seed(42)
    return {
        "gaussian": np.random.normal(100, 10, 1000),
        "poisson": np.random.poisson(5, 1000),
        "linear": np.linspace(0, 100, 100) + np.random.normal(0, 1, 100),
    }


@pytest.fixture
def sample_process_params():
    """Sample process parameters for testing."""
    return {
        "extrusion_temp": [200, 210, 220, 230, 240],
        "print_speed": [10, 15, 20, 25, 30],
        "layer_height": [0.1, 0.15, 0.2, 0.25, 0.3],
    }


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path / "test_data"


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path / "outputs"


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests (performance benchmarks)")
    config.addinivalue_line(
        "markers", "requires_data: Tests that require external data files"
    )
