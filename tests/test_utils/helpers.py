"""
Test utilities and helper functions.
"""

import numpy as np
from typing import Dict, Any, Tuple


def assert_metrics_close(
    computed: Dict[str, float],
    expected: Dict[str, float],
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> None:
    """Assert that computed metrics are close to expected values."""
    for key in expected:
        assert key in computed, f"Missing metric: {key}"
        np.testing.assert_allclose(
            computed[key],
            expected[key],
            rtol=rtol,
            atol=atol,
            err_msg=f"Metric {key} mismatch",
        )


def assert_volume_valid(volume: np.ndarray) -> None:
    """Assert that a volume has valid properties."""
    assert volume is not None, "Volume is None"
    assert isinstance(volume, np.ndarray), "Volume is not a numpy array"
    assert volume.ndim == 3, f"Volume should be 3D, got {volume.ndim}D"
    assert volume.size > 0, "Volume is empty"
    assert volume.dtype in [
        np.uint8,
        np.uint16,
        np.float32,
        np.float64,
    ], f"Invalid volume dtype: {volume.dtype}"


def create_test_volume(
    shape: Tuple[int, int, int] = (50, 50, 50), dtype: np.dtype = np.uint8
) -> np.ndarray:
    """Create a simple test volume."""
    volume = np.zeros(shape, dtype=dtype)
    # Add a small cube in the center
    center = tuple(s // 2 for s in shape)
    size = tuple(min(10, s // 4) for s in shape)
    corner = tuple(c - s // 2 for c, s in zip(center, size))

    i0, i1 = corner[0], corner[0] + size[0]
    j0, j1 = corner[1], corner[1] + size[1]
    k0, k1 = corner[2], corner[2] + size[2]

    if dtype in [np.uint8, np.uint16]:
        volume[i0:i1, j0:j1, k0:k1] = 255
    else:
        volume[i0:i1, j0:j1, k0:k1] = 1.0

    return volume


def get_known_sphere_volume(
    radius: float, voxel_size: Tuple[float, float, float]
) -> float:
    """Calculate known volume of a sphere."""
    return (4.0 / 3.0) * np.pi * (radius**3)


def get_known_cube_volume(side_length: float) -> float:
    """Calculate known volume of a cube."""
    return side_length**3
