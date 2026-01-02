"""
Synthetic volume generation for testing.
"""

import numpy as np
from typing import Tuple, Optional


def create_sphere_volume(
    shape: Tuple[int, int, int],
    center: Tuple[int, int, int],
    radius: float,
    value: int = 255,
) -> np.ndarray:
    """Create a volume with a sphere."""
    volume = np.zeros(shape, dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt(
                    (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
                )
                if dist <= radius:
                    volume[i, j, k] = value
    return volume


def create_cube_volume(
    shape: Tuple[int, int, int],
    corner: Tuple[int, int, int],
    size: Tuple[int, int, int],
    value: int = 255,
) -> np.ndarray:
    """Create a volume with a cube."""
    volume = np.zeros(shape, dtype=np.uint8)
    i0, j0, k0 = corner
    di, dj, dk = size
    volume[i0 : i0 + di, j0 : j0 + dj, k0 : k0 + dk] = value
    return volume


def create_cylinder_volume(
    shape: Tuple[int, int, int],
    center: Tuple[int, int],
    radius: float,
    axis: int = 2,
    value: int = 255,
) -> np.ndarray:
    """Create a volume with a cylinder along specified axis."""
    volume = np.zeros(shape, dtype=np.uint8)
    cx, cy = center

    if axis == 2:  # z-direction
        for i in range(shape[0]):
            for j in range(shape[1]):
                dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist <= radius:
                    volume[i, j, :] = value
    elif axis == 1:  # y-direction
        for i in range(shape[0]):
            for k in range(shape[2]):
                dist = np.sqrt((i - cx) ** 2 + (k - cy) ** 2)
                if dist <= radius:
                    volume[i, :, k] = value
    else:  # x-direction
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((j - cx) ** 2 + (k - cy) ** 2)
                if dist <= radius:
                    volume[:, j, k] = value

    return volume


def create_porous_volume(
    shape: Tuple[int, int, int], porosity: float = 0.3, seed: Optional[int] = 42
) -> np.ndarray:
    """Create a volume with specified porosity."""
    if seed is not None:
        np.random.seed(seed)

    volume = np.ones(shape, dtype=np.uint8) * 255
    n_voxels = np.prod(shape)
    n_pores = int(n_voxels * porosity)

    # Create random pores
    pore_positions = np.random.randint(
        (shape[0] // 10, shape[1] // 10, shape[2] // 10),
        (9 * shape[0] // 10, 9 * shape[1] // 10, 9 * shape[2] // 10),
        size=(n_pores, 3),
    )

    for pos in pore_positions:
        size = np.random.randint(1, 4)
        i, j, k = pos
        i0, i1 = max(0, i - size), min(shape[0], i + size)
        j0, j1 = max(0, j - size), min(shape[1], j + size)
        k0, k1 = max(0, k - size), min(shape[2], k + size)
        volume[i0:i1, j0:j1, k0:k1] = 0

    return volume


def create_filament_volume(
    shape: Tuple[int, int, int],
    n_filaments: int = 5,
    filament_radius: float = 3.0,
    direction: int = 2,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Create a volume with filaments."""
    if seed is not None:
        np.random.seed(seed)

    volume = np.zeros(shape, dtype=np.uint8)

    # Distribute filaments evenly
    if direction == 2:  # z-direction
        spacing = shape[0] // (n_filaments + 1)
        for f in range(n_filaments):
            center_x = spacing * (f + 1)
            center_y = shape[1] // 2
            for i in range(shape[0]):
                for j in range(shape[1]):
                    dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    if dist <= filament_radius:
                        volume[i, j, :] = 255

    return volume


def create_test_pattern_volume(
    shape: Tuple[int, int, int], pattern_type: str = "checkerboard"
) -> np.ndarray:
    """Create volumes with test patterns."""
    volume = np.zeros(shape, dtype=np.uint8)

    if pattern_type == "checkerboard":
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if (i + j + k) % 2 == 0:
                        volume[i, j, k] = 255

    return volume
