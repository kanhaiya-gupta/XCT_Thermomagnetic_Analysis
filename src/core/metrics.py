"""
Scalar Metrics Module

Computation of scalar quantities from segmented volumes:
- Volume
- Total surface area
- Void fraction (porosity)
- Relative density
- Specific surface area
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import ndimage
from skimage import measure
import logging

logger = logging.getLogger(__name__)


def compute_volume(volume: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
    """
    Calculate total material volume.

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm (dx, dy, dz)

    Returns:
        Total volume in mm³
    """
    voxel_volume = np.prod(voxel_size)
    material_voxels = np.sum(volume > 0)
    total_volume = material_voxels * voxel_volume

    logger.debug(
        f"Material voxels: {material_voxels}, Voxel volume: {voxel_volume:.6f} mm³, "
        f"Total volume: {total_volume:.3f} mm³"
    )

    return float(total_volume)


def compute_surface_area(
    volume: np.ndarray, voxel_size: Tuple[float, float, float]
) -> float:
    """
    Calculate total surface area using marching cubes algorithm.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing in mm (dx, dy, dz)

    Returns:
        Total surface area in mm²
    """
    # Use marching cubes to compute surface
    try:
        verts, faces, normals, values = measure.marching_cubes(
            volume.astype(float), level=0.5, spacing=voxel_size
        )

        # Calculate surface area from faces
        surface_area = measure.mesh_surface_area(verts, faces)

        logger.debug(f"Surface area: {surface_area:.3f} mm²")
        return float(surface_area)

    except Exception as e:
        logger.warning(f"Marching cubes failed: {e}, using voxel-based approximation")
        # Fallback: approximate surface area from voxel faces
        return compute_surface_area_voxel_based(volume, voxel_size)


def compute_surface_area_voxel_based(
    volume: np.ndarray, voxel_size: Tuple[float, float, float]
) -> float:
    """
    Approximate surface area by counting exposed voxel faces.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing in mm

    Returns:
        Approximate surface area in mm²
    """
    # Count exposed faces in each direction
    dx, dy, dz = voxel_size

    # Faces in x-direction
    diff_x = np.diff(volume, axis=2)
    faces_x = np.sum(np.abs(diff_x)) * (dy * dz)

    # Faces in y-direction
    diff_y = np.diff(volume, axis=1)
    faces_y = np.sum(np.abs(diff_y)) * (dx * dz)

    # Faces in z-direction
    diff_z = np.diff(volume, axis=0)
    faces_z = np.sum(np.abs(diff_z)) * (dx * dy)

    # Boundary faces
    boundary_x = (np.sum(volume[:, :, 0]) + np.sum(volume[:, :, -1])) * (dy * dz)
    boundary_y = (np.sum(volume[:, 0, :]) + np.sum(volume[:, -1, :])) * (dx * dz)
    boundary_z = (np.sum(volume[0, :, :]) + np.sum(volume[-1, :, :])) * (dx * dy)

    total_area = faces_x + faces_y + faces_z + boundary_x + boundary_y + boundary_z

    return float(total_area)


def compute_void_fraction(volume: np.ndarray) -> float:
    """
    Calculate void fraction (porosity) as percentage of void space.

    Args:
        volume: Binary volume (1 = material, 0 = void)

    Returns:
        Void fraction (0.0 to 1.0)
    """
    total_voxels = volume.size
    void_voxels = np.sum(volume == 0)
    void_fraction = void_voxels / total_voxels

    logger.debug(f"Void fraction: {void_fraction:.2%}")
    return float(void_fraction)


def compute_relative_density(volume: np.ndarray) -> float:
    """
    Calculate relative density (percentage of material).

    Args:
        volume: Binary volume (1 = material, 0 = void)

    Returns:
        Relative density (0.0 to 1.0)
    """
    return 1.0 - compute_void_fraction(volume)


def compute_specific_surface_area(
    volume: np.ndarray, voxel_size: Tuple[float, float, float]
) -> float:
    """
    Calculate specific surface area (surface area per unit volume).

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing in mm

    Returns:
        Specific surface area in mm²/mm³
    """
    total_volume = compute_volume(volume, voxel_size)
    if total_volume == 0:
        return 0.0

    surface_area = compute_surface_area(volume, voxel_size)
    specific_surface_area = surface_area / total_volume

    return float(specific_surface_area)


def compute_all_metrics(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    include_surface_area: bool = True,
) -> Dict[str, Any]:
    """
    Compute all scalar metrics at once.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing in mm
        include_surface_area: Whether to compute surface area (can be slow)

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Basic metrics
    metrics["volume"] = compute_volume(volume, voxel_size)
    metrics["void_fraction"] = compute_void_fraction(volume)
    metrics["relative_density"] = compute_relative_density(volume)

    # Surface area (optional, can be slow)
    if include_surface_area:
        metrics["surface_area"] = compute_surface_area(volume, voxel_size)
        metrics["specific_surface_area"] = compute_specific_surface_area(
            volume, voxel_size
        )
    else:
        metrics["surface_area"] = None
        metrics["specific_surface_area"] = None

    # Additional statistics
    metrics["total_voxels"] = int(volume.size)
    metrics["material_voxels"] = int(np.sum(volume > 0))
    metrics["void_voxels"] = int(np.sum(volume == 0))

    logger.info(
        f"Computed metrics: Volume={metrics['volume']:.3f} mm³, "
        f"Void fraction={metrics['void_fraction']:.2%}"
    )

    return metrics


def compute_volume_fraction_map(
    volume: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """
    Compute local volume fraction (density) map using sliding window.

    Args:
        volume: Binary volume
        window_size: Size of sliding window

    Returns:
        Volume fraction map (0.0 to 1.0)
    """
    from scipy.ndimage import uniform_filter

    # Convert to float for filtering
    volume_float = volume.astype(float)

    # Apply uniform filter (moving average)
    filtered = uniform_filter(volume_float, size=window_size)

    return filtered
