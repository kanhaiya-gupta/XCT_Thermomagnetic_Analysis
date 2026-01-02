"""
Preprocessing and Filtering Module

Filter and clean segmented XCT data before analysis:
- Filter by voxel size (volume)
- Filter by sphericity
- Filter by spatial coordinates (X, Y, Z)
- Filter by aspect ratio
- Remove edge objects
- Remove small/large objects
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology
import logging

logger = logging.getLogger(__name__)


def compute_sphericity(
    volume: np.ndarray, voxel_size: Tuple[float, float, float]
) -> float:
    """
    Compute sphericity of a 3D object.

    Sphericity = (36π * V²)^(1/3) / A
    where V is volume and A is surface area.

    Sphericity ranges from 0 to 1, where 1 is a perfect sphere.

    Args:
        volume: Binary volume (1 = object, 0 = background)
        voxel_size: Voxel spacing in mm (dx, dy, dz)

    Returns:
        Sphericity value (0.0 to 1.0)
    """
    voxel_volume = np.prod(voxel_size)

    # Compute volume
    object_volume = np.sum(volume > 0) * voxel_volume

    if object_volume == 0:
        return 0.0

    # Compute surface area using marching cubes
    try:
        from skimage import measure

        verts, faces, normals, values = measure.marching_cubes(
            volume.astype(float), level=0.5, spacing=voxel_size
        )
        surface_area = measure.mesh_surface_area(verts, faces)
    except Exception:
        # Fallback: approximate surface area
        # Count exposed faces
        diff_x = np.diff(volume, axis=2)
        diff_y = np.diff(volume, axis=1)
        diff_z = np.diff(volume, axis=0)

        dx, dy, dz = voxel_size
        faces_x = np.sum(np.abs(diff_x)) * (dy * dz)
        faces_y = np.sum(np.abs(diff_y)) * (dx * dz)
        faces_z = np.sum(np.abs(diff_z)) * (dx * dy)

        # Boundary faces
        boundary_x = (np.sum(volume[:, :, 0]) + np.sum(volume[:, :, -1])) * (dy * dz)
        boundary_y = (np.sum(volume[:, 0, :]) + np.sum(volume[:, -1, :])) * (dx * dz)
        boundary_z = (np.sum(volume[0, :, :]) + np.sum(volume[-1, :, :])) * (dx * dy)

        surface_area = (
            faces_x + faces_y + faces_z + boundary_x + boundary_y + boundary_z
        )

    if surface_area == 0:
        return 0.0

    # Sphericity formula
    sphericity = (36 * np.pi * object_volume**2) ** (1 / 3) / surface_area

    return float(sphericity)


def compute_aspect_ratio(
    volume: np.ndarray, voxel_size: Tuple[float, float, float]
) -> Tuple[float, float, float, float]:
    """
    Compute aspect ratios of a 3D object.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing

    Returns:
        Tuple of (max_ratio, x_y_ratio, x_z_ratio, y_z_ratio)
    """
    # Get bounding box
    coords = np.argwhere(volume > 0)
    if len(coords) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    # Convert to physical coordinates
    coords_physical = coords * np.array(voxel_size)

    # Compute dimensions (in physical coordinates)
    # Note: coords_physical is in (z, y, x) order from numpy
    dims_physical = coords_physical.max(axis=0) - coords_physical.min(axis=0)

    # Get dimensions for each axis (z, y, x)
    dim_z = dims_physical[0] if dims_physical[0] > 0 else 0.0
    dim_y = dims_physical[1] if dims_physical[1] > 0 else 0.0
    dim_x = dims_physical[2] if dims_physical[2] > 0 else 0.0

    # Get non-zero dimensions
    dims = np.array([dim_z, dim_y, dim_x])
    dims = dims[dims > 0]

    if len(dims) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    max_dim = np.max(dims)
    min_dim = np.min(dims)
    max_ratio = max_dim / min_dim if min_dim > 0 else 0.0

    # Individual ratios (x, y, z correspond to physical coordinates)
    # x_y_ratio: ratio of x to y dimensions
    if dim_x > 0 and dim_y > 0:
        x_y_ratio = dim_x / dim_y
    else:
        x_y_ratio = 0.0

    # x_z_ratio: ratio of x to z dimensions
    if dim_x > 0 and dim_z > 0:
        x_z_ratio = dim_x / dim_z
    else:
        x_z_ratio = 0.0

    # y_z_ratio: ratio of y to z dimensions
    if dim_y > 0 and dim_z > 0:
        y_z_ratio = dim_y / dim_z
    else:
        y_z_ratio = 0.0

    return (float(max_ratio), float(x_y_ratio), float(x_z_ratio), float(y_z_ratio))


def filter_by_volume(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    min_volume: Optional[float] = None,
    max_volume: Optional[float] = None,
) -> np.ndarray:
    """
    Filter objects by volume (in mm³).

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        min_volume: Minimum volume in mm³
        max_volume: Maximum volume in mm³

    Returns:
        Filtered binary volume
    """
    if min_volume is None and max_volume is None:
        return volume

    voxel_volume = np.prod(voxel_size)

    # Label connected components
    labeled, num_features = ndimage.label(volume)

    filtered_volume = np.zeros_like(volume)

    for label_id in range(1, num_features + 1):
        object_mask = labeled == label_id
        object_volume = np.sum(object_mask) * voxel_volume

        keep = True
        if min_volume is not None and object_volume < min_volume:
            keep = False
        if max_volume is not None and object_volume > max_volume:
            keep = False

        if keep:
            filtered_volume[object_mask] = 1

    return filtered_volume


def filter_by_sphericity(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    min_sphericity: float = 0.0,
    max_sphericity: float = 1.0,
) -> np.ndarray:
    """
    Filter objects by sphericity.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        min_sphericity: Minimum sphericity (0.0 to 1.0)
        max_sphericity: Maximum sphericity (0.0 to 1.0)

    Returns:
        Filtered binary volume
    """
    # Label connected components
    labeled, num_features = ndimage.label(volume)

    filtered_volume = np.zeros_like(volume)

    for label_id in range(1, num_features + 1):
        object_mask = labeled == label_id
        object_volume = np.zeros_like(volume)
        object_volume[object_mask] = 1

        sphericity = compute_sphericity(object_volume, voxel_size)

        if min_sphericity <= sphericity <= max_sphericity:
            filtered_volume[object_mask] = 1

    return filtered_volume


def filter_by_spatial_bounds(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Filter objects by spatial bounds (X, Y, Z coordinates).

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        x_range: (min_x, max_x) in mm
        y_range: (min_y, max_y) in mm
        z_range: (min_z, max_z) in mm

    Returns:
        Filtered binary volume
    """
    if x_range is None and y_range is None and z_range is None:
        return volume

    # Label connected components
    labeled, num_features = ndimage.label(volume)

    filtered_volume = np.zeros_like(volume)

    for label_id in range(1, num_features + 1):
        object_mask = labeled == label_id
        coords = np.argwhere(object_mask)

        if len(coords) == 0:
            continue

        # Convert to physical coordinates
        coords_physical = coords * np.array(voxel_size)

        # Check bounds
        # Note: coords_physical is in (z, y, x) order from numpy indexing
        # But we want to check if object is WITHIN bounds (not just touching)
        keep = True

        if x_range is not None:
            x_coords = coords_physical[:, 2]  # x is last dimension
            x_min_obj, x_max_obj = x_coords.min(), x_coords.max()
            # Object is within bounds if its entire range is within bounds
            if x_min_obj < x_range[0] or x_max_obj > x_range[1]:
                keep = False

        if y_range is not None:
            y_coords = coords_physical[:, 1]  # y is middle dimension
            y_min_obj, y_max_obj = y_coords.min(), y_coords.max()
            if y_min_obj < y_range[0] or y_max_obj > y_range[1]:
                keep = False

        if z_range is not None:
            z_coords = coords_physical[:, 0]  # z is first dimension
            z_min_obj, z_max_obj = z_coords.min(), z_coords.max()
            if z_min_obj < z_range[0] or z_max_obj > z_range[1]:
                keep = False

        if keep:
            filtered_volume[object_mask] = 1

    return filtered_volume


def filter_by_aspect_ratio(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    max_aspect_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    Filter objects by maximum aspect ratio.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        max_aspect_ratio: Maximum aspect ratio (ratio of longest to shortest dimension)

    Returns:
        Filtered binary volume
    """
    if max_aspect_ratio is None:
        return volume

    # Label connected components
    labeled, num_features = ndimage.label(volume)

    filtered_volume = np.zeros_like(volume)

    for label_id in range(1, num_features + 1):
        object_mask = labeled == label_id
        object_volume = np.zeros_like(volume)
        object_volume[object_mask] = 1

        max_ratio, _, _, _ = compute_aspect_ratio(object_volume, voxel_size)

        if max_ratio <= max_aspect_ratio:
            filtered_volume[object_mask] = 1

    return filtered_volume


def remove_edge_objects(volume: np.ndarray, margin: int = 1) -> np.ndarray:
    """
    Remove objects touching the edges of the volume.

    Args:
        volume: Binary volume
        margin: Margin from edge (in voxels)

    Returns:
        Filtered binary volume
    """
    # Create edge mask
    edge_mask = np.zeros_like(volume, dtype=bool)
    edge_mask[:margin, :, :] = True
    edge_mask[-margin:, :, :] = True
    edge_mask[:, :margin, :] = True
    edge_mask[:, -margin:, :] = True
    edge_mask[:, :, :margin] = True
    edge_mask[:, :, -margin:] = True

    # Label connected components
    labeled, num_features = ndimage.label(volume)

    # Find objects touching edges
    edge_labels = set(labeled[edge_mask])
    edge_labels.discard(0)  # Remove background

    # Create filtered volume
    filtered_volume = np.zeros_like(volume)
    for label_id in range(1, num_features + 1):
        if label_id not in edge_labels:
            filtered_volume[labeled == label_id] = 1

    return filtered_volume


def apply_filters(
    volume: np.ndarray, voxel_size: Tuple[float, float, float], filters: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply multiple filters to segmented volume.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        filters: Dictionary with filter parameters:
            - min_volume: Minimum volume in mm³
            - max_volume: Maximum volume in mm³
            - min_sphericity: Minimum sphericity
            - max_sphericity: Maximum sphericity
            - x_range: (min_x, max_x) in mm
            - y_range: (min_y, max_y) in mm
            - z_range: (min_z, max_z) in mm
            - max_aspect_ratio: Maximum aspect ratio
            - remove_edge_objects: Boolean, remove objects touching edges
            - edge_margin: Margin for edge removal (voxels)

    Returns:
        Filtered volume and statistics dictionary
    """
    filtered = volume.copy()
    stats = {
        "initial_objects": 0,
        "final_objects": 0,
        "removed_objects": 0,
        "filters_applied": [],
    }

    # Count initial objects
    labeled, num_features = ndimage.label(filtered)
    stats["initial_objects"] = num_features

    # Apply volume filter
    if "min_volume" in filters or "max_volume" in filters:
        filtered = filter_by_volume(
            filtered,
            voxel_size,
            min_volume=filters.get("min_volume"),
            max_volume=filters.get("max_volume"),
        )
        stats["filters_applied"].append("volume")

    # Apply sphericity filter
    if "min_sphericity" in filters or "max_sphericity" in filters:
        filtered = filter_by_sphericity(
            filtered,
            voxel_size,
            min_sphericity=filters.get("min_sphericity", 0.0),
            max_sphericity=filters.get("max_sphericity", 1.0),
        )
        stats["filters_applied"].append("sphericity")

    # Apply spatial bounds filter
    if "x_range" in filters or "y_range" in filters or "z_range" in filters:
        filtered = filter_by_spatial_bounds(
            filtered,
            voxel_size,
            x_range=filters.get("x_range"),
            y_range=filters.get("y_range"),
            z_range=filters.get("z_range"),
        )
        stats["filters_applied"].append("spatial_bounds")

    # Apply aspect ratio filter
    if "max_aspect_ratio" in filters:
        filtered = filter_by_aspect_ratio(
            filtered, voxel_size, max_aspect_ratio=filters.get("max_aspect_ratio")
        )
        stats["filters_applied"].append("aspect_ratio")

    # Remove edge objects
    if filters.get("remove_edge_objects", False):
        margin = filters.get("edge_margin", 1)
        filtered = remove_edge_objects(filtered, margin=margin)
        stats["filters_applied"].append("remove_edge_objects")

    # Count final objects
    labeled_final, num_features_final = ndimage.label(filtered)
    stats["final_objects"] = num_features_final
    stats["removed_objects"] = stats["initial_objects"] - stats["final_objects"]

    logger.info(
        f"Filtering complete: {stats['initial_objects']} → {stats['final_objects']} objects "
        f"({stats['removed_objects']} removed)"
    )

    return filtered, stats


def analyze_object_properties(
    volume: np.ndarray, voxel_size: Tuple[float, float, float]
) -> List[Dict[str, Any]]:
    """
    Analyze properties of all objects in segmented volume.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing

    Returns:
        List of dictionaries with object properties
    """
    # Label connected components
    labeled, num_features = ndimage.label(volume)

    objects = []
    voxel_volume = np.prod(voxel_size)

    for label_id in range(1, num_features + 1):
        object_mask = labeled == label_id
        object_volume = np.zeros_like(volume)
        object_volume[object_mask] = 1

        # Compute properties
        coords = np.argwhere(object_mask)
        coords_physical = coords * np.array(voxel_size)

        # Volume
        volume_mm3 = np.sum(object_mask) * voxel_volume

        # Centroid
        centroid = coords_physical.mean(axis=0)

        # Bounding box
        bbox_min = coords_physical.min(axis=0)
        bbox_max = coords_physical.max(axis=0)
        bbox_size = bbox_max - bbox_min

        # Sphericity
        sphericity = compute_sphericity(object_volume, voxel_size)

        # Aspect ratio
        max_ratio, x_y_ratio, x_z_ratio, y_z_ratio = compute_aspect_ratio(
            object_volume, voxel_size
        )

        # Check if on edge
        on_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == volume.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == volume.shape[1] - 1)
            or np.any(coords[:, 2] == 0)
            or np.any(coords[:, 2] == volume.shape[2] - 1)
        )

        objects.append(
            {
                "label_id": int(label_id),
                "volume_mm3": float(volume_mm3),
                "voxel_count": int(np.sum(object_mask)),
                "centroid_x": float(centroid[2]),  # Note: numpy uses (z, y, x)
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[0]),
                "bbox_min_x": float(bbox_min[2]),
                "bbox_min_y": float(bbox_min[1]),
                "bbox_min_z": float(bbox_min[0]),
                "bbox_max_x": float(bbox_max[2]),
                "bbox_max_y": float(bbox_max[1]),
                "bbox_max_z": float(bbox_max[0]),
                "bbox_size_x": float(bbox_size[2]),
                "bbox_size_y": float(bbox_size[1]),
                "bbox_size_z": float(bbox_size[0]),
                "sphericity": float(sphericity),
                "max_aspect_ratio": float(max_ratio),
                "x_y_ratio": float(x_y_ratio),
                "x_z_ratio": float(x_z_ratio),
                "y_z_ratio": float(y_z_ratio),
                "on_edge": bool(on_edge),
            }
        )

    return objects
