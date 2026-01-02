"""
Filament and Channel Analysis Module

Estimation of mean filament diameter and mean channel width from
segmented XCT volumes of 3D-printed structures.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy import ndimage
from skimage import measure, morphology
import logging

try:
    from ..preprocessing.statistics import fit_gaussian, compare_fits

    HAS_STATISTICS = True
except ImportError:
    HAS_STATISTICS = False

logger = logging.getLogger(__name__)


def estimate_filament_diameter(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    direction: str = "z",
    method: str = "distance_transform",
) -> Dict[str, Any]:
    """
    Estimate mean filament diameter along specified direction.

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm (dx, dy, dz)
        direction: Direction along which filaments extend ('x', 'y', 'z')
        method: Method for diameter estimation ('distance_transform', 'cross_section')

    Returns:
        Dictionary with diameter statistics
    """
    if method == "distance_transform":
        return _estimate_diameter_distance_transform(volume, voxel_size, direction)
    elif method == "cross_section":
        return _estimate_diameter_cross_section(volume, voxel_size, direction)
    else:
        raise ValueError(f"Unknown method: {method}")


def _estimate_diameter_distance_transform(
    volume: np.ndarray, voxel_size: Tuple[float, float, float], direction: str
) -> Dict[str, Any]:
    """
    Estimate diameter using distance transform (distance from center to edge).
    """
    from .morphology import distance_transform

    # Extract cross-sections perpendicular to filament direction
    if direction == "z":
        axis = 0
        spacing = (voxel_size[1], voxel_size[2])
    elif direction == "y":
        axis = 1
        spacing = (voxel_size[0], voxel_size[2])
    else:  # 'x'
        axis = 2
        spacing = (voxel_size[0], voxel_size[1])

    diameters = []

    # Sample slices along the direction
    n_slices = min(volume.shape[axis], 20)  # Sample up to 20 slices
    slice_indices = np.linspace(0, volume.shape[axis] - 1, n_slices, dtype=int)

    for idx in slice_indices:
        if direction == "z":
            slice_2d = volume[idx, :, :]
        elif direction == "y":
            slice_2d = volume[:, idx, :]
        else:
            slice_2d = volume[:, :, idx]

        # Label connected components (filaments)
        labeled, num_features = ndimage.label(slice_2d)

        for label_id in range(1, num_features + 1):
            filament_mask = labeled == label_id

            # Distance transform from center
            distance = distance_transform(filament_mask.astype(np.uint8))

            # Maximum distance (radius) in physical units
            max_distance = np.max(distance) * np.mean(spacing)
            diameter = max_distance * 2  # diameter = 2 * radius

            if diameter > 0:
                diameters.append(diameter)

    if len(diameters) == 0:
        logger.warning("No filaments detected")
        return {
            "mean_diameter": 0.0,
            "std_diameter": 0.0,
            "min_diameter": 0.0,
            "max_diameter": 0.0,
            "diameters": [],
        }

    diameters = np.array(diameters)

    return {
        "mean_diameter": float(np.mean(diameters)),
        "std_diameter": float(np.std(diameters)),
        "min_diameter": float(np.min(diameters)),
        "max_diameter": float(np.max(diameters)),
        "median_diameter": float(np.median(diameters)),
        "diameters": diameters.tolist(),
    }


def _estimate_diameter_cross_section(
    volume: np.ndarray, voxel_size: Tuple[float, float, float], direction: str
) -> Dict[str, Any]:
    """
    Estimate diameter by analyzing cross-sectional areas.
    """
    if direction == "z":
        axis = 0
    elif direction == "y":
        axis = 1
    else:  # 'x'
        axis = 2

    diameters = []

    # Sample slices
    n_slices = min(volume.shape[axis], 20)
    slice_indices = np.linspace(0, volume.shape[axis] - 1, n_slices, dtype=int)

    for idx in slice_indices:
        if direction == "z":
            slice_2d = volume[idx, :, :]
            pixel_area = voxel_size[1] * voxel_size[2]
        elif direction == "y":
            slice_2d = volume[:, idx, :]
            pixel_area = voxel_size[0] * voxel_size[2]
        else:
            slice_2d = volume[:, :, idx]
            pixel_area = voxel_size[0] * voxel_size[1]

        # Label connected components
        labeled, num_features = ndimage.label(slice_2d)

        for label_id in range(1, num_features + 1):
            filament_mask = labeled == label_id
            area = np.sum(filament_mask) * pixel_area

            # Estimate diameter from area (assuming circular cross-section)
            diameter = 2 * np.sqrt(area / np.pi)

            if diameter > 0:
                diameters.append(diameter)

    if len(diameters) == 0:
        return {
            "mean_diameter": 0.0,
            "std_diameter": 0.0,
            "min_diameter": 0.0,
            "max_diameter": 0.0,
            "diameters": [],
        }

    diameters = np.array(diameters)

    return {
        "mean_diameter": float(np.mean(diameters)),
        "std_diameter": float(np.std(diameters)),
        "min_diameter": float(np.min(diameters)),
        "max_diameter": float(np.max(diameters)),
        "median_diameter": float(np.median(diameters)),
        "diameters": diameters.tolist(),
    }


def estimate_channel_width(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    direction: str = "z",
    method: str = "distance_transform",
) -> Dict[str, Any]:
    """
    Estimate mean channel width (void space between filaments).

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm
        direction: Direction along which channels extend
        method: Estimation method

    Returns:
        Dictionary with channel width statistics
    """
    # Invert volume (void = 1, material = 0)
    void_volume = 1 - volume

    # Use same methods as filament diameter but on inverted volume
    if method == "distance_transform":
        return _estimate_diameter_distance_transform(void_volume, voxel_size, direction)
    elif method == "cross_section":
        return _estimate_diameter_cross_section(void_volume, voxel_size, direction)
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_cross_section(
    slice_2d: np.ndarray, voxel_size: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Analyze a 2D cross-section slice.

    Args:
        slice_2d: 2D binary slice
        voxel_size: Voxel spacing (dx, dy) in mm

    Returns:
        Dictionary with cross-section metrics
    """
    pixel_area = voxel_size[0] * voxel_size[1]

    # Material properties
    material_mask = slice_2d > 0
    material_area = np.sum(material_mask) * pixel_area
    material_fraction = np.mean(material_mask)

    # Void properties
    void_mask = slice_2d == 0
    void_area = np.sum(void_mask) * pixel_area
    void_fraction = np.mean(void_mask)

    # Connected components
    labeled_material, n_material = ndimage.label(material_mask)
    labeled_void, n_void = ndimage.label(void_mask)

    # Average sizes
    material_sizes = []
    void_sizes = []

    for label_id in range(1, n_material + 1):
        size = np.sum(labeled_material == label_id) * pixel_area
        material_sizes.append(size)

    for label_id in range(1, n_void + 1):
        size = np.sum(labeled_void == label_id) * pixel_area
        void_sizes.append(size)

    return {
        "material_area": float(material_area),
        "void_area": float(void_area),
        "material_fraction": float(material_fraction),
        "void_fraction": float(void_fraction),
        "n_material_regions": int(n_material),
        "n_void_regions": int(n_void),
        "mean_material_size": float(np.mean(material_sizes)) if material_sizes else 0.0,
        "mean_void_size": float(np.mean(void_sizes)) if void_sizes else 0.0,
        "material_sizes": material_sizes,
        "void_sizes": void_sizes,
    }


def compute_diameter_distribution(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    direction: str = "z",
    bins: int = 50,
    fit_distribution: bool = True,
) -> Dict[str, Any]:
    """
    Compute distribution of filament diameters.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        direction: Filament direction
        bins: Number of bins for histogram
        fit_distribution: Whether to fit statistical distribution

    Returns:
        Dictionary with distribution statistics
    """
    results = estimate_filament_diameter(volume, voxel_size, direction)
    diameters = results["diameters"]

    if len(diameters) == 0:
        return {"histogram": None, "bin_edges": None, "diameters": [], "fit": None}

    hist, bin_edges = np.histogram(diameters, bins=bins)

    result_dict = {
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "diameters": diameters,
        "mean": float(np.mean(diameters)),
        "std": float(np.std(diameters)),
        "median": float(np.median(diameters)),
    }

    # Fit statistical distribution
    if fit_distribution and HAS_STATISTICS and len(diameters) > 0:
        try:
            result_dict["fit"] = compare_fits(
                np.array(diameters), distributions=["gaussian"]
            )
        except Exception as e:
            logger.warning(f"Failed to fit diameter distribution: {e}")
            result_dict["fit"] = None
    else:
        result_dict["fit"] = None

    return result_dict


def compute_channel_distribution(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    direction: str = "z",
    bins: int = 50,
) -> Dict[str, Any]:
    """
    Compute distribution of channel widths.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        direction: Channel direction
        bins: Number of bins for histogram

    Returns:
        Dictionary with distribution statistics
    """
    results = estimate_channel_width(volume, voxel_size, direction)
    widths = results["diameters"]  # Reusing diameter estimation for channels

    if len(widths) == 0:
        return {"histogram": None, "bin_edges": None, "widths": []}

    hist, bin_edges = np.histogram(widths, bins=bins)

    return {
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "widths": widths,
        "mean": float(np.mean(widths)),
        "std": float(np.std(widths)),
        "median": float(np.median(widths)),
    }
