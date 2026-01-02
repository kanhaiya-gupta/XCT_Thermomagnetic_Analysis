"""
Slice Analysis Module

Analysis of 2D slices extracted from 3D volumes, both along and
perpendicular to water-flow direction. Evaluation of repeatability
and dimensional accuracy.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


def extract_slice(volume: np.ndarray, axis: int, position: int) -> np.ndarray:
    """
    Extract 2D slice from 3D volume.

    Args:
        volume: 3D volume
        axis: Axis along which to extract (0=z, 1=y, 2=x)
        position: Position along axis

    Returns:
        2D slice
    """
    if axis == 0:  # z-axis
        if position < 0 or position >= volume.shape[0]:
            raise ValueError(f"Position {position} out of range [0, {volume.shape[0]})")
        return volume[position, :, :]
    elif axis == 1:  # y-axis
        if position < 0 or position >= volume.shape[1]:
            raise ValueError(f"Position {position} out of range [0, {volume.shape[1]})")
        return volume[:, position, :]
    elif axis == 2:  # x-axis
        if position < 0 or position >= volume.shape[2]:
            raise ValueError(f"Position {position} out of range [0, {volume.shape[2]})")
        return volume[:, :, position]
    else:
        raise ValueError(f"Invalid axis: {axis} (must be 0, 1, or 2)")


def analyze_slice_along_flow(
    volume: np.ndarray,
    flow_direction: str = "z",
    n_slices: Optional[int] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Analyze slices along water-flow direction.

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow ('x', 'y', 'z')
        n_slices: Number of slices to analyze (if None, analyze all)
        voxel_size: Voxel spacing for metric calculations

    Returns:
        Dictionary with slice analysis results
    """
    if flow_direction == "z":
        axis = 0
        n_total = volume.shape[0]
    elif flow_direction == "y":
        axis = 1
        n_total = volume.shape[1]
    else:  # 'x'
        axis = 2
        n_total = volume.shape[2]

    if n_slices is None:
        n_slices = n_total

    # Sample slice positions
    slice_positions = np.linspace(0, n_total - 1, n_slices, dtype=int)

    slice_metrics = []

    for pos in slice_positions:
        slice_2d = extract_slice(volume, axis, pos)

        if voxel_size:
            if flow_direction == "z":
                pixel_size = (voxel_size[1], voxel_size[2])
            elif flow_direction == "y":
                pixel_size = (voxel_size[0], voxel_size[2])
            else:
                pixel_size = (voxel_size[0], voxel_size[1])

            from .filament_analysis import analyze_cross_section

            metrics = analyze_cross_section(slice_2d, pixel_size)
        else:
            # Basic metrics without physical units
            metrics = {
                "material_fraction": float(np.mean(slice_2d > 0)),
                "void_fraction": float(np.mean(slice_2d == 0)),
                "n_material_regions": int(ndimage.label(slice_2d > 0)[1]),
                "n_void_regions": int(ndimage.label(slice_2d == 0)[1]),
            }

        metrics["position"] = int(pos)
        slice_metrics.append(metrics)

    # Compute statistics across slices
    material_fractions = [m["material_fraction"] for m in slice_metrics]
    void_fractions = [m["void_fraction"] for m in slice_metrics]

    return {
        "flow_direction": flow_direction,
        "n_slices": len(slice_metrics),
        "slice_metrics": slice_metrics,
        "mean_material_fraction": float(np.mean(material_fractions)),
        "std_material_fraction": float(np.std(material_fractions)),
        "mean_void_fraction": float(np.mean(void_fractions)),
        "std_void_fraction": float(np.std(void_fractions)),
        "variation_coefficient": (
            float(np.std(material_fractions) / np.mean(material_fractions))
            if np.mean(material_fractions) > 0
            else 0.0
        ),
    }


def analyze_slice_perpendicular_flow(
    volume: np.ndarray,
    flow_direction: str = "z",
    positions: Optional[List[int]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Analyze slices perpendicular to water-flow direction.

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow ('x', 'y', 'z')
        positions: Specific slice positions to analyze (if None, analyze middle slice)
        voxel_size: Voxel spacing for metric calculations

    Returns:
        Dictionary with slice analysis results
    """
    if flow_direction == "z":
        axis = 0
        n_total = volume.shape[0]
        perpendicular_axes = (1, 2)  # y and x
    elif flow_direction == "y":
        axis = 1
        n_total = volume.shape[1]
        perpendicular_axes = (0, 2)  # z and x
    else:  # 'x'
        axis = 2
        n_total = volume.shape[2]
        perpendicular_axes = (0, 1)  # z and y

    if positions is None:
        positions = [n_total // 2]  # Middle slice

    slice_metrics = []

    for pos in positions:
        slice_2d = extract_slice(volume, axis, pos)

        if voxel_size:
            if flow_direction == "z":
                pixel_size = (voxel_size[1], voxel_size[2])
            elif flow_direction == "y":
                pixel_size = (voxel_size[0], voxel_size[2])
            else:
                pixel_size = (voxel_size[0], voxel_size[1])

            from .filament_analysis import analyze_cross_section

            metrics = analyze_cross_section(slice_2d, pixel_size)
        else:
            metrics = {
                "material_fraction": float(np.mean(slice_2d > 0)),
                "void_fraction": float(np.mean(slice_2d == 0)),
                "n_material_regions": int(ndimage.label(slice_2d > 0)[1]),
                "n_void_regions": int(ndimage.label(slice_2d == 0)[1]),
            }

        metrics["position"] = int(pos)
        slice_metrics.append(metrics)

    return {
        "flow_direction": flow_direction,
        "perpendicular_to_flow": True,
        "n_slices": len(slice_metrics),
        "slice_metrics": slice_metrics,
    }


def slice_metrics(
    slice_2d: np.ndarray, voxel_size: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a single slice.

    Args:
        slice_2d: 2D binary slice
        voxel_size: Voxel spacing (dx, dy) in mm

    Returns:
        Dictionary with slice metrics
    """
    if voxel_size:
        pixel_area = voxel_size[0] * voxel_size[1]
    else:
        pixel_area = 1.0

    # Basic fractions
    material_fraction = float(np.mean(slice_2d > 0))
    void_fraction = float(np.mean(slice_2d == 0))

    # Areas
    material_area = np.sum(slice_2d > 0) * pixel_area
    void_area = np.sum(slice_2d == 0) * pixel_area

    # Connected components
    labeled_material, n_material = ndimage.label(slice_2d > 0)
    labeled_void, n_void = ndimage.label(slice_2d == 0)

    # Perimeter (approximate)
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(slice_2d > 0)
    perimeter_voxels = np.sum((slice_2d > 0) & ~eroded)
    perimeter = (
        perimeter_voxels * np.mean(voxel_size) if voxel_size else perimeter_voxels
    )

    return {
        "material_fraction": material_fraction,
        "void_fraction": void_fraction,
        "material_area": float(material_area),
        "void_area": float(void_area),
        "n_material_regions": int(n_material),
        "n_void_regions": int(n_void),
        "perimeter": float(perimeter),
        "perimeter_voxels": int(perimeter_voxels),
    }


def compare_slices(
    slices: List[np.ndarray], voxel_size: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Compare multiple slices.

    Args:
        slices: List of 2D slices
        voxel_size: Voxel spacing

    Returns:
        Dictionary with comparison metrics
    """
    slice_metrics_list = [slice_metrics(s, voxel_size) for s in slices]

    # Extract metrics
    material_fractions = [m["material_fraction"] for m in slice_metrics_list]
    void_fractions = [m["void_fraction"] for m in slice_metrics_list]
    n_material_regions = [m["n_material_regions"] for m in slice_metrics_list]
    n_void_regions = [m["n_void_regions"] for m in slice_metrics_list]

    return {
        "n_slices": len(slices),
        "material_fraction": {
            "mean": float(np.mean(material_fractions)),
            "std": float(np.std(material_fractions)),
            "min": float(np.min(material_fractions)),
            "max": float(np.max(material_fractions)),
            "cv": (
                float(np.std(material_fractions) / np.mean(material_fractions))
                if np.mean(material_fractions) > 0
                else 0.0
            ),
        },
        "void_fraction": {
            "mean": float(np.mean(void_fractions)),
            "std": float(np.std(void_fractions)),
            "min": float(np.min(void_fractions)),
            "max": float(np.max(void_fractions)),
        },
        "n_material_regions": {
            "mean": float(np.mean(n_material_regions)),
            "std": float(np.std(n_material_regions)),
        },
        "n_void_regions": {
            "mean": float(np.mean(n_void_regions)),
            "std": float(np.std(n_void_regions)),
        },
        "slice_metrics": slice_metrics_list,
    }


def repeatability_analysis(
    slices: List[np.ndarray], voxel_size: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Evaluate repeatability across multiple slices.

    Args:
        slices: List of 2D slices (e.g., from multiple samples)
        voxel_size: Voxel spacing

    Returns:
        Dictionary with repeatability metrics
    """
    comparison = compare_slices(slices, voxel_size)

    # Coefficient of variation (CV) as repeatability metric
    cv_material = comparison["material_fraction"]["cv"]
    cv_void = comparison["void_fraction"]["cv"]

    # Repeatability: lower CV = better repeatability
    repeatability_score = 1.0 / (1.0 + cv_material + cv_void)  # Normalized score

    return {
        "coefficient_of_variation": {
            "material_fraction": cv_material,
            "void_fraction": cv_void,
        },
        "repeatability_score": float(repeatability_score),
        "comparison": comparison,
    }
