"""
Porosity Distribution Analysis Module

Analysis of internal porosity distribution along printing direction
and spatial distribution of pores in 3D-printed structures.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy import ndimage
from skimage import measure
import logging

try:
    from ..preprocessing.statistics import fit_gaussian, fit_poisson, compare_fits

    HAS_STATISTICS = True
except ImportError:
    HAS_STATISTICS = False

logger = logging.getLogger(__name__)


def porosity_along_direction(
    volume: np.ndarray,
    direction: str = "z",
    normalize: bool = True,
    fit_trend: bool = False,
) -> Dict[str, Any]:
    """
    Compute porosity profile along specified direction (typically printing direction).

    Args:
        volume: Binary volume (1 = material, 0 = void)
        direction: Direction along which to analyze ('x', 'y', 'z')
        normalize: If True, return fraction (0-1), else return voxel counts
        fit_trend: Whether to fit linear/quadratic trend to porosity profile

    Returns:
        Dictionary with porosity profile
    """
    if direction == "z":
        axis = 0
        n_slices = volume.shape[0]
    elif direction == "y":
        axis = 1
        n_slices = volume.shape[1]
    else:  # 'x'
        axis = 2
        n_slices = volume.shape[2]

    porosity_profile = []
    slice_positions = []

    for i in range(n_slices):
        if direction == "z":
            slice_2d = volume[i, :, :]
        elif direction == "y":
            slice_2d = volume[:, i, :]
        else:
            slice_2d = volume[:, :, i]

        void_fraction = np.mean(slice_2d == 0)

        if normalize:
            porosity_profile.append(float(void_fraction))
        else:
            void_voxels = np.sum(slice_2d == 0)
            porosity_profile.append(int(void_voxels))

        slice_positions.append(i)

    result = {
        "direction": direction,
        "porosity": porosity_profile,
        "positions": slice_positions,
        "mean_porosity": float(np.mean(porosity_profile)),
        "std_porosity": float(np.std(porosity_profile)),
        "min_porosity": float(np.min(porosity_profile)),
        "max_porosity": float(np.max(porosity_profile)),
    }

    # Fit trend (linear or quadratic)
    if fit_trend and HAS_STATISTICS and len(porosity_profile) > 2:
        try:
            from ..preprocessing.statistics import (
                fit_linear,
                fit_quadratic,
                compare_fits,
            )

            positions_array = np.array(slice_positions)
            porosity_array = np.array(porosity_profile)

            # Try linear and quadratic fits
            linear_fit = fit_linear(positions_array, porosity_array)
            quadratic_fit = fit_quadratic(positions_array, porosity_array)

            # Select best based on R-squared
            if quadratic_fit["r_squared"] > linear_fit["r_squared"]:
                result["trend_fit"] = quadratic_fit
                result["trend_type"] = "quadratic"
            else:
                result["trend_fit"] = linear_fit
                result["trend_type"] = "linear"
        except Exception as e:
            logger.warning(f"Failed to fit porosity trend: {e}")
            result["trend_fit"] = None

    return result


def local_porosity_map(volume: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Compute local porosity map using sliding window.

    Args:
        volume: Binary volume
        window_size: Size of sliding window (in voxels)

    Returns:
        Local porosity map (0.0 to 1.0)
    """
    from scipy.ndimage import uniform_filter

    # Convert to void fraction (0 = material, 1 = void)
    void_volume = 1 - volume.astype(float)

    # Apply uniform filter (moving average)
    local_porosity = uniform_filter(void_volume, size=window_size)

    return local_porosity


def porosity_gradient(volume: np.ndarray, direction: str = "z") -> Dict[str, Any]:
    """
    Compute porosity gradient along specified direction.

    Args:
        volume: Binary volume
        direction: Direction for gradient computation

    Returns:
        Dictionary with gradient information
    """
    profile = porosity_along_direction(volume, direction, normalize=True)
    porosity = np.array(profile["porosity"])

    # Compute gradient
    gradient = np.gradient(porosity)

    return {
        "direction": direction,
        "gradient": gradient.tolist(),
        "mean_gradient": float(np.mean(gradient)),
        "std_gradient": float(np.std(gradient)),
        "max_gradient": float(np.max(gradient)),
        "min_gradient": float(np.min(gradient)),
    }


def pore_size_distribution(
    volume: np.ndarray, voxel_size: Tuple[float, float, float], min_size: int = 1
) -> Dict[str, Any]:
    """
    Compute pore size distribution.

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm
        min_size: Minimum pore size (in voxels) to consider

    Returns:
        Dictionary with pore size distribution
    """
    # Invert volume (pores = 1, material = 0)
    pore_volume = 1 - volume

    # Label connected pore regions
    labeled, num_pores = ndimage.label(pore_volume)

    voxel_volume = np.prod(voxel_size)
    pore_sizes = []
    pore_volumes = []

    for label_id in range(1, num_pores + 1):
        pore_mask = labeled == label_id
        size_voxels = np.sum(pore_mask)

        if size_voxels >= min_size:
            volume_mm3 = size_voxels * voxel_volume
            pore_sizes.append(size_voxels)
            pore_volumes.append(volume_mm3)

    if len(pore_volumes) == 0:
        return {
            "n_pores": 0,
            "pore_volumes": [],
            "pore_sizes": [],
            "mean_pore_volume": 0.0,
            "total_pore_volume": 0.0,
        }

    pore_volumes = np.array(pore_volumes)
    pore_sizes = np.array(pore_sizes)

    # Estimate equivalent spherical diameter
    equivalent_diameters = 2 * np.power(3 * pore_volumes / (4 * np.pi), 1 / 3)

    return {
        "n_pores": int(num_pores),
        "pore_volumes": pore_volumes.tolist(),
        "pore_sizes": pore_sizes.tolist(),
        "equivalent_diameters": equivalent_diameters.tolist(),
        "mean_pore_volume": float(np.mean(pore_volumes)),
        "std_pore_volume": float(np.std(pore_volumes)),
        "median_pore_volume": float(np.median(pore_volumes)),
        "total_pore_volume": float(np.sum(pore_volumes)),
        "mean_equivalent_diameter": float(np.mean(equivalent_diameters)),
        "std_equivalent_diameter": float(np.std(equivalent_diameters)),
    }


def connectivity_analysis(volume: np.ndarray) -> Dict[str, Any]:
    """
    Analyze pore connectivity.

    Args:
        volume: Binary volume (1 = material, 0 = void)

    Returns:
        Dictionary with connectivity metrics
    """
    # Invert volume (pores = 1)
    pore_volume = 1 - volume

    # Label connected components
    labeled, num_components = ndimage.label(pore_volume)

    # Component sizes
    component_sizes = []
    for label_id in range(1, num_components + 1):
        size = np.sum(labeled == label_id)
        component_sizes.append(size)

    component_sizes = np.array(component_sizes)

    # Largest connected component
    if len(component_sizes) > 0:
        largest_idx = np.argmax(component_sizes)
        largest_size = component_sizes[largest_idx]
        largest_fraction = largest_size / np.sum(component_sizes)
    else:
        largest_size = 0
        largest_fraction = 0.0

    return {
        "n_connected_components": int(num_components),
        "component_sizes": component_sizes.tolist(),
        "largest_component_size": int(largest_size),
        "largest_component_fraction": float(largest_fraction),
        "mean_component_size": (
            float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0
        ),
    }


def analyze_porosity_distribution(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    printing_direction: str = "z",
    fit_distributions: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive porosity distribution analysis.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        printing_direction: Direction along which printing occurred
        fit_distributions: Whether to fit statistical distributions to pore sizes

    Returns:
        Dictionary with all porosity analysis results
    """
    results = {}

    # Porosity profile along printing direction
    results["porosity_profile"] = porosity_along_direction(volume, printing_direction)

    # Porosity gradient
    results["porosity_gradient"] = porosity_gradient(volume, printing_direction)

    # Pore size distribution
    results["pore_size_distribution"] = pore_size_distribution(volume, voxel_size)

    # Connectivity
    results["connectivity"] = connectivity_analysis(volume)

    # Local porosity map
    results["local_porosity_map"] = local_porosity_map(volume, window_size=10)

    # Fit statistical distributions to pore sizes
    if fit_distributions and HAS_STATISTICS:
        pore_volumes = results["pore_size_distribution"].get("pore_volumes", [])
        equivalent_diameters = results["pore_size_distribution"].get(
            "equivalent_diameters", []
        )

        if len(pore_volumes) > 0:
            try:
                results["pore_volume_fits"] = compare_fits(
                    np.array(pore_volumes), distributions=["gaussian"]
                )
            except Exception as e:
                logger.warning(f"Failed to fit pore volume distribution: {e}")

        if len(equivalent_diameters) > 0:
            try:
                results["pore_diameter_fits"] = compare_fits(
                    np.array(equivalent_diameters), distributions=["gaussian"]
                )
            except Exception as e:
                logger.warning(f"Failed to fit pore diameter distribution: {e}")

    logger.info(
        f"Porosity analysis complete: "
        f"Mean porosity={results['porosity_profile']['mean_porosity']:.2%}, "
        f"Pores={results['pore_size_distribution']['n_pores']}"
    )

    return results
