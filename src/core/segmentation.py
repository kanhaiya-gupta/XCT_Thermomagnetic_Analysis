"""
Image Segmentation Module

Segmentation of XCT volumes using Otsu's thresholding method and other
thresholding techniques for separating material and void regions.
"""

import numpy as np
from typing import Optional, Tuple, Union
from skimage import filters, segmentation
import logging

logger = logging.getLogger(__name__)


def otsu_threshold(
    volume: np.ndarray, return_threshold: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Apply Otsu's thresholding method to 3D volume.

    Otsu's method automatically determines the optimal threshold value
    by minimizing intra-class variance and maximizing inter-class variance.

    Args:
        volume: Input 3D volume (grayscale)
        return_threshold: If True, return threshold value

    Returns:
        Binary segmented volume (0 = void, 1 = material)
        If return_threshold=True: (binary_volume, threshold_value)
    """
    # Flatten volume for threshold calculation
    flat_volume = volume.flatten()

    # Remove any NaN or inf values
    valid_pixels = np.isfinite(flat_volume)
    if not np.all(valid_pixels):
        logger.warning(
            "Volume contains NaN or inf values, removing them for threshold calculation"
        )
        flat_volume = flat_volume[valid_pixels]

    # Handle edge cases
    if len(flat_volume) == 0:
        # Empty volume
        binary_volume = np.zeros_like(volume, dtype=np.uint8)
        threshold_value = 0.0
    elif len(np.unique(flat_volume)) == 1:
        # Constant volume - all same value
        unique_val = flat_volume[0]
        if unique_val > 127:
            # All bright - treat as material
            binary_volume = np.ones_like(volume, dtype=np.uint8)
            threshold_value = float(unique_val)
        else:
            # All dark - treat as void
            binary_volume = np.zeros_like(volume, dtype=np.uint8)
            threshold_value = float(unique_val)
    else:
        # Calculate Otsu threshold
        threshold_value = filters.threshold_otsu(flat_volume)
        # Convert to Python float to ensure isinstance works correctly
        threshold_value = float(threshold_value)

        # Apply threshold
        binary_volume = (volume > threshold_value).astype(np.uint8)

    logger.info(
        f"Otsu threshold: {threshold_value:.2f}, "
        f"Material fraction: {binary_volume.mean():.2%}"
    )

    if return_threshold:
        return binary_volume, threshold_value
    return binary_volume


def multi_threshold_segmentation(volume: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """
    Multi-class segmentation using multiple thresholds.

    Args:
        volume: Input 3D volume
        n_classes: Number of classes (e.g., 3 = void, material, high-density)

    Returns:
        Labeled volume (0, 1, 2, ...)
    """
    flat_volume = volume.flatten()
    valid_pixels = flat_volume[np.isfinite(flat_volume)]

    # Calculate thresholds
    thresholds = []
    for i in range(1, n_classes):
        # Use percentile-based approach for multi-threshold
        percentile = (i / n_classes) * 100
        threshold = np.percentile(valid_pixels, percentile)
        thresholds.append(threshold)

    # Create labeled volume
    labeled_volume = np.zeros_like(volume, dtype=np.uint8)
    for i, threshold in enumerate(thresholds):
        labeled_volume[volume > threshold] = i + 1

    logger.info(
        f"Multi-threshold segmentation: {n_classes} classes, "
        f"thresholds: {thresholds}"
    )

    return labeled_volume


def adaptive_threshold(
    volume: np.ndarray, block_size: int = 15, method: str = "gaussian"
) -> np.ndarray:
    """
    Adaptive thresholding for non-uniform illumination.

    Args:
        volume: Input 3D volume
        block_size: Size of local neighborhood
        method: 'gaussian' or 'mean'

    Returns:
        Binary segmented volume
    """
    # Apply adaptive threshold slice by slice (2D)
    binary_volume = np.zeros_like(volume, dtype=np.uint8)

    for z in range(volume.shape[0]):
        slice_2d = volume[z, :, :]

        if method == "gaussian":
            from skimage.filters import threshold_local

            threshold = threshold_local(slice_2d, block_size, method="gaussian")
        else:  # mean
            from skimage.filters import threshold_local

            threshold = threshold_local(slice_2d, block_size, method="mean")

        binary_volume[z, :, :] = (slice_2d > threshold).astype(np.uint8)

    logger.info(
        f"Adaptive threshold applied: block_size={block_size}, " f"method={method}"
    )

    return binary_volume


def segment_volume(volume: np.ndarray, method: str = "otsu", **kwargs) -> np.ndarray:
    """
    Main segmentation interface.

    Args:
        volume: Input 3D volume
        method: Segmentation method ('otsu', 'multi', 'adaptive', 'manual')
        **kwargs: Additional arguments for specific methods

    Returns:
        Segmented volume (binary or labeled)
    """
    if method == "otsu":
        return otsu_threshold(volume, return_threshold=False)

    elif method == "multi":
        n_classes = kwargs.get("n_classes", 3)
        return multi_threshold_segmentation(volume, n_classes)

    elif method == "adaptive":
        block_size = kwargs.get("block_size", 15)
        method_type = kwargs.get("method_type", "gaussian")
        return adaptive_threshold(volume, block_size, method_type)

    elif method == "manual":
        threshold = kwargs.get("threshold")
        if threshold is None:
            raise ValueError("Manual threshold requires 'threshold' parameter")
        return (volume > threshold).astype(np.uint8)

    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def refine_segmentation(
    binary_volume: np.ndarray,
    remove_small_objects: bool = True,
    min_size: int = 100,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Refine binary segmentation by removing small objects and filling holes.

    Args:
        binary_volume: Binary segmented volume
        remove_small_objects: Remove small disconnected regions
        min_size: Minimum size for objects to keep
        fill_holes: Fill internal holes

    Returns:
        Refined binary volume
    """
    from .morphology import remove_small_objects as rso, fill_holes as fh

    refined = binary_volume.copy()

    if remove_small_objects:
        refined = rso(refined, min_size=min_size)

    if fill_holes:
        refined = fh(refined)

    return refined
