"""
Morphological Operations Module

Morphological operations for cleaning and refining segmented volumes,
including erosion, dilation, opening, closing, and specialized operations.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage
from skimage import morphology
import logging

logger = logging.getLogger(__name__)


def erode(
    volume: np.ndarray,
    kernel_size: Union[int, Tuple[int, int, int]] = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Erosion operation - shrinks foreground objects.

    Args:
        volume: Binary volume (0 = background, 1 = foreground)
        kernel_size: Size of structuring element (int or (z, y, x))
        iterations: Number of iterations

    Returns:
        Eroded volume
    """
    if isinstance(kernel_size, int):
        kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
    else:
        kernel = np.ones(kernel_size, dtype=bool)

    eroded = volume.copy()
    for _ in range(iterations):
        eroded = ndimage.binary_erosion(eroded, structure=kernel)

    return eroded.astype(np.uint8)


def dilate(
    volume: np.ndarray,
    kernel_size: Union[int, Tuple[int, int, int]] = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Dilation operation - expands foreground objects.

    Args:
        volume: Binary volume (0 = background, 1 = foreground)
        kernel_size: Size of structuring element (int or (z, y, x))
        iterations: Number of iterations

    Returns:
        Dilated volume
    """
    if isinstance(kernel_size, int):
        kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
    else:
        kernel = np.ones(kernel_size, dtype=bool)

    dilated = volume.copy()
    for _ in range(iterations):
        dilated = ndimage.binary_dilation(dilated, structure=kernel)

    return dilated.astype(np.uint8)


def open_operation(
    volume: np.ndarray, kernel_size: Union[int, Tuple[int, int, int]] = 3
) -> np.ndarray:
    """
    Opening operation - erosion followed by dilation.
    Removes small objects and smooths boundaries.

    Args:
        volume: Binary volume
        kernel_size: Size of structuring element

    Returns:
        Opened volume
    """
    opened = erode(volume, kernel_size)
    opened = dilate(opened, kernel_size)
    return opened


def close_operation(
    volume: np.ndarray, kernel_size: Union[int, Tuple[int, int, int]] = 3
) -> np.ndarray:
    """
    Closing operation - dilation followed by erosion.
    Fills small holes and connects nearby objects.

    Args:
        volume: Binary volume
        kernel_size: Size of structuring element

    Returns:
        Closed volume
    """
    closed = dilate(volume, kernel_size)
    closed = erode(closed, kernel_size)
    return closed


def remove_small_objects(
    volume: np.ndarray, min_size: int = 100, connectivity: int = 1
) -> np.ndarray:
    """
    Remove small disconnected objects from binary volume.

    Args:
        volume: Binary volume
        min_size: Minimum size (in voxels) to keep
        connectivity: Connectivity (1 = 6-connected, 2 = 18-connected, 3 = 26-connected)

    Returns:
        Volume with small objects removed
    """
    cleaned = morphology.remove_small_objects(
        volume.astype(bool), min_size=min_size, connectivity=connectivity
    )

    return cleaned.astype(np.uint8)


def fill_holes(volume: np.ndarray) -> np.ndarray:
    """
    Fill holes (internal cavities) in binary volume.

    Args:
        volume: Binary volume

    Returns:
        Volume with holes filled
    """
    # Fill holes slice by slice (2D) for 3D volumes
    filled = np.zeros_like(volume)

    for z in range(volume.shape[0]):
        slice_2d = volume[z, :, :]
        filled_slice = ndimage.binary_fill_holes(slice_2d)
        filled[z, :, :] = filled_slice.astype(np.uint8)

    return filled


def skeletonize(volume: np.ndarray) -> np.ndarray:
    """
    Skeletonize binary volume - reduces objects to 1-voxel-wide skeletons.
    Useful for filament analysis.

    Note: Uses 2D skeletonization on each slice and combines results.
    For true 3D skeletonization, consider using specialized libraries.

    Args:
        volume: Binary volume

    Returns:
        Skeletonized volume
    """
    volume_bool = volume.astype(bool)
    skeleton = np.zeros_like(volume_bool, dtype=bool)

    # Apply 2D skeletonization to each slice along each axis and combine
    # This is an approximation - true 3D skeletonization requires more complex algorithms
    for axis in range(3):
        for i in range(volume.shape[axis]):
            if axis == 0:
                slice_2d = volume_bool[i, :, :]
            elif axis == 1:
                slice_2d = volume_bool[:, i, :]
            else:
                slice_2d = volume_bool[:, :, i]

            if np.any(slice_2d):
                try:
                    skeleton_2d = morphology.skeletonize(slice_2d)
                    if axis == 0:
                        skeleton[i, :, :] = skeleton[i, :, :] | skeleton_2d
                    elif axis == 1:
                        skeleton[:, i, :] = skeleton[:, i, :] | skeleton_2d
                    else:
                        skeleton[:, :, i] = skeleton[:, :, i] | skeleton_2d
                except Exception:
                    # If skeletonization fails on a slice, skip it
                    pass

    return skeleton.astype(np.uint8)


def distance_transform(volume: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute distance transform - distance from each foreground voxel
    to nearest background voxel.

    Args:
        volume: Binary volume
        metric: Distance metric ('euclidean', 'taxicab', 'chessboard')

    Returns:
        Distance map
    """
    if metric == "euclidean":
        return ndimage.distance_transform_edt(volume)
    elif metric == "taxicab":
        return ndimage.distance_transform_cdt(volume, metric="taxicab")
    elif metric == "chessboard":
        return ndimage.distance_transform_cdt(volume, metric="chessboard")
    else:
        raise ValueError(f"Unknown metric: {metric}")


def watershed_segmentation(
    volume: np.ndarray,
    markers: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Watershed segmentation for separating touching objects.

    Args:
        volume: Input volume (distance transform or gradient)
        markers: Seed markers (optional)
        mask: Mask region (optional)

    Returns:
        Labeled volume
    """
    from skimage.segmentation import watershed

    if markers is None:
        # Use distance transform peaks as markers
        distance = distance_transform(volume)
        from scipy.ndimage import maximum_filter

        local_maxima = maximum_filter(distance, size=5) == distance
        markers = ndimage.label(local_maxima)[0]

    labels = watershed(-volume, markers, mask=mask)
    return labels.astype(np.uint16)


def apply_morphological_operations(volume: np.ndarray, operations: list) -> np.ndarray:
    """
    Apply a sequence of morphological operations.

    Args:
        volume: Binary volume
        operations: List of operations, each as dict with 'op' and parameters
                   e.g., [{'op': 'open', 'kernel_size': 3}, {'op': 'close', 'kernel_size': 5}]

    Returns:
        Processed volume
    """
    result = volume.copy()

    for op_config in operations:
        op_name = op_config["op"]

        if op_name == "erode":
            result = erode(result, **{k: v for k, v in op_config.items() if k != "op"})
        elif op_name == "dilate":
            result = dilate(result, **{k: v for k, v in op_config.items() if k != "op"})
        elif op_name == "open":
            result = open_operation(
                result, **{k: v for k, v in op_config.items() if k != "op"}
            )
        elif op_name == "close":
            result = close_operation(
                result, **{k: v for k, v in op_config.items() if k != "op"}
            )
        elif op_name == "remove_small":
            result = remove_small_objects(
                result, **{k: v for k, v in op_config.items() if k != "op"}
            )
        elif op_name == "fill_holes":
            result = fill_holes(result)
        else:
            logger.warning(f"Unknown operation: {op_name}")

    return result
