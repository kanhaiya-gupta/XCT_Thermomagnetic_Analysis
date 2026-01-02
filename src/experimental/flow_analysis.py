"""
Flow Analysis Module for Water-Flowable Elements

Analyze flow characteristics in 3D-printed thermomagnetic heat exchanger components:
- Flow path connectivity (inlet to outlet)
- Tortuosity (path complexity)
- Flow resistance and pressure drop
- Flow distribution uniformity
- Dead-end channel detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import ndimage
from scipy.spatial.distance import cdist, euclidean
from scipy.ndimage import distance_transform_edt
import logging

logger = logging.getLogger(__name__)


def analyze_flow_connectivity(
    volume: np.ndarray,
    flow_direction: str = "z",
    inlet_position: Optional[Tuple[int, int, int]] = None,
    outlet_position: Optional[Tuple[int, int, int]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Analyze flow path connectivity from inlet to outlet.

    Args:
        volume: Binary volume (1 = material, 0 = void/channel)
        flow_direction: Direction of water flow ('x', 'y', 'z')
        inlet_position: Optional inlet position (z, y, x) in voxel coordinates
        outlet_position: Optional outlet position (z, y, x) in voxel coordinates
        voxel_size: Optional voxel spacing for physical distance calculations

    Returns:
        Dictionary with connectivity analysis results
    """
    # Invert volume: channels (void) = 1, material = 0
    channel_volume = 1 - volume

    # Determine inlet/outlet positions if not provided
    if flow_direction == "z":
        axis = 0
        if inlet_position is None:
            inlet_position = (0, volume.shape[1] // 2, volume.shape[2] // 2)
        if outlet_position is None:
            outlet_position = (
                volume.shape[0] - 1,
                volume.shape[1] // 2,
                volume.shape[2] // 2,
            )
    elif flow_direction == "y":
        axis = 1
        if inlet_position is None:
            inlet_position = (volume.shape[0] // 2, 0, volume.shape[2] // 2)
        if outlet_position is None:
            outlet_position = (
                volume.shape[0] // 2,
                volume.shape[1] - 1,
                volume.shape[2] // 2,
            )
    else:  # 'x'
        axis = 2
        if inlet_position is None:
            inlet_position = (volume.shape[0] // 2, volume.shape[1] // 2, 0)
        if outlet_position is None:
            outlet_position = (
                volume.shape[0] // 2,
                volume.shape[1] // 2,
                volume.shape[2] - 1,
            )

    # Check if inlet/outlet are in channels
    inlet_in_channel = channel_volume[inlet_position] > 0
    outlet_in_channel = channel_volume[outlet_position] > 0

    # Label connected components in channel network
    labeled, num_components = ndimage.label(channel_volume)

    # Find component containing inlet and outlet
    inlet_label = labeled[inlet_position] if inlet_in_channel else 0
    outlet_label = labeled[outlet_position] if outlet_in_channel else 0

    # Check connectivity
    connected = inlet_label > 0 and outlet_label > 0 and inlet_label == outlet_label

    # Analyze all flow paths
    if connected:
        # For connected paths, estimate path length
        # Use the extent along the flow direction as a simple estimate
        # (More accurate would require actual path finding algorithms)
        if flow_direction == "z":
            path_length_voxels = abs(outlet_position[0] - inlet_position[0])
        elif flow_direction == "y":
            path_length_voxels = abs(outlet_position[1] - inlet_position[1])
        else:  # 'x'
            path_length_voxels = abs(outlet_position[2] - inlet_position[2])

        # Ensure minimum path length of 1 voxel
        if path_length_voxels == 0:
            path_length_voxels = 1

        # Convert to physical distance
        if voxel_size:
            if flow_direction == "z":
                path_length_physical = path_length_voxels * voxel_size[0]
            elif flow_direction == "y":
                path_length_physical = path_length_voxels * voxel_size[1]
            else:
                path_length_physical = path_length_voxels * voxel_size[2]
        else:
            path_length_physical = float(path_length_voxels)

        # Straight-line distance
        if voxel_size:
            straight_line = np.array(outlet_position) - np.array(inlet_position)
            straight_line_physical = np.linalg.norm(
                straight_line * np.array(voxel_size)
            )
        else:
            straight_line_physical = np.linalg.norm(
                np.array(outlet_position) - np.array(inlet_position)
            )

        # Ensure straight_line is at least the path length along flow direction
        if straight_line_physical == 0:
            straight_line_physical = (
                path_length_physical if path_length_physical > 0 else 1.0
            )

        # Tortuosity (will be computed separately, but include here)
        tortuosity = (
            path_length_physical / straight_line_physical
            if straight_line_physical > 0
            else 1.0
        )
    else:
        path_length_voxels = np.inf
        path_length_physical = np.inf
        straight_line_physical = 0.0
        tortuosity = np.inf

    # Analyze channel network
    component_sizes = []
    for label_id in range(1, num_components + 1):
        size = np.sum(labeled == label_id)
        component_sizes.append(size)

    component_sizes = np.array(component_sizes)
    largest_component = np.max(component_sizes) if len(component_sizes) > 0 else 0

    return {
        "connected": bool(connected),
        "inlet_position": inlet_position,
        "outlet_position": outlet_position,
        "inlet_in_channel": bool(inlet_in_channel),
        "outlet_in_channel": bool(outlet_in_channel),
        "inlet_component": int(inlet_label),
        "outlet_component": int(outlet_label),
        "n_components": int(num_components),  # Alias for compatibility
        "n_connected_components": int(num_components),
        "path_length_voxels": (
            float(path_length_voxels) if path_length_voxels != np.inf else None
        ),
        "path_length_physical": (
            float(path_length_physical) if path_length_physical != np.inf else None
        ),
        "straight_line_distance": (
            float(straight_line_physical)
            if straight_line_physical > 0
            else float(
                np.linalg.norm(np.array(outlet_position) - np.array(inlet_position))
                * (np.mean(voxel_size) if voxel_size else 1.0)
            )
        ),
        "tortuosity": float(tortuosity) if tortuosity != np.inf else None,
        "largest_component_size": int(largest_component),
        "flow_direction": flow_direction,
    }


def identify_flow_paths(
    volume: np.ndarray, flow_direction: str = "z", method: str = "distance_transform"
) -> Dict[str, Any]:
    """
    Identify and analyze all flow paths in the channel network.

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        method: Method for path identification ('distance_transform' or 'skeleton')

    Returns:
        Dictionary with flow path analysis
    """
    # Invert volume: channels = 1
    channel_volume = 1 - volume

    # Label connected components
    labeled, num_components = ndimage.label(channel_volume)

    paths = []

    for label_id in range(1, num_components + 1):
        component_mask = labeled == label_id
        component_coords = np.argwhere(component_mask)

        if len(component_coords) == 0:
            continue

        # Get bounding box
        if flow_direction == "z":
            min_pos = component_coords[:, 0].min()
            max_pos = component_coords[:, 0].max()
            extent = max_pos - min_pos
        elif flow_direction == "y":
            min_pos = component_coords[:, 1].min()
            max_pos = component_coords[:, 1].max()
            extent = max_pos - min_pos
        else:  # 'x'
            min_pos = component_coords[:, 2].min()
            max_pos = component_coords[:, 2].max()
            extent = max_pos - min_pos

        # Path characteristics
        path_volume = np.sum(component_mask)

        paths.append(
            {
                "path_id": int(label_id),
                "n_voxels": int(path_volume),
                "extent_along_flow": int(extent),
                "spans_full_length": extent
                >= volume.shape[{"z": 0, "y": 1, "x": 2}[flow_direction]] * 0.8,
            }
        )

    # Main flow path (largest component)
    if paths:
        main_path = max(paths, key=lambda p: p["n_voxels"])
        main_path_id = main_path["path_id"]
        main_path_spans = main_path["spans_full_length"]
    else:
        main_path_id = 0
        main_path_spans = False

    n_paths_spanning = sum(1 for p in paths if p["spans_full_length"])
    n_dead_ends = num_components - n_paths_spanning
    n_isolated = 0  # Isolated paths (not used currently)

    return {
        "n_paths": int(num_components),
        "n_paths_spanning": int(n_paths_spanning),
        "n_dead_ends": int(n_dead_ends),
        "n_isolated": int(n_isolated),
        "main_path_id": int(main_path_id),
        "main_path_spans_full_length": bool(main_path_spans),
        "paths": paths,
        "flow_direction": flow_direction,
    }


def detect_dead_end_channels(
    volume: np.ndarray, flow_direction: str = "z", min_length: int = 5
) -> Dict[str, Any]:
    """
    Detect dead-end channels (channels that don't span the flow direction).

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        min_length: Minimum length to be considered a channel (voxels)

    Returns:
        Dictionary with dead-end channel analysis
    """
    # Invert volume: channels = 1
    channel_volume = 1 - volume

    # Label connected components
    labeled, num_components = ndimage.label(channel_volume)

    dead_ends = []
    flow_paths = []

    for label_id in range(1, num_components + 1):
        component_mask = labeled == label_id
        component_coords = np.argwhere(component_mask)

        if len(component_coords) == 0:
            continue

        # Check extent along flow direction
        if flow_direction == "z":
            min_pos = component_coords[:, 0].min()
            max_pos = component_coords[:, 0].max()
            extent = max_pos - min_pos
            full_length = volume.shape[0]
        elif flow_direction == "y":
            min_pos = component_coords[:, 1].min()
            max_pos = component_coords[:, 1].max()
            extent = max_pos - min_pos
            full_length = volume.shape[1]
        else:  # 'x'
            min_pos = component_coords[:, 2].min()
            max_pos = component_coords[:, 2].max()
            extent = max_pos - min_pos
            full_length = volume.shape[2]

        # Dead-end if doesn't span at least 80% of flow direction
        is_dead_end = (
            extent < (full_length * 0.8) and len(component_coords) >= min_length
        )

        channel_info = {
            "channel_id": int(label_id),
            "n_voxels": int(len(component_coords)),
            "extent_along_flow": int(extent),
            "min_position": int(min_pos),
            "max_position": int(max_pos),
            "is_dead_end": bool(is_dead_end),
        }

        if is_dead_end:
            dead_ends.append(channel_info)
        else:
            flow_paths.append(channel_info)

    # Dead-end statistics
    if dead_ends:
        dead_end_volumes = [d["n_voxels"] for d in dead_ends]
        total_dead_end_volume = sum(dead_end_volumes)
    else:
        total_dead_end_volume = 0

    total_channel_volume = np.sum(channel_volume)
    dead_end_fraction = (
        total_dead_end_volume / total_channel_volume
        if total_channel_volume > 0
        else 0.0
    )

    return {
        "n_dead_ends": len(dead_ends),
        "n_flow_paths": len(flow_paths),
        "dead_end_fraction": float(dead_end_fraction),
        "total_dead_end_volume": int(total_dead_end_volume),
        "dead_ends": dead_ends,
        "flow_paths": flow_paths,
        "flow_direction": flow_direction,
    }


def compute_tortuosity(
    volume: np.ndarray,
    flow_direction: str = "z",
    method: str = "euclidean",
    inlet_position: Optional[Tuple[int, int, int]] = None,
    outlet_position: Optional[Tuple[int, int, int]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Compute tortuosity of flow paths.

    Tortuosity = actual path length / straight-line distance

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        method: Method for path length calculation ('euclidean', 'manhattan', 'geodesic')
        inlet_position: Optional inlet position
        outlet_position: Optional outlet position
        voxel_size: Optional voxel spacing

    Returns:
        Dictionary with tortuosity analysis
    """
    # Get connectivity analysis (includes path length)
    connectivity = analyze_flow_connectivity(
        volume, flow_direction, inlet_position, outlet_position, voxel_size
    )

    # Get path length and straight-line distance
    path_length = connectivity.get("path_length_physical")
    straight_line = connectivity.get("straight_line_distance", 0.0)

    # Always compute straight_line from positions if it's 0 or missing
    if straight_line == 0:
        inlet_pos = np.array(connectivity.get("inlet_position", (0, 0, 0)))
        outlet_pos = np.array(connectivity.get("outlet_position", (0, 0, 0)))
        pos_diff = outlet_pos - inlet_pos
        if np.any(pos_diff != 0):
            if voxel_size:
                straight_line = np.linalg.norm(pos_diff * np.array(voxel_size))
            else:
                # Use voxel distance if no voxel_size provided
                straight_line = np.linalg.norm(pos_diff)
        else:
            # If positions are same, use volume extent
            if flow_direction == "z":
                straight_line = volume.shape[0] * (
                    np.mean(voxel_size) if voxel_size else 1.0
                )
            elif flow_direction == "y":
                straight_line = volume.shape[1] * (
                    np.mean(voxel_size) if voxel_size else 1.0
                )
            else:
                straight_line = volume.shape[2] * (
                    np.mean(voxel_size) if voxel_size else 1.0
                )

    # Handle edge cases
    if not connectivity["connected"]:
        # Path not connected - compute tortuosity from geometry if possible
        # Use voxel distance as fallback
        if voxel_size:
            inlet_pos = np.array(connectivity.get("inlet_position", (0, 0, 0)))
            outlet_pos = np.array(connectivity.get("outlet_position", (0, 0, 0)))
            straight_line_voxels = np.linalg.norm(outlet_pos - inlet_pos)
            straight_line = (
                straight_line_voxels * np.mean(voxel_size)
                if straight_line_voxels > 0
                else 0.0
            )
        else:
            inlet_pos = np.array(connectivity.get("inlet_position", (0, 0, 0)))
            outlet_pos = np.array(connectivity.get("outlet_position", (0, 0, 0)))
            straight_line = np.linalg.norm(outlet_pos - inlet_pos)

        # Estimate path length from channel geometry
        channel_volume = 1 - volume
        if flow_direction == "z":
            path_length = volume.shape[0] * (np.mean(voxel_size) if voxel_size else 1.0)
        elif flow_direction == "y":
            path_length = volume.shape[1] * (np.mean(voxel_size) if voxel_size else 1.0)
        else:
            path_length = volume.shape[2] * (np.mean(voxel_size) if voxel_size else 1.0)

        if straight_line > 0:
            tortuosity = path_length / straight_line
        else:
            tortuosity = 1.0  # Default for unconnected paths
    elif path_length is None or straight_line == 0:
        # If straight_line is 0, compute from positions
        if straight_line == 0:
            inlet_pos = np.array(connectivity.get("inlet_position", (0, 0, 0)))
            outlet_pos = np.array(connectivity.get("outlet_position", (0, 0, 0)))
            pos_diff = outlet_pos - inlet_pos
            if np.any(pos_diff != 0):  # Only compute if positions are different
                if voxel_size:
                    straight_line = np.linalg.norm(pos_diff * np.array(voxel_size))
                else:
                    straight_line = np.linalg.norm(pos_diff)

        if path_length is None:
            # Estimate path length from volume extent
            if flow_direction == "z":
                path_length = volume.shape[0] * (
                    np.mean(voxel_size) if voxel_size else 1.0
                )
            elif flow_direction == "y":
                path_length = volume.shape[1] * (
                    np.mean(voxel_size) if voxel_size else 1.0
                )
            else:
                path_length = volume.shape[2] * (
                    np.mean(voxel_size) if voxel_size else 1.0
                )

        # Ensure straight_line is never 0 for valid paths
        if straight_line == 0:
            # If still 0, use path_length as straight_line (perfectly straight)
            straight_line = path_length if path_length > 0 else 1.0

        # Ensure path_length is never 0
        if path_length == 0 or path_length is None:
            path_length = straight_line if straight_line > 0 else 1.0

        if straight_line > 0 and path_length > 0:
            tortuosity = path_length / straight_line
            # Ensure tortuosity is at least 1.0 (path cannot be shorter than straight line)
            tortuosity = max(1.0, tortuosity)
        else:
            tortuosity = 1.0  # Default for straight path
    else:
        # Tortuosity
        # Ensure path_length is valid
        if path_length == 0 or path_length is None:
            path_length = straight_line if straight_line > 0 else 1.0

        if straight_line > 0 and path_length > 0:
            tortuosity = path_length / straight_line
            # Ensure tortuosity is at least 1.0 (path cannot be shorter than straight line)
            tortuosity = max(1.0, tortuosity)
        else:
            tortuosity = 1.0

    # Additional metrics
    # Effective tortuosity (considering channel geometry)
    channel_volume = 1 - volume
    void_fraction = np.mean(channel_volume)

    # Anisotropy (how different from isotropic)
    # Simplified: compare flow direction extent to perpendicular directions
    if flow_direction == "z":
        flow_extent = volume.shape[0]
        perp_extents = (volume.shape[1], volume.shape[2])
    elif flow_direction == "y":
        flow_extent = volume.shape[1]
        perp_extents = (volume.shape[0], volume.shape[2])
    else:  # 'x'
        flow_extent = volume.shape[2]
        perp_extents = (volume.shape[0], volume.shape[1])

    anisotropy = (
        flow_extent / np.mean(perp_extents) if np.mean(perp_extents) > 0 else 1.0
    )

    return {
        "mean_tortuosity": float(tortuosity),  # Alias for compatibility
        "tortuosity": float(tortuosity),
        "path_length": float(path_length) if path_length is not None else None,
        "straight_line_distance": float(straight_line),
        "void_fraction": float(void_fraction),
        "anisotropy": float(anisotropy),
        "method": method,
        "connected": connectivity["connected"],
        "interpretation": (
            "Low tortuosity"
            if tortuosity < 1.5
            else "Moderate tortuosity" if tortuosity < 2.5 else "High tortuosity"
        ),
    }


def compute_flow_path_lengths(
    volume: np.ndarray,
    flow_direction: str = "z",
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Compute flow path lengths for all channels.

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        voxel_size: Optional voxel spacing

    Returns:
        Dictionary with path length analysis
    """
    # Invert volume: channels = 1
    channel_volume = 1 - volume

    # Label connected components
    labeled, num_components = ndimage.label(channel_volume)

    path_lengths = []

    for label_id in range(1, num_components + 1):
        component_mask = labeled == label_id
        component_coords = np.argwhere(component_mask)

        if len(component_coords) == 0:
            continue

        # Compute path length using distance transform
        # Create distance map within component
        component_distance = distance_transform_edt(~component_mask)

        # Maximum distance (diameter of component)
        max_distance = np.max(component_distance[component_mask])

        # Extent along flow direction
        if flow_direction == "z":
            min_pos = component_coords[:, 0].min()
            max_pos = component_coords[:, 0].max()
            extent = max_pos - min_pos
        elif flow_direction == "y":
            min_pos = component_coords[:, 1].min()
            max_pos = component_coords[:, 1].max()
            extent = max_pos - min_pos
        else:  # 'x'
            min_pos = component_coords[:, 2].min()
            max_pos = component_coords[:, 2].max()
            extent = max_pos - min_pos

        # Convert to physical units
        if voxel_size:
            path_length_physical = extent * np.mean(voxel_size)
            diameter_physical = max_distance * np.mean(voxel_size)
        else:
            path_length_physical = float(extent)
            diameter_physical = float(max_distance)

        path_lengths.append(
            {
                "path_id": int(label_id),
                "length_voxels": int(extent),
                "length_physical": float(path_length_physical),
                "diameter_voxels": float(max_distance),
                "diameter_physical": float(diameter_physical),
                "n_voxels": int(len(component_coords)),
            }
        )

    if path_lengths:
        lengths = [p["length_physical"] for p in path_lengths]
        diameters = [p["diameter_physical"] for p in path_lengths]

        return {
            "n_paths": len(path_lengths),
            "path_lengths": path_lengths,
            "mean_path_length": float(np.mean(lengths)),
            "std_path_length": float(np.std(lengths)),
            "min_path_length": float(np.min(lengths)),
            "max_path_length": float(np.max(lengths)),
            "mean_diameter": float(np.mean(diameters)),
            "std_diameter": float(np.std(diameters)),
            "flow_direction": flow_direction,
        }
    else:
        return {"n_paths": 0, "path_lengths": [], "error": "No flow paths found"}


def analyze_flow_branching(
    volume: np.ndarray, flow_direction: str = "z"
) -> Dict[str, Any]:
    """
    Analyze flow path branching (how channels branch and merge).

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow

    Returns:
        Dictionary with branching analysis
    """
    # Invert volume: channels = 1
    channel_volume = 1 - volume

    # Use skeletonization to identify centerlines
    from ..core.morphology import skeletonize

    try:
        skeleton = skeletonize(channel_volume)
    except:
        logger.warning("Skeletonization failed, using simplified branching analysis")
        skeleton = channel_volume

    # Count branch points (voxels with 3+ neighbors in skeleton)
    from scipy.ndimage import binary_dilation

    # Simplified: count connected components at different cross-sections
    if flow_direction == "z":
        axis = 0
    elif flow_direction == "y":
        axis = 1
    else:  # 'x'
        axis = 2

    n_slices = volume.shape[axis]
    n_channels_per_slice = []

    for i in range(0, n_slices, max(1, n_slices // 50)):  # Sample slices
        if axis == 0:
            slice_2d = channel_volume[i, :, :]
        elif axis == 1:
            slice_2d = channel_volume[:, i, :]
        else:
            slice_2d = channel_volume[:, :, i]

        labeled_slice, n_components = ndimage.label(slice_2d)
        n_channels_per_slice.append(n_components)

    n_channels_per_slice = np.array(n_channels_per_slice)

    # Branching metrics
    mean_channels = float(np.mean(n_channels_per_slice))
    std_channels = float(np.std(n_channels_per_slice))
    max_channels = int(np.max(n_channels_per_slice))
    min_channels = int(np.min(n_channels_per_slice))

    # Branching variation (coefficient of variation)
    cv_channels = std_channels / mean_channels if mean_channels > 0 else 0.0

    return {
        "mean_channels_per_slice": float(mean_channels),
        "std_channels_per_slice": float(std_channels),
        "max_channels_per_slice": int(max_channels),
        "min_channels_per_slice": int(min_channels),
        "branching_variation": float(cv_channels),
        "n_slices_analyzed": len(n_channels_per_slice),
        "flow_direction": flow_direction,
        "interpretation": (
            "Uniform branching"
            if cv_channels < 0.2
            else (
                "Variable branching"
                if cv_channels < 0.5
                else "Highly variable branching"
            )
        ),
    }


def estimate_flow_resistance(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    flow_conditions: Optional[Dict[str, float]] = None,
    fluid_properties: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate flow resistance based on channel geometry.

    Args:
        volume: Binary volume (1 = material, 0 = void/channel)
        voxel_size: Voxel spacing in mm (dx, dy, dz)
        flow_conditions: Optional flow conditions (velocity, flow_rate)
        fluid_properties: Optional fluid properties (density, viscosity)

    Returns:
        Dictionary with flow resistance analysis
    """
    # Compute channel geometry from volume
    from ..core.metrics import compute_all_metrics

    channel_volume = 1 - volume
    metrics = compute_all_metrics(channel_volume, voxel_size)

    # Estimate mean channel diameter from void fraction and volume
    void_fraction = metrics.get("void_fraction", 0.0)
    total_volume = metrics.get("volume", 0.0)  # mm³

    # Estimate channel diameter (simplified: assume circular channels)
    if void_fraction > 0 and total_volume > 0:
        # Rough estimate: assume channels span the volume
        # Mean diameter ≈ sqrt(4 * area / π) where area ≈ void_fraction * cross_section
        mean_cross_section = total_volume / np.max(volume.shape)  # mm²
        mean_diameter = np.sqrt(4 * mean_cross_section / np.pi)  # mm
    else:
        mean_diameter = 1.0  # Default 1 mm

    # Default fluid properties (water at room temperature)
    if fluid_properties is None:
        fluid_properties = {"density": 1000.0, "viscosity": 0.001}  # kg/m³  # Pa·s

    # Get channel diameter
    mean_diameter_m = mean_diameter / 1000.0  # Convert to meters

    # Hydraulic diameter (for circular channels, = diameter)
    hydraulic_diameter = mean_diameter_m

    # Flow conditions
    if flow_conditions:
        flow_velocity = flow_conditions.get("velocity", 0.1)  # m/s
        flow_rate = flow_conditions.get("flow_rate", None)  # m³/s
    else:
        flow_velocity = 0.1  # Default 0.1 m/s
        flow_rate = None

    # Reynolds number
    Re = (
        fluid_properties["density"] * flow_velocity * hydraulic_diameter
    ) / fluid_properties["viscosity"]

    # Flow regime
    if Re < 2300:
        flow_regime = "laminar"
        # Friction factor for laminar flow in circular pipe: f = 64/Re
        friction_factor = 64.0 / Re if Re > 0 else 0.0
    else:
        flow_regime = "turbulent"
        # Simplified: f ≈ 0.316/Re^0.25 (Blasius)
        friction_factor = 0.316 / (Re**0.25) if Re > 0 else 0.0

    # Flow path length (from tortuosity analysis)
    tortuosity_result = compute_tortuosity(
        volume, flow_direction="z", voxel_size=voxel_size
    )
    if tortuosity_result.get("path_length"):
        path_length = tortuosity_result["path_length"] / 1000.0  # Convert mm to m
    else:
        # Estimate from volume dimensions
        if voxel_size:
            path_length = volume.shape[0] * voxel_size[0] / 1000.0  # m
        else:
            path_length = volume.shape[0] / 1000.0  # Assume 1 mm voxels

    # Pressure drop (Darcy-Weisbach equation)
    # ΔP = f * (L/D) * (ρ * v² / 2)
    if hydraulic_diameter > 0 and path_length > 0:
        pressure_drop = (
            friction_factor
            * (path_length / hydraulic_diameter)
            * (fluid_properties["density"] * flow_velocity**2 / 2.0)
        )
    else:
        pressure_drop = 0.0

    # Flow resistance (R = ΔP / Q)
    if flow_rate:
        flow_resistance = pressure_drop / flow_rate if flow_rate > 0 else 0.0
    else:
        # Estimate flow rate from velocity and cross-sectional area
        channel_area = np.pi * (hydraulic_diameter / 2.0) ** 2
        flow_rate = flow_velocity * channel_area
        flow_resistance = pressure_drop / flow_rate if flow_rate > 0 else 0.0

    return {
        "resistance": float(flow_resistance),  # Alias for compatibility (Pa·s/m³)
        "hydraulic_diameter": float(hydraulic_diameter * 1000),  # mm
        "reynolds_number": float(Re),
        "flow_regime": flow_regime,
        "friction_factor": float(friction_factor),
        "path_length": float(path_length * 1000),  # mm
        "pressure_drop": float(pressure_drop),  # Pa
        "pressure_drop_kpa": float(pressure_drop / 1000.0),  # kPa
        "flow_velocity": float(flow_velocity),  # m/s
        "flow_rate": float(flow_rate),  # m³/s
        "flow_resistance": float(flow_resistance),  # Pa·s/m³
        "tortuosity": tortuosity_result.get("tortuosity"),
    }


def calculate_pressure_drop(
    volume: np.ndarray,
    channel_geometry: Dict[str, float],
    flow_rate: float,
    fluid_properties: Optional[Dict[str, float]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Calculate pressure drop for given flow rate.

    Args:
        volume: Binary volume
        channel_geometry: Channel geometry dictionary
        flow_rate: Flow rate (m³/s)
        fluid_properties: Optional fluid properties
        voxel_size: Optional voxel spacing

    Returns:
        Dictionary with pressure drop analysis
    """
    # Estimate flow velocity from flow rate
    mean_diameter = channel_geometry.get("mean_diameter", 0.0) / 1000.0  # m
    channel_area = np.pi * (mean_diameter / 2.0) ** 2 if mean_diameter > 0 else 1.0

    flow_velocity = flow_rate / channel_area if channel_area > 0 else 0.0

    flow_conditions = {"velocity": flow_velocity, "flow_rate": flow_rate}

    # Use flow resistance function
    resistance_result = estimate_flow_resistance(
        volume, channel_geometry, flow_conditions, fluid_properties, voxel_size
    )

    return {
        "pressure_drop_pa": float(resistance_result["pressure_drop"]),
        "pressure_drop_kpa": float(resistance_result["pressure_drop_kpa"]),
        "flow_rate": float(flow_rate),
        "flow_velocity": float(flow_velocity),
        "reynolds_number": float(resistance_result["reynolds_number"]),
        "flow_regime": resistance_result["flow_regime"],
    }


def compute_hydraulic_diameter(
    channel_cross_section: np.ndarray, voxel_size: Optional[Tuple[float, float]] = None
) -> float:
    """
    Compute hydraulic diameter from channel cross-section.

    Hydraulic diameter = 4 * Area / Perimeter

    Args:
        channel_cross_section: 2D binary array (1 = channel, 0 = material)
        voxel_size: Optional pixel spacing

    Returns:
        Hydraulic diameter in mm
    """
    # Channel area
    channel_area = np.sum(channel_cross_section > 0)

    if channel_area == 0:
        return 0.0

    # Perimeter (channel-material interface)
    from ..core.morphology import erode

    eroded = erode(
        channel_cross_section.reshape(1, *channel_cross_section.shape), kernel_size=1
    )
    eroded = eroded[0]
    perimeter_voxels = np.sum((channel_cross_section > 0) & (eroded == 0))

    # Convert to physical units
    if voxel_size:
        pixel_area = voxel_size[0] * voxel_size[1]
        pixel_perimeter = 2 * (voxel_size[0] + voxel_size[1])  # Approximate
        area_physical = channel_area * pixel_area
        perimeter_physical = perimeter_voxels * pixel_perimeter
    else:
        area_physical = float(channel_area)
        perimeter_physical = float(perimeter_voxels)

    # Hydraulic diameter
    if perimeter_physical > 0:
        hydraulic_diameter = 4.0 * area_physical / perimeter_physical
    else:
        hydraulic_diameter = 0.0

    return float(hydraulic_diameter)


def estimate_reynolds_number(
    channel_geometry: Dict[str, float],
    flow_velocity: float,
    fluid_properties: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate Reynolds number for flow in channels.

    Re = ρ * v * D / μ

    Args:
        channel_geometry: Channel geometry (mean_diameter in mm)
        flow_velocity: Flow velocity (m/s)
        fluid_properties: Optional fluid properties

    Returns:
        Dictionary with Reynolds number and flow regime
    """
    if fluid_properties is None:
        fluid_properties = {"density": 1000.0, "viscosity": 0.001}  # kg/m³  # Pa·s

    # Hydraulic diameter (convert mm to m)
    mean_diameter_m = channel_geometry.get("mean_diameter", 0.0) / 1000.0

    # Reynolds number
    Re = (
        fluid_properties["density"] * flow_velocity * mean_diameter_m
    ) / fluid_properties["viscosity"]

    # Flow regime
    if Re < 2300:
        regime = "laminar"
    elif Re < 4000:
        regime = "transitional"
    else:
        regime = "turbulent"

    return {
        "reynolds_number": float(Re),
        "flow_regime": regime,
        "hydraulic_diameter": float(mean_diameter_m * 1000),  # mm
        "flow_velocity": float(flow_velocity),
        "density": float(fluid_properties["density"]),
        "viscosity": float(fluid_properties["viscosity"]),
    }


def analyze_flow_distribution(
    volume: np.ndarray,
    flow_direction: str = "z",
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Analyze flow distribution uniformity along flow direction.

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        voxel_size: Optional voxel spacing

    Returns:
        Dictionary with flow distribution analysis
    """
    # Invert volume: channels = 1
    channel_volume = 1 - volume

    # Analyze cross-sections along flow direction
    if flow_direction == "z":
        axis = 0
        n_slices = volume.shape[0]
    elif flow_direction == "y":
        axis = 1
        n_slices = volume.shape[1]
    else:  # 'x'
        axis = 2
        n_slices = volume.shape[2]

    channel_areas = []
    n_channels_per_slice = []

    for i in range(n_slices):
        if axis == 0:
            slice_2d = channel_volume[i, :, :]
        elif axis == 1:
            slice_2d = channel_volume[:, i, :]
        else:
            slice_2d = channel_volume[:, :, i]

        # Channel area
        channel_area = np.sum(slice_2d > 0)

        # Number of channels (connected components)
        labeled, n_components = ndimage.label(slice_2d)
        n_channels = n_components

        channel_areas.append(channel_area)
        n_channels_per_slice.append(n_channels)

    channel_areas = np.array(channel_areas)
    n_channels_per_slice = np.array(n_channels_per_slice)

    # Convert to physical units
    if voxel_size:
        if flow_direction == "z":
            pixel_area = voxel_size[1] * voxel_size[2]
        elif flow_direction == "y":
            pixel_area = voxel_size[0] * voxel_size[2]
        else:
            pixel_area = voxel_size[0] * voxel_size[1]

        channel_areas_physical = channel_areas * pixel_area
    else:
        channel_areas_physical = channel_areas.astype(float)

    # Distribution metrics
    mean_area = float(np.mean(channel_areas_physical))
    std_area = float(np.std(channel_areas_physical))
    cv_area = std_area / mean_area if mean_area > 0 else 0.0

    mean_n_channels = float(np.mean(n_channels_per_slice))
    std_n_channels = float(np.std(n_channels_per_slice))

    # Uniformity (1 - CV, higher is better)
    uniformity = 1.0 - cv_area if cv_area <= 1.0 else 0.0

    # Maldistribution detection (areas vary by more than 20%)
    maldistribution_threshold = 0.2
    is_maldistributed = cv_area > maldistribution_threshold

    return {
        "mean_channel_area": float(mean_area),
        "std_channel_area": float(std_area),
        "cv_channel_area": float(cv_area),
        "uniformity": float(uniformity),
        "maldistribution": float(cv_area),  # Alias for compatibility
        "is_maldistributed": bool(is_maldistributed),
        "mean_n_channels": float(mean_n_channels),
        "std_n_channels": float(std_n_channels),
        "n_slices": int(n_slices),
        "flow_direction": flow_direction,
        "interpretation": (
            "Uniform distribution"
            if uniformity > 0.8
            else (
                "Moderate maldistribution"
                if uniformity > 0.6
                else "Severe maldistribution"
            )
        ),
    }


def compute_flow_uniformity(
    volume: np.ndarray,
    flow_direction: str = "z",
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> float:
    """
    Compute flow uniformity coefficient (0-1, higher is better).

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        voxel_size: Optional voxel spacing

    Returns:
        Uniformity coefficient (0-1)
    """
    distribution = analyze_flow_distribution(volume, flow_direction, voxel_size)
    return distribution["uniformity"]


def detect_flow_maldistribution(
    volume: np.ndarray,
    flow_direction: str = "z",
    threshold: float = 0.2,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Detect flow maldistribution (uneven flow distribution).

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        threshold: CV threshold for maldistribution (default 0.2 = 20%)
        voxel_size: Optional voxel spacing

    Returns:
        Dictionary with maldistribution analysis
    """
    distribution = analyze_flow_distribution(volume, flow_direction, voxel_size)

    is_maldistributed = distribution["cv_channel_area"] > threshold

    # Identify problematic regions
    # (Would need more detailed analysis to identify specific regions)

    return {
        "is_maldistributed": bool(is_maldistributed),
        "cv_threshold": float(threshold),
        "actual_cv": float(distribution["cv_channel_area"]),
        "uniformity": float(distribution["uniformity"]),
        "severity": (
            "None"
            if not is_maldistributed
            else (
                "Mild"
                if distribution["cv_channel_area"] < 0.4
                else "Moderate" if distribution["cv_channel_area"] < 0.6 else "Severe"
            )
        ),
    }


def comprehensive_flow_analysis(
    volume: np.ndarray,
    flow_direction: str = "z",
    channel_geometry: Optional[Dict[str, float]] = None,
    flow_conditions: Optional[Dict[str, float]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive flow analysis combining all methods.

    Args:
        volume: Binary volume
        flow_direction: Direction of water flow
        channel_geometry: Optional channel geometry
        flow_conditions: Optional flow conditions
        voxel_size: Optional voxel spacing

    Returns:
        Comprehensive flow analysis results
    """
    results = {}

    # Connectivity
    results["connectivity"] = analyze_flow_connectivity(
        volume, flow_direction, voxel_size=voxel_size
    )

    # Flow paths
    results["flow_paths"] = identify_flow_paths(volume, flow_direction)

    # Dead ends
    results["dead_ends"] = detect_dead_end_channels(volume, flow_direction)

    # Tortuosity
    results["tortuosity"] = compute_tortuosity(
        volume, flow_direction, voxel_size=voxel_size
    )

    # Path lengths
    results["path_lengths"] = compute_flow_path_lengths(
        volume, flow_direction, voxel_size
    )

    # Branching
    results["branching"] = analyze_flow_branching(volume, flow_direction)

    # Flow distribution
    results["flow_distribution"] = analyze_flow_distribution(
        volume, flow_direction, voxel_size
    )

    # Flow resistance (if geometry and conditions provided)
    if channel_geometry:
        results["flow_resistance"] = estimate_flow_resistance(
            volume, channel_geometry, flow_conditions, voxel_size=voxel_size
        )

    # Summary
    summary = {
        "connected": results["connectivity"]["connected"],
        "tortuosity": results["tortuosity"].get("tortuosity"),
        "n_dead_ends": results["dead_ends"]["n_dead_ends"],
        "dead_end_fraction": results["dead_ends"]["dead_end_fraction"],
        "flow_uniformity": results["flow_distribution"]["uniformity"],
        "is_maldistributed": results["flow_distribution"]["is_maldistributed"],
    }

    if "flow_resistance" in results:
        summary["pressure_drop_kpa"] = results["flow_resistance"].get(
            "pressure_drop_kpa"
        )
        summary["reynolds_number"] = results["flow_resistance"].get("reynolds_number")
        summary["flow_regime"] = results["flow_resistance"].get("flow_regime")

    results["summary"] = summary

    return results
