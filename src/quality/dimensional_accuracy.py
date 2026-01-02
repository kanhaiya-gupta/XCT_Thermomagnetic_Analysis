"""
Dimensional Accuracy Analysis Module

Analyze dimensional accuracy of 3D-printed parts by comparing:
- Actual vs. designed dimensions (CAD comparison)
- Geometric deviations
- Tolerance compliance
- Surface deviation mapping
- Build orientation effects
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import logging

logger = logging.getLogger(__name__)


def compute_geometric_deviations(
    volume: np.ndarray,
    design_specs: Dict[str, Any],
    voxel_size: Tuple[float, float, float],
) -> Dict[str, Any]:
    """
    Compute geometric deviations from design specifications.

    Args:
        volume: Binary volume (1 = material, 0 = void)
        design_specs: Dictionary with design specifications:
            - 'dimensions': {'x': float, 'y': float, 'z': float} (mm)
            - 'tolerance': float (mm) - default tolerance
            - 'features': List of feature specifications (optional)
        voxel_size: Voxel spacing (dx, dy, dz) in mm

    Returns:
        Dictionary with deviation analysis results
    """
    # Get actual dimensions from volume
    coords = np.argwhere(volume > 0)
    if len(coords) == 0:
        return {
            "error": "No material found in volume",
            "deviations": {},
            "within_tolerance": {},
        }

    # Convert to physical coordinates
    coords_physical = coords * np.array(voxel_size)

    # Compute actual dimensions
    dims_actual = {
        "x": float((coords_physical[:, 2].max() - coords_physical[:, 2].min())),
        "y": float((coords_physical[:, 1].max() - coords_physical[:, 1].min())),
        "z": float((coords_physical[:, 0].max() - coords_physical[:, 0].min())),
    }

    # Get design dimensions
    design_dims = design_specs.get("dimensions", {})
    tolerance = design_specs.get("tolerance", 0.1)  # Default 0.1 mm

    # Compute deviations
    deviations = {}
    within_tolerance = {}
    deviation_percentages = {}

    for axis in ["x", "y", "z"]:
        if axis in design_dims:
            design_dim = design_dims[axis]
            actual_dim = dims_actual[axis]
            deviation = actual_dim - design_dim
            deviation_pct = (deviation / design_dim * 100) if design_dim > 0 else 0.0

            deviations[axis] = float(deviation)
            deviation_percentages[axis] = float(deviation_pct)
            within_tolerance[axis] = abs(deviation) <= tolerance

    # Overall dimensional accuracy
    n_within_tolerance = sum(within_tolerance.values())
    n_total = len(within_tolerance)
    dimensional_accuracy = (n_within_tolerance / n_total * 100) if n_total > 0 else 0.0

    # RMS deviation
    if deviations:
        rms_deviation = np.sqrt(np.mean([d**2 for d in deviations.values()]))
        max_deviation = max([abs(d) for d in deviations.values()])
    else:
        rms_deviation = 0.0
        max_deviation = 0.0

    return {
        "actual_dimensions": dims_actual,
        "design_dimensions": design_dims,
        "deviations": deviations,
        "deviation_percentages": deviation_percentages,
        "within_tolerance": within_tolerance,
        "tolerance": tolerance,
        "dimensional_accuracy": float(dimensional_accuracy),
        "rms_deviation": float(rms_deviation),
        "max_deviation": float(max_deviation),
        "n_dimensions_checked": n_total,
    }


def analyze_tolerance_compliance(
    volume: np.ndarray,
    tolerances: Dict[str, Union[float, Dict[str, float]]],
    voxel_size: Tuple[float, float, float],
    design_specs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze tolerance compliance for different features/dimensions.

    Args:
        volume: Binary volume
        tolerances: Dictionary of tolerances:
            - 'default': float (mm) - default tolerance
            - 'x': float (mm) - tolerance for X dimension
            - 'y': float (mm) - tolerance for Y dimension
            'z': float (mm) - tolerance for Z dimension
            - 'features': Dict[str, float] - feature-specific tolerances
        voxel_size: Voxel spacing
        design_specs: Optional design specifications

    Returns:
        Dictionary with tolerance compliance analysis
    """
    # Get deviations if design specs provided
    if design_specs:
        deviation_results = compute_geometric_deviations(
            volume, design_specs, voxel_size
        )
        deviations = deviation_results["deviations"]
    else:
        # Compute basic dimensions
        coords = np.argwhere(volume > 0)
        if len(coords) == 0:
            return {"error": "No material found in volume"}

        coords_physical = coords * np.array(voxel_size)
        deviations = {
            "x": 0.0,  # Would need design specs to compute
            "y": 0.0,
            "z": 0.0,
        }

    # Get tolerances
    default_tolerance = tolerances.get("default", 0.1)

    compliance_results = {}
    overall_compliance = []

    # Check each dimension
    for axis in ["x", "y", "z"]:
        if axis in tolerances:
            tolerance = tolerances[axis]
        else:
            tolerance = default_tolerance

        if axis in deviations:
            deviation = abs(deviations[axis])
            within_tolerance = deviation <= tolerance
            compliance_pct = (
                (1.0 - min(deviation / tolerance, 1.0)) * 100 if tolerance > 0 else 0.0
            )

            compliance_results[axis] = {
                "deviation": float(deviations[axis]),
                "tolerance": float(tolerance),
                "within_tolerance": bool(within_tolerance),
                "compliance_percentage": float(compliance_pct),
                "margin": float(tolerance - deviation),
            }
            overall_compliance.append(within_tolerance)

    # Overall compliance
    overall_compliance_pct = (
        (np.mean(overall_compliance) * 100) if overall_compliance else 0.0
    )

    # Grade based on compliance
    if overall_compliance_pct >= 95:
        grade = "Excellent"
    elif overall_compliance_pct >= 85:
        grade = "Good"
    elif overall_compliance_pct >= 70:
        grade = "Acceptable"
    else:
        grade = "Needs Improvement"

    return {
        "compliance_by_dimension": compliance_results,
        "compliance_rate": float(overall_compliance_pct)
        / 100.0,  # As fraction for compatibility
        "overall_compliance_percentage": float(overall_compliance_pct),
        "compliant_dimensions": [
            axis
            for axis, result in compliance_results.items()
            if result.get("within_tolerance", False)
        ],
        "grade": grade,
        "n_dimensions_checked": len(compliance_results),
        "all_within_tolerance": (
            all(overall_compliance) if overall_compliance else False
        ),
    }


def surface_deviation_map(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    cad_surface: Optional[np.ndarray] = None,
    method: str = "distance_transform",
) -> Dict[str, Any]:
    """
    Compute surface deviation map comparing actual surface to CAD surface.

    If CAD surface not provided, computes surface roughness/deviation from ideal.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        cad_surface: Optional CAD surface mesh or volume
        method: Method for deviation computation ('distance_transform' or 'point_cloud')

    Returns:
        Dictionary with surface deviation analysis
    """
    # Extract surface from volume
    from ..core.morphology import erode

    # Get surface voxels (material voxels adjacent to void)
    eroded = erode(volume, kernel_size=1)
    surface_mask = (volume > 0) & (eroded == 0)
    surface_coords = np.argwhere(surface_mask)

    if len(surface_coords) == 0:
        # Return empty deviation map if no surface found
        return {
            "error": "No surface found",
            "deviation_map": np.zeros_like(volume, dtype=float),
            "mean_deviation": 0.0,
            "std_deviation": 0.0,
            "max_deviation": 0.0,
            "min_deviation": 0.0,
            "n_surface_voxels": 0,
        }

    # Convert to physical coordinates
    surface_coords_physical = surface_coords * np.array(voxel_size)

    if cad_surface is not None:
        # Compare with CAD surface
        if isinstance(cad_surface, np.ndarray):
            # CAD surface as volume
            cad_surface_mask = cad_surface > 0
            cad_surface_coords = np.argwhere(cad_surface_mask)
            cad_surface_coords_physical = cad_surface_coords * np.array(voxel_size)

            # Compute distances from actual surface to CAD surface
            if method == "distance_transform":
                # Use distance transform
                from scipy.ndimage import distance_transform_edt

                # Create distance map from CAD surface
                cad_distance = distance_transform_edt(~cad_surface_mask)

                # Get distances at actual surface points
                distances = []
                for coord in surface_coords:
                    distances.append(cad_distance[tuple(coord)])

                distances = np.array(distances) * np.mean(voxel_size)  # Convert to mm
            else:
                # Point cloud method
                distances = []
                for point in surface_coords_physical:
                    dists = cdist([point], cad_surface_coords_physical)
                    distances.append(np.min(dists))
                distances = np.array(distances)
        else:
            # CAD surface as mesh (would need meshio or similar)
            distances = np.zeros(len(surface_coords_physical))
            logger.warning("CAD surface mesh comparison not fully implemented")
    else:
        # No CAD - compute surface roughness (deviation from smooth surface)
        # Use local surface normal variation as proxy for roughness
        from scipy.ndimage import gaussian_filter

        # Smooth the surface
        smoothed_volume = gaussian_filter(volume.astype(float), sigma=1.0)
        smoothed_surface = (smoothed_volume > 0.5) & (
            erode(smoothed_volume > 0.5, kernel_size=1) == 0
        )

        # Compute deviation from smoothed surface
        distances = np.zeros(len(surface_coords))
        for i, coord in enumerate(surface_coords):
            # Distance to nearest smoothed surface point
            if smoothed_surface[tuple(coord)]:
                distances[i] = 0.0
            else:
                # Find distance to smoothed surface
                from scipy.ndimage import distance_transform_edt

                dist_map = distance_transform_edt(~smoothed_surface)
                distances[i] = dist_map[tuple(coord)] * np.mean(voxel_size)

    # Statistics
    mean_deviation = float(np.mean(np.abs(distances)))
    std_deviation = float(np.std(distances))
    max_deviation = float(np.max(np.abs(distances)))
    rms_deviation = float(np.sqrt(np.mean(distances**2)))

    # Deviation map (for visualization)
    deviation_map = np.zeros_like(volume, dtype=float)
    for i, coord in enumerate(surface_coords):
        deviation_map[tuple(coord)] = distances[i]

    return {
        "mean_deviation": mean_deviation,
        "std_deviation": std_deviation,
        "max_deviation": max_deviation,
        "rms_deviation": rms_deviation,
        "deviation_map": deviation_map,
        "surface_points": len(surface_coords),
        "deviation_distribution": {
            "mean": mean_deviation,
            "std": std_deviation,
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "percentiles": {
                "p5": float(np.percentile(distances, 5)),
                "p25": float(np.percentile(distances, 25)),
                "p50": float(np.percentile(distances, 50)),
                "p75": float(np.percentile(distances, 75)),
                "p95": float(np.percentile(distances, 95)),
            },
        },
    }


def build_orientation_effects(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    orientations: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Analyze effects of build orientation on dimensional accuracy.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        orientations: List of orientation dictionaries with 'x', 'y', 'z' angles (degrees)
                     If None, analyzes default orientations

    Returns:
        Dictionary with orientation effects analysis
    """
    if orientations is None:
        # Default orientations: 0°, 45°, 90° rotations
        orientations = [
            {"x": 0, "y": 0, "z": 0, "name": "0_0_0"},
            {"x": 0, "y": 45, "z": 0, "name": "0_45_0"},
            {"x": 0, "y": 90, "z": 0, "name": "0_90_0"},
            {"x": 45, "y": 0, "z": 0, "name": "45_0_0"},
        ]

    results = {}

    for orient in orientations:
        # Rotate volume (simplified - would need proper 3D rotation)
        # For now, analyze dimensions in different orientations
        name = orient.get(
            "name", f"{orient.get('x', 0)}_{orient.get('y', 0)}_{orient.get('z', 0)}"
        )

        # Get dimensions (simplified - actual rotation would be more complex)
        coords = np.argwhere(volume > 0)
        if len(coords) == 0:
            continue

        coords_physical = coords * np.array(voxel_size)

        dims = {
            "x": float(coords_physical[:, 2].max() - coords_physical[:, 2].min()),
            "y": float(coords_physical[:, 1].max() - coords_physical[:, 1].min()),
            "z": float(coords_physical[:, 0].max() - coords_physical[:, 0].min()),
        }

        # Compute aspect ratios
        dims_list = [dims["x"], dims["y"], dims["z"]]
        max_dim = max(dims_list)
        min_dim = min(dims_list)
        aspect_ratio = max_dim / min_dim if min_dim > 0 else 0.0

        results[name] = {
            "dimensions": dims,
            "aspect_ratio": float(aspect_ratio),
            "orientation": orient,
        }

    # Compare orientations
    if len(results) > 1:
        # Find orientation with best dimensional stability
        aspect_ratios = [r["aspect_ratio"] for r in results.values()]
        best_orientation = min(results.keys(), key=lambda k: results[k]["aspect_ratio"])

        return {
            "orientations": results,
            "best_orientation": best_orientation,
            "aspect_ratio_range": {
                "min": float(min(aspect_ratios)),
                "max": float(max(aspect_ratios)),
                "range": float(max(aspect_ratios) - min(aspect_ratios)),
            },
            "recommendation": f"Best orientation: {best_orientation} (lowest aspect ratio)",
        }

    return {"orientations": results}


def compare_to_cad(
    volume: np.ndarray,
    cad_model: Union[np.ndarray, Dict[str, Any]],
    voxel_size: Tuple[float, float, float],
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """
    Compare actual volume to CAD model.

    Args:
        volume: Binary volume (actual)
        cad_model: CAD model as binary volume or dictionary with CAD specifications
        voxel_size: Voxel spacing
        tolerance: Tolerance for comparison (mm)

    Returns:
        Dictionary with CAD comparison results
    """
    if isinstance(cad_model, dict):
        # CAD model as specifications
        return compute_geometric_deviations(volume, cad_model, voxel_size)
    elif isinstance(cad_model, np.ndarray):
        # CAD model as volume
        # Compute overlap and differences
        overlap = np.logical_and(volume > 0, cad_model > 0)
        actual_only = np.logical_and(volume > 0, cad_model == 0)
        cad_only = np.logical_and(volume == 0, cad_model > 0)

        voxel_volume = np.prod(voxel_size)

        overlap_volume = np.sum(overlap) * voxel_volume
        actual_only_volume = np.sum(actual_only) * voxel_volume
        cad_only_volume = np.sum(cad_only) * voxel_volume

        total_cad_volume = np.sum(cad_model > 0) * voxel_volume
        total_actual_volume = np.sum(volume > 0) * voxel_volume

        # Volume accuracy
        volume_error = total_actual_volume - total_cad_volume
        volume_error_pct = (
            (volume_error / total_cad_volume * 100) if total_cad_volume > 0 else 0.0
        )

        # Overlap percentage
        overlap_pct = (
            (overlap_volume / total_cad_volume * 100) if total_cad_volume > 0 else 0.0
        )

        # Surface deviation
        surface_dev = surface_deviation_map(volume, voxel_size, cad_model)

        return {
            "volume_comparison": {
                "cad_volume": float(total_cad_volume),
                "actual_volume": float(total_actual_volume),
                "volume_error": float(volume_error),
                "volume_error_percentage": float(volume_error_pct),
                "overlap_volume": float(overlap_volume),
                "overlap_percentage": float(overlap_pct),
                "actual_only_volume": float(actual_only_volume),
                "cad_only_volume": float(cad_only_volume),
            },
            "surface_deviation": surface_dev,
            "within_tolerance": (
                abs(volume_error_pct) <= (tolerance / total_cad_volume * 100)
                if total_cad_volume > 0
                else False
            ),
            "dimensional_accuracy": float(overlap_pct),
        }
    else:
        return {"error": "Invalid CAD model format"}


def comprehensive_dimensional_analysis(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    design_specs: Optional[Dict[str, Any]] = None,
    tolerances: Optional[Dict[str, Any]] = None,
    cad_model: Optional[Union[np.ndarray, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive dimensional accuracy analysis combining all methods.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        design_specs: Optional design specifications
        tolerances: Optional tolerance specifications
        cad_model: Optional CAD model for comparison

    Returns:
        Comprehensive dimensional analysis results
    """
    results = {}

    # Geometric deviations
    if design_specs:
        results["geometric_deviations"] = compute_geometric_deviations(
            volume, design_specs, voxel_size
        )

    # Tolerance compliance
    if tolerances:
        results["tolerance_compliance"] = analyze_tolerance_compliance(
            volume, tolerances, voxel_size, design_specs
        )

    # Surface deviation
    results["surface_deviation"] = surface_deviation_map(volume, voxel_size, cad_model)

    # Build orientation effects
    results["orientation_effects"] = build_orientation_effects(volume, voxel_size)

    # CAD comparison
    if cad_model is not None:
        results["cad_comparison"] = compare_to_cad(volume, cad_model, voxel_size)

    # Summary
    summary = {"analysis_complete": True, "n_analyses": len(results)}

    if "geometric_deviations" in results:
        summary["dimensional_accuracy"] = results["geometric_deviations"].get(
            "dimensional_accuracy", 0.0
        )
        summary["rms_deviation"] = results["geometric_deviations"].get(
            "rms_deviation", 0.0
        )

    if "tolerance_compliance" in results:
        summary["overall_compliance"] = results["tolerance_compliance"].get(
            "overall_compliance_percentage", 0.0
        )
        summary["grade"] = results["tolerance_compliance"].get("grade", "Unknown")

    if "surface_deviation" in results:
        summary["mean_surface_deviation"] = results["surface_deviation"].get(
            "mean_deviation", 0.0
        )

    results["summary"] = summary

    return results
