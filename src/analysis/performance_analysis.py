"""
Performance Analysis Module for Thermomagnetic Elements

Connects XCT-measured structure to performance metrics:
- Heat transfer efficiency (related to surface area, channel geometry)
- Magnetic properties (related to material distribution, defects)
- Mechanical properties (related to porosity, defects)

This module enables process-structure-property relationships for
thermomagnetic heat exchanger components.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def estimate_heat_transfer_efficiency(
    metrics: Dict[str, float],
    channel_geometry: Dict[str, float],
    flow_conditions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate heat transfer efficiency from XCT metrics.

    Heat transfer efficiency is related to:
    - Surface area (more surface = better heat transfer)
    - Channel geometry (channel width affects flow)
    - Void fraction (affects flow path)
    - Specific surface area

    Args:
        metrics: XCT metrics (volume, surface_area, void_fraction, etc.)
        channel_geometry: Channel width statistics
        flow_conditions: Optional flow conditions (velocity, etc.)

    Returns:
        Dictionary with heat transfer efficiency estimates
    """
    surface_area = metrics.get("surface_area", 0.0)
    volume = metrics.get("volume", 1.0)
    void_fraction = metrics.get("void_fraction", 0.0)
    specific_surface_area = metrics.get("specific_surface_area", 0.0)

    # Mean channel width
    mean_channel_width = channel_geometry.get("mean_diameter", 0.0)

    # Heat transfer efficiency factors
    # 1. Surface area factor (more surface = better)
    surface_factor = specific_surface_area / 10.0  # Normalize (typical ~10 mm²/mm³)
    surface_factor = np.clip(surface_factor, 0.0, 1.0)

    # 2. Channel geometry factor (optimal channel width for flow)
    # Assume optimal channel width around 0.5-1.0 mm for water flow
    optimal_channel = 0.75  # mm
    if mean_channel_width > 0:
        channel_factor = (
            1.0 - abs(mean_channel_width - optimal_channel) / optimal_channel
        )
        channel_factor = np.clip(channel_factor, 0.0, 1.0)
    else:
        channel_factor = 0.0

    # 3. Void fraction factor (need some void for flow, but not too much)
    # Optimal void fraction around 0.3-0.5 for heat exchangers
    optimal_void = 0.4
    void_factor = 1.0 - abs(void_fraction - optimal_void) / optimal_void
    void_factor = np.clip(void_factor, 0.0, 1.0)

    # Combined efficiency (weighted)
    heat_transfer_efficiency = (
        0.4 * surface_factor + 0.3 * channel_factor + 0.3 * void_factor
    )

    return {
        "heat_transfer_efficiency": float(heat_transfer_efficiency),
        "surface_factor": float(surface_factor),
        "channel_factor": float(channel_factor),
        "void_factor": float(void_factor),
        "specific_surface_area": float(specific_surface_area),
        "mean_channel_width": float(mean_channel_width),
        "void_fraction": float(void_fraction),
    }


def estimate_magnetic_property_impact(
    metrics: Dict[str, float],
    porosity_distribution: Optional[Dict[str, Any]] = None,
    defect_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate impact of structure on magnetic properties.

    Magnetic properties are affected by:
    - Material continuity (defects break magnetic paths)
    - Porosity distribution (affects magnetic flux)
    - Surface defects (affect domain structure)

    Args:
        metrics: XCT metrics
        porosity_distribution: Porosity distribution analysis
        defect_metrics: Optional defect metrics

    Returns:
        Dictionary with magnetic property impact estimates
    """
    void_fraction = metrics.get("void_fraction", 0.0)
    relative_density = metrics.get("relative_density", 1.0)

    # Porosity profile statistics
    if porosity_distribution is None:
        porosity_distribution = {}
    porosity_profile = porosity_distribution.get("porosity_profile", {})
    mean_porosity = porosity_profile.get("mean_porosity", void_fraction)
    std_porosity = porosity_profile.get("std_porosity", 0.0)

    # Material continuity factor (higher density = better)
    continuity_factor = relative_density

    # Porosity uniformity factor (more uniform = better for magnetic properties)
    # Lower std = more uniform
    if mean_porosity > 0:
        uniformity_factor = 1.0 - min(1.0, std_porosity / mean_porosity)
    else:
        uniformity_factor = 1.0

    # Defect impact (if available)
    if defect_metrics:
        defect_factor = 1.0 - min(
            1.0, defect_metrics.get("defect_volume_fraction", 0.0)
        )
    else:
        defect_factor = 1.0

    # Combined magnetic property factor
    magnetic_property_factor = (
        0.5 * continuity_factor + 0.3 * uniformity_factor + 0.2 * defect_factor
    )

    return {
        "magnetic_property_factor": float(magnetic_property_factor),
        "continuity_factor": float(continuity_factor),
        "uniformity_factor": float(uniformity_factor),
        "defect_factor": float(defect_factor),
        "relative_density": float(relative_density),
        "porosity_std": float(std_porosity),
    }


def estimate_mechanical_property_impact(
    metrics: Dict[str, float],
    porosity_distribution: Optional[Dict[str, Any]] = None,
    filament_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate impact of structure on mechanical properties.

    Mechanical properties are affected by:
    - Porosity (reduces strength)
    - Filament connectivity
    - Defect distribution
    - Dimensional accuracy

    Args:
        metrics: XCT metrics
        porosity_distribution: Porosity distribution
        filament_analysis: Optional filament analysis results

    Returns:
        Dictionary with mechanical property impact estimates
    """
    void_fraction = metrics.get("void_fraction", 0.0)
    relative_density = metrics.get("relative_density", 1.0)

    # Porosity impact (more porosity = lower strength)
    # Typical relationship: strength ~ (1 - porosity)^n
    n = 2.0  # Empirical exponent
    strength_factor = (1.0 - void_fraction) ** n

    # Filament connectivity (if available)
    if filament_analysis:
        filament_diameters = filament_analysis.get("filament_diameter", {}).get(
            "diameters", []
        )
        if len(filament_diameters) > 0:
            # More consistent diameters = better connectivity
            std_diameter = np.std(filament_diameters)
            mean_diameter = np.mean(filament_diameters)
            if mean_diameter > 0:
                connectivity_factor = 1.0 - min(1.0, std_diameter / mean_diameter)
            else:
                connectivity_factor = 0.5
        else:
            connectivity_factor = 0.5
    else:
        connectivity_factor = relative_density  # Use density as proxy

    # Combined mechanical property factor
    mechanical_property_factor = 0.7 * strength_factor + 0.3 * connectivity_factor

    return {
        "mechanical_property_factor": float(mechanical_property_factor),
        "strength_factor": float(strength_factor),
        "connectivity_factor": float(connectivity_factor),
        "relative_density": float(relative_density),
        "void_fraction": float(void_fraction),
    }


def process_structure_performance_relationship(
    process_params: pd.DataFrame,
    structure_metrics: pd.DataFrame,
    performance_metrics: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Analyze relationships between process parameters, structure, and performance.

    For thermomagnetic elements:
    - Process: extrusion temp, speed, layer height, heat treatment
    - Structure: porosity, filament diameter, channel width, surface area
    - Performance: heat transfer efficiency, magnetic properties, mechanical properties

    Args:
        process_params: Process parameters (extrusion temp, speed, etc.)
        structure_metrics: Structure metrics from XCT
        performance_metrics: Optional measured performance data

    Returns:
        Dictionary with correlation analysis and relationships
    """
    # Merge data
    data = pd.merge(
        process_params, structure_metrics, left_index=True, right_index=True
    )

    # Calculate performance estimates if not provided
    if performance_metrics is None:
        # Estimate performance from structure
        performance_estimates = []
        for idx, row in data.iterrows():
            metrics = row.to_dict()
            # Estimate heat transfer efficiency
            # Estimate magnetic properties
            # Estimate mechanical properties
            # (Would need full analysis results)
            pass

    # Calculate correlations
    process_cols = process_params.columns.tolist()
    structure_cols = structure_metrics.columns.tolist()

    correlations = {}

    # Process -> Structure correlations
    for proc_col in process_cols:
        for struct_col in structure_cols:
            if proc_col in data.columns and struct_col in data.columns:
                corr = data[proc_col].corr(data[struct_col])
                if not np.isnan(corr):
                    correlations[f"Process[{proc_col}] -> Structure[{struct_col}]"] = (
                        float(corr)
                    )

    # Structure -> Performance correlations (if available)
    if performance_metrics is not None:
        data = pd.merge(data, performance_metrics, left_index=True, right_index=True)
        perf_cols = performance_metrics.columns.tolist()

        for struct_col in structure_cols:
            for perf_col in perf_cols:
                if struct_col in data.columns and perf_col in data.columns:
                    corr = data[struct_col].corr(data[perf_col])
                    if not np.isnan(corr):
                        correlations[
                            f"Structure[{struct_col}] -> Performance[{perf_col}]"
                        ] = float(corr)

    return {
        "correlations": correlations,
        "n_samples": len(data),
        "process_parameters": process_cols,
        "structure_metrics": structure_cols,
        "performance_metrics": perf_cols if performance_metrics is not None else [],
    }


def optimize_for_performance(
    process_param_bounds: Dict[str, Tuple[float, float]],
    structure_simulator: Callable,
    performance_objective: str = "heat_transfer",
    constraints: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Optimize process parameters to maximize performance.

    For thermomagnetic elements, optimize for:
    - Heat transfer efficiency (maximize)
    - Magnetic properties (maximize)
    - Mechanical properties (maximize, or constraint)

    Args:
        process_param_bounds: Process parameter bounds
            e.g., {'extrusion_temp': (200, 250), 'print_speed': (10, 50)}
        structure_simulator: Function that predicts structure from process params
        performance_objective: 'heat_transfer', 'magnetic', 'mechanical', or 'combined'
        constraints: Optional constraints (e.g., minimum mechanical strength)

    Returns:
        Dictionary with optimization results
    """
    from .virtual_experiments import optimize_process_parameters

    def objective_function(params):
        # Predict structure from process parameters
        structure = structure_simulator(params)

        # Calculate performance
        if performance_objective == "heat_transfer":
            # Estimate heat transfer efficiency
            return estimate_heat_transfer_efficiency(
                structure["metrics"], structure.get("channel_geometry", {})
            )["heat_transfer_efficiency"]
        elif performance_objective == "magnetic":
            return estimate_magnetic_property_impact(
                structure["metrics"], structure.get("porosity_distribution", {})
            )["magnetic_property_factor"]
        elif performance_objective == "mechanical":
            return estimate_mechanical_property_impact(
                structure["metrics"], structure.get("porosity_distribution", {})
            )["mechanical_property_factor"]
        else:  # combined
            ht = estimate_heat_transfer_efficiency(
                structure["metrics"], structure.get("channel_geometry", {})
            )["heat_transfer_efficiency"]
            mag = estimate_magnetic_property_impact(
                structure["metrics"], structure.get("porosity_distribution", {})
            )["magnetic_property_factor"]
            mech = estimate_mechanical_property_impact(
                structure["metrics"], structure.get("porosity_distribution", {})
            )["mechanical_property_factor"]
            return (ht + mag + mech) / 3.0

    return optimize_process_parameters(
        process_param_bounds, objective_function, constraints=constraints, maximize=True
    )
