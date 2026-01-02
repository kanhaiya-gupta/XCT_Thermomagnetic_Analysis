"""
Thermal Analysis Module

Analyze thermal characteristics for heat exchanger components:
- Thermal resistance (conduction and convection)
- Temperature gradient analysis
- Heat transfer coefficient estimation
- Overall thermal performance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def compute_thermal_resistance(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    material_properties: Optional[Dict[str, float]] = None,
    flow_conditions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute overall thermal resistance.

    Thermal resistance includes:
    - Conduction resistance (through material)
    - Convection resistance (at surfaces)

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm (dx, dy, dz)
        material_properties: Optional dictionary with:
            - 'thermal_conductivity': k (W/m·K)
            - 'density': ρ (kg/m³)
            - 'specific_heat': cp (J/kg·K)
        flow_conditions: Optional flow conditions for convection

    Returns:
        Dictionary with thermal resistance analysis
    """
    # Default material properties
    if material_properties is None:
        material_properties = {}

    # Get material properties
    k = material_properties.get("thermal_conductivity", 50.0)  # W/m·K (default: steel)
    density = material_properties.get("density", 7850.0)  # kg/m³
    cp = material_properties.get("specific_heat", 500.0)  # J/kg·K

    # Compute metrics
    from ..core.metrics import compute_all_metrics

    metrics = compute_all_metrics(volume, voxel_size)

    volume_m3 = metrics.get("volume", 0.0) / 1e9  # Convert mm³ to m³
    surface_area_m2 = metrics.get("surface_area", 0.0) / 1e6  # Convert mm² to m²

    # Characteristic length (for conduction)
    # Use volume/surface_area ratio
    if surface_area_m2 > 0:
        characteristic_length = 3.0 * volume_m3 / surface_area_m2  # m
    else:
        characteristic_length = 0.0

    # Conduction resistance: R_cond = L / (k * A)
    # Simplified: R = characteristic_length / (k * effective_area)
    effective_area = surface_area_m2 / 2.0  # Approximate
    if k > 0 and effective_area > 0:
        conduction_resistance = characteristic_length / (k * effective_area)  # K/W
    else:
        conduction_resistance = np.inf

    # Convection resistance: R_conv = 1 / (h * A)
    if flow_conditions:
        # Estimate heat transfer coefficient
        htc_result = estimate_heat_transfer_coefficient(
            volume, voxel_size, flow_conditions, material_properties
        )
        h = htc_result["htc"]
    else:
        # Default: natural convection ~10 W/m²·K
        h = 10.0  # W/m²·K

    if h > 0 and surface_area_m2 > 0:
        convection_resistance = 1.0 / (h * surface_area_m2)  # K/W
    else:
        convection_resistance = np.inf

    # Overall thermal resistance (series: R_total = R_cond + R_conv)
    if conduction_resistance != np.inf and convection_resistance != np.inf:
        total_resistance = conduction_resistance + convection_resistance
    elif conduction_resistance != np.inf:
        total_resistance = conduction_resistance
    elif convection_resistance != np.inf:
        total_resistance = convection_resistance
    else:
        total_resistance = np.inf

    # Thermal conductance (inverse of resistance)
    if total_resistance > 0 and total_resistance != np.inf:
        thermal_conductance = 1.0 / total_resistance  # W/K
    else:
        thermal_conductance = 0.0

    return {
        "conduction_resistance": (
            float(conduction_resistance) if conduction_resistance != np.inf else None
        ),
        "convection_resistance": (
            float(convection_resistance) if convection_resistance != np.inf else None
        ),
        "total_resistance": (
            float(total_resistance) if total_resistance != np.inf else None
        ),
        "thermal_conductance": float(thermal_conductance),
        "heat_transfer_coefficient": float(h),
        "characteristic_length": float(characteristic_length * 1000),  # mm
        "surface_area": float(surface_area_m2 * 1e6),  # mm²
        "volume": float(volume_m3 * 1e9),  # mm³
    }


def analyze_conduction_resistance(
    volume: np.ndarray,
    thermal_conductivity: float,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Analyze conduction resistance through material.

    Args:
        volume: Binary volume
        thermal_conductivity: Thermal conductivity (W/m·K)
        voxel_size: Optional voxel spacing

    Returns:
        Dictionary with conduction resistance analysis
    """
    from ..core.metrics import compute_all_metrics

    if voxel_size:
        metrics = compute_all_metrics(volume, voxel_size)
    else:
        metrics = compute_all_metrics(volume, (0.001, 0.001, 0.001))

    volume_m3 = metrics.get("volume", 0.0) / 1e9  # m³
    surface_area_m2 = metrics.get("surface_area", 0.0) / 1e6  # m²

    # Characteristic length
    if surface_area_m2 > 0:
        L = 3.0 * volume_m3 / surface_area_m2  # m
    else:
        L = 0.0

    # Conduction resistance: R = L / (k * A)
    effective_area = surface_area_m2 / 2.0
    if thermal_conductivity > 0 and effective_area > 0:
        resistance = L / (thermal_conductivity * effective_area)  # K/W
    else:
        resistance = np.inf

    return {
        "conduction_resistance": float(resistance) if resistance != np.inf else None,
        "characteristic_length": float(L * 1000),  # mm
        "thermal_conductivity": float(thermal_conductivity),
        "effective_area": float(effective_area * 1e6),  # mm²
    }


def analyze_convection_resistance(
    surface_area: float,
    flow_conditions: Dict[str, float],
    heat_transfer_coefficient: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze convection resistance at surfaces.

    Args:
        surface_area: Surface area (mm²)
        flow_conditions: Flow conditions dictionary
        heat_transfer_coefficient: Optional heat transfer coefficient (W/m²·K)

    Returns:
        Dictionary with convection resistance analysis
    """
    surface_area_m2 = surface_area / 1e6  # Convert to m²

    # Estimate heat transfer coefficient if not provided
    if heat_transfer_coefficient is None:
        # Simplified: h ≈ 100-1000 W/m²·K for forced convection
        flow_velocity = flow_conditions.get("velocity", 0.1)  # m/s
        # Rough estimate: h ≈ 100 * v^0.8 (for water)
        heat_transfer_coefficient = 100.0 * (flow_velocity**0.8)

    # Convection resistance: R = 1 / (h * A)
    if heat_transfer_coefficient > 0 and surface_area_m2 > 0:
        resistance = 1.0 / (heat_transfer_coefficient * surface_area_m2)  # K/W
    else:
        resistance = np.inf

    return {
        "convection_resistance": float(resistance) if resistance != np.inf else None,
        "heat_transfer_coefficient": float(heat_transfer_coefficient),
        "surface_area": float(surface_area),  # mm²
        "surface_area_m2": float(surface_area_m2),  # m²
    }


def estimate_heat_transfer_coefficient(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    flow_conditions: Optional[Dict[str, float]] = None,
    material_properties: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate heat transfer coefficient for forced convection.

    Uses simplified correlations based on flow conditions.

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm (dx, dy, dz)
        flow_conditions: Optional flow conditions (velocity, etc.)
        material_properties: Optional material properties

    Returns:
        Dictionary with heat transfer coefficient analysis
    """
    # Default flow conditions
    if flow_conditions is None:
        flow_conditions = {}

    flow_velocity = flow_conditions.get("velocity", 0.1)  # m/s

    # Simplified correlation for forced convection in channels
    # For water: h ≈ Nu * k / D, where Nu ≈ 0.023 * Re^0.8 * Pr^0.4

    # Fluid properties (water at room temperature)
    k_fluid = 0.6  # W/m·K (water)
    Pr = 7.0  # Prandtl number (water)

    # Get channel diameter (simplified)
    from ..core.filament_analysis import estimate_channel_width

    if voxel_size:
        channel_result = estimate_channel_width(volume, voxel_size)
        mean_diameter = channel_result.get("mean_diameter", 0.001)  # m
    else:
        mean_diameter = 0.001  # Default 1 mm

    # Reynolds number
    rho = 1000.0  # kg/m³
    mu = 0.001  # Pa·s
    Re = (rho * flow_velocity * mean_diameter) / mu

    # Nusselt number (Dittus-Boelter for turbulent, simplified)
    if Re > 2300:
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)
    else:
        # Laminar: Nu ≈ 3.66 (constant)
        Nu = 3.66

    # Heat transfer coefficient
    h = Nu * k_fluid / mean_diameter if mean_diameter > 0 else 100.0

    return {
        "htc": float(h),  # Alias for compatibility
        "heat_transfer_coefficient": float(h),  # W/m²·K
        "reynolds_number": float(Re),
        "nusselt_number": float(Nu),
        "mean_diameter": float(mean_diameter * 1000),  # mm
        "flow_velocity": float(flow_velocity),  # m/s
    }


def estimate_temperature_gradient(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    thermal_properties: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate temperature gradient for given heat flux.

    Args:
        volume: Binary volume (1 = material, 0 = void)
        voxel_size: Voxel spacing in mm (dx, dy, dz)
        thermal_properties: Optional thermal properties (conductivity, etc.)

    Returns:
        Dictionary with temperature gradient analysis
    """
    # Default thermal properties
    if thermal_properties is None:
        thermal_properties = {}

    k = thermal_properties.get("thermal_conductivity", 50.0)  # W/m·K
    heat_flux = thermal_properties.get("heat_flux", 1000.0)  # W/m² (default)

    from ..core.metrics import compute_all_metrics

    if voxel_size:
        metrics = compute_all_metrics(volume, voxel_size)
    else:
        metrics = compute_all_metrics(volume, (0.001, 0.001, 0.001))

    surface_area_m2 = metrics.get("surface_area", 0.0) / 1e6  # m²
    volume_m3 = metrics.get("volume", 0.0) / 1e9  # m³

    # Characteristic length
    if surface_area_m2 > 0:
        L = 3.0 * volume_m3 / surface_area_m2  # m
    else:
        L = 0.0

    # Temperature gradient: ΔT = q * L / k
    # where q is heat flux, L is length, k is conductivity
    if k > 0:
        temperature_gradient = (heat_flux * L) / k  # K
    else:
        temperature_gradient = 0.0

    # Heat transfer rate
    heat_transfer_rate = heat_flux * surface_area_m2  # W

    return {
        "gradient": float(temperature_gradient),  # Alias for compatibility
        "temperature_gradient": float(temperature_gradient),  # K
        "direction": "z",  # Default direction
        "heat_flux": float(heat_flux),  # W/m²
        "heat_transfer_rate": float(heat_transfer_rate),  # W
        "characteristic_length": float(L * 1000),  # mm
        "thermal_conductivity": float(k),  # W/m·K
    }


def comprehensive_thermal_analysis(
    volume: np.ndarray,
    material_properties: Dict[str, float],
    flow_conditions: Optional[Dict[str, float]] = None,
    heat_flux: Optional[float] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive thermal analysis combining all methods.

    Args:
        volume: Binary volume
        material_properties: Material thermal properties
        flow_conditions: Optional flow conditions
        heat_flux: Optional heat flux (W/m²)
        voxel_size: Optional voxel spacing

    Returns:
        Comprehensive thermal analysis results
    """
    results = {}

    # Thermal resistance
    results["thermal_resistance"] = compute_thermal_resistance(
        volume, voxel_size, material_properties, flow_conditions
    )

    # Conduction resistance
    results["conduction"] = analyze_conduction_resistance(
        volume, material_properties.get("thermal_conductivity", 50.0), voxel_size
    )

    # Convection resistance
    if flow_conditions:
        from ..core.metrics import compute_all_metrics

        metrics = compute_all_metrics(volume, voxel_size)

        surface_area = metrics.get("surface_area", 0.0)
        htc_result = estimate_heat_transfer_coefficient(
            volume, voxel_size, flow_conditions, material_properties
        )
        h = htc_result["htc"]

        results["convection"] = analyze_convection_resistance(
            surface_area, flow_conditions, h
        )

    # Temperature gradient
    if heat_flux is not None:
        thermal_props = material_properties.copy() if material_properties else {}
        thermal_props["heat_flux"] = heat_flux
        results["temperature_gradient"] = estimate_temperature_gradient(
            volume, voxel_size, thermal_props
        )

    # Summary
    summary = {
        "total_thermal_resistance": results["thermal_resistance"].get(
            "total_resistance"
        ),
        "thermal_conductance": results["thermal_resistance"].get("thermal_conductance"),
        "heat_transfer_coefficient": results["thermal_resistance"].get(
            "heat_transfer_coefficient"
        ),
    }

    if "temperature_gradient" in results:
        summary["temperature_gradient"] = results["temperature_gradient"].get(
            "temperature_gradient"
        )
        summary["heat_transfer_rate"] = results["temperature_gradient"].get(
            "heat_transfer_rate"
        )

    results["summary"] = summary

    return results
