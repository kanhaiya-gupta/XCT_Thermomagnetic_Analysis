"""
Tests for thermal analysis module.
"""

import numpy as np
import pytest
from src.experimental.thermal_analysis import (
    compute_thermal_resistance,
    estimate_heat_transfer_coefficient,
    estimate_temperature_gradient,
)


class TestComputeThermalResistance:
    """Test thermal resistance computation."""

    @pytest.mark.unit
    def test_compute_thermal_resistance_basic(self, simple_volume, voxel_size):
        """Test basic thermal resistance computation."""
        volume = (simple_volume > 128).astype(np.uint8)

        result = compute_thermal_resistance(volume, voxel_size)

        assert "total_resistance" in result
        assert "conduction_resistance" in result
        assert "convection_resistance" in result
        assert result["total_resistance"] > 0

    @pytest.mark.unit
    def test_compute_thermal_resistance_with_material_properties(
        self, simple_volume, voxel_size
    ):
        """Test thermal resistance with material properties."""
        volume = (simple_volume > 128).astype(np.uint8)

        material_properties = {
            "thermal_conductivity": 50.0,  # W/(m·K)
            "heat_transfer_coefficient": 1000.0,  # W/(m²·K)
        }

        result = compute_thermal_resistance(
            volume, voxel_size, material_properties=material_properties
        )

        assert "total_resistance" in result
        assert result["total_resistance"] > 0


class TestEstimateHeatTransferCoefficient:
    """Test heat transfer coefficient estimation."""

    @pytest.mark.unit
    def test_estimate_htc_basic(self, simple_volume, voxel_size):
        """Test basic HTC estimation."""
        volume = (simple_volume > 128).astype(np.uint8)

        result = estimate_heat_transfer_coefficient(volume, voxel_size)

        assert "htc" in result
        assert result["htc"] > 0
        # Typical HTC for water flow: 100-10000 W/(m²·K)
        assert 100 <= result["htc"] <= 10000


class TestEstimateTemperatureGradient:
    """Test temperature gradient estimation."""

    @pytest.mark.unit
    def test_estimate_temperature_gradient_basic(self, simple_volume, voxel_size):
        """Test basic temperature gradient estimation."""
        volume = (simple_volume > 128).astype(np.uint8)

        result = estimate_temperature_gradient(volume, voxel_size)

        assert "gradient" in result
        assert "direction" in result
        assert result["gradient"] >= 0
