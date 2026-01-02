"""
Tests for energy conversion module.
"""

import numpy as np
import pytest
from src.experimental.energy_conversion import (
    estimate_power_output,
    calculate_energy_conversion_efficiency,
    compute_power_density,
)


class TestEstimatePowerOutput:
    """Test power output estimation."""

    @pytest.mark.unit
    def test_estimate_power_output_basic(self, simple_volume, voxel_size):
        """Test basic power output estimation."""
        volume = (simple_volume > 128).astype(np.uint8)

        # Temperature difference
        delta_t = 10.0  # K

        result = estimate_power_output(volume, voxel_size, delta_t)

        assert "power_output" in result
        assert result["power_output"] >= 0

    @pytest.mark.unit
    def test_estimate_power_output_with_properties(self, simple_volume, voxel_size):
        """Test power output with material properties."""
        volume = (simple_volume > 128).astype(np.uint8)

        delta_t = 10.0
        material_properties = {
            "thermal_conductivity": 50.0,
            "magnetic_susceptibility": 0.1,
        }

        result = estimate_power_output(
            volume, voxel_size, delta_t, material_properties=material_properties
        )

        assert "power_output" in result
        assert result["power_output"] >= 0


class TestCalculateEnergyConversionEfficiency:
    """Test energy conversion efficiency calculation."""

    @pytest.mark.unit
    def test_calculate_efficiency_basic(self, simple_volume, voxel_size):
        """Test basic efficiency calculation."""
        volume = (simple_volume > 128).astype(np.uint8)

        # Heat input
        heat_input = 100.0  # W

        result = calculate_energy_conversion_efficiency(volume, voxel_size, heat_input)

        assert "efficiency" in result
        assert 0 <= result["efficiency"] <= 1
        # Typical efficiency for thermomagnetic generators: 1-5%
        assert result["efficiency"] <= 0.1


class TestComputePowerDensity:
    """Test power density computation."""

    @pytest.mark.unit
    def test_compute_power_density_basic(self, simple_volume, voxel_size):
        """Test basic power density computation."""
        volume = (simple_volume > 128).astype(np.uint8)

        power_output = 10.0  # W

        result = compute_power_density(volume, voxel_size, power_output)

        assert "power_density" in result
        assert result["power_density"] >= 0
        # Power density in W/m³
        assert result["power_density"] < 1e6  # Reasonable upper bound
