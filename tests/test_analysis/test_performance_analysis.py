"""
Tests for performance analysis module.
"""

import numpy as np
import pytest
from src.analysis.performance_analysis import (
    estimate_heat_transfer_efficiency,
    estimate_magnetic_property_impact,
    estimate_mechanical_property_impact,
)


class TestEstimateHeatTransferEfficiency:
    """Test heat transfer efficiency estimation."""

    @pytest.mark.unit
    def test_heat_transfer_efficiency_basic(self):
        """Test basic heat transfer efficiency estimation."""
        metrics = {
            "surface_area": 1000.0,
            "volume": 10000.0,
            "void_fraction": 0.4,
            "specific_surface_area": 0.1,
        }
        channel_geometry = {"mean_diameter": 0.75}

        result = estimate_heat_transfer_efficiency(metrics, channel_geometry)

        assert "heat_transfer_efficiency" in result
        assert 0 <= result["heat_transfer_efficiency"] <= 1
        assert "surface_factor" in result
        assert "channel_factor" in result
        assert "void_factor" in result

    @pytest.mark.unit
    def test_heat_transfer_efficiency_optimal_conditions(self):
        """Test efficiency with optimal conditions."""
        metrics = {
            "surface_area": 1000.0,
            "volume": 10000.0,
            "void_fraction": 0.4,  # Optimal
            "specific_surface_area": 0.1,
        }
        channel_geometry = {"mean_diameter": 0.75}  # Optimal

        result = estimate_heat_transfer_efficiency(metrics, channel_geometry)

        # Should have high efficiency with optimal conditions
        assert result["heat_transfer_efficiency"] > 0.5


class TestEstimateMagneticPropertyImpact:
    """Test magnetic property impact estimation."""

    @pytest.mark.unit
    def test_magnetic_property_impact_basic(self):
        """Test basic magnetic property impact estimation."""
        metrics = {"relative_density": 0.7, "void_fraction": 0.3}

        result = estimate_magnetic_property_impact(metrics)

        assert "magnetic_property_factor" in result
        assert 0 <= result["magnetic_property_factor"] <= 1

    @pytest.mark.unit
    def test_magnetic_property_high_density(self):
        """Test magnetic impact with high density."""
        metrics = {"relative_density": 0.95, "void_fraction": 0.05}  # High density

        result = estimate_magnetic_property_impact(metrics)

        # High density should have high magnetic property factor
        assert result["magnetic_property_factor"] > 0.8


class TestEstimateMechanicalPropertyImpact:
    """Test mechanical property impact estimation."""

    @pytest.mark.unit
    def test_mechanical_property_impact_basic(self):
        """Test basic mechanical property impact estimation."""
        metrics = {"relative_density": 0.7, "void_fraction": 0.3}

        result = estimate_mechanical_property_impact(metrics)

        assert "mechanical_property_factor" in result
        assert 0 <= result["mechanical_property_factor"] <= 1

    @pytest.mark.unit
    def test_mechanical_property_high_density(self):
        """Test mechanical impact with high density."""
        metrics = {"relative_density": 0.95, "void_fraction": 0.05}  # High density

        result = estimate_mechanical_property_impact(metrics)

        # High density should have high mechanical property factor
        assert result["mechanical_property_factor"] > 0.8
