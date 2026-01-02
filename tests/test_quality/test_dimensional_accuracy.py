"""
Tests for dimensional accuracy module.
"""

import numpy as np
import pytest
from src.quality.dimensional_accuracy import (
    compute_geometric_deviations,
    analyze_tolerance_compliance,
    surface_deviation_map,
)


class TestComputeGeometricDeviations:
    """Test geometric deviation computation."""

    @pytest.mark.unit
    def test_compute_geometric_deviations_basic(self, simple_volume, voxel_size):
        """Test basic geometric deviation computation."""
        volume = (simple_volume > 128).astype(np.uint8)

        design_specs = {
            "dimensions": {"x": 5.0, "y": 5.0, "z": 5.0},  # mm
            "tolerance": 0.1,  # mm
        }

        result = compute_geometric_deviations(volume, design_specs, voxel_size)

        assert "deviations" in result
        assert "within_tolerance" in result
        assert "x" in result["deviations"] or "error" in result

    @pytest.mark.unit
    def test_compute_geometric_deviations_empty_volume(self, voxel_size):
        """Test with empty volume."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)

        design_specs = {"dimensions": {"x": 5.0, "y": 5.0, "z": 5.0}, "tolerance": 0.1}

        result = compute_geometric_deviations(volume, design_specs, voxel_size)

        assert "error" in result


class TestAnalyzeToleranceCompliance:
    """Test tolerance compliance analysis."""

    @pytest.mark.unit
    def test_analyze_tolerance_compliance_basic(self, simple_volume, voxel_size):
        """Test basic tolerance compliance analysis."""
        volume = (simple_volume > 128).astype(np.uint8)

        design_specs = {"dimensions": {"x": 5.0, "y": 5.0, "z": 5.0}, "tolerance": 0.1}

        result = analyze_tolerance_compliance(volume, design_specs, voxel_size)

        assert "compliance_rate" in result
        assert 0 <= result["compliance_rate"] <= 1
        assert "compliant_dimensions" in result


class TestSurfaceDeviationMap:
    """Test surface deviation mapping."""

    @pytest.mark.unit
    def test_surface_deviation_map_basic(self, simple_volume, voxel_size):
        """Test basic surface deviation map."""
        volume = (simple_volume > 128).astype(np.uint8)

        # Create simple CAD reference (slightly larger)
        cad_volume = np.zeros_like(volume)
        cad_volume[10:40, 10:40, 10:40] = 1

        result = surface_deviation_map(volume, voxel_size, cad_volume)

        assert "deviation_map" in result
        assert "mean_deviation" in result
        assert "std_deviation" in result
