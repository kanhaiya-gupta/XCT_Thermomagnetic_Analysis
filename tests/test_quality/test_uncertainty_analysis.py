"""
Tests for uncertainty analysis module.
"""

import numpy as np
import pytest
from src.quality.uncertainty_analysis import (
    measurement_uncertainty,
    segmentation_uncertainty,
    confidence_intervals,
    uncertainty_budget,
)


class TestMeasurementUncertainty:
    """Test measurement uncertainty computation."""

    @pytest.mark.unit
    def test_measurement_uncertainty_basic(self, simple_volume, voxel_size):
        """Test basic measurement uncertainty."""
        volume = (simple_volume > 128).astype(np.uint8)

        result = measurement_uncertainty(volume, voxel_size)

        assert "volume_uncertainty" in result
        assert "surface_area_uncertainty" in result
        assert result["volume_uncertainty"] >= 0
        assert result["surface_area_uncertainty"] >= 0

    @pytest.mark.unit
    def test_measurement_uncertainty_custom(self, simple_volume, voxel_size):
        """Test measurement uncertainty with custom uncertainties."""
        volume = (simple_volume > 128).astype(np.uint8)

        voxel_size_uncertainty = (0.001, 0.001, 0.001)  # 0.001 mm
        segmentation_uncertainty = 0.5  # voxels

        result = measurement_uncertainty(
            volume,
            voxel_size,
            voxel_size_uncertainty=voxel_size_uncertainty,
            segmentation_uncertainty=segmentation_uncertainty,
        )

        assert "volume_uncertainty" in result
        assert result["volume_uncertainty"] >= 0


class TestSegmentationUncertainty:
    """Test segmentation uncertainty analysis."""

    @pytest.mark.unit
    def test_segmentation_uncertainty_basic(self, simple_volume):
        """Test basic segmentation uncertainty."""
        volume = simple_volume.astype(np.float32)

        # Create multiple segmentations with different thresholds
        thresholds = [100, 120, 140, 160, 180]
        segmentations = [(volume > t).astype(np.uint8) for t in thresholds]

        result = segmentation_uncertainty(segmentations)

        assert "volume_uncertainty" in result
        assert "surface_area_uncertainty" in result
        assert result["volume_uncertainty"] >= 0


class TestConfidenceIntervals:
    """Test confidence interval computation."""

    @pytest.mark.unit
    def test_confidence_intervals_basic(self):
        """Test basic confidence interval computation."""
        values = np.random.normal(100, 10, 100)

        result = confidence_intervals(values, confidence_level=0.95)

        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["mean"]
        assert result["ci_upper"] > result["mean"]

    @pytest.mark.unit
    def test_confidence_intervals_different_levels(self):
        """Test confidence intervals with different confidence levels."""
        values = np.random.normal(100, 10, 100)

        result_90 = confidence_intervals(values, confidence_level=0.90)
        result_95 = confidence_intervals(values, confidence_level=0.95)
        result_99 = confidence_intervals(values, confidence_level=0.99)

        # Higher confidence should have wider intervals
        width_90 = result_90["ci_upper"] - result_90["ci_lower"]
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]

        assert width_90 < width_95 < width_99


class TestUncertaintyBudget:
    """Test uncertainty budget computation."""

    @pytest.mark.unit
    def test_uncertainty_budget_basic(self, simple_volume, voxel_size):
        """Test basic uncertainty budget."""
        volume = (simple_volume > 128).astype(np.uint8)

        result = uncertainty_budget(volume, voxel_size)

        assert "components" in result
        assert "total_uncertainty" in result
        assert (
            "contribution" in result["components"][0]
        )  # First component should have contribution
