"""
Tests for validation module.
"""

import numpy as np
import pytest
from src.quality.validation import compute_accuracy_metrics, validate_segmentation


class TestComputeAccuracyMetrics:
    """Test accuracy metrics computation."""

    @pytest.mark.unit
    def test_compute_accuracy_metrics_basic(self):
        """Test basic accuracy metrics computation."""
        # Create ground truth and predicted volumes
        ground_truth = np.zeros((50, 50, 50), dtype=np.uint8)
        ground_truth[10:30, 10:30, 10:30] = 1

        predicted = np.zeros((50, 50, 50), dtype=np.uint8)
        predicted[12:28, 12:28, 12:28] = 1  # Slightly different

        metrics = compute_accuracy_metrics(ground_truth, predicted)

        assert "dice_coefficient" in metrics
        assert "jaccard_index" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["dice_coefficient"] <= 1
        assert 0 <= metrics["jaccard_index"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    @pytest.mark.unit
    def test_compute_accuracy_metrics_perfect_match(self):
        """Test accuracy metrics with perfect match."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        volume[10:30, 10:30, 10:30] = 1

        metrics = compute_accuracy_metrics(volume, volume)

        assert metrics["dice_coefficient"] == 1.0
        assert metrics["jaccard_index"] == 1.0
        assert metrics["accuracy"] == 1.0


class TestValidateSegmentation:
    """Test segmentation validation."""

    @pytest.mark.unit
    def test_validate_segmentation_basic(self, simple_volume):
        """Test basic segmentation validation."""
        segmented = (simple_volume > 128).astype(np.uint8)

        result = validate_segmentation(segmented)

        assert "is_valid" in result
        assert "checks" in result
        assert isinstance(result["is_valid"], bool)

    @pytest.mark.unit
    def test_validate_segmentation_empty(self):
        """Test validation of empty segmentation."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)

        result = validate_segmentation(volume)

        # Empty volume should fail validation
        assert result["is_valid"] == False
