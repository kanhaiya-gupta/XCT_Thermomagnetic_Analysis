"""
Tests for main XCTAnalyzer class.
"""

import numpy as np
import pytest
from src.analyzer import XCTAnalyzer


class TestXCTAnalyzer:
    """Test XCTAnalyzer class."""

    @pytest.mark.unit
    def test_analyzer_initialization(self, voxel_size):
        """Test analyzer initialization."""
        analyzer = XCTAnalyzer(voxel_size=voxel_size)

        assert analyzer.voxel_size == voxel_size

    @pytest.mark.unit
    def test_analyzer_segmentation(self, simple_volume, voxel_size):
        """Test analyzer segmentation."""
        analyzer = XCTAnalyzer(voxel_size=voxel_size)

        segmented = analyzer.segment(simple_volume, method="otsu")

        assert segmented is not None
        assert segmented.shape == simple_volume.shape

    @pytest.mark.unit
    def test_analyzer_compute_metrics(self, simple_volume, voxel_size):
        """Test analyzer metrics computation."""
        analyzer = XCTAnalyzer(voxel_size=voxel_size)
        segmented = (simple_volume > 128).astype(np.uint8)

        metrics = analyzer.compute_metrics(segmented)

        assert "volume" in metrics
        assert "surface_area" in metrics
        assert "void_fraction" in metrics

    @pytest.mark.integration
    def test_analyzer_complete_workflow(self, simple_volume, voxel_size):
        """Test complete analyzer workflow."""
        analyzer = XCTAnalyzer(voxel_size=voxel_size)

        # Load volume (using simple_volume directly)
        # Segment
        segmented = analyzer.segment(simple_volume, method="otsu")

        # Compute metrics
        metrics = analyzer.compute_metrics(segmented)

        # Check results
        assert metrics is not None
        assert "volume" in metrics
        assert metrics["volume"] >= 0
