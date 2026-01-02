"""
Tests for flow analysis module.
"""

import numpy as np
import pytest
from src.experimental.flow_analysis import (
    analyze_flow_connectivity,
    compute_tortuosity,
    estimate_flow_resistance,
    analyze_flow_distribution,
)


class TestAnalyzeFlowConnectivity:
    """Test flow connectivity analysis."""

    @pytest.mark.unit
    def test_analyze_flow_connectivity_basic(self):
        """Test basic flow connectivity analysis."""
        # Create a volume with a channel along z-direction
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        volume[:, 25, 25] = 0  # Channel along z

        result = analyze_flow_connectivity(volume, flow_direction="z")

        assert "connected" in result
        assert "n_components" in result
        assert isinstance(result["connected"], bool)

    @pytest.mark.unit
    def test_analyze_flow_connectivity_disconnected(self):
        """Test connectivity with disconnected channels."""
        # Create volume with disconnected channels
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        volume[10:20, 25, 25] = 0  # Channel 1
        volume[30:40, 25, 25] = 0  # Channel 2 (disconnected)

        result = analyze_flow_connectivity(volume, flow_direction="z")

        # Should detect disconnection
        assert result["n_components"] >= 2


class TestComputeTortuosity:
    """Test tortuosity computation."""

    @pytest.mark.unit
    def test_compute_tortuosity_straight_path(self):
        """Test tortuosity for straight path (should be ~1.0)."""
        # Create straight channel
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        volume[:, 25, 25] = 0  # Straight channel

        result = compute_tortuosity(volume, flow_direction="z")

        assert "mean_tortuosity" in result
        assert result["mean_tortuosity"] >= 1.0
        # Straight path should have tortuosity close to 1.0
        assert result["mean_tortuosity"] < 1.5

    @pytest.mark.unit
    def test_compute_tortuosity_curved_path(self):
        """Test tortuosity for curved path (should be > 1.0)."""
        # Create curved channel
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        for z in range(50):
            y = 25 + int(5 * np.sin(z * np.pi / 25))
            volume[z, y, 25] = 0

        result = compute_tortuosity(volume, flow_direction="z")

        assert "mean_tortuosity" in result
        # Curved path should have higher tortuosity
        assert result["mean_tortuosity"] > 1.0


class TestEstimateFlowResistance:
    """Test flow resistance estimation."""

    @pytest.mark.unit
    def test_estimate_flow_resistance_basic(self, voxel_size):
        """Test basic flow resistance estimation."""
        # Create channel volume
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        volume[:, 25, 25] = 0  # Channel

        result = estimate_flow_resistance(volume, voxel_size)

        assert "resistance" in result
        assert "pressure_drop" in result
        assert result["resistance"] > 0
        assert result["pressure_drop"] >= 0


class TestAnalyzeFlowDistribution:
    """Test flow distribution analysis."""

    @pytest.mark.unit
    def test_analyze_flow_distribution_basic(self):
        """Test basic flow distribution analysis."""
        # Create volume with multiple channels
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        volume[:, 20, 25] = 0  # Channel 1
        volume[:, 30, 25] = 0  # Channel 2

        result = analyze_flow_distribution(volume, flow_direction="z")

        assert "uniformity" in result
        assert "maldistribution" in result
        assert 0 <= result["uniformity"] <= 1
