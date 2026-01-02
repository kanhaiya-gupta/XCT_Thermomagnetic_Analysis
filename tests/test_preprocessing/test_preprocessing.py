"""
Tests for preprocessing module.
"""

import numpy as np
import pytest
from src.preprocessing.preprocessing import (
    filter_by_volume,
    filter_by_sphericity,
    filter_by_spatial_bounds,
    filter_by_aspect_ratio,
    remove_edge_objects,
    apply_filters,
    analyze_object_properties,
    compute_sphericity,
    compute_aspect_ratio,
)
from tests.test_utils import assert_volume_valid


class TestFilterByVolume:
    """Test volume filtering."""

    @pytest.mark.unit
    def test_filter_by_volume_min(self, simple_volume, voxel_size):
        """Test filtering by minimum volume."""
        volume = (simple_volume > 128).astype(np.uint8)
        filtered = filter_by_volume(volume, voxel_size, min_volume=1000)
        assert_volume_valid(filtered)

    @pytest.mark.unit
    def test_filter_by_volume_max(self, simple_volume, voxel_size):
        """Test filtering by maximum volume."""
        volume = (simple_volume > 128).astype(np.uint8)
        filtered = filter_by_volume(volume, voxel_size, max_volume=100000)
        assert_volume_valid(filtered)


class TestFilterBySphericity:
    """Test sphericity filtering."""

    @pytest.mark.unit
    def test_filter_by_sphericity(self, simple_volume, voxel_size):
        """Test filtering by sphericity."""
        volume = (simple_volume > 128).astype(np.uint8)
        filtered = filter_by_sphericity(volume, voxel_size, min_sphericity=0.5)
        assert_volume_valid(filtered)


class TestComputeSphericity:
    """Test sphericity computation."""

    @pytest.mark.unit
    def test_compute_sphericity_sphere(self, voxel_size):
        """Test sphericity for a sphere (should be close to 1)."""
        # Create a sphere
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        center = (25, 25, 25)
        radius = 10
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    dist = np.sqrt(
                        (i - center[0]) ** 2
                        + (j - center[1]) ** 2
                        + (k - center[2]) ** 2
                    )
                    if dist <= radius:
                        volume[i, j, k] = 1

        sphericity = compute_sphericity(volume, voxel_size)
        assert 0 <= sphericity <= 1
        # Sphere should have high sphericity
        assert sphericity > 0.8


class TestApplyFilters:
    """Test filter application."""

    @pytest.mark.unit
    def test_apply_filters_basic(self, simple_volume, voxel_size):
        """Test applying multiple filters."""
        volume = (simple_volume > 128).astype(np.uint8)
        filters_config = {"min_volume": 100, "min_sphericity": 0.3}
        filtered, stats = apply_filters(volume, voxel_size, filters_config)
        assert_volume_valid(filtered)
