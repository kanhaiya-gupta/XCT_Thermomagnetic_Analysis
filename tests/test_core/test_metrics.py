"""
Tests for metrics module.
"""

import numpy as np
import pytest
from src.core.metrics import (
    compute_volume,
    compute_surface_area,
    compute_void_fraction,
    compute_relative_density,
    compute_specific_surface_area,
    compute_all_metrics,
)
from tests.test_utils import (
    assert_volume_valid,
    get_known_sphere_volume,
    get_known_cube_volume,
)
from tests.fixtures.synthetic_volumes import create_sphere_volume, create_cube_volume


class TestComputeVolume:
    """Test volume computation."""

    @pytest.mark.unit
    def test_compute_volume_sphere(self, voxel_size):
        """Test volume calculation for known sphere."""
        # Create sphere with radius 2 mm (fits in 50x50x50 volume with 0.1 mm voxels)
        radius_mm = 2.0
        radius_voxels = int(radius_mm / voxel_size[0])
        shape = (50, 50, 50)
        center = (25, 25, 25)

        volume = create_sphere_volume(shape, center, radius_voxels)
        segmented = (volume > 0).astype(np.uint8)

        computed_volume = compute_volume(segmented, voxel_size)
        expected_volume = get_known_sphere_volume(radius_mm, voxel_size)

        # Allow 15% tolerance due to voxel discretization
        assert abs(computed_volume - expected_volume) / expected_volume < 0.15

    @pytest.mark.unit
    def test_compute_volume_cube(self, voxel_size):
        """Test volume calculation for known cube."""
        # Create cube with side length 3 mm (fits in 50x50x50 volume)
        side_length_mm = 3.0
        side_length_voxels = int(side_length_mm / voxel_size[0])
        shape = (50, 50, 50)
        corner = (20, 20, 20)
        size = (side_length_voxels, side_length_voxels, side_length_voxels)

        volume = create_cube_volume(shape, corner, size)
        segmented = (volume > 0).astype(np.uint8)

        computed_volume = compute_volume(segmented, voxel_size)
        expected_volume = get_known_cube_volume(side_length_mm)

        # Should be very close for cube (within 5% due to discretization)
        assert abs(computed_volume - expected_volume) / expected_volume < 0.05

    @pytest.mark.unit
    def test_compute_volume_empty(self, voxel_size):
        """Test volume calculation for empty volume."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        computed_volume = compute_volume(volume, voxel_size)
        assert computed_volume == 0.0

    @pytest.mark.unit
    def test_compute_volume_full(self, voxel_size):
        """Test volume calculation for full volume."""
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        computed_volume = compute_volume(volume, voxel_size)
        expected_volume = 50 * 50 * 50 * np.prod(voxel_size)
        assert abs(computed_volume - expected_volume) < 1e-6


class TestComputeSurfaceArea:
    """Test surface area computation."""

    @pytest.mark.unit
    def test_compute_surface_area_basic(self, simple_volume, voxel_size):
        """Test surface area calculation."""
        segmented = (simple_volume > 128).astype(np.uint8)
        surface_area = compute_surface_area(segmented, voxel_size)
        assert surface_area > 0
        assert isinstance(surface_area, float)

    @pytest.mark.unit
    def test_compute_surface_area_empty(self, voxel_size):
        """Test surface area for empty volume."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        surface_area = compute_surface_area(volume, voxel_size)
        # Empty volume should have zero or very small surface area
        assert surface_area >= 0


class TestComputeVoidFraction:
    """Test void fraction computation."""

    @pytest.mark.unit
    def test_compute_void_fraction_basic(self, porous_volume, voxel_size):
        """Test void fraction calculation."""
        segmented = (porous_volume > 128).astype(np.uint8)
        void_fraction = compute_void_fraction(segmented)
        assert 0 <= void_fraction <= 1

    @pytest.mark.unit
    def test_compute_void_fraction_empty(self, voxel_size):
        """Test void fraction for empty volume."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        void_fraction = compute_void_fraction(volume)
        assert void_fraction == 1.0  # All void

    @pytest.mark.unit
    def test_compute_void_fraction_full(self, voxel_size):
        """Test void fraction for full volume."""
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        void_fraction = compute_void_fraction(volume)
        assert void_fraction == 0.0  # No void


class TestComputeRelativeDensity:
    """Test relative density computation."""

    @pytest.mark.unit
    def test_compute_relative_density_basic(self, porous_volume, voxel_size):
        """Test relative density calculation."""
        segmented = (porous_volume > 128).astype(np.uint8)
        relative_density = compute_relative_density(segmented)
        assert 0 <= relative_density <= 1

    @pytest.mark.unit
    def test_relative_density_void_fraction_relationship(
        self, porous_volume, voxel_size
    ):
        """Test that relative density = 1 - void fraction."""
        segmented = (porous_volume > 128).astype(np.uint8)
        void_fraction = compute_void_fraction(segmented)
        relative_density = compute_relative_density(segmented)
        assert abs(relative_density - (1 - void_fraction)) < 1e-6


class TestComputeSpecificSurfaceArea:
    """Test specific surface area computation."""

    @pytest.mark.unit
    def test_compute_specific_surface_area_basic(self, simple_volume, voxel_size):
        """Test specific surface area calculation."""
        segmented = (simple_volume > 128).astype(np.uint8)
        ssa = compute_specific_surface_area(segmented, voxel_size)
        assert ssa >= 0
        assert isinstance(ssa, float)


class TestComputeAllMetrics:
    """Test comprehensive metrics computation."""

    @pytest.mark.unit
    def test_compute_all_metrics_basic(self, simple_volume, voxel_size):
        """Test computing all metrics."""
        segmented = (simple_volume > 128).astype(np.uint8)
        metrics = compute_all_metrics(segmented, voxel_size)

        assert isinstance(metrics, dict)
        assert "volume" in metrics
        assert "surface_area" in metrics
        assert "void_fraction" in metrics
        assert "relative_density" in metrics
        assert "specific_surface_area" in metrics

        # Check all values are valid
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)

    @pytest.mark.unit
    def test_compute_all_metrics_consistency(self, simple_volume, voxel_size):
        """Test that individual and all metrics are consistent."""
        segmented = (simple_volume > 128).astype(np.uint8)

        # Compute individually
        volume = compute_volume(segmented, voxel_size)
        void_fraction = compute_void_fraction(segmented)
        relative_density = compute_relative_density(segmented)

        # Compute all at once
        all_metrics = compute_all_metrics(segmented, voxel_size)

        # Check consistency
        assert abs(all_metrics["volume"] - volume) < 1e-6
        assert abs(all_metrics["void_fraction"] - void_fraction) < 1e-6
        assert abs(all_metrics["relative_density"] - relative_density) < 1e-6
