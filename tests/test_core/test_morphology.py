"""
Tests for morphology module.
"""

import numpy as np
import pytest
from src.core.morphology import (
    erode,
    dilate,
    open_operation,
    close_operation,
    remove_small_objects,
    fill_holes,
    skeletonize,
    distance_transform,
)
from tests.test_utils import assert_volume_valid


class TestErode:
    """Test erosion operation."""

    @pytest.mark.unit
    def test_erode_basic(self, simple_volume):
        """Test basic erosion."""
        volume = (simple_volume > 128).astype(np.uint8)
        eroded = erode(volume, kernel_size=3)
        assert_volume_valid(eroded)
        # Erosion should not increase volume
        assert np.sum(eroded) <= np.sum(volume)

    @pytest.mark.unit
    def test_erode_iterations(self, simple_volume):
        """Test erosion with multiple iterations."""
        volume = (simple_volume > 128).astype(np.uint8)
        eroded_1 = erode(volume, kernel_size=3, iterations=1)
        eroded_2 = erode(volume, kernel_size=3, iterations=2)
        assert_volume_valid(eroded_1)
        assert_volume_valid(eroded_2)
        # More iterations should erode more
        assert np.sum(eroded_2) <= np.sum(eroded_1)


class TestDilate:
    """Test dilation operation."""

    @pytest.mark.unit
    def test_dilate_basic(self, simple_volume):
        """Test basic dilation."""
        volume = (simple_volume > 128).astype(np.uint8)
        dilated = dilate(volume, kernel_size=3)
        assert_volume_valid(dilated)
        # Dilation should not decrease volume
        assert np.sum(dilated) >= np.sum(volume)

    @pytest.mark.unit
    def test_dilate_iterations(self, simple_volume):
        """Test dilation with multiple iterations."""
        volume = (simple_volume > 128).astype(np.uint8)
        dilated_1 = dilate(volume, kernel_size=3, iterations=1)
        dilated_2 = dilate(volume, kernel_size=3, iterations=2)
        assert_volume_valid(dilated_1)
        assert_volume_valid(dilated_2)
        # More iterations should dilate more
        assert np.sum(dilated_2) >= np.sum(dilated_1)


class TestOpenOperation:
    """Test opening operation."""

    @pytest.mark.unit
    def test_open_operation_basic(self, simple_volume):
        """Test basic opening operation."""
        volume = (simple_volume > 128).astype(np.uint8)
        opened = open_operation(volume, kernel_size=3)
        assert_volume_valid(opened)

    @pytest.mark.unit
    def test_open_removes_small_objects(self):
        """Test that opening removes small objects."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        # Add a large object
        volume[10:30, 10:30, 10:30] = 1
        # Add a small object
        volume[40:42, 40:42, 40:42] = 1

        opened = open_operation(volume, kernel_size=5)
        # Small object should be removed
        assert np.sum(opened[40:42, 40:42, 40:42]) == 0
        # Large object should remain
        assert np.sum(opened[10:30, 10:30, 10:30]) > 0


class TestCloseOperation:
    """Test closing operation."""

    @pytest.mark.unit
    def test_close_operation_basic(self, simple_volume):
        """Test basic closing operation."""
        volume = (simple_volume > 128).astype(np.uint8)
        closed = close_operation(volume, kernel_size=3)
        assert_volume_valid(closed)

    @pytest.mark.unit
    def test_close_fills_holes(self):
        """Test that closing fills small holes."""
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        # Add a small hole
        volume[25:27, 25:27, 25:27] = 0

        closed = close_operation(volume, kernel_size=5)
        # Hole should be filled
        assert np.sum(closed[25:27, 25:27, 25:27]) > 0


class TestRemoveSmallObjects:
    """Test small object removal."""

    @pytest.mark.unit
    def test_remove_small_objects_basic(self):
        """Test basic small object removal."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        # Add a large object
        volume[10:30, 10:30, 10:30] = 1
        # Add a small object
        volume[40:42, 40:42, 40:42] = 1

        cleaned = remove_small_objects(volume, min_size=1000)
        # Small object should be removed
        assert np.sum(cleaned[40:42, 40:42, 40:42]) == 0
        # Large object should remain
        assert np.sum(cleaned[10:30, 10:30, 10:30]) > 0

    @pytest.mark.unit
    def test_remove_small_objects_all_removed(self):
        """Test when all objects are small."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        volume[25:27, 25:27, 25:27] = 1

        cleaned = remove_small_objects(volume, min_size=1000)
        assert np.sum(cleaned) == 0


class TestFillHoles:
    """Test hole filling."""

    @pytest.mark.unit
    def test_fill_holes_basic(self):
        """Test basic hole filling."""
        volume = np.ones((50, 50, 50), dtype=np.uint8)
        # Add a hole
        volume[20:30, 20:30, 20:30] = 0

        filled = fill_holes(volume)
        assert_volume_valid(filled)
        # Hole should be filled
        assert np.sum(filled[20:30, 20:30, 20:30]) > 0

    @pytest.mark.unit
    def test_fill_holes_no_holes(self, simple_volume):
        """Test hole filling on volume without holes."""
        volume = (simple_volume > 128).astype(np.uint8)
        filled = fill_holes(volume)
        assert_volume_valid(filled)


class TestSkeletonize:
    """Test skeletonization."""

    @pytest.mark.unit
    def test_skeletonize_basic(self, simple_volume):
        """Test basic skeletonization."""
        volume = (simple_volume > 128).astype(np.uint8)
        skeleton = skeletonize(volume)
        assert_volume_valid(skeleton)
        # Skeleton should be subset of original
        assert np.all(skeleton <= volume)

    @pytest.mark.unit
    def test_skeletonize_preserves_connectivity(self):
        """Test that skeletonization preserves connectivity."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        # Create a connected structure
        volume[20:30, 25, 25] = 1

        skeleton = skeletonize(volume)
        # Should still be connected
        assert np.sum(skeleton) > 0


class TestDistanceTransform:
    """Test distance transform."""

    @pytest.mark.unit
    def test_distance_transform_basic(self, simple_volume):
        """Test basic distance transform."""
        volume = (simple_volume > 128).astype(np.uint8)
        dist = distance_transform(volume)
        assert_volume_valid(dist)
        assert dist.dtype in [np.float32, np.float64]
        assert np.all(dist >= 0)

    @pytest.mark.unit
    def test_distance_transform_empty(self):
        """Test distance transform on empty volume."""
        volume = np.zeros((50, 50, 50), dtype=np.uint8)
        dist = distance_transform(volume)
        assert_volume_valid(dist)
        assert np.all(dist == 0)
