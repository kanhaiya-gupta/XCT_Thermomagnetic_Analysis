"""
Tests for segmentation module.
"""

import numpy as np
import pytest
from src.core.segmentation import (
    otsu_threshold,
    multi_threshold_segmentation,
    adaptive_threshold,
    segment_volume,
    refine_segmentation,
)
from tests.test_utils import assert_volume_valid


class TestOtsuThreshold:
    """Test Otsu thresholding functionality."""

    @pytest.mark.unit
    def test_otsu_basic(self, simple_volume):
        """Test basic Otsu thresholding."""
        segmented = otsu_threshold(simple_volume)
        assert_volume_valid(segmented)
        assert segmented.dtype == np.uint8
        assert np.all((segmented == 0) | (segmented == 1))

    @pytest.mark.unit
    def test_otsu_return_threshold(self, simple_volume):
        """Test Otsu thresholding with threshold return."""
        segmented, threshold = otsu_threshold(simple_volume, return_threshold=True)
        assert_volume_valid(segmented)
        assert isinstance(threshold, (int, float))
        assert 0 <= threshold <= 255

    @pytest.mark.unit
    def test_otsu_porous_volume(self, porous_volume):
        """Test Otsu on porous volume."""
        segmented = otsu_threshold(porous_volume)
        assert_volume_valid(segmented)
        # Should have both material and void
        assert segmented.min() == 0
        assert segmented.max() == 1

    @pytest.mark.unit
    def test_otsu_edge_cases(self, edge_case_volumes):
        """Test Otsu on edge case volumes."""
        # Empty volume
        segmented = otsu_threshold(edge_case_volumes["empty"])
        assert_volume_valid(segmented)
        assert np.all(segmented == 0)

        # Full volume
        segmented = otsu_threshold(edge_case_volumes["full"])
        assert_volume_valid(segmented)
        assert np.all(segmented == 1)

    @pytest.mark.unit
    def test_otsu_different_dtypes(self):
        """Test Otsu with different data types."""
        volume_uint8 = np.random.randint(0, 255, (50, 50, 50), dtype=np.uint8)
        volume_float32 = volume_uint8.astype(np.float32)

        seg1 = otsu_threshold(volume_uint8)
        seg2 = otsu_threshold(volume_float32)

        assert_volume_valid(seg1)
        assert_volume_valid(seg2)
        # Results should be similar (allowing for threshold differences)
        assert seg1.shape == seg2.shape


class TestMultiThreshold:
    """Test multi-threshold segmentation."""

    @pytest.mark.unit
    def test_multi_threshold_basic(self, simple_volume):
        """Test basic multi-threshold segmentation."""
        segmented = multi_threshold_segmentation(simple_volume, n_classes=3)
        assert_volume_valid(segmented)
        assert segmented.max() <= 2  # 0, 1, 2

    @pytest.mark.unit
    def test_multi_threshold_n_classes(self, simple_volume):
        """Test different number of classes."""
        for n_classes in [2, 3, 4, 5]:
            segmented = multi_threshold_segmentation(simple_volume, n_classes=n_classes)
            assert_volume_valid(segmented)
            assert segmented.max() <= n_classes - 1


class TestAdaptiveThreshold:
    """Test adaptive thresholding."""

    @pytest.mark.unit
    def test_adaptive_threshold_gaussian(self, simple_volume):
        """Test adaptive threshold with Gaussian method."""
        segmented = adaptive_threshold(simple_volume, block_size=15, method="gaussian")
        assert_volume_valid(segmented)
        assert segmented.dtype == np.uint8

    @pytest.mark.unit
    def test_adaptive_threshold_mean(self, simple_volume):
        """Test adaptive threshold with mean method."""
        segmented = adaptive_threshold(simple_volume, block_size=15, method="mean")
        assert_volume_valid(segmented)
        assert segmented.dtype == np.uint8


class TestSegmentVolume:
    """Test main segmentation interface."""

    @pytest.mark.unit
    def test_segment_volume_otsu(self, simple_volume):
        """Test segment_volume with Otsu method."""
        segmented = segment_volume(simple_volume, method="otsu")
        assert_volume_valid(segmented)

    @pytest.mark.unit
    def test_segment_volume_multi(self, simple_volume):
        """Test segment_volume with multi-threshold method."""
        segmented = segment_volume(simple_volume, method="multi", n_classes=3)
        assert_volume_valid(segmented)

    @pytest.mark.unit
    def test_segment_volume_adaptive(self, simple_volume):
        """Test segment_volume with adaptive method."""
        segmented = segment_volume(simple_volume, method="adaptive", block_size=15)
        assert_volume_valid(segmented)

    @pytest.mark.unit
    def test_segment_volume_manual(self, simple_volume):
        """Test segment_volume with manual threshold."""
        threshold = 128
        segmented = segment_volume(simple_volume, method="manual", threshold=threshold)
        assert_volume_valid(segmented)

    @pytest.mark.unit
    def test_segment_volume_invalid_method(self, simple_volume):
        """Test segment_volume with invalid method."""
        with pytest.raises(ValueError):
            segment_volume(simple_volume, method="invalid")

    @pytest.mark.unit
    def test_segment_volume_manual_no_threshold(self, simple_volume):
        """Test segment_volume with manual method but no threshold."""
        with pytest.raises(ValueError):
            segment_volume(simple_volume, method="manual")


class TestRefineSegmentation:
    """Test segmentation refinement."""

    @pytest.mark.unit
    def test_refine_segmentation_basic(self, simple_volume):
        """Test basic segmentation refinement."""
        segmented = otsu_threshold(simple_volume)
        refined = refine_segmentation(
            segmented, remove_small_objects=True, fill_holes=True
        )
        assert_volume_valid(refined)
        assert refined.shape == segmented.shape

    @pytest.mark.unit
    def test_refine_segmentation_no_removal(self, simple_volume):
        """Test refinement without small object removal."""
        segmented = otsu_threshold(simple_volume)
        refined = refine_segmentation(
            segmented, remove_small_objects=False, fill_holes=True
        )
        assert_volume_valid(refined)

    @pytest.mark.unit
    def test_refine_segmentation_no_fill(self, simple_volume):
        """Test refinement without hole filling."""
        segmented = otsu_threshold(simple_volume)
        refined = refine_segmentation(
            segmented, remove_small_objects=True, fill_holes=False
        )
        assert_volume_valid(refined)
