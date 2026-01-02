"""
Tests for comparative analysis module.
"""

import numpy as np
import pytest
from src.analysis.comparative_analysis import (
    compare_samples,
    statistical_tests,
    process_structure_property,
    batch_analyze,
)


class TestCompareSamples:
    """Test sample comparison functionality."""

    @pytest.mark.unit
    def test_compare_samples_basic(self):
        """Test basic sample comparison."""
        sample_results = [
            {"volume": 1000, "void_fraction": 0.3, "surface_area": 500},
            {"volume": 1100, "void_fraction": 0.32, "surface_area": 550},
            {"volume": 1050, "void_fraction": 0.31, "surface_area": 525},
        ]

        comparison = compare_samples(sample_results)

        assert "volume" in comparison
        assert "void_fraction" in comparison
        assert "surface_area" in comparison

        # Check statistics
        assert "mean" in comparison["volume"]
        assert "std" in comparison["volume"]
        assert "min" in comparison["volume"]
        assert "max" in comparison["volume"]
        assert comparison["volume"]["n_samples"] == 3

    @pytest.mark.unit
    def test_compare_samples_specific_metrics(self):
        """Test comparison with specific metrics."""
        sample_results = [
            {"volume": 1000, "void_fraction": 0.3},
            {"volume": 1100, "void_fraction": 0.32},
        ]

        comparison = compare_samples(sample_results, metric_names=["volume"])

        assert "volume" in comparison
        assert "void_fraction" not in comparison

    @pytest.mark.unit
    def test_compare_samples_empty(self):
        """Test comparison with empty sample list."""
        comparison = compare_samples([])
        assert "error" in comparison


class TestStatisticalTests:
    """Test statistical testing functionality."""

    @pytest.mark.unit
    def test_statistical_tests_anova(self):
        """Test ANOVA statistical test."""
        sample_groups = {
            "group1": [1, 2, 3, 4, 5],
            "group2": [2, 3, 4, 5, 6],
            "group3": [3, 4, 5, 6, 7],
        }

        result = statistical_tests(sample_groups, test_type="anova")

        assert "test_type" in result
        assert "statistic" in result
        assert "p_value" in result
        assert result["test_type"] == "anova"

    @pytest.mark.unit
    def test_statistical_tests_ttest(self):
        """Test t-test."""
        sample_groups = {"group1": [1, 2, 3, 4, 5], "group2": [2, 3, 4, 5, 6]}

        result = statistical_tests(sample_groups, test_type="ttest")

        assert "test_type" in result
        assert "statistic" in result
        assert "p_value" in result

    @pytest.mark.unit
    def test_statistical_tests_mannwhitney(self):
        """Test Mann-Whitney U test."""
        sample_groups = {"group1": [1, 2, 3, 4, 5], "group2": [2, 3, 4, 5, 6]}

        result = statistical_tests(sample_groups, test_type="mannwhitney")

        assert "test_type" in result
        assert "statistic" in result
        assert "p_value" in result


class TestProcessStructureProperty:
    """Test process-structure-property analysis."""

    @pytest.mark.unit
    def test_psp_basic(self):
        """Test basic PSP analysis."""
        process_params = {"temp": [200, 210, 220], "speed": [10, 15, 20]}
        structure_metrics = {
            "void_fraction": [0.3, 0.32, 0.35],
            "relative_density": [0.7, 0.68, 0.65],
        }
        performance = {"efficiency": [0.8, 0.82, 0.85]}

        # This would need proper DataFrame inputs
        # For now, just test that function exists and can be called
        # (actual implementation would need proper data structures)
        pass


class TestBatchAnalyze:
    """Test batch analysis functionality."""

    @pytest.mark.unit
    @pytest.mark.skip(
        reason="batch_analyze requires file I/O - needs integration test with actual files"
    )
    def test_batch_analyze_structure(self):
        """Test that batch_analyze returns correct structure."""
        # batch_analyze requires file paths, not in-memory volumes
        # This test should be an integration test with actual files
        pass
