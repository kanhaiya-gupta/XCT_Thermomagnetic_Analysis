"""
Tests for sensitivity analysis module.
"""

import numpy as np
import pytest
from src.analysis.sensitivity_analysis import (
    parameter_sweep,
    local_sensitivity,
    morris_screening,
    sobol_indices,
    uncertainty_propagation,
)


class TestParameterSweep:
    """Test parameter sweep functionality."""

    @pytest.mark.unit
    def test_parameter_sweep_basic(self):
        """Test basic parameter sweep."""

        def analysis_function(params):
            # Simple test function: y = a*x + b
            return {"result": params["a"] * params["x"] + params["b"]}

        base_params = {"a": 1.0, "b": 0.0}
        param_ranges = {"x": np.array([1, 2, 3]), "a": np.array([1.0, 2.0])}

        result = parameter_sweep(
            base_params, param_ranges, analysis_function, metric_name="result"
        )

        assert "results" in result
        assert "sensitivity" in result
        assert len(result["results"]) > 0
        assert "x" in result["sensitivity"]
        assert "a" in result["sensitivity"]

    @pytest.mark.unit
    def test_parameter_sweep_single_param(self):
        """Test parameter sweep with single parameter."""

        def analysis_function(params):
            return {"value": params["x"] ** 2}

        base_params = {}
        param_ranges = {"x": np.array([1, 2, 3, 4, 5])}

        result = parameter_sweep(
            base_params, param_ranges, analysis_function, metric_name="value"
        )

        assert len(result["results"]) == 5
        assert "x" in result["sensitivity"]


class TestLocalSensitivity:
    """Test local sensitivity analysis."""

    @pytest.mark.unit
    def test_local_sensitivity_basic(self):
        """Test basic local sensitivity calculation."""

        def analysis_function(params):
            # y = x^2
            return {"y": params["x"] ** 2}

        base_params = {"x": 2.0}
        sensitivity = local_sensitivity(
            base_params, "x", analysis_function, metric_name="y"
        )

        assert "sensitivity" in sensitivity
        assert "metric_value" in sensitivity
        # For y = x^2, derivative at x=2 is 4
        assert abs(sensitivity["sensitivity"] - 4.0) < 0.5


class TestMorrisScreening:
    """Test Morris screening method."""

    @pytest.mark.unit
    def test_morris_screening_basic(self):
        """Test basic Morris screening."""

        def analysis_function(params):
            # Simple function with known sensitivities
            return {"result": params["x1"] + 2 * params["x2"]}

        param_bounds = {"x1": (0, 10), "x2": (0, 10)}

        result = morris_screening(
            param_bounds, analysis_function, n_trajectories=10, metric_name="result"
        )

        assert "mu_star" in result
        assert "sigma" in result
        assert "x1" in result["mu_star"]
        assert "x2" in result["mu_star"]
        # x2 should have higher sensitivity (coefficient 2 vs 1)
        assert result["mu_star"]["x2"] > result["mu_star"]["x1"]


class TestSobolIndices:
    """Test Sobol indices calculation."""

    @pytest.mark.unit
    def test_sobol_indices_basic(self):
        """Test basic Sobol indices calculation."""

        def analysis_function(params):
            # Simple additive function
            return {"result": params["x1"] + params["x2"]}

        param_bounds = {"x1": (0, 1), "x2": (0, 1)}

        result = sobol_indices(
            param_bounds, analysis_function, n_samples=100, metric_name="result"
        )

        assert "first_order" in result
        assert "total_order" in result
        assert "x1" in result["first_order"]
        assert "x2" in result["first_order"]


class TestUncertaintyPropagation:
    """Test uncertainty propagation."""

    @pytest.mark.unit
    def test_uncertainty_propagation_basic(self):
        """Test basic uncertainty propagation."""

        def analysis_function(params):
            return {"result": params["x"] ** 2}

        param_distributions = {"x": {"type": "normal", "mean": 2.0, "std": 0.1}}

        result = uncertainty_propagation(
            param_distributions, analysis_function, n_samples=100, metric_name="result"
        )

        assert "mean" in result
        assert "std" in result
        assert "confidence_interval" in result
        assert result["mean"] > 0
