"""
Tests for statistics module.
"""

import numpy as np
import pytest
from src.preprocessing.statistics import (
    fit_gaussian,
    fit_poisson,
    fit_linear,
    fit_quadratic,
    fit_distribution,
    compare_fits,
    evaluate_fit_quality,
)


class TestFitGaussian:
    """Test Gaussian distribution fitting."""

    @pytest.mark.unit
    def test_fit_gaussian_basic(self, sample_distributions):
        """Test basic Gaussian fitting."""
        data = sample_distributions["gaussian"]
        result = fit_gaussian(data)

        assert "params" in result
        assert "mean" in result["params"]
        assert "std" in result["params"]
        assert "r_squared" in result

    @pytest.mark.unit
    def test_fit_gaussian_known_params(self):
        """Test Gaussian fitting with known parameters."""
        # Generate data with known mean and std
        np.random.seed(42)
        true_mean = 100
        true_std = 10
        data = np.random.normal(true_mean, true_std, 1000)

        result = fit_gaussian(data)
        fitted_mean = result["params"]["mean"]
        fitted_std = result["params"]["std"]

        # Should be close to true values
        assert abs(fitted_mean - true_mean) < 1.0
        assert abs(fitted_std - true_std) < 1.0


class TestFitPoisson:
    """Test Poisson distribution fitting."""

    @pytest.mark.unit
    def test_fit_poisson_basic(self, sample_distributions):
        """Test basic Poisson fitting."""
        data = sample_distributions["poisson"]
        result = fit_poisson(data)

        assert "params" in result
        assert "lambda" in result["params"]
        assert "r_squared" in result


class TestFitLinear:
    """Test linear regression."""

    @pytest.mark.unit
    def test_fit_linear_basic(self):
        """Test basic linear fitting."""
        x = np.linspace(0, 100, 100)
        y = 2 * x + 5 + np.random.normal(0, 1, 100)

        result = fit_linear(x, y)

        assert "params" in result
        assert "slope" in result["params"]
        assert "intercept" in result["params"]
        assert "r_squared" in result

        # Check that slope is close to 2
        assert abs(result["params"]["slope"] - 2) < 0.5


class TestFitQuadratic:
    """Test quadratic regression."""

    @pytest.mark.unit
    def test_fit_quadratic_basic(self):
        """Test basic quadratic fitting."""
        x = np.linspace(0, 100, 100)
        y = 0.1 * x**2 + 2 * x + 5 + np.random.normal(0, 1, 100)

        result = fit_quadratic(x, y)

        assert "params" in result
        assert "a" in result["params"]
        assert "b" in result["params"]
        assert "c" in result["params"]
        assert "r_squared" in result


class TestEvaluateFitQuality:
    """Test fit quality evaluation."""

    @pytest.mark.unit
    def test_evaluate_fit_quality(self):
        """Test fit quality evaluation."""
        x = np.linspace(0, 100, 100)
        y = 2 * x + 5 + np.random.normal(0, 0.1, 100)  # Low noise

        result = fit_linear(x, y)
        quality = evaluate_fit_quality(x, y, result)

        assert "r_squared" in quality
        assert "rmse" in quality
        assert "mae" in quality
        assert quality["r_squared"] > 0.9  # Good fit
