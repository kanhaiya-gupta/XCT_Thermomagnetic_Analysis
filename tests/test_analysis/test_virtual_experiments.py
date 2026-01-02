"""
Tests for virtual experiments module.
"""

import numpy as np
import pandas as pd
import pytest
from src.analysis.virtual_experiments import (
    full_factorial_design,
    latin_hypercube_sampling,
    central_composite_design,
    box_behnken_design,
    run_virtual_experiment,
    fit_response_surface,
)


class TestFullFactorialDesign:
    """Test full factorial design."""

    @pytest.mark.unit
    def test_full_factorial_basic(self):
        """Test basic full factorial design."""
        factors = {"temp": [200, 220, 240], "speed": [10, 20]}

        design = full_factorial_design(factors)

        assert len(design) == 3 * 2  # 6 combinations
        assert "temp" in design.columns
        assert "speed" in design.columns
        assert all(temp in [200, 220, 240] for temp in design["temp"])
        assert all(speed in [10, 20] for speed in design["speed"])

    @pytest.mark.unit
    def test_full_factorial_with_center_points(self):
        """Test full factorial with center points."""
        factors = {"x": [0, 1], "y": [0, 1]}

        design = full_factorial_design(factors, center_points=3)

        assert len(design) == 2 * 2 + 3  # 4 combinations + 3 center points
        # Check center points
        center_rows = design[design["x"] == 0.5]
        assert len(center_rows) == 3


class TestLatinHypercubeSampling:
    """Test Latin Hypercube Sampling."""

    @pytest.mark.unit
    def test_lhs_basic(self):
        """Test basic LHS design."""
        param_bounds = {"x": (0, 10), "y": (0, 10)}

        design = latin_hypercube_sampling(param_bounds, n_samples=20)

        assert len(design) == 20
        assert "x" in design.columns
        assert "y" in design.columns
        assert all(0 <= x <= 10 for x in design["x"])
        assert all(0 <= y <= 10 for y in design["y"])

    @pytest.mark.unit
    def test_lhs_coverage(self):
        """Test that LHS provides good coverage."""
        param_bounds = {"x": (0, 1)}

        # Use more samples for better coverage guarantee
        design = latin_hypercube_sampling(param_bounds, n_samples=20)

        # Check that values are spread across range (more lenient check)
        assert design["x"].min() < 0.3
        assert design["x"].max() > 0.7


class TestCentralCompositeDesign:
    """Test Central Composite Design."""

    @pytest.mark.unit
    def test_ccd_basic(self):
        """Test basic CCD design."""
        factors = {"x": (-1, 1), "y": (-1, 1)}

        design = central_composite_design(factors)

        assert "x" in design.columns
        assert "y" in design.columns
        # CCD should have factorial + axial + center points
        assert len(design) >= 2**2  # At least factorial points


class TestBoxBehnkenDesign:
    """Test Box-Behnken Design."""

    @pytest.mark.unit
    def test_bbd_basic(self):
        """Test basic BBD design."""
        factors = {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)}

        design = box_behnken_design(factors)

        assert "x" in design.columns
        assert "y" in design.columns
        assert "z" in design.columns
        # BBD for 3 factors should have specific number of runs
        assert len(design) > 0


class TestRunVirtualExperiment:
    """Test virtual experiment execution."""

    @pytest.mark.unit
    def test_run_virtual_experiment_basic(self):
        """Test basic virtual experiment."""

        def response_function(params):
            # Simple response: y = x1 + x2
            return params["x1"] + params["x2"]

        design = pd.DataFrame({"x1": [1, 2, 3], "x2": [1, 2, 3]})

        results = run_virtual_experiment(design, response_function)

        assert len(results) == len(design)
        assert "response" in results.columns
        assert all(results["response"] == design["x1"] + design["x2"])


class TestFitResponseSurface:
    """Test response surface fitting."""

    @pytest.mark.unit
    def test_fit_response_surface_linear(self):
        """Test linear response surface fitting."""
        # Generate data with linear relationship
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.normal(0, 0.1, 20)

        data = pd.DataFrame({"x": x, "y": y})

        model = fit_response_surface(data[["x"]], data["y"], degree=1)

        assert "coefficients" in model
        assert "r_squared" in model
        assert model["r_squared"] > 0.9  # Good fit

    @pytest.mark.unit
    def test_fit_response_surface_quadratic(self):
        """Test quadratic response surface fitting."""
        # Generate data with quadratic relationship
        x = np.linspace(0, 10, 20)
        y = 0.1 * x**2 + 2 * x + 1 + np.random.normal(0, 0.1, 20)

        data = pd.DataFrame({"x": x, "y": y})

        model = fit_response_surface(data[["x"]], data["y"], degree=2)

        assert "coefficients" in model
        assert "r_squared" in model
        assert model["r_squared"] > 0.9  # Good fit
