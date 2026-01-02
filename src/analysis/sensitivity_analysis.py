"""
Sensitivity Analysis Module for XCT Data

Analyzes sensitivity of XCT metrics to:
- Segmentation parameters (threshold, method)
- Preprocessing parameters (filters, bounds)
- Voxel size and resolution
- Analysis parameters (directions, methods)

Methods:
- Parameter sweep
- Local sensitivity (derivatives)
- Global sensitivity (Sobol indices, Morris screening)
- Uncertainty quantification
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
import logging
from itertools import product

logger = logging.getLogger(__name__)


def parameter_sweep(
    base_params: Dict[str, Any],
    param_ranges: Dict[str, np.ndarray],
    analysis_function: Callable,
    metric_name: str = "void_fraction",
) -> Dict[str, Any]:
    """
    Perform parameter sweep to analyze sensitivity.

    Args:
        base_params: Base parameter values
        param_ranges: Dictionary of parameter names and value ranges to sweep
        analysis_function: Function that takes params and returns metrics dict
        metric_name: Name of metric to track

    Returns:
        Dictionary with sweep results
    """
    results = []
    param_names = list(param_ranges.keys())

    # Generate all combinations
    param_combinations = list(product(*param_ranges.values()))

    for i, param_values in enumerate(param_combinations):
        # Create parameter dict
        params = base_params.copy()
        for j, param_name in enumerate(param_names):
            params[param_name] = param_values[j]

        try:
            # Run analysis
            metrics = analysis_function(params)
            metric_value = metrics.get(metric_name, np.nan)

            result = {"run_id": i, metric_name: metric_value}
            for param_name, param_value in zip(param_names, param_values):
                result[param_name] = param_value

            results.append(result)
        except Exception as e:
            logger.warning(f"Analysis failed for combination {i}: {e}")
            continue

    df = pd.DataFrame(results)

    # Calculate sensitivity metrics
    sensitivity = {}
    for param_name in param_names:
        if len(df) > 0:
            correlation = df[param_name].corr(df[metric_name])
            sensitivity[param_name] = {
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "range_effect": (
                    float(df[metric_name].max() - df[metric_name].min())
                    if len(df) > 0
                    else 0.0
                ),
                "std_effect": (
                    float(df.groupby(param_name)[metric_name].std().mean())
                    if len(df) > 0
                    else 0.0
                ),
            }

    return {
        "results": df,
        "sensitivity": sensitivity,
        "param_names": param_names,
        "metric_name": metric_name,
        "n_runs": len(results),
    }


def local_sensitivity(
    base_params: Dict[str, Any],
    param_name: str,
    analysis_function: Callable,
    metric_name: str = "void_fraction",
    perturbation: float = 0.01,
) -> Dict[str, Any]:
    """
    Calculate local sensitivity (derivative) of metric to parameter.

    Args:
        base_params: Base parameter values
        param_name: Parameter to perturb
        perturbation: Relative perturbation (e.g., 0.01 = 1%)
        analysis_function: Function that takes params and returns metrics dict
        metric_name: Name of metric to track

    Returns:
        Dictionary with sensitivity results
    """
    if param_name not in base_params:
        raise ValueError(f"Parameter {param_name} not in base_params")

    base_value = base_params[param_name]

    # Forward difference
    params_forward = base_params.copy()
    if isinstance(base_value, (int, float)):
        params_forward[param_name] = base_value * (1 + perturbation)
    else:
        params_forward[param_name] = base_value

    # Backward difference
    params_backward = base_params.copy()
    if isinstance(base_value, (int, float)):
        params_backward[param_name] = base_value * (1 - perturbation)
    else:
        params_backward[param_name] = base_value

    # Central difference
    try:
        metric_base = analysis_function(base_params).get(metric_name, np.nan)
        metric_forward = analysis_function(params_forward).get(metric_name, np.nan)
        metric_backward = analysis_function(params_backward).get(metric_name, np.nan)

        # Calculate derivative
        if isinstance(base_value, (int, float)) and base_value != 0:
            delta_param = base_value * 2 * perturbation
            derivative = (metric_forward - metric_backward) / delta_param
            relative_sensitivity = (
                derivative * base_value / metric_base if metric_base != 0 else 0.0
            )
        else:
            derivative = (metric_forward - metric_backward) / (2 * perturbation)
            relative_sensitivity = derivative / metric_base if metric_base != 0 else 0.0

        return {
            "parameter": param_name,
            "base_value": base_value,
            "base_metric": float(metric_base),
            "metric_value": float(metric_base),  # Alias for compatibility
            "sensitivity": float(derivative),  # Alias for compatibility
            "forward_metric": float(metric_forward),
            "backward_metric": float(metric_backward),
            "derivative": float(derivative),
            "relative_sensitivity": float(relative_sensitivity),
            "perturbation": perturbation,
        }
    except Exception as e:
        logger.error(f"Error in local sensitivity: {e}")
        return {"parameter": param_name, "error": str(e)}


def morris_screening(
    param_bounds: Dict[str, Tuple[float, float]],
    analysis_function: Callable,
    n_trajectories: int = 10,
    metric_name: str = "void_fraction",
) -> Dict[str, Any]:
    """
    Morris screening method for global sensitivity analysis.

    Args:
        param_bounds: Dictionary of parameter names and (min, max) bounds
        n_trajectories: Number of trajectories
        analysis_function: Function that takes params and returns metrics dict
        metric_name: Name of metric to track

    Returns:
        Dictionary with Morris indices (μ*, σ)
    """
    param_names = list(param_bounds.keys())
    n_params = len(param_names)

    # Generate trajectories
    trajectories = []
    elementary_effects = {name: [] for name in param_names}

    for traj in range(n_trajectories):
        # Random starting point
        start_point = {}
        for name, (min_val, max_val) in param_bounds.items():
            start_point[name] = np.random.uniform(min_val, max_val)

        # Random order of parameters
        param_order = np.random.permutation(param_names)

        # Calculate step size
        delta = 1.0 / (n_params + 1)

        # Current point
        current_point = start_point.copy()
        current_metric = analysis_function(current_point).get(metric_name, np.nan)

        for param_name in param_order:
            # Move to next point
            min_val, max_val = param_bounds[param_name]
            step = delta * (max_val - min_val)

            next_point = current_point.copy()
            if next_point[param_name] + step <= max_val:
                next_point[param_name] += step
            else:
                next_point[param_name] -= step

            next_metric = analysis_function(next_point).get(metric_name, np.nan)

            # Calculate elementary effect
            if not np.isnan(current_metric) and not np.isnan(next_metric):
                ee = (next_metric - current_metric) / step
                elementary_effects[param_name].append(ee)

            current_point = next_point
            current_metric = next_metric

    # Calculate Morris indices
    morris_indices = {}
    for param_name in param_names:
        ees = np.array(elementary_effects[param_name])
        if len(ees) > 0:
            mu_star = np.mean(np.abs(ees))
            sigma = np.std(ees)
            mu = np.mean(ees)

            morris_indices[param_name] = {
                "mu_star": float(mu_star),  # Mean absolute elementary effect
                "sigma": float(sigma),  # Std of elementary effects
                "mu": float(mu),  # Mean elementary effect
                "n_samples": len(ees),
            }
        else:
            morris_indices[param_name] = {
                "mu_star": 0.0,
                "sigma": 0.0,
                "mu": 0.0,
                "n_samples": 0,
            }

    # Flatten mu_star and sigma for easier access
    mu_star = {name: idx["mu_star"] for name, idx in morris_indices.items()}
    sigma = {name: idx["sigma"] for name, idx in morris_indices.items()}

    return {
        "morris_indices": morris_indices,
        "mu_star": mu_star,  # Top-level for compatibility
        "sigma": sigma,  # Top-level for compatibility
        "n_trajectories": n_trajectories,
        "n_params": n_params,
        "metric_name": metric_name,
    }


def sobol_indices(
    param_bounds: Dict[str, Tuple[float, float]],
    analysis_function: Callable,
    n_samples: int = 1000,
    metric_name: str = "void_fraction",
) -> Dict[str, Any]:
    """
    Calculate Sobol sensitivity indices (first-order and total).

    Args:
        param_bounds: Dictionary of parameter names and (min, max) bounds
        n_samples: Number of samples for Monte Carlo
        analysis_function: Function that takes params and returns metrics dict
        metric_name: Name of metric to track

    Returns:
        Dictionary with Sobol indices
    """
    param_names = list(param_bounds.keys())
    n_params = len(param_names)

    # Generate samples using Saltelli's method
    # A matrix: base samples
    A = np.random.uniform(0, 1, (n_samples, n_params))

    # B matrix: resampled
    B = np.random.uniform(0, 1, (n_samples, n_params))

    # Convert to parameter space
    def to_param_space(samples):
        params_list = []
        for sample in samples:
            params = {}
            for i, param_name in enumerate(param_names):
                min_val, max_val = param_bounds[param_name]
                params[param_name] = min_val + sample[i] * (max_val - min_val)
            params_list.append(params)
        return params_list

    A_params = to_param_space(A)
    B_params = to_param_space(B)

    # Evaluate function
    f_A = np.array([analysis_function(p).get(metric_name, np.nan) for p in A_params])
    f_B = np.array([analysis_function(p).get(metric_name, np.nan) for p in B_params])

    # Remove NaN values
    valid = np.isfinite(f_A) & np.isfinite(f_B)
    f_A = f_A[valid]
    f_B = f_B[valid]
    n_valid = len(f_A)

    if n_valid < 10:
        logger.warning("Too few valid samples for Sobol indices")
        return {
            "first_order": {name: 0.0 for name in param_names},
            "total_order": {name: 0.0 for name in param_names},
            "n_valid_samples": n_valid,
        }

    # Calculate variance
    f0 = np.mean(f_A)
    V = np.var(f_A)

    if V == 0:
        logger.warning("Zero variance in output")
        return {
            "first_order": {name: 0.0 for name in param_names},
            "total_order": {name: 0.0 for name in param_names},
            "n_valid_samples": n_valid,
        }

    # First-order indices
    first_order = {}
    total_order = {}

    for i, param_name in enumerate(param_names):
        # Create C_i: A with B values for parameter i
        C_i = A.copy()
        C_i[:, i] = B[:, i]
        C_i_params = to_param_space(C_i)
        f_Ci = np.array(
            [analysis_function(p).get(metric_name, np.nan) for p in C_i_params]
        )
        f_Ci = f_Ci[valid]

        # First-order: S_i = V[E(Y|X_i)] / V(Y)
        # Approximated as: E[Y_A * Y_Ci] - f0^2 / V
        if len(f_Ci) == len(f_A):
            S_i = (np.mean(f_A * f_Ci) - f0**2) / V
            first_order[param_name] = float(max(0, min(1, S_i)))  # Clamp to [0, 1]
        else:
            first_order[param_name] = 0.0

        # Total-order: S_Ti = 1 - V[E(Y|X_~i)] / V(Y)
        # Approximated as: 1 - E[Y_B * Y_Ci] - f0^2 / V
        if len(f_Ci) == len(f_B):
            S_Ti = 1 - (np.mean(f_B * f_Ci) - f0**2) / V
            total_order[param_name] = float(max(0, min(1, S_Ti)))  # Clamp to [0, 1]
        else:
            total_order[param_name] = 0.0

    return {
        "first_order": first_order,
        "total_order": total_order,
        "variance": float(V),
        "mean": float(f0),
        "n_valid_samples": n_valid,
        "n_samples": n_samples,
    }


def uncertainty_propagation(
    param_distributions: Dict[str, Callable],
    analysis_function: Callable,
    n_samples: int = 1000,
    metric_name: str = "void_fraction",
) -> Dict[str, Any]:
    """
    Propagate uncertainty through analysis to quantify output uncertainty.

    Args:
        param_distributions: Dictionary of parameter names and distribution functions
        n_samples: Number of Monte Carlo samples
        analysis_function: Function that takes params and returns metrics dict
        metric_name: Name of metric to track

    Returns:
        Dictionary with uncertainty statistics
    """
    param_names = list(param_distributions.keys())

    # Helper function to sample from distribution
    def sample_from_dist(dist_spec):
        """Sample from distribution specification (dict or callable)."""
        if callable(dist_spec):
            return dist_spec()
        elif isinstance(dist_spec, dict):
            dist_type = dist_spec.get("type", "normal")
            if dist_type == "normal":
                mean = dist_spec.get("mean", 0.0)
                std = dist_spec.get("std", 1.0)
                return np.random.normal(mean, std)
            elif dist_type == "uniform":
                low = dist_spec.get("low", 0.0)
                high = dist_spec.get("high", 1.0)
                return np.random.uniform(low, high)
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        else:
            raise ValueError("Distribution must be callable or dict")

    # Sample parameters
    samples = []
    for _ in range(n_samples):
        params = {}
        for param_name, dist_spec in param_distributions.items():
            params[param_name] = sample_from_dist(dist_spec)
        samples.append(params)

    # Evaluate function
    metric_values = []
    for params in samples:
        try:
            metric = analysis_function(params).get(metric_name, np.nan)
            if np.isfinite(metric):
                metric_values.append(metric)
        except Exception:
            continue

    metric_values = np.array(metric_values)

    if len(metric_values) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_95": (np.nan, np.nan), "n_valid": 0}

    # Calculate statistics
    mean = np.mean(metric_values)
    std = np.std(metric_values)
    ci_95 = np.percentile(metric_values, [2.5, 97.5])

    return {
        "mean": float(mean),
        "std": float(std),
        "ci_95": (float(ci_95[0]), float(ci_95[1])),
        "confidence_interval": (
            float(ci_95[0]),
            float(ci_95[1]),
        ),  # Alias for compatibility
        "min": float(np.min(metric_values)),
        "max": float(np.max(metric_values)),
        "median": float(np.median(metric_values)),
        "n_valid": len(metric_values),
        "n_samples": n_samples,
    }


def analyze_segmentation_sensitivity(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    base_threshold: float = 0.5,
    threshold_range: np.ndarray = None,
    metric_name: str = "void_fraction",
) -> Dict[str, Any]:
    """
    Analyze sensitivity of metrics to segmentation threshold.

    Args:
        volume: Input volume
        voxel_size: Voxel spacing
        base_threshold: Base threshold value
        threshold_range: Range of thresholds to test
        metric_name: Metric to track

    Returns:
        Sensitivity analysis results
    """
    from ..core.segmentation import otsu_threshold
    from ..core.metrics import compute_all_metrics

    if threshold_range is None:
        threshold_range = np.linspace(0.1, 0.9, 20)

    results = []
    for threshold in threshold_range:
        try:
            # Segment
            segmented = (volume > threshold).astype(np.uint8)

            # Compute metrics
            metrics = compute_all_metrics(segmented, voxel_size)

            results.append(
                {
                    "threshold": float(threshold),
                    metric_name: metrics.get(metric_name, np.nan),
                }
            )
        except Exception as e:
            logger.warning(f"Failed for threshold {threshold}: {e}")
            continue

    df = pd.DataFrame(results)

    # Calculate sensitivity
    if len(df) > 1:
        correlation = df["threshold"].corr(df[metric_name])
        sensitivity_coef = np.polyfit(df["threshold"], df[metric_name], 1)[0]
    else:
        correlation = 0.0
        sensitivity_coef = 0.0

    return {
        "results": df,
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "sensitivity_coefficient": float(sensitivity_coef),
        "threshold_range": threshold_range.tolist(),
        "metric_name": metric_name,
    }
