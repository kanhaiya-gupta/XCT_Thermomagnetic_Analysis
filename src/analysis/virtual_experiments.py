"""
Virtual Experiments Module for XCT Analysis

Enables virtual experiments similar to PBF process optimization:
- Design of Experiments (DoE)
- Process parameter optimization
- What-if scenario analysis
- Process-structure-property relationships
- Multi-objective optimization

Design Types:
- Full factorial
- Central composite
- Latin Hypercube Sampling
- Box-Behnken
- D-optimal
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from itertools import product
import logging

logger = logging.getLogger(__name__)


def full_factorial_design(
    factors: Dict[str, List[float]], center_points: int = 0
) -> pd.DataFrame:
    """
    Generate full factorial design.

    Args:
        factors: Dictionary of factor names and levels
        center_points: Number of center point replicates

    Returns:
        DataFrame with experimental design
    """
    factor_names = list(factors.keys())
    factor_levels = [factors[name] for name in factor_names]

    # Generate all combinations
    combinations = list(product(*factor_levels))

    # Create design matrix
    design = []
    for combo in combinations:
        design.append(dict(zip(factor_names, combo)))

    # Add center points
    if center_points > 0:
        center = {name: np.mean(levels) for name, levels in factors.items()}
        for _ in range(center_points):
            design.append(center)

    return pd.DataFrame(design)


def latin_hypercube_sampling(
    param_bounds: Dict[str, Tuple[float, float]], n_samples: int = 100
) -> pd.DataFrame:
    """
    Generate Latin Hypercube Sampling (LHS) design.

    Args:
        param_bounds: Dictionary of parameter names and (min, max) bounds
        n_samples: Number of samples

    Returns:
        DataFrame with LHS design
    """
    param_names = list(param_bounds.keys())
    n_params = len(param_names)

    # Generate LHS
    lhs = np.random.uniform(0, 1, (n_samples, n_params))

    # Permute each column
    for i in range(n_params):
        lhs[:, i] = np.random.permutation(lhs[:, i])

    # Scale to parameter bounds
    design = []
    for sample in lhs:
        params = {}
        for i, param_name in enumerate(param_names):
            min_val, max_val = param_bounds[param_name]
            params[param_name] = min_val + sample[i] * (max_val - min_val)
        design.append(params)

    return pd.DataFrame(design)


def central_composite_design(
    factors: Dict[str, Tuple[float, float]],
    alpha: str = "orthogonal",
    center_points: int = 3,
) -> pd.DataFrame:
    """
    Generate Central Composite Design (CCD).

    Args:
        factors: Dictionary of factor names and (min, max) bounds
        alpha: Type of alpha ('orthogonal', 'rotatable', or 'face-centered')
        center_points: Number of center point replicates

    Returns:
        DataFrame with CCD design
    """
    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    # Calculate alpha
    if alpha == "orthogonal":
        alpha_val = 2 ** (n_factors / 4)
    elif alpha == "rotatable":
        alpha_val = 2 ** (n_factors / 4)
    elif alpha == "face-centered":
        alpha_val = 1.0
    else:
        alpha_val = 1.0

    design = []

    # Factorial points (2^n)
    factorial_levels = [-1, 1]
    for combo in product(*[factorial_levels] * n_factors):
        point = {}
        for i, factor_name in enumerate(factor_names):
            min_val, max_val = factors[factor_name]
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            point[factor_name] = center + combo[i] * half_range
        design.append(point)

    # Axial points
    for i, factor_name in enumerate(factor_names):
        min_val, max_val = factors[factor_name]
        center = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2

        # +alpha
        point_plus = {
            name: (factors[name][0] + factors[name][1]) / 2 for name in factor_names
        }
        point_plus[factor_name] = center + alpha_val * half_range
        design.append(point_plus)

        # -alpha
        point_minus = {
            name: (factors[name][0] + factors[name][1]) / 2 for name in factor_names
        }
        point_minus[factor_name] = center - alpha_val * half_range
        design.append(point_minus)

    # Center points
    center = {name: (factors[name][0] + factors[name][1]) / 2 for name in factor_names}
    for _ in range(center_points):
        design.append(center.copy())

    return pd.DataFrame(design)


def box_behnken_design(
    factors: Dict[str, Tuple[float, float]], center_points: int = 3
) -> pd.DataFrame:
    """
    Generate Box-Behnken Design (BBD).

    Args:
        factors: Dictionary of factor names and (min, max) bounds
        center_points: Number of center point replicates

    Returns:
        DataFrame with BBD design
    """
    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    if n_factors < 3:
        logger.warning("Box-Behnken requires at least 3 factors")
        return pd.DataFrame()

    # BBD for 3 factors (base case)
    # For more factors, use combinations
    design = []

    # Generate BBD points
    levels = [-1, 0, 1]

    # For 3 factors, BBD has specific structure
    if n_factors == 3:
        # All combinations where exactly one factor is 0
        for i in range(n_factors):
            for combo in product(
                *[[-1, 1] if j != i else [0] for j in range(n_factors)]
            ):
                point = {}
                for k, factor_name in enumerate(factor_names):
                    min_val, max_val = factors[factor_name]
                    center = (min_val + max_val) / 2
                    half_range = (max_val - min_val) / 2
                    point[factor_name] = center + combo[k] * half_range
                design.append(point)
    else:
        # Simplified BBD for more factors
        # Use combinations of factors
        for combo in product(*[levels] * n_factors):
            # Count zeros
            n_zeros = sum(1 for x in combo if x == 0)
            # BBD: exactly one or two factors at 0
            if 1 <= n_zeros <= 2:
                point = {}
                for k, factor_name in enumerate(factor_names):
                    min_val, max_val = factors[factor_name]
                    center = (min_val + max_val) / 2
                    half_range = (max_val - min_val) / 2
                    point[factor_name] = center + combo[k] * half_range
                design.append(point)

    # Center points
    center = {name: (factors[name][0] + factors[name][1]) / 2 for name in factor_names}
    for _ in range(center_points):
        design.append(center.copy())

    return pd.DataFrame(design)


def run_virtual_experiment(
    design: pd.DataFrame, process_simulator: Callable, response_names: List[str] = None
) -> pd.DataFrame:
    """
    Run virtual experiments using design matrix.

    Args:
        design: Experimental design (DataFrame)
        process_simulator: Function that takes parameter dict and returns response dict
        response_names: Names of responses to track

    Returns:
        DataFrame with design and responses
    """
    results = []

    for idx, row in design.iterrows():
        params = row.to_dict()

        try:
            responses = process_simulator(params)

            # Handle case where response is a scalar
            if not isinstance(responses, dict):
                if response_names and len(response_names) == 1:
                    responses = {response_names[0]: responses}
                elif response_names:
                    # Multiple response names but scalar response - use first name
                    responses = {response_names[0]: responses}
                else:
                    responses = {"response": responses}

            result = params.copy()
            if response_names:
                for name in response_names:
                    result[name] = responses.get(name, np.nan)
            else:
                # Update with all responses
                if isinstance(responses, dict):
                    result.update(responses)
                else:
                    result["response"] = responses

            results.append(result)
        except Exception as e:
            logger.warning(f"Experiment {idx} failed: {e}")
            result = params.copy()
            if response_names:
                for name in response_names:
                    result[name] = np.nan
            results.append(result)

    return pd.DataFrame(results)


def fit_response_surface(
    factors: Union[pd.DataFrame, List[str]],
    response: Union[pd.Series, str],
    degree: int = 2,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Fit response surface model to experimental data.

    Args:
        factors: DataFrame with factors or list of factor names (if data provided)
        response: Series with response or response name (if data provided)
        degree: Polynomial degree (1 or 2)
        data: Optional DataFrame (if factors/response are names)

    Returns:
        Dictionary with model coefficients and statistics
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    # Handle two calling conventions:
    # 1. fit_response_surface(X_df, y_series, degree) - factors and response as DataFrames/Series
    # 2. fit_response_surface(factor_names, response_name, degree, data) - names with data DataFrame

    if data is not None:
        # Convention 2: factors and response are column names
        X = data[factors].values
        y = data[response].values
    else:
        # Convention 1: factors and response are DataFrames/Series
        if isinstance(factors, pd.DataFrame):
            X = factors.values
        else:
            raise ValueError("If data is not provided, factors must be a DataFrame")

        if isinstance(response, pd.Series):
            y = response.values
        else:
            raise ValueError("If data is not provided, response must be a Series")

    # Remove NaN
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]

    # Check if we have enough data points
    n_features = X.shape[1] if len(X.shape) > 1 else 1
    min_points = max(degree + 1, n_features + 1)
    if len(X) < min_points:
        return {"error": "Insufficient data", "r_squared": 0.0}

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predictions
    y_pred = model.predict(X_poly)

    # Statistics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Feature names - need to get column names properly
    if data is not None:
        # factors is a list of column names
        input_features = factors if isinstance(factors, list) else list(factors)
    else:
        # factors is a DataFrame - get column names
        if isinstance(factors, pd.DataFrame):
            input_features = list(factors.columns)
        else:
            input_features = [f"x{i}" for i in range(n_features)]

    feature_names = poly.get_feature_names_out(input_features)

    return {
        "model": model,
        "poly_features": poly,
        "coefficients": dict(zip(feature_names, model.coef_)),
        "intercept": float(model.intercept_),
        "r_squared": float(r2),
        "rmse": float(rmse),
        "n_samples": len(X),
        "factors": factors,
        "response": response,
        "degree": degree,
    }


def optimize_process_parameters(
    param_bounds: Dict[str, Tuple[float, float]],
    objective_function: Callable,
    constraints: Optional[List[Callable]] = None,
    method: str = "differential_evolution",
    maximize: bool = False,
) -> Dict[str, Any]:
    """
    Optimize process parameters to achieve desired outcomes.

    Args:
        param_bounds: Parameter bounds
        objective_function: Function that takes params and returns objective value
        constraints: List of constraint functions (return <= 0 when satisfied)
        method: Optimization method
        maximize: Whether to maximize (True) or minimize (False)

    Returns:
        Dictionary with optimization results
    """
    param_names = list(param_bounds.keys())
    bounds_list = [param_bounds[name] for name in param_names]

    def objective(x):
        params = dict(zip(param_names, x))
        value = objective_function(params)
        return -value if maximize else value

    def constraint_wrapper(constraint_func):
        def constraint(x):
            params = dict(zip(param_names, x))
            return constraint_func(params)

        return constraint

    # Set up constraints
    constraint_list = []
    if constraints:
        for constraint_func in constraints:
            constraint_list.append(
                {"type": "ineq", "fun": constraint_wrapper(constraint_func)}
            )

    try:
        if method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds_list,
                constraints=constraint_list if constraint_list else None,
                seed=42,
            )
        else:
            # Initial guess (center of bounds)
            x0 = [(b[0] + b[1]) / 2 for b in bounds_list]
            result = minimize(
                objective,
                x0,
                bounds=bounds_list,
                constraints=constraint_list if constraint_list else None,
                method=method,
            )

        optimal_params = dict(zip(param_names, result.x))
        optimal_value = result.fun if not maximize else -result.fun

        return {
            "success": result.success,
            "optimal_parameters": optimal_params,
            "optimal_value": float(optimal_value),
            "message": result.message,
            "n_iterations": getattr(result, "nit", None),
            "n_evaluations": getattr(result, "nfev", None),
        }
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {"success": False, "error": str(e)}


def multi_objective_optimization(
    param_bounds: Dict[str, Tuple[float, float]],
    objectives: Dict[str, Callable],
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_sum",
) -> Dict[str, Any]:
    """
    Multi-objective optimization (e.g., minimize porosity while maximizing strength).

    Args:
        param_bounds: Parameter bounds
        objectives: Dictionary of objective names and functions
        weights: Weights for each objective (default: equal)
        method: Method ('weighted_sum' or 'pareto')

    Returns:
        Dictionary with optimization results
    """
    if weights is None:
        weights = {name: 1.0 / len(objectives) for name in objectives.keys()}

    if method == "weighted_sum":

        def combined_objective(params):
            total = 0.0
            for obj_name, obj_func in objectives.items():
                value = obj_func(params)
                total += weights.get(obj_name, 0.0) * value
            return total

        result = optimize_process_parameters(
            param_bounds, combined_objective, maximize=False
        )

        # Evaluate individual objectives at optimum
        if result["success"]:
            individual_objectives = {}
            for obj_name, obj_func in objectives.items():
                individual_objectives[obj_name] = obj_func(result["optimal_parameters"])
            result["individual_objectives"] = individual_objectives

        return result
    else:
        # Pareto frontier (simplified - would need more sophisticated algorithm)
        logger.warning("Pareto optimization not fully implemented, using weighted sum")
        return multi_objective_optimization(
            param_bounds, objectives, weights, "weighted_sum"
        )
