"""
Statistical Fitting Module

Statistical distribution fitting and analysis for XCT data:
- Gaussian (normal) distribution
- Poisson distribution
- Linear and quadratic regression
- Goodness of fit evaluation
- Distribution parameter extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy import stats
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


def fit_gaussian(
    data: np.ndarray, weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Fit Gaussian (normal) distribution to data.

    Args:
        data: Input data array
        weights: Optional weights for weighted fitting

    Returns:
        Dictionary with fit parameters and statistics
    """
    data = np.asarray(data)
    data_clean = data[np.isfinite(data)]

    if len(data_clean) == 0:
        raise ValueError("No valid data points for fitting")

    # Fit using scipy.stats
    if weights is not None:
        weights_clean = weights[np.isfinite(data)]
        # Weighted mean and std
        mean = np.average(data_clean, weights=weights_clean)
        variance = np.average((data_clean - mean) ** 2, weights=weights_clean)
        std = np.sqrt(variance)
    else:
        mean, std = stats.norm.fit(data_clean)

    # Calculate additional statistics
    median = np.median(data_clean)
    mode = mean  # For normal distribution, mode = mean

    # Goodness of fit tests
    ks_statistic, ks_pvalue = stats.kstest(data_clean, "norm", args=(mean, std))

    # AIC and BIC (simplified)
    n = len(data_clean)
    log_likelihood = np.sum(stats.norm.logpdf(data_clean, mean, std))
    aic = 2 * 2 - 2 * log_likelihood  # 2 parameters: mean, std
    bic = 2 * np.log(n) - 2 * log_likelihood

    # R-squared (coefficient of determination)
    # Compare observed vs expected frequencies
    hist, bin_edges = np.histogram(data_clean, bins=min(50, len(data_clean) // 10))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_freq = (
        stats.norm.pdf(bin_centers, mean, std) * n * (bin_edges[1] - bin_edges[0])
    )
    observed_freq = hist.astype(float)

    # Avoid division by zero
    expected_freq = np.maximum(expected_freq, 1e-10)

    ss_res = np.sum((observed_freq - expected_freq) ** 2)
    ss_tot = np.sum((observed_freq - np.mean(observed_freq)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "distribution": "gaussian",
        "params": {"mean": float(mean), "std": float(std)},
        "mean": float(mean),
        "std": float(std),
        "variance": float(std**2),
        "median": float(median),
        "mode": float(mode),
        "n_samples": int(n),
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "aic": float(aic),
        "bic": float(bic),
        "r_squared": float(r_squared),
        "log_likelihood": float(log_likelihood),
        "fitted": True,
    }


def fit_poisson(
    data: np.ndarray, weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Fit Poisson distribution to data.

    Note: Poisson is for count data (non-negative integers).
    Data will be rounded to integers if needed.

    Args:
        data: Input data array (counts)
        weights: Optional weights for weighted fitting

    Returns:
        Dictionary with fit parameters and statistics
    """
    data = np.asarray(data)
    data_clean = data[np.isfinite(data)]

    if len(data_clean) == 0:
        raise ValueError("No valid data points for fitting")

    # Round to integers for Poisson
    data_int = np.round(data_clean).astype(int)
    data_int = data_int[data_int >= 0]  # Poisson requires non-negative

    if len(data_int) == 0:
        raise ValueError("No valid non-negative integer data points")

    # Fit lambda (mean) parameter
    if weights is not None:
        weights_clean = weights[np.isfinite(data)][data_clean >= 0]
        lambda_param = np.average(data_int, weights=weights_clean)
    else:
        lambda_param = np.mean(data_int)

    # Calculate additional statistics
    mean = lambda_param
    variance = lambda_param  # For Poisson, variance = mean
    std = np.sqrt(lambda_param)
    median = np.median(data_int)

    # Goodness of fit tests
    ks_statistic, ks_pvalue = stats.kstest(data_int, "poisson", args=(lambda_param,))

    # AIC and BIC
    n = len(data_int)
    log_likelihood = np.sum(stats.poisson.logpmf(data_int, lambda_param))
    aic = 2 * 1 - 2 * log_likelihood  # 1 parameter: lambda
    bic = 1 * np.log(n) - 2 * log_likelihood

    # Calculate R-squared for Poisson
    # Compare observed vs expected frequencies
    max_val = int(np.max(data_int))
    observed = np.bincount(data_int, minlength=max_val + 1)
    expected = stats.poisson.pmf(np.arange(max_val + 1), lambda_param) * n
    expected = np.maximum(expected, 1e-10)  # Avoid division by zero

    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "distribution": "poisson",
        "params": {"lambda": float(lambda_param)},
        "lambda": float(lambda_param),
        "mean": float(mean),
        "variance": float(variance),
        "std": float(std),
        "median": float(median),
        "n_samples": int(n),
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "aic": float(aic),
        "bic": float(bic),
        "r_squared": float(r_squared),
        "log_likelihood": float(log_likelihood),
        "fitted": True,
    }


def fit_linear(
    x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Fit linear regression: y = a*x + b.

    Args:
        x: Independent variable
        y: Dependent variable
        weights: Optional weights for weighted regression

    Returns:
        Dictionary with fit parameters and statistics
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove invalid values
    valid = np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid]
    y_clean = y[valid]

    if len(x_clean) < 2:
        raise ValueError("Need at least 2 data points for linear fit")

    # Fit using scipy
    if weights is not None:
        weights_clean = weights[valid]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

        # For weighted regression, use curve_fit
        def linear_func(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(
            linear_func,
            x_clean,
            y_clean,
            sigma=1.0 / weights_clean,
            absolute_sigma=True,
        )
        slope, intercept = popt
        r_value = np.corrcoef(x_clean, y_clean)[0, 1]
        std_err = np.sqrt(np.diag(pcov))[0]
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    # Calculate predictions
    y_pred = slope * x_clean + intercept

    # R-squared
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residuals
    residuals = y_clean - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    return {
        "distribution": "linear",
        "params": {"slope": float(slope), "intercept": float(intercept)},
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_samples": int(len(x_clean)),
        "fitted": True,
    }


def fit_quadratic(
    x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Fit quadratic regression: y = a*x² + b*x + c.

    Args:
        x: Independent variable
        y: Dependent variable
        weights: Optional weights for weighted regression

    Returns:
        Dictionary with fit parameters and statistics
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove invalid values
    valid = np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid]
    y_clean = y[valid]

    if len(x_clean) < 3:
        raise ValueError("Need at least 3 data points for quadratic fit")

    # Define quadratic function
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit using curve_fit
    if weights is not None:
        weights_clean = weights[valid]
        popt, pcov = curve_fit(
            quadratic_func,
            x_clean,
            y_clean,
            sigma=1.0 / weights_clean,
            absolute_sigma=True,
        )
    else:
        popt, pcov = curve_fit(quadratic_func, x_clean, y_clean)

    a, b, c = popt
    param_errors = np.sqrt(np.diag(pcov))

    # Calculate predictions
    y_pred = quadratic_func(x_clean, a, b, c)

    # R-squared
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residuals
    residuals = y_clean - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    return {
        "distribution": "quadratic",
        "params": {
            "a": float(a),  # x² coefficient
            "b": float(b),  # x coefficient
            "c": float(c),  # constant
        },
        "a": float(a),  # x² coefficient
        "b": float(b),  # x coefficient
        "c": float(c),  # constant
        "a_err": float(param_errors[0]),
        "b_err": float(param_errors[1]),
        "c_err": float(param_errors[2]),
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_samples": int(len(x_clean)),
        "fitted": True,
    }


def fit_distribution(
    data: np.ndarray,
    distribution: str = "gaussian",
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit specified distribution to data.

    Args:
        data: Input data array
        distribution: Distribution type ('gaussian', 'poisson')
        weights: Optional weights

    Returns:
        Dictionary with fit parameters
    """
    if distribution.lower() in ["gaussian", "normal", "norm"]:
        return fit_gaussian(data, weights)
    elif distribution.lower() in ["poisson", "poiss"]:
        return fit_poisson(data, weights)
    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Supported: 'gaussian', 'poisson'"
        )


def compare_fits(
    data: np.ndarray,
    distributions: List[str] = ["gaussian", "poisson"],
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare multiple distribution fits and select best one.

    Args:
        data: Input data array
        distributions: List of distribution names to compare
        weights: Optional weights

    Returns:
        Dictionary with all fits and best fit recommendation
    """
    fits = {}

    for dist in distributions:
        try:
            fits[dist] = fit_distribution(data, dist, weights)
        except Exception as e:
            logger.warning(f"Failed to fit {dist}: {e}")
            fits[dist] = {"fitted": False, "error": str(e)}

    # Select best fit based on AIC (lower is better)
    valid_fits = {k: v for k, v in fits.items() if v.get("fitted", False)}

    if len(valid_fits) == 0:
        return {"fits": fits, "best_fit": None, "best_distribution": None}

    best_dist = min(valid_fits.keys(), key=lambda k: valid_fits[k].get("aic", np.inf))

    return {
        "fits": fits,
        "best_fit": valid_fits[best_dist],
        "best_distribution": best_dist,
        "comparison": {
            dist: {
                "aic": fit.get("aic", np.inf),
                "bic": fit.get("bic", np.inf),
                "r_squared": fit.get("r_squared", 0.0),
                "ks_pvalue": fit.get("ks_pvalue", 0.0),
            }
            for dist, fit in valid_fits.items()
        },
    }


def evaluate_fit_quality(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    fit_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate quality of a distribution fit.

    Args:
        x: Optional x values (for regression fits)
        y: Optional y values (for regression fits)
        fit_result: Result from fit function

    Returns:
        Dictionary with quality metrics and interpretation
    """
    if fit_result is None:
        raise ValueError("fit_result is required")

    quality = {"good": False, "acceptable": False, "poor": False, "metrics": {}}

    if not fit_result.get("fitted", False):
        quality["poor"] = True
        quality["interpretation"] = "Fit failed"
        return quality

    # If x and y provided, calculate additional metrics
    if x is not None and y is not None:
        x = np.asarray(x)
        y = np.asarray(y)
        valid = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid]
        y_clean = y[valid]

        # Calculate predictions based on fit type
        if fit_result.get("distribution") == "linear":
            slope = fit_result.get("slope", 0)
            intercept = fit_result.get("intercept", 0)
            y_pred = slope * x_clean + intercept
        elif fit_result.get("distribution") == "quadratic":
            a = fit_result.get("a", 0)
            b = fit_result.get("b", 0)
            c = fit_result.get("c", 0)
            y_pred = a * x_clean**2 + b * x_clean + c
        else:
            y_pred = None

        if y_pred is not None:
            residuals = y_clean - y_pred
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            quality["rmse"] = float(rmse)
            quality["mae"] = float(mae)
            quality["metrics"]["rmse"] = float(rmse)
            quality["metrics"]["mae"] = float(mae)

    # Check R-squared (for regression)
    if "r_squared" in fit_result:
        r_sq = fit_result["r_squared"]
        quality["r_squared"] = r_sq
        quality["metrics"]["r_squared"] = r_sq
        if r_sq > 0.95:
            quality["good"] = True
        elif r_sq > 0.80:
            quality["acceptable"] = True
        else:
            quality["poor"] = True

    # Check KS test p-value (for distributions)
    if "ks_pvalue" in fit_result:
        p_value = fit_result["ks_pvalue"]
        quality["metrics"]["ks_pvalue"] = p_value
        if p_value > 0.05:
            quality["good"] = True
        elif p_value > 0.01:
            quality["acceptable"] = True
        else:
            quality["poor"] = True

    # Check AIC/BIC (lower is better, but relative)
    if "aic" in fit_result:
        quality["metrics"]["aic"] = fit_result["aic"]

    if "bic" in fit_result:
        quality["metrics"]["bic"] = fit_result["bic"]

    # Overall interpretation
    if quality["good"]:
        quality["interpretation"] = "Good fit"
    elif quality["acceptable"]:
        quality["interpretation"] = "Acceptable fit"
    else:
        quality["interpretation"] = "Poor fit - consider alternative distributions"

    return quality


def generate_fit_samples(
    fit_result: Dict[str, Any], n_samples: int = 1000
) -> np.ndarray:
    """
    Generate random samples from fitted distribution.

    Args:
        fit_result: Result from fit function
        n_samples: Number of samples to generate

    Returns:
        Array of generated samples
    """
    if not fit_result.get("fitted", False):
        raise ValueError("Cannot generate samples from failed fit")

    dist_type = fit_result["distribution"]

    if dist_type == "gaussian":
        return np.random.normal(fit_result["mean"], fit_result["std"], size=n_samples)
    elif dist_type == "poisson":
        return np.random.poisson(fit_result["lambda"], size=n_samples)
    else:
        raise ValueError(f"Cannot generate samples for {dist_type}")


def predict_from_fit(x: np.ndarray, fit_result: Dict[str, Any]) -> np.ndarray:
    """
    Predict y values from fitted regression model.

    Args:
        x: Independent variable values
        fit_result: Result from fit_linear or fit_quadratic

    Returns:
        Predicted y values
    """
    if not fit_result.get("fitted", False):
        raise ValueError("Cannot predict from failed fit")

    dist_type = fit_result["distribution"]

    if dist_type == "linear":
        return fit_result["slope"] * x + fit_result["intercept"]
    elif dist_type == "quadratic":
        return fit_result["a"] * x**2 + fit_result["b"] * x + fit_result["c"]
    else:
        raise ValueError(f"Cannot predict for {dist_type}")


# ============================================================================
# Advanced Statistical Methods
# ============================================================================


def non_parametric_tests(
    data1: np.ndarray, data2: np.ndarray, test_type: str = "mann_whitney"
) -> Dict[str, Any]:
    """
    Perform non-parametric statistical tests.

    Args:
        data1: First data array
        data2: Second data array
        test_type: Type of test ('mann_whitney', 'ks', 'wilcoxon', 'kruskal')

    Returns:
        Dictionary with test results
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    data1_clean = data1[np.isfinite(data1)]
    data2_clean = data2[np.isfinite(data2)]

    if len(data1_clean) == 0 or len(data2_clean) == 0:
        return {"error": "Insufficient valid data"}

    results = {"test_type": test_type, "n1": len(data1_clean), "n2": len(data2_clean)}

    if test_type == "mann_whitney":
        # Mann-Whitney U test (independent samples)
        statistic, pvalue = stats.mannwhitneyu(
            data1_clean, data2_clean, alternative="two-sided"
        )
        results.update(
            {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant": pvalue < 0.05,
                "interpretation": (
                    "Significant difference"
                    if pvalue < 0.05
                    else "No significant difference"
                ),
            }
        )

    elif test_type == "ks":
        # Kolmogorov-Smirnov test (distribution comparison)
        statistic, pvalue = stats.ks_2samp(data1_clean, data2_clean)
        results.update(
            {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant": pvalue < 0.05,
                "interpretation": (
                    "Distributions differ" if pvalue < 0.05 else "Distributions similar"
                ),
            }
        )

    elif test_type == "wilcoxon":
        # Wilcoxon signed-rank test (paired samples)
        if len(data1_clean) != len(data2_clean):
            return {"error": "Wilcoxon test requires paired samples (same length)"}
        statistic, pvalue = stats.wilcoxon(data1_clean, data2_clean)
        results.update(
            {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant": pvalue < 0.05,
                "interpretation": (
                    "Significant difference"
                    if pvalue < 0.05
                    else "No significant difference"
                ),
            }
        )

    elif test_type == "kruskal":
        # Kruskal-Wallis H-test (multiple groups)
        # For two groups, equivalent to Mann-Whitney
        statistic, pvalue = stats.kruskal(data1_clean, data2_clean)
        results.update(
            {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant": pvalue < 0.05,
                "interpretation": (
                    "Groups differ" if pvalue < 0.05 else "Groups similar"
                ),
            }
        )

    else:
        return {"error": f"Unknown test type: {test_type}"}

    return results


def spatial_autocorrelation(
    volume: np.ndarray,
    metric: str = "moran",
    distance_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute spatial autocorrelation for volume data.

    Args:
        volume: 3D volume
        metric: Autocorrelation metric ('moran', 'geary', 'variogram')
        distance_threshold: Optional distance threshold for analysis

    Returns:
        Dictionary with spatial autocorrelation metrics
    """
    # Flatten volume for analysis
    volume_flat = volume.flatten()

    # Get coordinates
    coords = (
        np.array(
            np.meshgrid(
                np.arange(volume.shape[0]),
                np.arange(volume.shape[1]),
                np.arange(volume.shape[2]),
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )

    # Simplified spatial autocorrelation (Moran's I approximation)
    if metric == "moran":
        # Compute mean
        mean_val = np.mean(volume_flat)

        # Compute spatial lag (simplified - using nearest neighbors)
        n_neighbors = 6  # 6-connected neighborhood in 3D
        spatial_lag = np.zeros_like(volume_flat)

        # Simplified: use local mean as spatial lag
        from scipy.ndimage import uniform_filter

        spatial_lag_volume = uniform_filter(volume.astype(float), size=3)
        spatial_lag = spatial_lag_volume.flatten()

        # Moran's I
        numerator = np.sum((volume_flat - mean_val) * (spatial_lag - mean_val))
        denominator = np.sum((volume_flat - mean_val) ** 2)
        morans_i = (numerator / denominator) if denominator > 0 else 0.0

        return {
            "morans_i": float(morans_i),
            "interpretation": (
                "Positive autocorrelation"
                if morans_i > 0
                else (
                    "Negative autocorrelation" if morans_i < 0 else "No autocorrelation"
                )
            ),
        }

    elif metric == "variogram":
        # Simplified variogram
        # Sample pairs at different distances
        n_samples = min(1000, len(volume_flat))
        indices = np.random.choice(len(volume_flat), n_samples, replace=False)

        distances = []
        squared_diffs = []

        for i in range(n_samples):
            for j in range(i + 1, min(i + 10, n_samples)):  # Limit pairs
                idx1, idx2 = indices[i], indices[j]
                coord1, coord2 = coords[idx1], coords[idx2]
                dist = np.linalg.norm(coord1 - coord2)
                sq_diff = (volume_flat[idx1] - volume_flat[idx2]) ** 2

                distances.append(dist)
                squared_diffs.append(sq_diff)

        if len(distances) > 0:
            distances = np.array(distances)
            squared_diffs = np.array(squared_diffs)

            # Bin distances
            if distance_threshold is None:
                distance_threshold = np.percentile(distances, 50)

            # Variogram value
            variogram = np.mean(squared_diffs[distances <= distance_threshold]) / 2.0

            return {
                "variogram": float(variogram),
                "distance_threshold": float(distance_threshold),
                "n_pairs": len(distances),
            }

    return {"error": f"Unknown metric: {metric}"}


def principal_component_analysis(
    metrics_df: Union[pd.DataFrame, Dict[str, np.ndarray]],
    n_components: Optional[int] = None,
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis (PCA) on metrics.

    Args:
        metrics_df: DataFrame or dictionary with metrics
        n_components: Number of components (if None, use all)
        standardize: Whether to standardize data

    Returns:
        Dictionary with PCA results
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn required for PCA")
        return {"error": "scikit-learn not available"}

    # Convert to DataFrame if needed
    if isinstance(metrics_df, dict):
        metrics_df = pd.DataFrame(metrics_df)

    # Remove non-numeric columns
    numeric_df = metrics_df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {"error": "No numeric columns found"}

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_df)
    else:
        data_scaled = numeric_df.values
        scaler = None

    # Perform PCA
    if n_components is None:
        n_components = min(data_scaled.shape[1], data_scaled.shape[0])

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    return {
        "n_components": n_components,
        "n_samples": data_scaled.shape[0],
        "n_features": data_scaled.shape[1],
        "explained_variance_ratio": explained_variance.tolist(),
        "cumulative_variance_ratio": cumulative_variance.tolist(),
        "components": pca.components_.tolist(),
        "principal_components": pca_result.tolist(),
        "feature_names": numeric_df.columns.tolist(),
        "n_components_95_variance": (
            int(np.argmax(cumulative_variance >= 0.95) + 1)
            if np.any(cumulative_variance >= 0.95)
            else n_components
        ),
    }


def bayesian_uncertainty(
    measurements: np.ndarray,
    priors: Optional[Dict[str, float]] = None,
    distribution: str = "normal",
) -> Dict[str, Any]:
    """
    Bayesian uncertainty quantification (simplified).

    Args:
        measurements: Array of measurements
        priors: Optional prior parameters
        distribution: Distribution assumption ('normal', 'gamma')

    Returns:
        Dictionary with Bayesian uncertainty results
    """
    measurements = np.asarray(measurements)
    measurements_clean = measurements[np.isfinite(measurements)]

    if len(measurements_clean) == 0:
        return {"error": "No valid measurements"}

    if distribution == "normal":
        # Bayesian inference for normal distribution
        # Using conjugate prior (normal-inverse-gamma)

        # Prior parameters (default)
        if priors is None:
            mu_0 = np.mean(measurements_clean)  # Prior mean
            lambda_0 = 1.0  # Prior precision (inverse variance) weight
            alpha_0 = 1.0  # Prior shape
            beta_0 = 1.0  # Prior scale
        else:
            mu_0 = priors.get("mu", np.mean(measurements_clean))
            lambda_0 = priors.get("lambda", 1.0)
            alpha_0 = priors.get("alpha", 1.0)
            beta_0 = priors.get("beta", 1.0)

        n = len(measurements_clean)
        sample_mean = np.mean(measurements_clean)
        sample_var = np.var(measurements_clean, ddof=1)

        # Posterior parameters
        lambda_n = lambda_0 + n
        mu_n = (lambda_0 * mu_0 + n * sample_mean) / lambda_n
        alpha_n = alpha_0 + n / 2.0
        beta_n = beta_0 + 0.5 * (
            n * sample_var + (lambda_0 * n * (sample_mean - mu_0) ** 2) / lambda_n
        )

        # Posterior mean and variance
        posterior_mean = mu_n
        posterior_var = beta_n / (alpha_n - 1) if alpha_n > 1 else beta_n

        # Credible intervals (95%)
        from scipy.stats import t

        t_critical = t.ppf(0.975, 2 * alpha_n)
        credible_interval = (
            posterior_mean - t_critical * np.sqrt(posterior_var / lambda_n),
            posterior_mean + t_critical * np.sqrt(posterior_var / lambda_n),
        )

        return {
            "posterior_mean": float(posterior_mean),
            "posterior_std": float(np.sqrt(posterior_var)),
            "posterior_variance": float(posterior_var),
            "credible_interval_95": (
                float(credible_interval[0]),
                float(credible_interval[1]),
            ),
            "n_measurements": n,
            "prior_mean": float(mu_0),
            "sample_mean": float(sample_mean),
            "distribution": "normal",
        }

    else:
        return {"error": f"Unsupported distribution: {distribution}"}


def time_series_analysis(
    data: np.ndarray, analysis_type: str = "trend"
) -> Dict[str, Any]:
    """
    Time series analysis for printing direction trends.

    Args:
        data: Time series data (e.g., porosity along printing direction)
        analysis_type: Type of analysis ('trend', 'autocorrelation', 'stationarity')

    Returns:
        Dictionary with time series analysis results
    """
    data = np.asarray(data)
    data_clean = data[np.isfinite(data)]

    if len(data_clean) < 3:
        return {"error": "Insufficient data for time series analysis"}

    results = {"n_points": len(data_clean), "analysis_type": analysis_type}

    if analysis_type == "trend":
        # Linear trend
        x = np.arange(len(data_clean))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data_clean)

        results.update(
            {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "trend_significant": p_value < 0.05,
                "trend_direction": (
                    "increasing"
                    if slope > 0
                    else "decreasing" if slope < 0 else "stable"
                ),
            }
        )

    elif analysis_type == "autocorrelation":
        # Autocorrelation (lag-1)
        if len(data_clean) > 1:
            autocorr = np.corrcoef(data_clean[:-1], data_clean[1:])[0, 1]
            results["autocorrelation_lag1"] = float(autocorr)
            results["interpretation"] = (
                "Positive autocorrelation"
                if autocorr > 0
                else "Negative autocorrelation"
            )

    elif analysis_type == "stationarity":
        # Simplified stationarity test (Augmented Dickey-Fuller would require statsmodels)
        # Use variance ratio test
        first_half = data_clean[: len(data_clean) // 2]
        second_half = data_clean[len(data_clean) // 2 :]

        var1 = np.var(first_half)
        var2 = np.var(second_half)
        variance_ratio = var2 / var1 if var1 > 0 else 0.0

        results.update(
            {
                "variance_ratio": float(variance_ratio),
                "stationary": 0.5 < variance_ratio < 2.0,  # Simplified criterion
                "first_half_var": float(var1),
                "second_half_var": float(var2),
            }
        )

    return results
