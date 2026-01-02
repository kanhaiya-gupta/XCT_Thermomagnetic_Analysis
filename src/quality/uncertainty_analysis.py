"""
Uncertainty Analysis Module

Comprehensive uncertainty quantification for XCT measurements:
- Measurement uncertainty (voxel size, segmentation)
- Systematic vs. random errors
- Confidence intervals
- Uncertainty budgets
- Monte Carlo uncertainty propagation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from scipy import stats
from scipy.ndimage import gaussian_filter
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def measurement_uncertainty(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    voxel_size_uncertainty: Optional[Tuple[float, float, float]] = None,
    segmentation_uncertainty: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute measurement uncertainty from voxel size and segmentation uncertainty.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing (dx, dy, dz) in mm
        voxel_size_uncertainty: Uncertainty in voxel size (default: 1% of voxel size)
        segmentation_uncertainty: Uncertainty in segmentation (default: 0.5 voxels)

    Returns:
        Dictionary with measurement uncertainty analysis
    """
    # Default uncertainties
    if voxel_size_uncertainty is None:
        voxel_size_uncertainty = tuple(v * 0.01 for v in voxel_size)  # 1% uncertainty

    if segmentation_uncertainty is None:
        segmentation_uncertainty = 0.5  # 0.5 voxels

    # Compute volume
    n_voxels = np.sum(volume > 0)
    voxel_volume = np.prod(voxel_size)
    total_volume = n_voxels * voxel_volume

    # Volume uncertainty from voxel size uncertainty
    # Using error propagation: δV = V * sqrt((δdx/dx)² + (δdy/dy)² + (δdz/dz)²)
    relative_uncertainties = [
        voxel_size_uncertainty[i] / voxel_size[i] for i in range(3)
    ]
    volume_uncertainty_voxel_size = total_volume * np.sqrt(
        sum(u**2 for u in relative_uncertainties)
    )

    # Volume uncertainty from segmentation uncertainty
    # Surface voxels contribute most to uncertainty
    from ..core.morphology import erode

    eroded = erode(volume, kernel_size=1)
    surface_voxels = np.sum((volume > 0) & (eroded == 0))
    volume_uncertainty_segmentation = (
        surface_voxels * segmentation_uncertainty * voxel_volume
    )

    # Combined volume uncertainty (RSS)
    volume_uncertainty_total = np.sqrt(
        volume_uncertainty_voxel_size**2 + volume_uncertainty_segmentation**2
    )

    # Surface area uncertainty (approximate)
    # Surface area depends on surface voxels
    surface_area_uncertainty = (
        surface_voxels
        * segmentation_uncertainty
        * np.mean(
            [
                voxel_size[1] * voxel_size[2],  # y-z face
                voxel_size[0] * voxel_size[2],  # x-z face
                voxel_size[0] * voxel_size[1],  # x-y face
            ]
        )
    )

    return {
        "volume_uncertainty": float(
            volume_uncertainty_total
        ),  # Top-level for compatibility
        "volume_uncertainty_breakdown": {
            "from_voxel_size": float(volume_uncertainty_voxel_size),
            "from_segmentation": float(volume_uncertainty_segmentation),
            "total": float(volume_uncertainty_total),
            "relative": (
                float(volume_uncertainty_total / total_volume * 100)
                if total_volume > 0
                else 0.0
            ),
        },
        "surface_area_uncertainty": float(surface_area_uncertainty),
        "voxel_size_uncertainty": voxel_size_uncertainty,
        "segmentation_uncertainty": segmentation_uncertainty,
        "n_surface_voxels": int(surface_voxels),
        "n_total_voxels": int(n_voxels),
    }


def segmentation_uncertainty(
    volume_or_segmentations: Union[np.ndarray, List[np.ndarray]],
    threshold_range: Optional[Tuple[float, float]] = None,
    n_samples: int = 100,
) -> Dict[str, Any]:
    """
    Quantify segmentation uncertainty by varying threshold or comparing multiple segmentations.

    Args:
        volume_or_segmentations: Grayscale volume (before segmentation) or list of binary segmentations
        threshold_range: (min_threshold, max_threshold) range to test (only if volume provided)
        n_samples: Number of threshold values to test (only if volume provided)

    Returns:
        Dictionary with segmentation uncertainty analysis
    """
    # Handle case where list of segmentations is provided
    if isinstance(volume_or_segmentations, list):
        segmentations = volume_or_segmentations
        # Compute statistics across segmentations
        volumes = [np.sum(seg > 0) for seg in segmentations]
        surface_areas = []
        for seg in segmentations:
            from ..core.morphology import erode

            eroded = erode(seg, kernel_size=1)
            surface_voxels = np.sum((seg > 0) & (eroded == 0))
            surface_areas.append(surface_voxels)

        volume_std = float(np.std(volumes))
        surface_area_std = float(np.std(surface_areas))

        return {
            "volume_uncertainty": volume_std,
            "surface_area_uncertainty": surface_area_std,
            "n_segmentations": len(segmentations),
            "volume_mean": float(np.mean(volumes)),
            "volume_std": volume_std,
        }

    volume = volume_or_segmentations
    if volume.dtype == np.uint8 and volume.max() <= 1:
        # Already binary, can't analyze threshold uncertainty
        return {
            "error": "Volume is already binary. Provide grayscale volume for threshold uncertainty analysis.",
            "threshold_uncertainty": 0.0,
        }

    # Default threshold range if not provided
    if threshold_range is None:
        threshold_range = (float(volume.min()), float(volume.max()))

    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_samples)

    volumes = []
    surface_areas = []
    void_fractions = []

    for threshold in thresholds:
        segmented = (volume > threshold).astype(np.uint8)

        # Compute metrics
        n_voxels = np.sum(segmented > 0)
        volumes.append(n_voxels)

        # Surface area (approximate)
        from ..core.morphology import erode

        eroded = erode(segmented, kernel_size=1)
        surface_voxels = np.sum((segmented > 0) & (eroded == 0))
        surface_areas.append(surface_voxels)

        # Void fraction
        void_fractions.append(1.0 - (n_voxels / segmented.size))

    # Compute statistics
    volumes = np.array(volumes)
    surface_areas = np.array(surface_areas)
    void_fractions = np.array(void_fractions)

    return {
        "threshold_range": threshold_range,
        "n_samples": n_samples,
        "volume_std": float(np.std(volumes)),
        "volume_mean": float(np.mean(volumes)),
        "volume_cv": (
            float(np.std(volumes) / np.mean(volumes) * 100)
            if np.mean(volumes) > 0
            else 0.0
        ),
        "surface_area_std": float(np.std(surface_areas)),
        "void_fraction_std": float(np.std(void_fractions)),
        "void_fraction_range": (
            float(np.min(void_fractions)),
            float(np.max(void_fractions)),
        ),
        "sensitivity": {
            "volume_per_threshold": float(
                np.std(volumes) / (threshold_range[1] - threshold_range[0])
            ),
            "surface_area_per_threshold": float(
                np.std(surface_areas) / (threshold_range[1] - threshold_range[0])
            ),
        },
    }


def confidence_intervals(
    metrics: Union[Dict[str, float], np.ndarray, List[float]],
    uncertainties: Optional[Dict[str, float]] = None,
    confidence: float = 0.95,
    confidence_level: Optional[float] = None,  # Alias for compatibility
    distribution: str = "normal",
) -> Dict[str, Any]:
    """
    Compute confidence intervals for metrics given uncertainties.

    Args:
        metrics: Dictionary of metric values, or array/list of values
        uncertainties: Dictionary of uncertainties (standard deviations) or None to compute from data
        confidence: Confidence level (0.95 for 95% CI)
        confidence_level: Alias for confidence (for compatibility)
        distribution: Distribution assumption ('normal' or 't')

    Returns:
        Dictionary with confidence intervals
    """
    # Use confidence_level if provided
    if confidence_level is not None:
        confidence = confidence_level

    # Handle array/list input
    if isinstance(metrics, (np.ndarray, list)):
        values = np.asarray(metrics)
        mean = float(np.mean(values))
        std = float(np.std(values))

        alpha = 1 - confidence
        if distribution == "normal":
            z_score = stats.norm.ppf(1 - alpha / 2)
        else:
            z_score = stats.norm.ppf(1 - alpha / 2)

        margin = z_score * std

        return {
            "mean": mean,
            "std": std,
            "ci_lower": float(mean - margin),
            "ci_upper": float(mean + margin),
            "confidence": confidence,
        }

    # Handle dict input
    alpha = 1 - confidence
    if distribution == "normal":
        z_score = stats.norm.ppf(1 - alpha / 2)
    else:  # t-distribution (would need degrees of freedom)
        z_score = stats.norm.ppf(1 - alpha / 2)  # Approximate with normal

    intervals = {}

    for metric_name, value in metrics.items():
        if uncertainties and metric_name in uncertainties:
            uncertainty = uncertainties[metric_name]
            margin = z_score * uncertainty

            intervals[metric_name] = (float(value - margin), float(value + margin))
        else:
            intervals[metric_name] = (float(value), float(value))

    return intervals


def uncertainty_budget(
    analysis_results: Union[Dict[str, Any], np.ndarray],
    uncertainty_sources: Optional[
        Union[Dict[str, float], Tuple[float, float, float]]
    ] = None,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Create uncertainty budget showing contribution of each uncertainty source.

    Args:
        analysis_results: Dictionary with analysis results, or volume array
        uncertainty_sources: Dictionary of uncertainty sources and values, or voxel_size tuple

    Returns:
        DataFrame or Dict with uncertainty budget
    """
    # Handle case where volume and voxel_size are passed (for compatibility)
    if isinstance(analysis_results, np.ndarray):
        volume = analysis_results
        if isinstance(uncertainty_sources, tuple):
            voxel_size = uncertainty_sources
        else:
            voxel_size = (0.1, 0.1, 0.1)  # Default

        # Compute basic uncertainties
        meas_unc = measurement_uncertainty(volume, voxel_size)

        return {
            "components": [
                {
                    "source": "Voxel Size",
                    "contribution": meas_unc.get(
                        "volume_uncertainty_breakdown", {}
                    ).get("from_voxel_size", 0.0),
                },
                {
                    "source": "Segmentation",
                    "contribution": meas_unc.get(
                        "volume_uncertainty_breakdown", {}
                    ).get("from_segmentation", 0.0),
                },
            ],
            "total_uncertainty": meas_unc.get("volume_uncertainty", 0.0),
        }

    if uncertainty_sources is None:
        # Default uncertainty sources
        uncertainty_sources = {
            "Voxel Size": 0.01,  # 1% relative
            "Segmentation": 0.5,  # 0.5 voxels
            "Image Noise": 0.02,  # 2% relative
            "Calibration": 0.005,  # 0.5% relative
            "Processing": 0.01,  # 1% relative
        }
    elif isinstance(uncertainty_sources, tuple):
        # Convert tuple to dict
        uncertainty_sources = {
            "Voxel Size": uncertainty_sources[0],
            "Segmentation": (
                uncertainty_sources[1] if len(uncertainty_sources) > 1 else 0.5
            ),
            "Image Noise": (
                uncertainty_sources[2] if len(uncertainty_sources) > 2 else 0.02
            ),
        }

    budget_data = []

    for source, value in uncertainty_sources.items():
        budget_data.append(
            {
                "Source": source,
                "Uncertainty": value,
                "Type": "Relative" if source != "Segmentation" else "Absolute (voxels)",
                "Contribution (%)": (
                    value / sum(uncertainty_sources.values()) * 100
                    if sum(uncertainty_sources.values()) > 0
                    else 0.0
                ),
            }
        )

    # Total uncertainty (RSS)
    total_uncertainty = np.sqrt(
        sum(
            u**2
            for u in uncertainty_sources.values()
            if isinstance(u, (int, float)) and u < 1
        )
    )

    budget_data.append(
        {
            "Source": "Total (RSS)",
            "Uncertainty": total_uncertainty,
            "Type": "Combined",
            "Contribution (%)": 100.0,
        }
    )

    return pd.DataFrame(budget_data)


def monte_carlo_uncertainty(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    analysis_function: Callable,
    n_samples: int = 1000,
    voxel_size_uncertainty: Optional[Tuple[float, float, float]] = None,
    segmentation_uncertainty: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Monte Carlo uncertainty propagation.

    Args:
        volume: Binary or grayscale volume
        voxel_size: Voxel spacing
        analysis_function: Function that takes (volume, voxel_size) and returns metrics dict
        n_samples: Number of Monte Carlo samples
        voxel_size_uncertainty: Uncertainty in voxel size
        segmentation_uncertainty: Uncertainty in segmentation
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with uncertainty analysis results
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Default uncertainties
    if voxel_size_uncertainty is None:
        voxel_size_uncertainty = tuple(v * 0.01 for v in voxel_size)

    if segmentation_uncertainty is None:
        segmentation_uncertainty = 0.5

    # Run Monte Carlo simulation
    results_list = []

    for i in range(n_samples):
        # Perturb voxel size
        perturbed_voxel_size = tuple(
            np.random.normal(v, u) for v, u in zip(voxel_size, voxel_size_uncertainty)
        )
        perturbed_voxel_size = tuple(
            max(0.001, v) for v in perturbed_voxel_size
        )  # Ensure positive

        # Perturb volume (if grayscale, add noise; if binary, erode/dilate slightly)
        if volume.dtype == np.uint8 and volume.max() <= 1:
            # Binary volume - random erosion/dilation
            from ..core.morphology import erode, dilate

            if np.random.rand() > 0.5:
                perturbed_volume = erode(volume, kernel_size=1)
            else:
                perturbed_volume = dilate(volume, kernel_size=1)
        else:
            # Grayscale - add noise
            noise = np.random.normal(
                0, segmentation_uncertainty * volume.std(), volume.shape
            )
            perturbed_volume = np.clip(volume + noise, 0, volume.max())

        # Run analysis
        try:
            metrics = analysis_function(perturbed_volume, perturbed_voxel_size)
            results_list.append(metrics)
        except Exception as e:
            logger.warning(f"Monte Carlo sample {i} failed: {e}")
            continue

    if len(results_list) == 0:
        return {"error": "All Monte Carlo samples failed"}

    # Aggregate results
    all_metrics = {}
    for result in results_list:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

    # Compute statistics
    uncertainty_results = {}
    for metric_name, values in all_metrics.items():
        values_array = np.array(values)
        uncertainty_results[metric_name] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "cv": (
                float(np.std(values_array) / np.mean(values_array) * 100)
                if np.mean(values_array) > 0
                else 0.0
            ),
            "percentiles": {
                "p5": float(np.percentile(values_array, 5)),
                "p25": float(np.percentile(values_array, 25)),
                "p50": float(np.percentile(values_array, 50)),
                "p75": float(np.percentile(values_array, 75)),
                "p95": float(np.percentile(values_array, 95)),
            },
        }

    return {
        "n_samples": len(results_list),
        "n_successful": len(results_list),
        "uncertainties": uncertainty_results,
        "confidence_intervals_95": {
            name: (
                (
                    results["percentiles"]["p2.5"]
                    if "p2.5" in results["percentiles"]
                    else results["mean"] - 1.96 * results["std"]
                ),
                (
                    results["percentiles"]["p97.5"]
                    if "p97.5" in results["percentiles"]
                    else results["mean"] + 1.96 * results["std"]
                ),
            )
            for name, results in uncertainty_results.items()
        },
    }


def systematic_vs_random_errors(
    measurements: List[Dict[str, float]],
    reference_values: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Analyze systematic vs. random errors in measurements.

    Args:
        measurements: List of measurement dictionaries (e.g., from multiple samples)
        reference_values: Optional reference/true values

    Returns:
        Dictionary with systematic and random error analysis
    """
    if len(measurements) == 0:
        return {"error": "No measurements provided"}

    # Aggregate measurements
    all_metrics = {}
    for measurement in measurements:
        for key, value in measurement.items():
            if isinstance(value, (int, float)):
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

    results = {}

    for metric_name, values in all_metrics.items():
        values_array = np.array(values)

        # Random error (precision) - standard deviation
        random_error = float(np.std(values_array))

        # Systematic error (bias) - difference from reference
        if reference_values and metric_name in reference_values:
            systematic_error = float(
                np.mean(values_array) - reference_values[metric_name]
            )
            bias_percentage = (
                (systematic_error / reference_values[metric_name] * 100)
                if reference_values[metric_name] != 0
                else 0.0
            )
        else:
            systematic_error = 0.0
            bias_percentage = 0.0

        # Total error (RMSE if reference available)
        if reference_values and metric_name in reference_values:
            total_error = float(
                np.sqrt(np.mean((values_array - reference_values[metric_name]) ** 2))
            )
        else:
            total_error = random_error

        results[metric_name] = {
            "mean": float(np.mean(values_array)),
            "random_error": random_error,
            "systematic_error": systematic_error,
            "bias_percentage": bias_percentage,
            "total_error": total_error,
            "precision": random_error,
            "trueness": abs(systematic_error),
            "accuracy": total_error,
        }

    return {
        "n_measurements": len(measurements),
        "errors_by_metric": results,
        "summary": {
            "mean_random_error": float(
                np.mean([r["random_error"] for r in results.values()])
            ),
            "mean_systematic_error": float(
                np.mean([abs(r["systematic_error"]) for r in results.values()])
            ),
            "mean_total_error": float(
                np.mean([r["total_error"] for r in results.values()])
            ),
        },
    }


def comprehensive_uncertainty_analysis(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    metrics: Dict[str, float],
    voxel_size_uncertainty: Optional[Tuple[float, float, float]] = None,
    segmentation_uncertainty: Optional[float] = None,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Comprehensive uncertainty analysis combining all methods.

    Args:
        volume: Binary volume
        voxel_size: Voxel spacing
        metrics: Dictionary of computed metrics
        voxel_size_uncertainty: Optional voxel size uncertainty
        segmentation_uncertainty: Optional segmentation uncertainty
        confidence: Confidence level for intervals

    Returns:
        Comprehensive uncertainty analysis results
    """
    results = {}

    # Measurement uncertainty
    results["measurement_uncertainty"] = measurement_uncertainty(
        volume, voxel_size, voxel_size_uncertainty, segmentation_uncertainty
    )

    # Confidence intervals
    uncertainties = {}
    for metric_name in metrics.keys():
        if "volume" in metric_name.lower():
            uncertainties[metric_name] = results["measurement_uncertainty"][
                "volume_uncertainty"
            ]["total"]
        elif "surface" in metric_name.lower():
            uncertainties[metric_name] = results["measurement_uncertainty"][
                "surface_area_uncertainty"
            ]
        else:
            # Default uncertainty (1% relative)
            uncertainties[metric_name] = metrics[metric_name] * 0.01

    results["confidence_intervals"] = confidence_intervals(
        metrics, uncertainties, confidence
    )

    # Uncertainty budget
    results["uncertainty_budget"] = uncertainty_budget(
        metrics,
        {
            "Voxel Size": results["measurement_uncertainty"]["voxel_size_uncertainty"][
                0
            ]
            / voxel_size[0]
            * 100,
            "Segmentation": segmentation_uncertainty or 0.5,
        },
    )

    # Summary
    results["summary"] = {
        "total_volume_uncertainty": results["measurement_uncertainty"][
            "volume_uncertainty"
        ]["total"],
        "relative_volume_uncertainty": results["measurement_uncertainty"][
            "volume_uncertainty"
        ]["relative"],
        "confidence_level": confidence,
        "n_uncertainty_sources": len(results["uncertainty_budget"])
        - 1,  # Exclude total
    }

    return results
