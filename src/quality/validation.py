"""
Validation Framework

Validate analysis results through:
- Comparison with DragonFly software
- Validation against ground truth
- Cross-validation with other tools
- Benchmark datasets
- Accuracy metrics (bias, precision, trueness)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compare_with_dragonfly(
    our_results: Dict[str, Any],
    dragonfly_results: Union[Dict[str, Any], str, Path],
    tolerance: float = 0.01,  # 1% relative tolerance
) -> Dict[str, Any]:
    """
    Compare our analysis results with DragonFly software results.

    Args:
        our_results: Our analysis results dictionary
        dragonfly_results: DragonFly results (dict, CSV path, or JSON path)
        tolerance: Relative tolerance for comparison (default 1%)

    Returns:
        Dictionary with comparison results
    """
    # Load DragonFly results if path provided
    if isinstance(dragonfly_results, (str, Path)):
        dragonfly_path = Path(dragonfly_results)
        if dragonfly_path.suffix.lower() == ".csv":
            dragonfly_data = pd.read_csv(dragonfly_path)
            dragonfly_results = (
                dragonfly_data.to_dict("records")[0] if len(dragonfly_data) > 0 else {}
            )
        elif dragonfly_path.suffix.lower() == ".json":
            import json

            with open(dragonfly_path, "r") as f:
                dragonfly_results = json.load(f)
        else:
            return {"error": f"Unsupported file format: {dragonfly_path.suffix}"}

    # Common metrics to compare
    metrics_to_compare = [
        "volume",
        "surface_area",
        "void_fraction",
        "relative_density",
        "mean_filament_diameter",
        "mean_channel_width",
        "porosity",
    ]

    comparisons = {}
    agreements = []

    for metric in metrics_to_compare:
        # Try different naming conventions
        our_value = None
        dragonfly_value = None

        # Our results
        if metric in our_results:
            our_value = our_results[metric]
        elif f"{metric}_mm3" in our_results:
            our_value = our_results[f"{metric}_mm3"]
        elif f"{metric}_mm2" in our_results:
            our_value = our_results[f"{metric}_mm2"]

        # DragonFly results
        if metric in dragonfly_results:
            dragonfly_value = dragonfly_results[metric]
        elif f"{metric}_mm3" in dragonfly_results:
            dragonfly_value = dragonfly_results[f"{metric}_mm3"]
        elif f"{metric}_mm2" in dragonfly_results:
            dragonfly_value = dragonfly_results[f"{metric}_mm2"]

        if our_value is not None and dragonfly_value is not None:
            # Compute difference
            absolute_diff = abs(our_value - dragonfly_value)
            relative_diff = (
                (absolute_diff / abs(dragonfly_value) * 100)
                if dragonfly_value != 0
                else 0.0
            )

            # Check agreement
            within_tolerance = relative_diff <= (tolerance * 100)

            comparisons[metric] = {
                "our_value": float(our_value),
                "dragonfly_value": float(dragonfly_value),
                "absolute_difference": float(absolute_diff),
                "relative_difference_percent": float(relative_diff),
                "within_tolerance": within_tolerance,
            }

            agreements.append(within_tolerance)

    # Overall agreement
    agreement_percentage = (np.mean(agreements) * 100) if agreements else 0.0

    return {
        "comparisons": comparisons,
        "n_metrics_compared": len(comparisons),
        "n_agreements": sum(agreements),
        "agreement_percentage": float(agreement_percentage),
        "tolerance": tolerance,
        "overall_agreement": agreement_percentage >= 95.0,
    }


def validate_against_ground_truth(
    volume: np.ndarray,
    ground_truth: Union[np.ndarray, Dict[str, float]],
    voxel_size: Tuple[float, float, float],
    metrics_function: Optional[callable] = None,
) -> Dict[str, float]:
    """
    Validate analysis against ground truth (known geometry).

    Args:
        volume: Analyzed volume
        ground_truth: Ground truth volume or dictionary with true values
        voxel_size: Voxel spacing
        metrics_function: Optional function to compute metrics

    Returns:
        Dictionary with validation metrics (bias, precision, trueness, accuracy)
    """
    if metrics_function is None:
        from ..core.metrics import compute_all_metrics

        metrics_function = compute_all_metrics

    # Compute metrics from volume
    computed_metrics = metrics_function(volume, voxel_size)

    if isinstance(ground_truth, np.ndarray):
        # Ground truth as volume - compute metrics
        ground_truth_metrics = metrics_function(ground_truth, voxel_size)
    else:
        # Ground truth as dictionary
        ground_truth_metrics = ground_truth

    # Compute validation metrics
    validation_results = {}

    for metric_name in computed_metrics.keys():
        if metric_name in ground_truth_metrics:
            computed_value = computed_metrics[metric_name]
            true_value = ground_truth_metrics[metric_name]

            # Bias (systematic error)
            bias = computed_value - true_value
            bias_percent = (bias / true_value * 100) if true_value != 0 else 0.0

            # Trueness (closeness to true value)
            trueness = abs(bias)
            trueness_percent = abs(bias_percent)

            # Accuracy (would need multiple measurements for precision)
            # For single measurement, accuracy = trueness
            accuracy = trueness
            accuracy_percent = trueness_percent

            validation_results[metric_name] = {
                "computed": float(computed_value),
                "true": float(true_value),
                "bias": float(bias),
                "bias_percent": float(bias_percent),
                "trueness": float(trueness),
                "trueness_percent": float(trueness_percent),
                "accuracy": float(accuracy),
                "accuracy_percent": float(accuracy_percent),
            }

    # Overall validation score
    if validation_results:
        mean_accuracy = np.mean(
            [r["accuracy_percent"] for r in validation_results.values()]
        )
        validation_results["_overall"] = {
            "mean_accuracy_percent": float(mean_accuracy),
            "validation_grade": (
                "Excellent"
                if mean_accuracy < 1.0
                else (
                    "Good"
                    if mean_accuracy < 5.0
                    else "Acceptable" if mean_accuracy < 10.0 else "Needs Improvement"
                )
            ),
        }

    return validation_results


def cross_validate_tools(
    results_dict: Dict[str, Dict[str, Any]], reference_tool: str = "dragonfly"
) -> pd.DataFrame:
    """
    Cross-validate results from multiple analysis tools.

    Args:
        results_dict: Dictionary with tool names as keys and results as values
        reference_tool: Name of reference tool for comparison

    Returns:
        DataFrame with cross-validation results
    """
    if reference_tool not in results_dict:
        logger.warning(f"Reference tool '{reference_tool}' not found, using first tool")
        reference_tool = list(results_dict.keys())[0]

    reference_results = results_dict[reference_tool]

    comparison_data = []

    for tool_name, tool_results in results_dict.items():
        if tool_name == reference_tool:
            continue

        # Compare with reference
        comparison = compare_with_dragonfly(tool_results, reference_results)

        for metric, comp_data in comparison.get("comparisons", {}).items():
            comparison_data.append(
                {
                    "tool": tool_name,
                    "metric": metric,
                    "value": comp_data["our_value"],
                    "reference_value": comp_data["dragonfly_value"],
                    "relative_difference_percent": comp_data[
                        "relative_difference_percent"
                    ],
                    "within_tolerance": comp_data["within_tolerance"],
                }
            )

    df = pd.DataFrame(comparison_data)

    return df


def benchmark_analysis(
    volume: np.ndarray,
    benchmark_specs: Dict[str, Any],
    voxel_size: Tuple[float, float, float],
    analysis_function: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Run benchmark analysis against known specifications.

    Args:
        volume: Volume to analyze
        benchmark_specs: Benchmark specifications with expected values
        voxel_size: Voxel spacing
        analysis_function: Optional custom analysis function

    Returns:
        Dictionary with benchmark results
    """
    if analysis_function is None:
        from ..core.metrics import compute_all_metrics

        analysis_function = compute_all_metrics

    # Run analysis
    results = analysis_function(volume, voxel_size)

    # Compare with benchmark
    benchmark_results = {}
    passed_tests = []

    for metric_name, expected_value in benchmark_specs.items():
        if metric_name in results:
            computed_value = results[metric_name]

            # Check if within tolerance
            if isinstance(expected_value, dict):
                # Expected value with tolerance
                expected = expected_value.get("value", 0)
                tolerance = expected_value.get("tolerance", 0.01)  # 1% default
                tolerance_type = expected_value.get("type", "relative")
            else:
                # Simple expected value (1% tolerance)
                expected = expected_value
                tolerance = 0.01
                tolerance_type = "relative"

            if tolerance_type == "relative":
                diff = (
                    abs(computed_value - expected) / abs(expected)
                    if expected != 0
                    else abs(computed_value)
                )
                passed = diff <= tolerance
            else:  # absolute
                diff = abs(computed_value - expected)
                passed = diff <= tolerance

            benchmark_results[metric_name] = {
                "computed": float(computed_value),
                "expected": float(expected),
                "difference": float(abs(computed_value - expected)),
                "relative_difference_percent": float(
                    (abs(computed_value - expected) / abs(expected) * 100)
                    if expected != 0
                    else 0.0
                ),
                "passed": passed,
            }

            passed_tests.append(passed)

    # Overall benchmark score
    pass_rate = (np.mean(passed_tests) * 100) if passed_tests else 0.0

    return {
        "benchmark_results": benchmark_results,
        "n_tests": len(benchmark_results),
        "n_passed": sum(passed_tests),
        "pass_rate": float(pass_rate),
        "benchmark_grade": "Pass" if pass_rate >= 95.0 else "Fail",
    }


def compute_accuracy_metrics(
    measured_values: np.ndarray, true_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute accuracy metrics: bias, precision, trueness, accuracy.
    Also computes Dice coefficient and Jaccard index for binary volumes.

    Args:
        measured_values: Array of measured values or binary volume
        true_values: Array of true/reference values or binary volume

    Returns:
        Dictionary with accuracy metrics
    """
    # Handle binary volumes (for Dice/Jaccard)
    if (
        measured_values.ndim == 3
        and true_values.ndim == 3
        and measured_values.dtype in [np.uint8, bool]
        and true_values.dtype in [np.uint8, bool]
    ):
        # Binary volumes - compute Dice and Jaccard
        seg_binary = (measured_values > 0).astype(bool)
        gt_binary = (true_values > 0).astype(bool)

        intersection = np.sum(seg_binary & gt_binary)
        union = np.sum(seg_binary | gt_binary)
        seg_sum = np.sum(seg_binary)
        gt_sum = np.sum(gt_binary)

        dice_coefficient = (
            (2.0 * intersection) / (seg_sum + gt_sum) if (seg_sum + gt_sum) > 0 else 0.0
        )
        jaccard_index = intersection / union if union > 0 else 0.0

        # Also compute pixel-wise accuracy
        accuracy = (
            np.sum(seg_binary == gt_binary) / seg_binary.size
            if seg_binary.size > 0
            else 0.0
        )

        return {
            "dice_coefficient": float(dice_coefficient),
            "jaccard_index": float(jaccard_index),
            "accuracy": float(accuracy),
            "intersection": int(intersection),
            "union": int(union),
        }

    # Handle arrays of values
    if len(measured_values) != len(true_values):
        raise ValueError("Measured and true values must have same length")

    errors = measured_values - true_values

    # Bias (systematic error) - mean error
    bias = float(np.mean(errors))
    bias_percent = float(
        (bias / np.mean(true_values) * 100) if np.mean(true_values) != 0 else 0.0
    )

    # Precision (random error) - standard deviation of errors
    precision = float(np.std(errors))
    precision_percent = float(
        (precision / np.mean(true_values) * 100) if np.mean(true_values) != 0 else 0.0
    )

    # Trueness (closeness to true value) - mean absolute error
    trueness = float(np.mean(np.abs(errors)))
    trueness_percent = float(
        (trueness / np.mean(true_values) * 100) if np.mean(true_values) != 0 else 0.0
    )

    # Accuracy (RMSE)
    accuracy = float(np.sqrt(np.mean(errors**2)))
    accuracy_percent = float(
        (accuracy / np.mean(true_values) * 100) if np.mean(true_values) != 0 else 0.0
    )

    return {
        "bias": bias,
        "bias_percent": bias_percent,
        "precision": precision,
        "precision_percent": precision_percent,
        "trueness": trueness,
        "trueness_percent": trueness_percent,
        "accuracy": accuracy,
        "accuracy_percent": accuracy_percent,
        "rmse": accuracy,
        "mae": trueness,
    }


def validate_segmentation(
    segmented_volume: np.ndarray, ground_truth_segmentation: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Validate segmentation against ground truth using common metrics, or perform basic validation.

    Args:
        segmented_volume: Our segmentation (binary)
        ground_truth_segmentation: Optional ground truth segmentation (binary)

    Returns:
        Dictionary with segmentation validation metrics
    """
    seg_binary = (segmented_volume > 0).astype(bool)

    # If no ground truth provided, do basic validation checks
    if ground_truth_segmentation is None:
        checks = {
            "has_material": bool(np.any(seg_binary)),
            "has_void": bool(np.any(~seg_binary)),
            "is_binary": bool(
                np.all((segmented_volume == 0) | (segmented_volume == 1))
            ),
            "volume_fraction": float(np.mean(seg_binary)),
            "is_empty": not bool(np.any(seg_binary)),
            "is_full": bool(np.all(seg_binary)),
        }

        is_valid = (
            checks["has_material"]
            and checks["has_void"]
            and checks["is_binary"]
            and not checks["is_empty"]
            and not checks["is_full"]
        )

        return {"is_valid": is_valid, "checks": checks}

    # Compare with ground truth
    gt_binary = (ground_truth_segmentation > 0).astype(bool)

    # True positives, false positives, false negatives, true negatives
    tp = np.sum(seg_binary & gt_binary)
    fp = np.sum(seg_binary & ~gt_binary)
    fn = np.sum(~seg_binary & gt_binary)
    tn = np.sum(~seg_binary & ~gt_binary)

    # Metrics
    n_total = seg_binary.size

    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Jaccard index (IoU)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score),
        "dice_coefficient": float(dice),
        "jaccard_index": float(iou),
        "iou": float(iou),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
