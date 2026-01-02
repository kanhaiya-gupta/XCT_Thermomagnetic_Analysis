"""
Comparative Analysis Module

Compare multiple XCT samples:
- Batch processing
- Statistical comparison
- Process-structure-property relationships
- Quality control analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compare_samples(
    sample_results: List[Dict[str, Any]], metric_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compare metrics across multiple samples.

    Args:
        sample_results: List of result dictionaries from analysis
        metric_names: List of metric names to compare

    Returns:
        Dictionary with comparison statistics
    """
    if metric_names is None:
        # Extract all metric names from first sample
        if len(sample_results) > 0:
            metric_names = list(sample_results[0].keys())
        else:
            return {"error": "No samples provided"}

    comparison = {}

    for metric_name in metric_names:
        values = []
        for sample in sample_results:
            value = sample.get(metric_name, np.nan)
            if np.isfinite(value):
                values.append(value)

        if len(values) > 0:
            values = np.array(values)
            comparison[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "cv": (
                    float(np.std(values) / np.mean(values))
                    if np.mean(values) != 0
                    else 0.0
                ),  # Coefficient of variation
                "n_samples": len(values),
            }
        else:
            comparison[metric_name] = {"mean": np.nan, "std": np.nan, "n_samples": 0}

    return comparison


def statistical_tests(
    sample_groups: Dict[str, List[float]], test_type: str = "anova"
) -> Dict[str, Any]:
    """
    Perform statistical tests to compare groups.

    Args:
        sample_groups: Dictionary of group names and value lists
        test_type: Type of test ('anova', 'ttest', 'mannwhitney')

    Returns:
        Dictionary with test results
    """
    groups = list(sample_groups.keys())
    group_values = [sample_groups[g] for g in groups]

    if test_type == "anova":
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_values)
        return {
            "test_type": "anova",
            "test": "one_way_anova",
            "statistic": float(f_stat),
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "groups": groups,
        }
    elif test_type == "ttest" and len(groups) == 2:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(group_values[0], group_values[1])
        return {
            "test_type": "ttest",
            "test": "independent_ttest",
            "statistic": float(t_stat),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "groups": groups,
        }
    elif test_type == "mannwhitney" and len(groups) == 2:
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group_values[0], group_values[1])
        return {
            "test_type": "mannwhitney",
            "test": "mann_whitney_u",
            "statistic": float(u_stat),
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "groups": groups,
        }
    else:
        return {
            "error": f"Test type {test_type} not supported for {len(groups)} groups"
        }


def process_structure_property(
    process_params: pd.DataFrame,
    structure_metrics: pd.DataFrame,
    property_metrics: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Analyze process-structure-property relationships.

    Args:
        process_params: DataFrame with process parameters
        structure_metrics: DataFrame with structure metrics (from XCT)
        property_metrics: Optional DataFrame with property measurements

    Returns:
        Dictionary with correlation analysis
    """
    # Merge data
    if property_metrics is not None:
        data = pd.merge(
            process_params, structure_metrics, left_index=True, right_index=True
        )
        data = pd.merge(data, property_metrics, left_index=True, right_index=True)
    else:
        data = pd.merge(
            process_params, structure_metrics, left_index=True, right_index=True
        )

    # Calculate correlations
    process_cols = process_params.columns.tolist()
    structure_cols = structure_metrics.columns.tolist()

    correlations = {}

    for proc_col in process_cols:
        for struct_col in structure_cols:
            if proc_col in data.columns and struct_col in data.columns:
                corr = data[proc_col].corr(data[struct_col])
                if not np.isnan(corr):
                    correlations[f"{proc_col} -> {struct_col}"] = float(corr)

    # If property metrics available, also correlate
    if property_metrics is not None:
        property_cols = property_metrics.columns.tolist()
        for struct_col in structure_cols:
            for prop_col in property_cols:
                if struct_col in data.columns and prop_col in data.columns:
                    corr = data[struct_col].corr(data[prop_col])
                    if not np.isnan(corr):
                        correlations[f"{struct_col} -> {prop_col}"] = float(corr)

    return {
        "correlations": correlations,
        "n_samples": len(data),
        "process_parameters": process_cols,
        "structure_metrics": structure_cols,
        "property_metrics": property_cols if property_metrics is not None else [],
    }


def batch_analyze(
    volume_paths: List[str],
    voxel_sizes: List[Tuple[float, float, float]],
    analysis_config: Dict[str, Any],
    sample_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Batch analyze multiple XCT volumes.

    Args:
        volume_paths: List of file paths
        voxel_sizes: List of voxel sizes (one per volume)
        analysis_config: Configuration for analysis
        sample_names: Optional names for samples

    Returns:
        DataFrame with results for all samples
    """
    from ..analyzer import XCTAnalyzer

    if sample_names is None:
        sample_names = [f"Sample_{i+1}" for i in range(len(volume_paths))]

    results = []

    for i, (path, voxel_size, name) in enumerate(
        zip(volume_paths, voxel_sizes, sample_names)
    ):
        try:
            logger.info(f"Analyzing {name} ({i+1}/{len(volume_paths)})")

            # Initialize analyzer
            analyzer = XCTAnalyzer(voxel_size=voxel_size, target_unit="mm")
            analyzer.load_volume(path, normalize=True)

            # Segment
            segmented = analyzer.segment(
                method=analysis_config.get("segmentation_method", "otsu"),
                refine=analysis_config.get("refine", True),
            )

            # Compute metrics
            metrics = analyzer.compute_metrics()

            # Add sample name
            metrics["sample_name"] = name
            metrics["file_path"] = path

            results.append(metrics)
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}")
            results.append({"sample_name": name, "file_path": path, "error": str(e)})

    return pd.DataFrame(results)
