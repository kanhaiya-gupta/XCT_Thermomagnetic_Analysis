"""
XCT Thermomagnetic Analysis Package

A modular framework for analyzing X-ray Computed Tomography (XCT) images
of 3D-printed thermomagnetic elements.

Package Structure:
- core: Core analysis modules (segmentation, morphology, metrics, etc.)
- preprocessing: Data preprocessing and statistics
- analysis: Advanced analysis (sensitivity, virtual experiments, etc.)
- quality: Quality control and validation
- experimental: Experiment-specific analysis (flow, thermal, energy)
- integration: External tool integration (DragonFly)
- utils: Utility functions
"""

__version__ = "1.0.0"

# Main analyzer class
from .analyzer import XCTAnalyzer

# Core modules - direct imports for convenience
from .core.segmentation import otsu_threshold, segment_volume
from .core.morphology import (
    erode,
    dilate,
    open_operation,
    close_operation,
    remove_small_objects,
    fill_holes,
    skeletonize,
    distance_transform,
)
from .core.metrics import (
    compute_all_metrics,
    compute_volume,
    compute_surface_area,
    compute_void_fraction,
    compute_relative_density,
    compute_specific_surface_area,
)
from .core.filament_analysis import (
    estimate_filament_diameter,
    estimate_channel_width,
    analyze_cross_section,
)
from .core.porosity import (
    analyze_porosity_distribution,
    porosity_along_direction,
    local_porosity_map,
    pore_size_distribution,
)
from .core.slice_analysis import (
    analyze_slice_along_flow,
    analyze_slice_perpendicular_flow,
)
from .core.visualization import (
    visualize_3d_volume,
    visualize_slice,
    plot_porosity_profile,
    plot_metrics_comparison,
    create_analysis_report,
    publication_quality_plot,
    multi_panel_figure,
    export_3d_for_publication,
    create_figure_caption,
    apply_publication_style,
    JOURNAL_STYLES,
)

# Preprocessing modules
from .preprocessing.preprocessing import (
    filter_by_volume,
    filter_by_sphericity,
    filter_by_spatial_bounds,
    filter_by_aspect_ratio,
    remove_edge_objects,
    apply_filters,
    analyze_object_properties,
    compute_sphericity,
    compute_aspect_ratio,
)
from .preprocessing.statistics import (
    fit_gaussian,
    fit_poisson,
    fit_linear,
    fit_quadratic,
    fit_distribution,
    compare_fits,
    evaluate_fit_quality,
    generate_fit_samples,
    predict_from_fit,
    non_parametric_tests,
    spatial_autocorrelation,
    principal_component_analysis,
    bayesian_uncertainty,
    time_series_analysis,
)

# Analysis modules
from .analysis.sensitivity_analysis import (
    parameter_sweep,
    local_sensitivity,
    morris_screening,
    sobol_indices,
    uncertainty_propagation,
    analyze_segmentation_sensitivity,
)
from .analysis.virtual_experiments import (
    full_factorial_design,
    latin_hypercube_sampling,
    central_composite_design,
    box_behnken_design,
    run_virtual_experiment,
    fit_response_surface,
    optimize_process_parameters,
    multi_objective_optimization,
)
from .analysis.comparative_analysis import (
    compare_samples,
    statistical_tests,
    process_structure_property,
    batch_analyze,
)
from .analysis.performance_analysis import (
    estimate_heat_transfer_efficiency,
    estimate_magnetic_property_impact,
    estimate_mechanical_property_impact,
    process_structure_performance_relationship,
    optimize_for_performance,
)

# Quality modules
from .quality.dimensional_accuracy import (
    compute_geometric_deviations,
    analyze_tolerance_compliance,
    surface_deviation_map,
    build_orientation_effects,
    compare_to_cad,
    comprehensive_dimensional_analysis,
)
from .quality.uncertainty_analysis import (
    measurement_uncertainty,
    segmentation_uncertainty,
    confidence_intervals,
    uncertainty_budget,
    monte_carlo_uncertainty,
    systematic_vs_random_errors,
    comprehensive_uncertainty_analysis,
)
from .quality.reproducibility import (
    AnalysisConfig,
    ProvenanceTracker,
    SeedManager,
    track_analysis_provenance,
    export_reproducibility_package,
    save_analysis_config,
    load_analysis_config,
)
from .quality.validation import (
    compare_with_dragonfly,
    validate_against_ground_truth,
    cross_validate_tools,
    benchmark_analysis,
    compute_accuracy_metrics,
    validate_segmentation,
)

# Experimental modules
from .experimental.flow_analysis import (
    analyze_flow_connectivity,
    identify_flow_paths,
    detect_dead_end_channels,
    compute_tortuosity,
    compute_flow_path_lengths,
    analyze_flow_branching,
    estimate_flow_resistance,
    calculate_pressure_drop,
    compute_hydraulic_diameter,
    estimate_reynolds_number,
    analyze_flow_distribution,
    compute_flow_uniformity,
    detect_flow_maldistribution,
    comprehensive_flow_analysis,
)
from .experimental.thermal_analysis import (
    compute_thermal_resistance,
    analyze_conduction_resistance,
    analyze_convection_resistance,
    estimate_heat_transfer_coefficient,
    estimate_temperature_gradient,
    comprehensive_thermal_analysis,
)
from .experimental.energy_conversion import (
    estimate_power_output,
    calculate_energy_conversion_efficiency,
    analyze_temperature_dependent_performance,
    compute_power_density,
    estimate_cycle_efficiency,
    comprehensive_energy_conversion_analysis,
)

# Integration modules
from .integration.dragonfly_integration import (
    import_dragonfly_volume,
    export_to_dragonfly_volume,
    import_dragonfly_segmentation,
    export_segmentation_to_dragonfly,
    import_dragonfly_results,
    export_results_to_dragonfly,
    create_dragonfly_project_file,
    convert_to_dragonfly_workflow,
    comprehensive_dragonfly_integration,
)

# Utility functions
from .utils.utils import (
    load_volume,
    save_volume,
    load_segmented_data,
    save_segmented_data,
    convert_to_mm,
    convert_from_mm,
    normalize_units,
    parse_voxel_size_with_unit,
    create_output_directory,
    get_voxel_volume,
    normalize_volume,
)

__all__ = [
    # Main class
    "XCTAnalyzer",
    # Core - Segmentation
    "otsu_threshold",
    "segment_volume",
    # Core - Morphology
    "erode",
    "dilate",
    "open_operation",
    "close_operation",
    "remove_small_objects",
    "fill_holes",
    "skeletonize",
    "distance_transform",
    # Core - Metrics
    "compute_all_metrics",
    "compute_volume",
    "compute_surface_area",
    "compute_void_fraction",
    "compute_relative_density",
    "compute_specific_surface_area",
    # Core - Filament Analysis
    "estimate_filament_diameter",
    "estimate_channel_width",
    "analyze_cross_section",
    # Core - Porosity
    "analyze_porosity_distribution",
    "porosity_along_direction",
    "local_porosity_map",
    "pore_size_distribution",
    # Core - Slice Analysis
    "analyze_slice_along_flow",
    "analyze_slice_perpendicular_flow",
    # Core - Visualization
    "visualize_3d_volume",
    "visualize_slice",
    "plot_porosity_profile",
    "plot_metrics_comparison",
    "create_analysis_report",
    "publication_quality_plot",
    "multi_panel_figure",
    "export_3d_for_publication",
    "create_figure_caption",
    "apply_publication_style",
    "JOURNAL_STYLES",
    # Preprocessing
    "filter_by_volume",
    "filter_by_sphericity",
    "filter_by_spatial_bounds",
    "filter_by_aspect_ratio",
    "remove_edge_objects",
    "apply_filters",
    "analyze_object_properties",
    "compute_sphericity",
    "compute_aspect_ratio",
    # Statistics
    "fit_gaussian",
    "fit_poisson",
    "fit_linear",
    "fit_quadratic",
    "fit_distribution",
    "compare_fits",
    "evaluate_fit_quality",
    "generate_fit_samples",
    "predict_from_fit",
    "non_parametric_tests",
    "spatial_autocorrelation",
    "principal_component_analysis",
    "bayesian_uncertainty",
    "time_series_analysis",
    # Sensitivity Analysis
    "parameter_sweep",
    "local_sensitivity",
    "morris_screening",
    "sobol_indices",
    "uncertainty_propagation",
    "analyze_segmentation_sensitivity",
    # Virtual Experiments
    "full_factorial_design",
    "latin_hypercube_sampling",
    "central_composite_design",
    "box_behnken_design",
    "run_virtual_experiment",
    "fit_response_surface",
    "optimize_process_parameters",
    "multi_objective_optimization",
    # Comparative Analysis
    "compare_samples",
    "statistical_tests",
    "process_structure_property",
    "batch_analyze",
    # Performance Analysis
    "estimate_heat_transfer_efficiency",
    "estimate_magnetic_property_impact",
    "estimate_mechanical_property_impact",
    "process_structure_performance_relationship",
    "optimize_for_performance",
    # Dimensional Accuracy
    "compute_geometric_deviations",
    "analyze_tolerance_compliance",
    "surface_deviation_map",
    "build_orientation_effects",
    "compare_to_cad",
    "comprehensive_dimensional_analysis",
    # Uncertainty Analysis
    "measurement_uncertainty",
    "segmentation_uncertainty",
    "confidence_intervals",
    "uncertainty_budget",
    "monte_carlo_uncertainty",
    "systematic_vs_random_errors",
    "comprehensive_uncertainty_analysis",
    # Reproducibility
    "AnalysisConfig",
    "ProvenanceTracker",
    "SeedManager",
    "track_analysis_provenance",
    "export_reproducibility_package",
    "save_analysis_config",
    "load_analysis_config",
    # Validation
    "compare_with_dragonfly",
    "validate_against_ground_truth",
    "cross_validate_tools",
    "benchmark_analysis",
    "compute_accuracy_metrics",
    "validate_segmentation",
    # Flow Analysis
    "analyze_flow_connectivity",
    "identify_flow_paths",
    "detect_dead_end_channels",
    "compute_tortuosity",
    "compute_flow_path_lengths",
    "analyze_flow_branching",
    "estimate_flow_resistance",
    "calculate_pressure_drop",
    "compute_hydraulic_diameter",
    "estimate_reynolds_number",
    "analyze_flow_distribution",
    "compute_flow_uniformity",
    "detect_flow_maldistribution",
    "comprehensive_flow_analysis",
    # Thermal Analysis
    "compute_thermal_resistance",
    "analyze_conduction_resistance",
    "analyze_convection_resistance",
    "estimate_heat_transfer_coefficient",
    "estimate_temperature_gradient",
    "comprehensive_thermal_analysis",
    # Energy Conversion
    "estimate_power_output",
    "calculate_energy_conversion_efficiency",
    "analyze_temperature_dependent_performance",
    "compute_power_density",
    "estimate_cycle_efficiency",
    "comprehensive_energy_conversion_analysis",
    # DragonFly Integration
    "import_dragonfly_volume",
    "export_to_dragonfly_volume",
    "import_dragonfly_segmentation",
    "export_segmentation_to_dragonfly",
    "import_dragonfly_results",
    "export_results_to_dragonfly",
    "create_dragonfly_project_file",
    "convert_to_dragonfly_workflow",
    "comprehensive_dragonfly_integration",
    # Utilities
    "load_volume",
    "save_volume",
    "load_segmented_data",
    "save_segmented_data",
    "convert_to_mm",
    "convert_from_mm",
    "normalize_units",
    "parse_voxel_size_with_unit",
    "create_output_directory",
    "get_voxel_volume",
    "normalize_volume",
]
