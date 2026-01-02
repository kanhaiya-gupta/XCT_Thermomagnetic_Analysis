# Module Reference

Complete documentation of all modules in the XCT Thermomagnetic Analysis Framework.

## Core Modules

### `core.segmentation`

Image segmentation using various thresholding methods.

**Functions:**
- `otsu_threshold(volume, return_threshold=False)` - Otsu's automatic thresholding
- `multi_threshold_segmentation(volume, n_classes=3)` - Multi-class segmentation
- `adaptive_threshold(volume, block_size=11)` - Adaptive thresholding
- `segment_volume(volume, method='otsu', **kwargs)` - Main segmentation interface

**Example:**
```python
from src.core.segmentation import otsu_threshold

segmented = otsu_threshold(volume)
```

### `core.morphology`

Morphological operations for cleaning and refining segmented volumes.

**Functions:**
- `erode(volume, kernel_size=3, iterations=1)` - Erosion
- `dilate(volume, kernel_size=3, iterations=1)` - Dilation
- `opening(volume, kernel_size=3)` - Opening (erosion + dilation)
- `closing(volume, kernel_size=3)` - Closing (dilation + erosion)
- `remove_small_objects(volume, min_size=100)` - Remove small objects
- `fill_holes(volume)` - Fill internal holes
- `skeletonize(volume)` - Skeletonization

**Example:**
```python
from src.core.morphology import remove_small_objects, fill_holes

cleaned = remove_small_objects(segmented, min_size=100)
filled = fill_holes(cleaned)
```

### `core.metrics`

Scalar quantity estimation from segmented volumes.

**Functions:**
- `compute_all_metrics(volume, voxel_size)` - Compute all metrics
- `compute_volume(volume, voxel_size)` - Volume calculation
- `compute_surface_area(volume, voxel_size)` - Surface area
- `compute_void_fraction(volume)` - Void fraction
- `compute_relative_density(volume)` - Relative density
- `compute_specific_surface_area(volume, voxel_size)` - Specific surface area

**Example:**
```python
from src.core.metrics import compute_all_metrics

metrics = compute_all_metrics(segmented, (0.1, 0.1, 0.1))
print(f"Void fraction: {metrics['void_fraction']:.3f}")
```

### `core.filament_analysis`

Filament diameter and channel width estimation.

**Functions:**
- `estimate_filament_diameter(volume, voxel_size, direction='z')` - Filament diameter
- `estimate_channel_width(volume, voxel_size, direction='z')` - Channel width
- `analyze_cross_section(slice_2d, pixel_size)` - Cross-section analysis

**Example:**
```python
from src.core.filament_analysis import estimate_filament_diameter

result = estimate_filament_diameter(segmented, (0.1, 0.1, 0.1))
print(f"Mean diameter: {result['mean_diameter']:.2f} mm")
```

### `core.porosity`

Porosity distribution analysis along printing direction.

**Functions:**
- `analyze_porosity_distribution(volume, direction='z')` - Comprehensive porosity analysis
- `porosity_along_direction(volume, direction='z')` - Porosity profile
- `local_porosity_map(volume, window_size=10)` - Local porosity map
- `pore_size_distribution(volume, voxel_size)` - Pore size distribution

**Example:**
```python
from src.core.porosity import porosity_along_direction

profile = porosity_along_direction(segmented, direction='z')
```

### `core.slice_analysis`

Slice analysis along and perpendicular to water-flow direction.

**Functions:**
- `analyze_slice_along_flow(volume, flow_direction='z')` - Analysis along flow
- `analyze_slice_perpendicular_flow(volume, flow_direction='z')` - Perpendicular analysis

**Example:**
```python
from src.core.slice_analysis import analyze_slice_along_flow

results = analyze_slice_along_flow(segmented, flow_direction='z')
```

### `core.visualization`

3D visualization and plotting utilities.

**Functions:**
- `visualize_3d_volume(volume, **kwargs)` - 3D volume visualization
- `visualize_slice(volume, slice_idx, axis=0)` - Slice visualization
- `plot_porosity_profile(profile_data)` - Porosity profile plot
- `plot_metrics_comparison(metrics_dict)` - Metrics comparison
- `publication_quality_plot(fig, style='nature')` - Publication-quality plots

**Example:**
```python
from src.core.visualization import visualize_3d_volume

visualize_3d_volume(segmented)
```

## Preprocessing Modules

### `preprocessing.preprocessing`

Data filtering and object property analysis.

**Functions:**
- `filter_by_volume(volume, voxel_size, min_volume, max_volume)` - Volume filter
- `filter_by_sphericity(volume, voxel_size, min_sphericity, max_sphericity)` - Sphericity filter
- `filter_by_spatial_bounds(volume, voxel_size, x_range, y_range, z_range)` - Spatial filter
- `filter_by_aspect_ratio(volume, voxel_size, max_aspect_ratio)` - Aspect ratio filter
- `remove_edge_objects(volume, margin=5)` - Remove edge objects
- `apply_filters(volume, voxel_size, filters)` - Apply multiple filters
- `analyze_object_properties(volume, voxel_size)` - Object property analysis

**Example:**
```python
from src.preprocessing.preprocessing import filter_by_volume, remove_edge_objects

cleaned = remove_edge_objects(segmented, margin=5)
filtered = filter_by_volume(cleaned, (0.1, 0.1, 0.1), min_volume=0.1)
```

### `preprocessing.statistics`

Statistical fitting and analysis.

**Functions:**
- `fit_gaussian(data)` - Gaussian distribution fitting
- `fit_poisson(data)` - Poisson distribution fitting
- `fit_linear(x, y)` - Linear regression
- `fit_quadratic(x, y)` - Quadratic regression
- `compare_fits(fits)` - Compare multiple fits
- `evaluate_fit_quality(fit, data)` - Fit quality evaluation

**Example:**
```python
from src.preprocessing.statistics import fit_gaussian

fit = fit_gaussian(pore_sizes)
print(f"Mean: {fit['mean']:.2f}, Std: {fit['std']:.2f}")
```

## Analysis Modules

### `analysis.sensitivity_analysis`

Parameter sensitivity analysis.

**Functions:**
- `parameter_sweep(base_volume, parameters, metric_function)` - Parameter sweep
- `local_sensitivity(volume, parameter, metric_function)` - Local sensitivity
- `morris_screening(volume, parameters, metric_function)` - Morris screening
- `sobol_indices(volume, parameters, metric_function)` - Sobol indices
- `uncertainty_propagation(volume, uncertainties)` - Uncertainty propagation

**Example:**
```python
from src.analysis.sensitivity_analysis import parameter_sweep

results = parameter_sweep(
    base_volume=volume,
    parameters={'threshold': [0.3, 0.4, 0.5]},
    metric_function=compute_all_metrics
)
```

### `analysis.virtual_experiments`

Design of Experiments and optimization.

**Functions:**
- `full_factorial_design(parameters)` - Full factorial design
- `latin_hypercube_sampling(parameters, n_samples)` - LHS sampling
- `central_composite_design(parameters)` - Central composite design
- `box_behnken_design(parameters)` - Box-Behnken design
- `fit_response_surface(data, model_type='polynomial')` - Response surface fitting
- `optimize_process_parameters(response_surface, objective)` - Parameter optimization

**Example:**
```python
from src.analysis.virtual_experiments import latin_hypercube_sampling

design = latin_hypercube_sampling(
    parameters={'temp': (200, 300), 'speed': (10, 50)},
    n_samples=50
)
```

### `analysis.comparative_analysis`

Batch processing and statistical comparison.

**Functions:**
- `compare_samples(samples, metrics)` - Compare multiple samples
- `statistical_tests(data1, data2, test_type='t-test')` - Statistical tests
- `batch_analyze(volume_paths, voxel_size)` - Batch analysis
- `process_structure_property(process_params, structure_metrics)` - PSP analysis

**Example:**
```python
from src.analysis.comparative_analysis import batch_analyze

results = batch_analyze(
    volume_paths=['data/sample1.dcm', 'data/sample2.dcm'],
    voxel_size=(0.1, 0.1, 0.1)
)
```

### `analysis.performance_analysis`

Process-Structure-Performance relationships.

**Functions:**
- `estimate_heat_transfer_efficiency(structure_metrics)` - Heat transfer efficiency
- `estimate_magnetic_property_impact(structure_metrics)` - Magnetic property impact
- `estimate_mechanical_property_impact(structure_metrics)` - Mechanical property impact
- `process_structure_performance_relationship(process, structure, performance)` - PSP relationships

**Example:**
```python
from src.analysis.performance_analysis import estimate_heat_transfer_efficiency

efficiency = estimate_heat_transfer_efficiency(metrics)
```

## Quality Modules

### `quality.dimensional_accuracy`

Dimensional accuracy and tolerance analysis.

**Functions:**
- `compute_geometric_deviations(volume, cad_model, voxel_size)` - Geometric deviations
- `analyze_tolerance_compliance(deviations, tolerances)` - Tolerance compliance
- `surface_deviation_map(volume, cad_model)` - Surface deviation map
- `compare_to_cad(volume, cad_model, voxel_size)` - CAD comparison

**Example:**
```python
from src.quality.dimensional_accuracy import compare_to_cad

results = compare_to_cad(segmented, cad_model, (0.1, 0.1, 0.1))
```

### `quality.uncertainty_analysis`

Uncertainty quantification.

**Functions:**
- `measurement_uncertainty(measurements, uncertainties)` - Measurement uncertainty
- `segmentation_uncertainty(volume, n_iterations=100)` - Segmentation uncertainty
- `confidence_intervals(data, confidence=0.95)` - Confidence intervals
- `monte_carlo_uncertainty(analysis_function, uncertainties, n_samples=1000)` - Monte Carlo uncertainty

**Example:**
```python
from src.quality.uncertainty_analysis import confidence_intervals

ci = confidence_intervals(measurements, confidence=0.95)
```

### `quality.validation`

Validation against ground truth and other tools.

**Functions:**
- `validate_against_ground_truth(results, ground_truth)` - Ground truth validation
- `compare_with_dragonfly(results, dragonfly_results)` - DragonFly comparison
- `compute_accuracy_metrics(predicted, actual)` - Accuracy metrics

**Example:**
```python
from src.quality.validation import compare_with_dragonfly

comparison = compare_with_dragonfly(our_results, dragonfly_results)
```

## Experimental Modules

### `experimental.flow_analysis`

Flow path connectivity, tortuosity, and resistance analysis.

**Functions:**
- `analyze_flow_connectivity(volume, flow_direction='z')` - Flow connectivity
- `compute_tortuosity(volume, flow_direction='z')` - Tortuosity
- `estimate_flow_resistance(volume, channel_geometry)` - Flow resistance
- `analyze_flow_distribution(volume, flow_direction='z')` - Flow distribution

**Example:**
```python
from src.experimental.flow_analysis import compute_tortuosity

result = compute_tortuosity(segmented, flow_direction='z')
print(f"Tortuosity: {result['tortuosity']:.3f}")
```

### `experimental.thermal_analysis`

Thermal resistance and heat transfer analysis.

**Functions:**
- `compute_thermal_resistance(volume, material_properties)` - Thermal resistance
- `estimate_heat_transfer_coefficient(volume, flow_conditions)` - Heat transfer coefficient
- `estimate_temperature_gradient(volume, heat_flux)` - Temperature gradient

**Example:**
```python
from src.experimental.thermal_analysis import compute_thermal_resistance

resistance = compute_thermal_resistance(
    segmented,
    material_properties={'thermal_conductivity': 50.0}
)
```

### `experimental.energy_conversion`

Energy conversion efficiency and power output.

**Functions:**
- `estimate_power_output(volume, conditions)` - Power output estimation
- `calculate_energy_conversion_efficiency(volume, conditions)` - Efficiency calculation
- `analyze_temperature_dependent_performance(volume, temperature_range)` - Temperature-dependent analysis

**Example:**
```python
from src.experimental.energy_conversion import estimate_power_output

power = estimate_power_output(segmented, flow_conditions)
```

## Integration Modules

### `integration.dragonfly_integration`

DragonFly software integration.

**Functions:**
- `import_dragonfly_volume(file_path)` - Import DragonFly volume
- `export_to_dragonfly_volume(volume, output_path)` - Export to DragonFly
- `import_dragonfly_segmentation(file_path)` - Import segmentation
- `export_segmentation_to_dragonfly(volume, output_path)` - Export segmentation

**Example:**
```python
from src.integration.dragonfly_integration import import_dragonfly_volume

volume, metadata = import_dragonfly_volume('dragonfly_data.raw')
```

## Utils

### `utils.utils`

Utility functions for I/O and data conversion.

**Functions:**
- `load_volume(file_path, file_format='auto')` - Load volume from file
- `save_volume(volume, file_path, file_format='auto')` - Save volume to file
- `load_segmented_data(file_path)` - Load segmented data CSV
- `save_segmented_data(volume, file_path, metadata)` - Save segmented data
- `convert_to_mm(value, unit)` - Convert to millimeters
- `normalize_units(data, target_unit='mm')` - Normalize units

**Example:**
```python
from src.utils.utils import load_volume, convert_to_mm

volume, metadata = load_volume('data/sample.dcm')
voxel_size_mm = convert_to_mm(metadata['voxel_size'], metadata['unit'])
```

