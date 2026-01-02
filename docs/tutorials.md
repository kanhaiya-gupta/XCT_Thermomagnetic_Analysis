# Tutorials

Step-by-step tutorials for using the XCT Thermomagnetic Analysis Framework.

## Tutorial 1: Basic Analysis

### Step 1: Load and Segment

```python
from src.utils.utils import load_volume
from src.core.segmentation import otsu_threshold

# Load volume
volume, metadata = load_volume('data/sample.dcm')
print(f"Volume shape: {volume.shape}")
print(f"Voxel size: {metadata.get('voxel_size', 'Unknown')}")

# Segment
segmented = otsu_threshold(volume)
print(f"Segmented shape: {segmented.shape}")
```

### Step 2: Compute Metrics

```python
from src.core.metrics import compute_all_metrics

voxel_size = metadata.get('voxel_size', (0.1, 0.1, 0.1))
metrics = compute_all_metrics(segmented, voxel_size)

print(f"Volume: {metrics['volume']:.2f} mm³")
print(f"Surface area: {metrics['surface_area']:.2f} mm²")
print(f"Void fraction: {metrics['void_fraction']:.3f}")
print(f"Relative density: {metrics['relative_density']:.3f}")
```

### Step 3: Visualize

```python
from src.core.visualization import visualize_3d_volume, visualize_slice

# 3D visualization
visualize_3d_volume(segmented)

# Slice visualization
visualize_slice(segmented, slice_idx=50, axis=0)
```

## Tutorial 2: Preprocessing and Filtering

### Step 1: Load Segmented Data

```python
from src.utils.utils import load_segmented_data

# Load CSV with segmented data
volume, metadata = load_segmented_data('data/segmented_data/Sample_01_segmented.csv')
```

### Step 2: Filter Objects

```python
from src.preprocessing.preprocessing import (
    remove_edge_objects,
    filter_by_volume,
    filter_by_sphericity
)

voxel_size = (0.1, 0.1, 0.1)

# Remove edge objects
cleaned = remove_edge_objects(volume, margin=5)

# Filter by volume (remove small artifacts)
filtered = filter_by_volume(
    cleaned,
    voxel_size,
    min_volume=0.1,  # mm³
    max_volume=1000.0
)

# Filter by sphericity (remove non-spherical objects)
filtered = filter_by_sphericity(
    filtered,
    voxel_size,
    min_sphericity=0.5
)
```

### Step 3: Analyze Object Properties

```python
from src.preprocessing.preprocessing import analyze_object_properties

properties = analyze_object_properties(filtered, voxel_size)

print(f"Number of objects: {len(properties)}")
print(f"Mean volume: {np.mean([p['volume_mm3'] for p in properties]):.2f} mm³")
print(f"Mean sphericity: {np.mean([p['sphericity'] for p in properties]):.3f}")
```

## Tutorial 3: Porosity Analysis

### Step 1: Analyze Porosity Distribution

```python
from src.core.porosity import analyze_porosity_distribution

result = analyze_porosity_distribution(
    segmented,
    direction='z',  # Printing direction
    voxel_size=(0.1, 0.1, 0.1)
)

print(f"Mean porosity: {result['mean_porosity']:.3f}")
print(f"Porosity variation: {result['porosity_variation']:.3f}")
```

### Step 2: Plot Porosity Profile

```python
from src.core.porosity import porosity_along_direction
from src.core.visualization import plot_porosity_profile

profile = porosity_along_direction(segmented, direction='z')
plot_porosity_profile(profile)
```

### Step 3: Fit Distribution

```python
from src.preprocessing.statistics import fit_gaussian

pore_sizes = [p['volume_mm3'] for p in properties]
fit = fit_gaussian(pore_sizes)

print(f"Mean pore size: {fit['mean']:.2f} mm³")
print(f"Std pore size: {fit['std']:.2f} mm³")
```

## Tutorial 4: Flow Analysis

### Step 1: Analyze Flow Connectivity

```python
from src.experimental.flow_analysis import analyze_flow_connectivity

connectivity = analyze_flow_connectivity(
    segmented,
    flow_direction='z',
    voxel_size=(0.1, 0.1, 0.1)
)

print(f"Connected: {connectivity['is_connected']}")
print(f"Number of paths: {connectivity['n_paths']}")
```

### Step 2: Compute Tortuosity

```python
from src.experimental.flow_analysis import compute_tortuosity

tortuosity_result = compute_tortuosity(
    segmented,
    flow_direction='z',
    voxel_size=(0.1, 0.1, 0.1)
)

print(f"Tortuosity: {tortuosity_result['tortuosity']:.3f}")
print(f"Path length: {tortuosity_result['path_length']:.2f} mm")
```

### Step 3: Estimate Flow Resistance

```python
from src.experimental.flow_analysis import estimate_flow_resistance

channel_geometry = {
    'mean_diameter': 1.0  # mm
}

flow_conditions = {
    'velocity': 0.1,  # m/s
    'flow_rate': 0.001  # m³/s
}

resistance = estimate_flow_resistance(
    segmented,
    channel_geometry,
    flow_conditions,
    voxel_size=(0.1, 0.1, 0.1)
)

print(f"Pressure drop: {resistance['pressure_drop_kpa']:.2f} kPa")
print(f"Reynolds number: {resistance['reynolds_number']:.1f}")
```

## Tutorial 5: Batch Processing

### Step 1: Batch Analyze Multiple Samples

```python
from src.analysis.comparative_analysis import batch_analyze

results = batch_analyze(
    volume_paths=[
        'data/sample1.dcm',
        'data/sample2.dcm',
        'data/sample3.dcm'
    ],
    voxel_size=(0.1, 0.1, 0.1)
)

print(f"Analyzed {len(results)} samples")
```

### Step 2: Statistical Comparison

```python
from src.analysis.comparative_analysis import statistical_tests

void_fractions = [r['void_fraction'] for r in results]

# Compare first two samples
test_result = statistical_tests(
    void_fractions[0],
    void_fractions[1],
    test_type='t-test'
)

print(f"t-statistic: {test_result['statistic']:.3f}")
print(f"p-value: {test_result['p_value']:.3f}")
```

### Step 3: Process-Structure-Performance Analysis

```python
from src.analysis.performance_analysis import process_structure_performance_relationship

process_params = pd.DataFrame({
    'extrusion_temp': [200, 220, 240],
    'print_speed': [20, 30, 40]
})

structure_metrics = pd.DataFrame({
    'void_fraction': [0.3, 0.25, 0.2],
    'relative_density': [0.7, 0.75, 0.8]
})

performance = pd.DataFrame({
    'heat_transfer_efficiency': [0.6, 0.7, 0.8]
})

psp = process_structure_performance_relationship(
    process_params,
    structure_metrics,
    performance
)

print(f"Key correlations: {psp['correlations']}")
```

## Tutorial 6: Using the Analyzer Class

### Complete Analysis Pipeline

```python
from src.analyzer import XCTAnalyzer

# Initialize analyzer
analyzer = XCTAnalyzer(voxel_size=(0.1, 0.1, 0.1))

# Comprehensive analysis
results = analyzer.analyze_comprehensive(
    volume_path='data/sample.dcm',
    output_dir='outputs/analysis'
)

# Access results
print(f"Void fraction: {results['metrics']['void_fraction']:.3f}")
print(f"Porosity: {results['porosity']['mean_porosity']:.3f}")
print(f"Filament diameter: {results['filament']['mean_diameter']:.2f} mm")
```

## Next Steps

- Explore the [Workflows](workflows.md) for more complex analyses
- Check the [Module Reference](modules.md) for detailed function documentation
- See the [Jupyter Notebooks](../notebooks/) for interactive examples

