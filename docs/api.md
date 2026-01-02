# API Reference

Complete API documentation for all functions in the XCT Thermomagnetic Analysis Framework.

## Core Modules

### `src.core.segmentation`

#### `otsu_threshold(volume, return_threshold=False)`

Apply Otsu's thresholding method to 3D volume.

**Parameters:**
- `volume` (np.ndarray): Input 3D volume (grayscale)
- `return_threshold` (bool): If True, return threshold value

**Returns:**
- `np.ndarray` or `tuple`: Binary segmented volume, or (volume, threshold) if return_threshold=True

**Example:**
```python
segmented = otsu_threshold(volume)
```

---

#### `segment_volume(volume, method='otsu', **kwargs)`

Main segmentation interface supporting multiple methods.

**Parameters:**
- `volume` (np.ndarray): Input 3D volume
- `method` (str): Segmentation method ('otsu', 'adaptive', 'multi-threshold')
- `**kwargs`: Method-specific parameters

**Returns:**
- `np.ndarray`: Segmented volume

---

### `src.core.metrics`

#### `compute_all_metrics(volume, voxel_size)`

Compute all scalar metrics from segmented volume.

**Parameters:**
- `volume` (np.ndarray): Binary segmented volume
- `voxel_size` (tuple): Voxel spacing (z, y, x) in mm

**Returns:**
- `dict`: Dictionary with all metrics:
  - `volume` (float): Volume in mm³
  - `surface_area` (float): Surface area in mm²
  - `void_fraction` (float): Void fraction (0-1)
  - `relative_density` (float): Relative density (0-1)
  - `specific_surface_area` (float): Specific surface area in mm²/mm³

**Example:**
```python
metrics = compute_all_metrics(segmented, (0.1, 0.1, 0.1))
```

---

### `src.core.filament_analysis`

#### `estimate_filament_diameter(volume, voxel_size, direction='z', method='distance_transform')`

Estimate mean filament diameter along specified direction.

**Parameters:**
- `volume` (np.ndarray): Binary volume (1 = material)
- `voxel_size` (tuple): Voxel spacing (z, y, x) in mm
- `direction` (str): Direction ('x', 'y', 'z')
- `method` (str): Method ('distance_transform', 'cross_section')

**Returns:**
- `dict`: Dictionary with:
  - `mean_diameter` (float): Mean diameter in mm
  - `std_diameter` (float): Standard deviation in mm
  - `min_diameter` (float): Minimum diameter in mm
  - `max_diameter` (float): Maximum diameter in mm
  - `diameters` (list): List of all diameters

---

### `src.core.porosity`

#### `analyze_porosity_distribution(volume, direction='z', voxel_size=None)`

Comprehensive porosity distribution analysis.

**Parameters:**
- `volume` (np.ndarray): Binary volume
- `direction` (str): Analysis direction ('x', 'y', 'z')
- `voxel_size` (tuple, optional): Voxel spacing

**Returns:**
- `dict`: Dictionary with porosity analysis results

---

## Preprocessing Modules

### `src.preprocessing.preprocessing`

#### `filter_by_volume(volume, voxel_size, min_volume=None, max_volume=None)`

Filter objects by volume.

**Parameters:**
- `volume` (np.ndarray): Binary or labeled volume
- `voxel_size` (tuple): Voxel spacing (z, y, x) in mm
- `min_volume` (float, optional): Minimum volume in mm³
- `max_volume` (float, optional): Maximum volume in mm³

**Returns:**
- `np.ndarray`: Filtered volume

---

#### `analyze_object_properties(volume, voxel_size)`

Analyze properties of all objects in volume.

**Parameters:**
- `volume` (np.ndarray): Binary or labeled volume
- `voxel_size` (tuple): Voxel spacing (z, y, x) in mm

**Returns:**
- `list`: List of dictionaries with object properties:
  - `label_id` (int): Object label
  - `volume_mm3` (float): Volume in mm³
  - `sphericity` (float): Sphericity (0-1)
  - `max_aspect_ratio` (float): Maximum aspect ratio
  - `on_edge` (bool): Whether object touches edge

---

## Analysis Modules

### `src.analysis.sensitivity_analysis`

#### `parameter_sweep(base_volume, parameters, metric_function)`

Perform parameter sweep analysis.

**Parameters:**
- `base_volume` (np.ndarray): Base volume for analysis
- `parameters` (dict): Dictionary of parameter ranges, e.g., `{'threshold': [0.3, 0.4, 0.5]}`
- `metric_function` (callable): Function to compute metrics

**Returns:**
- `dict`: Dictionary with sweep results

---

### `src.analysis.virtual_experiments`

#### `latin_hypercube_sampling(parameters, n_samples)`

Generate Latin Hypercube Sampling design.

**Parameters:**
- `parameters` (dict): Parameter ranges, e.g., `{'temp': (200, 300), 'speed': (10, 50)}`
- `n_samples` (int): Number of samples

**Returns:**
- `np.ndarray`: Design matrix (n_samples × n_parameters)

---

## Experimental Modules

### `src.experimental.flow_analysis`

#### `compute_tortuosity(volume, flow_direction='z', voxel_size=None)`

Compute flow path tortuosity.

**Parameters:**
- `volume` (np.ndarray): Binary volume
- `flow_direction` (str): Flow direction ('x', 'y', 'z')
- `voxel_size` (tuple, optional): Voxel spacing

**Returns:**
- `dict`: Dictionary with:
  - `tortuosity` (float): Tortuosity factor
  - `path_length` (float): Path length in mm
  - `straight_line_distance` (float): Straight-line distance in mm

---

#### `estimate_flow_resistance(volume, channel_geometry, flow_conditions=None, fluid_properties=None, voxel_size=None)`

Estimate flow resistance based on channel geometry.

**Parameters:**
- `volume` (np.ndarray): Binary volume
- `channel_geometry` (dict): Channel geometry (mean_diameter in mm)
- `flow_conditions` (dict, optional): Flow conditions (velocity, flow_rate)
- `fluid_properties` (dict, optional): Fluid properties (density, viscosity)
- `voxel_size` (tuple, optional): Voxel spacing

**Returns:**
- `dict`: Dictionary with flow resistance analysis

---

### `src.experimental.thermal_analysis`

#### `compute_thermal_resistance(volume, material_properties, flow_conditions=None, voxel_size=None)`

Compute overall thermal resistance.

**Parameters:**
- `volume` (np.ndarray): Binary volume
- `material_properties` (dict): Material properties (thermal_conductivity, density, specific_heat)
- `flow_conditions` (dict, optional): Flow conditions
- `voxel_size` (tuple, optional): Voxel spacing

**Returns:**
- `dict`: Dictionary with thermal resistance analysis

---

## Quality Modules

### `src.quality.uncertainty_analysis`

#### `confidence_intervals(data, confidence=0.95)`

Compute confidence intervals for data.

**Parameters:**
- `data` (array-like): Data array
- `confidence` (float): Confidence level (0-1)

**Returns:**
- `dict`: Dictionary with:
  - `mean` (float): Mean value
  - `lower` (float): Lower bound
  - `upper` (float): Upper bound
  - `confidence` (float): Confidence level

---

## Utils

### `src.utils.utils`

#### `load_volume(file_path, file_format='auto', **kwargs)`

Load volume from various file formats.

**Parameters:**
- `file_path` (str or Path): Path to volume file
- `file_format` (str): File format ('auto', 'dicom', 'tiff', 'raw', etc.)
- `**kwargs`: Format-specific parameters

**Returns:**
- `tuple`: (volume array, metadata dictionary)

**Supported formats:**
- DICOM (.dcm, .dicom)
- TIFF (.tif, .tiff)
- RAW (.raw)
- NIfTI (.nii, .nii.gz)
- MetaImage (.mhd, .mha)
- NumPy (.npy, .npz)

---

#### `convert_to_mm(value, unit)`

Convert value to millimeters.

**Parameters:**
- `value` (float): Value to convert
- `unit` (str): Source unit ('mm', 'cm', 'um', 'micrometer', 'm')

**Returns:**
- `float`: Value in millimeters

---

## Main Analyzer Class

### `src.analyzer.XCTAnalyzer`

Main analyzer class that integrates all modules.

#### `__init__(voxel_size, output_dir=None)`

Initialize analyzer.

**Parameters:**
- `voxel_size` (tuple): Voxel spacing (z, y, x) in mm
- `output_dir` (str or Path, optional): Output directory

#### `analyze_comprehensive(volume_path, output_dir=None)`

Perform comprehensive analysis.

**Parameters:**
- `volume_path` (str or Path): Path to volume file
- `output_dir` (str or Path, optional): Output directory

**Returns:**
- `dict`: Dictionary with all analysis results

---

For more detailed documentation, see the [Module Reference](modules.md).

