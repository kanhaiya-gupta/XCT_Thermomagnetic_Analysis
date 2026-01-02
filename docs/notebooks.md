# Notebook Documentation

## Overview

This document describes the interactive Jupyter notebooks available in the XCT Thermomagnetic Analysis framework. All notebooks have been created and are ready for use. They provide comprehensive, interactive workflows for analyzing X-ray Computed Tomography (XCT) data of 3D-printed thermomagnetic elements.

### Key Features

1. **Interactive Widgets**: All notebooks use `ipywidgets` for real-time parameter adjustment and visualization
2. **Cross-Platform Path Handling**: Universal path normalization supports Windows, Linux, macOS, and WSL
3. **Comprehensive Coverage**: Cover all framework modules from basic analysis to advanced optimization
4. **Modular Structure**: Aligned with the framework's modular architecture
5. **Complete Workflows**: Demonstrate end-to-end analysis pipelines

## Notebook Structure

### Notebook 01: Getting Started - Basic Analysis
**File**: `01_Getting_Started_Basic_Analysis.ipynb`

**Purpose**: Introduction to the framework with basic analysis workflow

**Content**:
- Setup and installation
- Loading XCT data (multiple formats)
- Basic segmentation (Otsu thresholding)
- Computing core metrics (volume, surface area, void fraction)
- Simple visualization
- Using the `XCTAnalyzer` class

**Modules Covered**:
- `src.analyzer` (XCTAnalyzer)
- `src.core.segmentation`
- `src.core.metrics`
- `src.core.visualization`
- `src.utils.utils`

**Target Audience**: Beginners, first-time users

**Widget Requirements**:
- File path input widget
- File format dropdown
- Voxel size input widgets
- Segmentation method selector
- Threshold slider (for manual segmentation)
- Segmentation preview (interactive)
- Metrics display widget
- 3D visualization widget (interactive)
- Progress bar
- Status display

---

### Notebook 02: Preprocessing and Data Cleaning
**File**: `02_Preprocessing_Data_Cleaning.ipynb`

**Purpose**: Data preprocessing, filtering, and object property analysis

**Content**:
- Loading segmented data (CSV/Excel)
- Object filtering (volume, sphericity, spatial bounds, aspect ratio)
- Edge object removal
- Object property analysis
- Statistical fitting (Gaussian, Poisson distributions)
- Quality assessment of filtered data

**Modules Covered**:
- `src.preprocessing.preprocessing`
- `src.preprocessing.statistics`
- `src.core.morphology`

**Target Audience**: Users working with segmented data, quality control

**Widget Requirements**:
- File path input for segmented data
- Filter parameter widgets (volume, sphericity, spatial bounds, aspect ratio)
- Filter enable/disable checkboxes
- Object property table widget (interactive)
- Statistical fitting parameter widgets
- Distribution selector
- Fit results display widget
- Filter statistics display

---

### Notebook 03: Core Analysis - Morphology and Porosity
**File**: `03_Core_Analysis_Morphology_Porosity.ipynb`

**Purpose**: Comprehensive morphological and porosity analysis

**Content**:
- Filament diameter estimation
- Channel width analysis
- Porosity distribution analysis
- Slice analysis (along and perpendicular to flow)
- Statistical fitting of distributions
- Visualization of results

**Modules Covered**:
- `src.core.filament_analysis`
- `src.core.porosity`
- `src.core.slice_analysis`
- `src.preprocessing.statistics`
- `src.core.visualization`

**Target Audience**: Users analyzing structure morphology

**Widget Requirements**:
- Printing direction selector
- Flow direction selector
- Analysis parameter widgets
- Filament diameter visualization (interactive)
- Channel width visualization (interactive)
- Porosity profile plot (interactive)
- Slice selector for visualization
- Distribution fitting widgets
- Results tabs (Metrics, Filaments, Porosity, Slices)

---

### Notebook 04: Experimental Analysis - Flow, Thermal, Energy
**File**: `04_Experimental_Analysis_Flow_Thermal_Energy.ipynb`

**Purpose**: Flow, thermal, and energy conversion analysis for thermomagnetic elements

**Content**:
- Flow connectivity analysis
- Tortuosity computation
- Flow resistance estimation
- Thermal resistance analysis
- Heat transfer coefficient estimation
- Energy conversion efficiency
- Power output estimation
- Integrated analysis workflow

**Modules Covered**:
- `src.experimental.flow_analysis`
- `src.experimental.thermal_analysis`
- `src.experimental.energy_conversion`

**Target Audience**: Researchers analyzing performance characteristics

**Widget Requirements**:
- Flow direction selector
- Flow condition input widgets (velocity, flow rate)
- Material property input widgets
- Analysis type selector (flow, thermal, energy, or all)
- Results tabs (Flow, Thermal, Energy, Integrated)
- Interactive 3D flow path visualization
- Parameter sweep widgets for sensitivity
- Performance metric displays

---

### Notebook 05: Advanced Analysis - Sensitivity and Virtual Experiments
**File**: `05_Advanced_Analysis_Sensitivity_Virtual_Experiments.ipynb`

**Purpose**: Parameter sensitivity analysis and process optimization

**Content**:
- Parameter sensitivity analysis (local, Morris screening, Sobol indices)
- Design of Experiments (DoE) - Full factorial, LHS, CCD, Box-Behnken
- Virtual experiments
- Response surface modeling
- Process parameter optimization
- Multi-objective optimization
- Process-Structure-Performance relationships

**Modules Covered**:
- `src.analysis.sensitivity_analysis`
- `src.analysis.virtual_experiments`
- `src.analysis.performance_analysis`

**Target Audience**: Researchers optimizing process parameters

**Widget Requirements**:
- Parameter range input widgets
- Sensitivity method selector (local, Morris, Sobol)
- DoE design selector (Factorial, LHS, CCD, Box-Behnken)
- Number of samples input
- Optimization objective selector
- Parameter bounds input widgets
- Results visualization (interactive plots)
- Sensitivity ranking display
- Optimization results display

---

### Notebook 06: Comparative Analysis and Batch Processing
**File**: `06_Comparative_Analysis_Batch_Processing.ipynb`

**Purpose**: Batch processing and statistical comparison of multiple samples

**Content**:
- Batch analysis of multiple volumes
- Statistical comparison (ANOVA, t-tests, Mann-Whitney)
- Group comparison
- Process-Structure-Performance analysis across samples
- Quality control metrics (repeatability, process capability)
- Comprehensive reporting

**Modules Covered**:
- `src.analysis.comparative_analysis`
- `src.analysis.performance_analysis`
- `src.quality.reproducibility`

**Target Audience**: Users analyzing multiple samples, quality control

**Widget Requirements**:
- Batch file selector (multiple files)
- Batch processing progress bar
- Group assignment widgets
- Statistical test selector
- Comparison metric selector
- Results table widget (sortable, filterable)
- Group comparison visualization (interactive)
- Process capability widgets
- Export results button

---

### Notebook 07: Quality Control and Validation
**File**: `07_Quality_Control_Validation.ipynb`

**Purpose**: Quality control, validation, and uncertainty analysis

**Content**:
- Dimensional accuracy analysis
- CAD comparison
- Tolerance compliance
- Uncertainty quantification
- Confidence intervals
- Validation against ground truth
- DragonFly comparison
- Reproducibility analysis
- Provenance tracking

**Modules Covered**:
- `src.quality.dimensional_accuracy`
- `src.quality.uncertainty_analysis`
- `src.quality.validation`
- `src.quality.reproducibility`
- `src.integration.dragonfly_integration`

**Target Audience**: Quality control engineers, validation studies

**Widget Requirements**:
- CAD file input widget
- Tolerance input widgets
- Validation method selector
- Uncertainty parameter widgets
- Confidence level selector
- DragonFly import/export widgets
- Validation results display
- Uncertainty visualization (interactive)
- Comparison plots (interactive)

---

### Notebook 08: Complete Analysis Pipeline
**File**: `08_Complete_Analysis_Pipeline.ipynb`

**Purpose**: End-to-end comprehensive analysis workflow

**Content**:
- Complete workflow from data loading to reporting
- Integration of all analysis modules
- Best practices and workflow optimization
- Exporting results
- Generating comprehensive reports
- Reproducibility package creation

**Modules Covered**:
- All modules (comprehensive example)

**Target Audience**: Advanced users, complete workflow reference

**Widget Requirements**:
- Workflow step selector
- Configuration widgets for each step
- Progress tracking widget
- Results summary dashboard
- Export options widgets
- Reproducibility package generator
- Report generator widget

---

## Getting Started

### Recommended Learning Path

1. **Start with Notebook 01** if you're new to the framework
2. **Follow the sequence** (01 → 02 → 03 → 04) for a complete learning experience
3. **Use Notebook 08** as a reference for complete workflows
4. **Jump to specific notebooks** based on your analysis needs

### Prerequisites

- Python 3.8+ with required packages (see `requirements.txt`)
- Jupyter Notebook or JupyterLab
- `ipywidgets` installed for interactive features
- XCT data files (DICOM, TIFF, RAW, NIfTI, NumPy, or CSV/Excel formats)

### Quick Start

1. Open Jupyter Notebook or JupyterLab
2. Navigate to the `notebooks/` directory
3. Open `01_Getting_Started_Basic_Analysis.ipynb`
4. Run all cells to initialize the interactive dashboard
5. Follow the instructions in the notebook

## Notebook Template Structure

Each notebook should follow this structure:

```markdown
# Notebook Title

## Overview
Brief description of what this notebook covers

## Learning Objectives
- Objective 1
- Objective 2
- ...

## Prerequisites
- Required packages
- Data requirements
- Previous notebooks (if any)

## 1. Setup and Imports
- Import statements
- Configuration
- Helper functions

## 2. Section Title
- Code cells
- Explanations
- Visualizations

## 3. Summary
- Key takeaways
- Next steps
- References

## Appendix
- Additional examples
- Troubleshooting
- Advanced topics
```

## Key Requirements for All Notebooks

1. **Interactive Widgets**: ⭐ **REQUIRED** - All notebooks must include interactive ipywidgets
   - Use `ipywidgets` for interactive controls
   - Provide real-time parameter adjustment
   - Enable interactive visualization
   - See "Widget Requirements" section below for details

2. **Correct Import Paths**: Use new modular structure
   - `from src.core.segmentation import ...`
   - `from src.analysis.sensitivity_analysis import ...`
   - etc.

3. **Clear Documentation**: Each cell should have markdown explanations

4. **Reproducibility**: 
   - Set random seeds
   - Use relative paths
   - Include data loading examples

5. **Error Handling**: Show common errors and solutions

6. **Visualization**: Include plots and visualizations (with interactive widgets)

7. **Best Practices**: Follow framework conventions

8. **Links to Documentation**: Reference relevant docs

## Notebook Status

All notebooks have been created and are available:

- [x] Notebook 01: Getting Started - Basic Analysis ✅
- [x] Notebook 02: Preprocessing and Data Cleaning ✅
- [x] Notebook 03: Core Analysis - Morphology and Porosity ✅
- [x] Notebook 04: Experimental Analysis - Flow, Thermal, Energy ✅
- [x] Notebook 05: Advanced Analysis - Sensitivity and Virtual Experiments ✅
- [x] Notebook 06: Comparative Analysis and Batch Processing ✅
- [x] Notebook 07: Quality Control and Validation ✅
- [x] Notebook 08: Complete Analysis Pipeline ✅

### Recent Updates

- **Cross-Platform Path Support**: All notebooks now use `normalize_path()` for universal path handling (Windows, Linux, macOS, WSL)
- **Interactive Widgets**: All notebooks include comprehensive interactive widgets for parameter adjustment
- **Comprehensive Testing**: Path normalization has been thoroughly tested across platforms

## Widget Requirements (All Notebooks)

### Required Widget Libraries
```python
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Output, Tab, interactive
from IPython.display import display, clear_output, HTML
```

### Common Widget Patterns

1. **File Input Widgets**:
   - `widgets.Text()` for file paths
   - `widgets.Dropdown()` for file formats
   - `widgets.FileUpload()` for direct upload (optional)

2. **Parameter Input Widgets**:
   - `widgets.FloatSlider()` for continuous parameters
   - `widgets.IntSlider()` for integer parameters
   - `widgets.FloatText()` / `IntText()` for precise values
   - `widgets.Dropdown()` for categorical choices

3. **Control Widgets**:
   - `widgets.Button()` for actions
   - `widgets.Checkbox()` for toggles
   - `widgets.Progress()` / `IntProgress()` for progress

4. **Display Widgets**:
   - `widgets.Output()` for code output
   - `widgets.HTML()` for formatted text
   - `widgets.Tab()` for organized results
   - `widgets.VBox()` / `HBox()` for layout

5. **Interactive Visualization**:
   - Use `@interactive` decorator for automatic widget binding
   - Real-time plot updates with `Output()` widget
   - 3D visualization with PyVista (if available)

### Widget Best Practices

1. **Layout**: Use `VBox` and `HBox` for organized layouts
2. **Feedback**: Always show progress bars for long operations
3. **Status**: Display status messages using `HTML` widget
4. **Error Handling**: Show errors in status widgets, not just exceptions
5. **Responsiveness**: Disable widgets during processing
6. **Documentation**: Add tooltips/descriptions to widgets
7. **Default Values**: Set sensible defaults for all parameters

### Example Widget Structure

```python
# Create widgets
file_path = widgets.Text(description='File Path:', value='')
format_dropdown = widgets.Dropdown(options=['DICOM', 'TIFF', 'RAW'], value='DICOM')
load_button = widgets.Button(description='Load', button_style='primary')
progress = widgets.IntProgress(min=0, max=100, value=0)
status = widgets.HTML(value='Ready')
output = widgets.Output()

# Layout
controls = VBox([
    HBox([file_path, format_dropdown, load_button]),
    progress,
    status
])

# Display
display(controls, output)

# Callback
def on_load(button):
    with output:
        clear_output()
        # Process and display results
        ...

load_button.on_click(on_load)
```

## Technical Details

### Path Handling

All notebooks use the `normalize_path()` function from `src.utils.utils` for cross-platform compatibility:

```python
from src.utils.utils import normalize_path

# Automatically handles:
# - Windows paths: C:\Users\...
# - Linux/macOS paths: /home/user/...
# - WSL paths: /mnt/c/... or /c/...
# - Relative paths with data directory fallback
```

### Widget Requirements

- **Widgets are REQUIRED**: All notebooks include interactive widgets
- **Real-time Updates**: Parameters can be adjusted and results update immediately
- **Progress Tracking**: Long operations show progress bars
- **Error Handling**: User-friendly error messages in status widgets

### Best Practices

- Keep notebooks focused (30-60 minutes execution time)
- Use relative paths for data files
- Set random seeds for reproducibility
- Include both simple and advanced examples
- Link to relevant documentation
- Test in both Jupyter Notebook and JupyterLab

## Additional Resources

- [Framework Documentation](README.md)
- [Module Documentation](modules.md)
- [API Reference](../src/README.md)
- [Troubleshooting Guide](troubleshooting.md)

