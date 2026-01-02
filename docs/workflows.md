# Workflows

This document describes common workflows for using the XCT Thermomagnetic Analysis Framework, with visual flowcharts using Mermaid.

## Standard Analysis Workflow

```mermaid
flowchart TD
    Start([Start Analysis]) --> Load[Load XCT Volume]
    Load --> Preprocess[Preprocess Data]
    Preprocess --> Segment[Segment Volume]
    Segment --> Morph[Morphological Operations]
    Morph --> Metrics[Compute Metrics]
    Metrics --> Analyze[Advanced Analysis]
    Analyze --> Quality[Quality Control]
    Quality --> Visualize[Visualize Results]
    Visualize --> Export[Export Results]
    Export --> End([End])
    
    Preprocess --> Pre1[Filter Objects]
    Preprocess --> Pre2[Unit Conversion]
    
    Segment --> Seg1[Otsu Thresholding]
    Segment --> Seg2[Adaptive Thresholding]
    
    Morph --> Morph1[Remove Small Objects]
    Morph --> Morph2[Fill Holes]
    
    Metrics --> Met1[Volume]
    Metrics --> Met2[Surface Area]
    Metrics --> Met3[Void Fraction]
    
    Analyze --> An1[Porosity Analysis]
    Analyze --> An2[Filament Analysis]
    Analyze --> An3[Slice Analysis]
    
    Quality --> Qual1[Dimensional Accuracy]
    Quality --> Qual2[Uncertainty Analysis]
```

## Batch Processing Workflow

```mermaid
flowchart TD
    Start([Start Batch Processing]) --> LoadBatch[Load Multiple Volumes]
    LoadBatch --> ForEach{For Each Volume}
    ForEach --> Process[Process Volume]
    Process --> Store[Store Results]
    Store --> Next{More Volumes?}
    Next -->|Yes| ForEach
    Next -->|No| Compare[Compare Results]
    Compare --> Stats[Statistical Analysis]
    Stats --> Report[Generate Report]
    Report --> End([End])
    
    Process --> P1[Segment]
    Process --> P2[Compute Metrics]
    Process --> P3[Advanced Analysis]
    
    Compare --> C1[ANOVA]
    Compare --> C2[t-tests]
    Compare --> C3[Correlation]
```

## Sensitivity Analysis Workflow

```mermaid
flowchart TD
    Start([Start Sensitivity Analysis]) --> Define[Define Parameters]
    Define --> Design[Design Experiment]
    Design --> Run[Run Parameter Sweep]
    Run --> Analyze[Analyze Results]
    Analyze --> Identify[Identify Sensitive Parameters]
    Identify --> Visualize[Visualize Sensitivity]
    Visualize --> Report[Generate Report]
    Report --> End([End])
    
    Design --> D1[Full Factorial]
    Design --> D2[Latin Hypercube]
    Design --> D3[Central Composite]
    
    Analyze --> A1[Local Sensitivity]
    Analyze --> A2[Sobol Indices]
    Analyze --> A3[Morris Screening]
```

## Virtual Experiments Workflow

```mermaid
flowchart TD
    Start([Start Virtual Experiments]) --> Design[Design of Experiments]
    Design --> Generate[Generate Parameter Sets]
    Generate --> Simulate[Simulate Experiments]
    Simulate --> Fit[Fit Response Surface]
    Fit --> Optimize[Optimize Parameters]
    Optimize --> Validate[Validate Results]
    Validate --> Report[Generate Report]
    Report --> End([End])
    
    Design --> D1[Full Factorial]
    Design --> D2[LHS]
    Design --> D3[CCD]
    Design --> D4[Box-Behnken]
    
    Simulate --> S1[Run Analysis]
    Simulate --> S2[Collect Metrics]
    
    Fit --> F1[Polynomial]
    Fit --> F2[Gaussian Process]
    Fit --> F3[Neural Network]
    
    Optimize --> O1[Single Objective]
    Optimize --> O2[Multi-Objective]
```

## Flow Analysis Workflow

```mermaid
flowchart TD
    Start([Start Flow Analysis]) --> Load[Load Segmented Volume]
    Load --> Invert[Invert Volume]
    Invert --> Connectivity[Analyze Connectivity]
    Connectivity --> Paths[Identify Flow Paths]
    Paths --> Tortuosity[Compute Tortuosity]
    Tortuosity --> Resistance[Estimate Flow Resistance]
    Resistance --> Distribution[Analyze Distribution]
    Distribution --> Report[Generate Report]
    Report --> End([End])
    
    Connectivity --> C1[Inlet Detection]
    Connectivity --> C2[Outlet Detection]
    Connectivity --> C3[Path Finding]
    
    Paths --> P1[Main Path]
    Paths --> P2[Branching]
    Paths --> P3[Dead Ends]
    
    Tortuosity --> T1[Path Length]
    Tortuosity --> T2[Tortuosity Factor]
    
    Resistance --> R1[Pressure Drop]
    Resistance --> R2[Reynolds Number]
    Resistance --> R3[Hydraulic Diameter]
```

## Thermal Analysis Workflow

```mermaid
flowchart TD
    Start([Start Thermal Analysis]) --> Load[Load Volume & Properties]
    Load --> Conduction[Analyze Conduction]
    Conduction --> Convection[Analyze Convection]
    Convection --> Combined[Compute Total Resistance]
    Combined --> Coefficient[Estimate Heat Transfer Coefficient]
    Coefficient --> Gradient[Estimate Temperature Gradient]
    Gradient --> Report[Generate Report]
    Report --> End([End])
    
    Conduction --> C1[Thermal Conductivity]
    Conduction --> C2[Characteristic Length]
    Conduction --> C3[Effective Area]
    
    Convection --> Cv1[Flow Conditions]
    Convection --> Cv2[Surface Area]
    Convection --> Cv3[Heat Transfer Coefficient]
```

## Energy Conversion Analysis Workflow

```mermaid
flowchart TD
    Start([Start Energy Conversion Analysis]) --> Load[Load Volume & Conditions]
    Load --> Power[Estimate Power Output]
    Power --> Efficiency[Calculate Efficiency]
    Efficiency --> Temperature[Temperature-Dependent Analysis]
    Temperature --> Density[Compute Power Density]
    Density --> Cycle[Estimate Cycle Efficiency]
    Cycle --> Report[Generate Report]
    Report --> End([End])
    
    Power --> P1[Thermal Gradient]
    Power --> P2[Magnetic Properties]
    Power --> P3[Flow Conditions]
    
    Efficiency --> E1[Energy Conversion]
    Efficiency --> E2[Losses]
    
    Temperature --> T1[Performance vs Temperature]
    Temperature --> T2[Optimal Temperature]
```

## Quality Control Workflow

```mermaid
flowchart TD
    Start([Start Quality Control]) --> Dimensional[Dimensional Accuracy]
    Dimensional --> Uncertainty[Uncertainty Analysis]
    Uncertainty --> Validation[Validation]
    Validation --> Reproducibility[Reproducibility Check]
    Reproducibility --> Report[Generate QC Report]
    Report --> End([End])
    
    Dimensional --> D1[Compare to CAD]
    Dimensional --> D2[Tolerance Compliance]
    Dimensional --> D3[Surface Deviation]
    
    Uncertainty --> U1[Measurement Uncertainty]
    Uncertainty --> U2[Confidence Intervals]
    Uncertainty --> U3[Uncertainty Budget]
    
    Validation --> V1[Ground Truth]
    Validation --> V2[DragonFly Comparison]
    Validation --> V3[Cross-Validation]
    
    Reproducibility --> R1[Provenance Tracking]
    Reproducibility --> R2[Seed Management]
    Reproducibility --> R3[Config Export]
```

## Complete Analysis Pipeline

```mermaid
flowchart TD
    Start([Start]) --> Load[Load Data]
    Load --> Preprocess[Preprocess]
    Preprocess --> Segment[Segment]
    Segment --> Core[Core Analysis]
    Core --> Advanced[Advanced Analysis]
    Advanced --> Experimental[Experimental Analysis]
    Experimental --> Quality[Quality Control]
    Quality --> Visualize[Visualize]
    Visualize --> Export[Export]
    Export --> End([End])
    
    Core --> C1[Metrics]
    Core --> C2[Porosity]
    Core --> C3[Filament]
    Core --> C4[Slice]
    
    Advanced --> A1[Sensitivity]
    Advanced --> A2[Virtual Experiments]
    Advanced --> A3[Comparative]
    Advanced --> A4[Performance]
    
    Experimental --> E1[Flow]
    Experimental --> E2[Thermal]
    Experimental --> E3[Energy]
    
    Quality --> Q1[Dimensional]
    Quality --> Q2[Uncertainty]
    Quality --> Q3[Validation]
    Quality --> Q4[Reproducibility]
```

## Code Examples

### Standard Analysis

```python
from src.analyzer import XCTAnalyzer
from src.core.segmentation import otsu_threshold
from src.core.metrics import compute_all_metrics

# Initialize
analyzer = XCTAnalyzer(voxel_size=(0.1, 0.1, 0.1))

# Load and segment
volume, metadata = analyzer.load_volume('data/sample.dcm')
segmented = otsu_threshold(volume)

# Compute metrics
metrics = compute_all_metrics(segmented, (0.1, 0.1, 0.1))
```

### Batch Processing

```python
from src.analysis.comparative_analysis import batch_analyze

# Analyze multiple samples
results = batch_analyze(
    volume_paths=['data/sample1.dcm', 'data/sample2.dcm'],
    voxel_size=(0.1, 0.1, 0.1)
)
```

### Sensitivity Analysis

```python
from src.analysis.sensitivity_analysis import parameter_sweep

# Parameter sweep
results = parameter_sweep(
    base_volume=volume,
    parameters={'threshold': [0.3, 0.4, 0.5]},
    metric_function=compute_all_metrics
)
```

See [Tutorials](tutorials.md) for more detailed examples.

