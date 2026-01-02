# XCT Thermomagnetic Analysis Framework - Documentation

Welcome to the comprehensive documentation for the XCT Thermomagnetic Analysis Framework. This documentation provides detailed information about the framework's architecture, modules, workflows, and usage.

## 📚 Documentation Structure

- **[Architecture Overview](architecture.md)** - Framework architecture, design principles, and module organization
- **[Module Reference](modules.md)** - Detailed documentation of all modules and their functions
- **[Workflows](workflows.md)** - Step-by-step workflows with Mermaid diagrams
- **[Advanced Analysis Use Cases](advanced_analysis_use_cases.md)** - Sensitivity analysis, virtual experiments, comparative analysis, and performance prediction
- **[Experimental Use Cases](experimental_use_cases.md)** - Real-world use cases for thermomagnetic generator research
- **[External Image Analysis Tools](external_image_analysis_tools.md)** - Comparison with other tools (DragonFly, ImageJ, etc.) and integration guides
- **[Statistical Fitting Guide](statistical_fitting.md)** - Comprehensive guide to distribution and regression fitting
- **[Data Generation Guide](data_generation.md)** - Guide to generating synthetic XCT data for testing and demonstration
- **[API Reference](api.md)** - Complete API documentation
- **[Installation Guide](installation.md)** - Setup and installation instructions
- **[Tutorials](tutorials.md)** - Getting started tutorials and examples
- **[Notebooks Guide](notebooks.md)** - Interactive Jupyter notebook documentation
- **[Local CI/CD Testing](local_cicd_testing.md)** - Guide to testing GitHub Actions workflows locally
- **[Contributing](contributing.md)** - Guidelines for contributing to the framework
- **[Test Plan](tests.md)** - Comprehensive testing strategy and test plan

### Framework Information

- **[Research Alignment](RESEARCH_ALIGNMENT.md)** - How the framework aligns with thermomagnetic generator research


## 🚀 Quick Start

```python
from src.analyzer import XCTAnalyzer
from src.core.segmentation import otsu_threshold
from src.core.metrics import compute_all_metrics

# Initialize analyzer
analyzer = XCTAnalyzer(voxel_size=(0.1, 0.1, 0.1))

# Load and segment volume
volume, metadata = analyzer.load_volume('data/sample.dcm')
segmented = otsu_threshold(volume)

# Compute metrics
metrics = compute_all_metrics(segmented, (0.1, 0.1, 0.1))
print(f"Void fraction: {metrics['void_fraction']:.3f}")
```

## 📊 Framework Overview

The XCT Thermomagnetic Analysis Framework is organized into seven main categories:

1. **Core** - Fundamental analysis operations (segmentation, morphology, metrics)
2. **Preprocessing** - Data cleaning and statistical analysis
3. **Analysis** - Advanced analysis (sensitivity, virtual experiments, comparative)
4. **Quality** - Quality control and validation
5. **Experimental** - Experiment-specific analysis (flow, thermal, energy)
6. **Integration** - External tool integration (DragonFly)
7. **Utils** - Utility functions

See [Architecture Overview](architecture.md) for detailed information.

## 🔗 External Resources

- [Research Alignment](RESEARCH_ALIGNMENT.md) - How the framework aligns with research goals
- [Module Reference](modules.md) - Complete module listing and documentation
- [Notebooks](../notebooks/) - Interactive Jupyter notebooks

