# Installation Guide

## Requirements

- Python 3.8 or higher
- NumPy, SciPy, scikit-image
- Matplotlib, PyVista (for visualization)
- Pandas (for data handling)
- Jupyter (for notebooks)

## Installation

### 1. Clone or Download

If using git:
```bash
git clone https://github.com/kanhaiya-gupta/XCT_Thermomagnetic_Analysis.git
cd XCT_Thermomagnetic_Analysis
```

Or download and extract the framework.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
from src.analyzer import XCTAnalyzer
from src.core.segmentation import otsu_threshold

print("✅ Installation successful!")
```

## Dependencies

The framework requires the following packages:

- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **scikit-image** - Image processing
- **matplotlib** - Plotting
- **pyvista** - 3D visualization
- **pandas** - Data manipulation
- **jupyter** - Notebooks
- **ipywidgets** - Interactive widgets

See `requirements.txt` for specific versions.

## Optional Dependencies

- **dragonfly** - For DragonFly integration (if available)
- **plotly** - For interactive visualizations
- **seaborn** - For enhanced plotting

## Development Setup

For development:

```bash
pip install -r requirements.txt
pip install pytest  # For testing
pip install black   # For code formatting
pip install flake8  # For linting
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Ensure you're in the correct directory
2. Add the XCT_Thermomagnetic_Analysis root to Python path:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent))
   ```

### Visualization Issues

If 3D visualization doesn't work:

1. Install PyVista: `pip install pyvista`
2. For Jupyter: `pip install ipywidgets`

### Memory Issues

For large volumes:

1. Use chunked processing
2. Reduce volume resolution
3. Use memory-efficient data types

## Platform-Specific Notes

### Windows

- Ensure Visual C++ Redistributable is installed (for some packages)
- Use Anaconda/Miniconda for easier package management

### Linux

- May need to install system packages:
  ```bash
  sudo apt-get install python3-dev python3-pip
  ```

### macOS

- Use Homebrew for system dependencies if needed
- Anaconda recommended for scientific packages

## Next Steps

After installation:

1. Read the [Tutorials](tutorials.md)
2. Explore the [Workflows](workflows.md)
3. Check the [Module Reference](modules.md)

