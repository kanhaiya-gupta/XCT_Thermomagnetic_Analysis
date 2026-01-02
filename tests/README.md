# Test Suite for XCT Thermomagnetic Analysis Framework

This directory contains the comprehensive test suite for the XCT Thermomagnetic Analysis Framework.

## 📁 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and shared fixtures
├── test_utils.py            # Test utility functions
├── test_core/               # Core module tests
│   ├── test_segmentation.py
│   ├── test_morphology.py
│   └── test_metrics.py
├── test_preprocessing/       # Preprocessing tests
│   ├── test_preprocessing.py
│   └── test_statistics.py
├── test_analysis/           # Advanced analysis tests
│   ├── test_sensitivity_analysis.py
│   ├── test_virtual_experiments.py
│   ├── test_comparative_analysis.py
│   └── test_performance_analysis.py
├── test_quality/            # Quality control tests
│   ├── test_dimensional_accuracy.py
│   ├── test_uncertainty_analysis.py
│   ├── test_reproducibility.py
│   └── test_validation.py
├── test_experimental/       # Experiment-specific tests
│   ├── test_flow_analysis.py
│   ├── test_thermal_analysis.py
│   └── test_energy_conversion.py
├── test_integration/        # Integration tests
│   └── test_dragonfly_integration.py
├── test_utils/              # Utility tests
│   └── test_utils.py
├── test_analyzer/           # Main analyzer tests
│   └── test_analyzer.py
├── fixtures/                # Test data fixtures
│   └── synthetic_volumes.py
└── performance/             # Performance benchmarks (future)
```

## 🚀 Running Tests

### Install Dependencies

```bash
# Install all dependencies (including testing)
pip install -r requirements.txt
```

### Run All Tests

```bash
# From project root
pytest

# Or from tests directory
cd tests && pytest
```

**Note:** Pytest configuration is in `pytest.ini` at the project root.

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Core module tests
pytest tests/test_core/

# Preprocessing tests
pytest tests/test_preprocessing/

# Analysis tests
pytest tests/test_analysis/

# Quality tests
pytest tests/test_quality/

# Experimental tests
pytest tests/test_experimental/

# Specific test file
pytest tests/test_core/test_segmentation.py

# Specific test class or function
pytest tests/test_core/test_segmentation.py::TestOtsuThreshold
pytest tests/test_core/test_segmentation.py::TestOtsuThreshold::test_otsu_basic
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Run Performance Benchmarks

```bash
pytest tests/performance/ --benchmark-only
```

## 📊 Test Markers

- **`@pytest.mark.unit`**: Fast unit tests
- **`@pytest.mark.integration`**: Integration tests
- **`@pytest.mark.slow`**: Slow tests (performance benchmarks)
- **`@pytest.mark.requires_data`**: Tests requiring external data files

## 🔧 Test Fixtures

Common fixtures are defined in `conftest.py`:

- `simple_volume`: Basic test volume with sphere
- `porous_volume`: Volume with known porosity
- `filament_volume`: Volume with filaments
- `edge_case_volumes`: Edge case volumes (empty, full, etc.)
- `voxel_size`: Standard voxel size (0.1 mm)
- `sample_metrics`: Sample metric values
- `sample_distributions`: Sample data with known distributions

## 📝 Writing Tests

### Example Test Structure

```python
import numpy as np
import pytest
from src.core.segmentation import otsu_threshold
from tests.test_utils import assert_volume_valid

class TestOtsuThreshold:
    """Test Otsu thresholding functionality."""
    
    @pytest.mark.unit
    def test_otsu_basic(self, simple_volume):
        """Test basic Otsu thresholding."""
        segmented = otsu_threshold(simple_volume)
        assert_volume_valid(segmented)
        assert segmented.dtype == np.uint8
```

### Test Utilities

Use helper functions from `tests/test_utils.py`:

- `assert_volume_valid(volume)`: Validate volume properties (shape, dtype, etc.)
- `assert_metrics_close(computed, expected, rtol, atol)`: Compare metrics with tolerance
- `create_test_volume(shape, dtype)`: Generate test volumes
- `get_known_sphere_volume(radius, voxel_size)`: Calculate known sphere volume
- `get_known_cube_volume(side_length)`: Calculate known cube volume

### Synthetic Volume Fixtures

Use functions from `tests/fixtures/synthetic_volumes.py`:

- `create_sphere_volume(shape, center, radius)`: Create sphere test volume
- `create_cube_volume(shape, corner, size)`: Create cube test volume
- `create_cylinder_volume(shape, center, radius, axis)`: Create cylinder test volume
- `create_porous_volume(shape, porosity)`: Create volume with known porosity
- `create_filament_volume(shape, n_filaments, filament_radius)`: Create filament test volume

## ✅ Test Coverage Goals

- **Core Modules**: 90%+ coverage
- **Preprocessing**: 85%+ coverage
- **Analysis Modules**: 80%+ coverage
- **Quality Modules**: 85%+ coverage
- **Experimental Modules**: 75%+ coverage
- **Overall**: 80%+ coverage

## 📈 Continuous Integration

Tests are designed to run in CI/CD pipelines:

- Pre-commit: Fast unit tests
- Pull Request: All unit + integration tests
- Release: Full test suite + performance benchmarks

## 🔍 Test Validation

Tests use:

- **Known Value Validation**: Compare with analytical solutions
- **Cross-Validation**: Compare with other tools
- **Regression Testing**: Compare with reference results

## 📚 Documentation

- **[Test Plan](../docs/tests.md)** - Comprehensive testing strategy and detailed test plan
- **[Framework Documentation](../docs/README.md)** - Complete framework documentation
- **[API Reference](../docs/api.md)** - Function and class documentation

## 🔧 Configuration

Pytest is configured via `pytest.ini` at the project root. Key settings:

- **Test discovery**: Automatically finds `test_*.py` files
- **Markers**: `unit`, `integration`, `slow`, `requires_data`
- **Output**: Verbose mode with colored output
- **Coverage**: Can be enabled with `--cov=src` flag

## 📊 Test Statistics

- **Total test files**: 20+
- **Test categories**: 8 (core, preprocessing, analysis, quality, experimental, integration, utils, analyzer)
- **Test fixtures**: 10+ shared fixtures
- **Coverage**: All major modules have test coverage

