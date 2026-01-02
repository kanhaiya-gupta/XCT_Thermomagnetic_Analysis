# Contributing Guide

Thank you for your interest in contributing to the XCT Thermomagnetic Analysis Framework!

## Development Setup

1. Fork the repository: [https://github.com/kanhaiya-gupta/XCT_Thermomagnetic_Analysis](https://github.com/kanhaiya-gupta/XCT_Thermomagnetic_Analysis)
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/XCT_Thermomagnetic_Analysis.git
   cd XCT_Thermomagnetic_Analysis
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

## Code Style

- Follow PEP 8 style guide
- Use type hints for function parameters and returns
- Write docstrings for all functions and classes
- Keep functions focused and modular

## Adding New Modules

### 1. Choose the Right Category

- **Core** (`src/core/`): Fundamental analysis operations
- **Preprocessing** (`src/preprocessing/`): Data cleaning and statistics
- **Analysis** (`src/analysis/`): Advanced analysis
- **Quality** (`src/quality/`): Quality control and validation
- **Experimental** (`src/experimental/`): Experiment-specific analysis
- **Integration** (`src/integration/`): External tool integration
- **Utils** (`src/utils/`): Utility functions

### 2. Create Module File

Create a new Python file in the appropriate directory:

```python
"""
Module Name

Brief description of what this module does.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def your_function(param1: np.ndarray, param2: float) -> Dict[str, Any]:
    """
    Function description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Dictionary with results
    """
    # Implementation
    pass
```

### 3. Update `__init__.py`

Add imports to the appropriate `__init__.py`:

```python
# In src/category/__init__.py
from .your_module import your_function

__all__ = ['your_function']
```

### 4. Update Main `__init__.py`

Add to `src/__init__.py`:

```python
from .category.your_module import your_function

__all__ = [
    # ... existing exports
    'your_function',
]
```

### 5. Write Tests

Create tests in a `tests/` directory:

```python
def test_your_function():
    # Test implementation
    pass
```

### 6. Update Documentation

- Add module to [Module Reference](modules.md)
- Add examples to [Tutorials](tutorials.md) if applicable
- Update [API Reference](api.md)

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Write/update tests
4. Update documentation
5. Ensure all tests pass
6. Submit pull request with clear description

## Documentation Standards

- All functions must have docstrings
- Include parameter descriptions
- Include return value descriptions
- Add usage examples where helpful

## Testing

Run tests before submitting:

```bash
pytest tests/
```

## Questions?

Feel free to open an issue for questions or discussions.

