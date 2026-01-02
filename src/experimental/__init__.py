"""
Experimental module package

This package contains experimental modules for XCT analysis.
"""

# Import all modules for convenience
from .energy_conversion import *
from .flow_analysis import *
from .thermal_analysis import *

__all__ = ["flow_analysis", "thermal_analysis", "energy_conversion"]
