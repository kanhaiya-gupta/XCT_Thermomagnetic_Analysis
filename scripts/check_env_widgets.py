#!/usr/bin/env python
"""
XCT Thermomagnetic Analysis Framework - Environment Check Script

Checks if ipywidgets and ipykernel are installed for Jupyter notebook support.
Also provides instructions for registering the kernel.
"""

import sys
import subprocess
from pathlib import Path

# Project information
PROJECT_NAME = "XCT Thermomagnetic Analysis Framework"
PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 70)
print(f"{PROJECT_NAME}")
print("Jupyter Environment Check")
print("=" * 70)
print()

# Check current Python
print("Python Environment:")
print(f"  Executable: {sys.executable}")
print(f"  Version: {sys.version.split()[0]}")
print(f"  Path: {Path(sys.executable).parent}")
print()

# Try to import ipywidgets
print("Checking ipywidgets...")
try:
    import ipywidgets as widgets
    print(f"✅ ipywidgets is installed!")
    print(f"   Version: {widgets.__version__}")
    print(f"   Location: {widgets.__file__}")
    widgets_ok = True
except ImportError as e:
    print(f"❌ ipywidgets is NOT installed in this environment!")
    print(f"   Error: {e}")
    print()
    print("   Install with:")
    print("   pip install ipywidgets")
    print("   jupyter nbextension enable --py widgetsnbextension --sys-prefix")
    widgets_ok = False

print()

# Check if ipykernel is installed
print("Checking ipykernel...")
try:
    import ipykernel
    print(f"✅ ipykernel is installed!")
    print(f"   Version: {ipykernel.__version__}")
    kernel_ok = True
except ImportError:
    print(f"❌ ipykernel is NOT installed!")
    print("   Install with: pip install ipykernel")
    kernel_ok = False

print()

# Check if jupyter is installed
print("Checking Jupyter...")
try:
    import jupyter
    print(f"✅ Jupyter is installed!")
    print(f"   Version: {jupyter.__version__}")
    jupyter_ok = True
except ImportError:
    print(f"⚠️  Jupyter is NOT installed!")
    print("   Install with: pip install jupyter")
    jupyter_ok = False

print()
print("=" * 70)
print("Kernel Registration Instructions")
print("=" * 70)
print()

if not kernel_ok:
    print("1. Install ipykernel first:")
    print("   pip install ipykernel")
    print()

print("2. Register the kernel for Jupyter:")
print("   python -m ipykernel install --user --name amenv --display-name 'Python (amenv)'")
print()
print("   Or with a custom name for this project:")
print(f"   python -m ipykernel install --user --name xct-thermomagnetic --display-name 'Python (XCT Thermomagnetic)'")
print()

print("3. Verify kernel registration:")
print("   jupyter kernelspec list")
print()

print("4. Start Jupyter:")
print("   jupyter notebook")
print("   # or")
print("   jupyter lab")
print()

print("5. In Jupyter, select the kernel:")
print("   - Click 'Kernel' -> 'Change Kernel' -> Select your registered kernel")
print()

if widgets_ok and kernel_ok:
    print("=" * 70)
    print("✅ Environment is ready for Jupyter notebooks!")
    print("=" * 70)
    print()
    print("You can now use the notebooks in the notebooks/ directory:")
    print("  - notebooks/01_XCT_Data_Explorer.ipynb")
    print("  - notebooks/02_Sensitivity_Virtual_Experiments.ipynb")
    print("  - notebooks/03_Comparative_Analysis_Batch_Processing.ipynb")
else:
    print("=" * 70)
    print("⚠️  Please install missing packages and register the kernel")
    print("=" * 70)

print()
