#!/usr/bin/env python
"""
Environment Check Script for XCT Thermomagnetic Analysis Framework

Checks if all required dependencies are installed and the environment is properly configured.
"""

import sys
import subprocess
from pathlib import Path
from importlib import import_module

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Required packages and their import names
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'scikit-image': 'skimage',
    'scikit-learn': 'sklearn',
    'Pillow': 'PIL',
    'matplotlib': 'matplotlib',
    'pyvista': 'pyvista',
    'SimpleITK': 'SimpleITK',
    'pydicom': 'pydicom',
    'nibabel': 'nibabel',
    'imageio': 'imageio',
    'pandas': 'pandas',
    'PyYAML': 'yaml',
    'SALib': 'SALib',
    'scikit-optimize': 'skopt',
}

# Optional packages
OPTIONAL_PACKAGES = {
    'pytest': 'pytest',
    'pytest-cov': 'pytest',
    'black': 'black',
    'flake8': 'flake8',
    'pylint': 'pylint',
    'mypy': 'mypy',
}

def check_package(package_name, import_name, required=True):
    """Check if a package is installed."""
    try:
        module = import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        location = getattr(module, '__file__', 'unknown')
        status = "✅" if required else "✓"
        print(f"{status} {package_name:20s} - Version: {version}")
        if required:
            print(f"   Location: {location}")
        return True
    except ImportError:
        status = "❌" if required else "⚠️"
        print(f"{status} {package_name:20s} - NOT INSTALLED")
        if required:
            print(f"   Install with: pip install {package_name}")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "=" * 60)
    print("Checking Project Structure")
    print("=" * 60)
    
    required_dirs = ['src', 'tests', 'docs', 'notebooks']
    all_present = True
    
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory is missing")
            all_present = False
    
    # Check for key source modules
    key_modules = [
        'src/core',
        'src/preprocessing',
        'src/analysis',
        'src/quality',
        'src/experimental',
        'src/utils',
    ]
    
    print("\nKey source modules:")
    for module in key_modules:
        module_path = PROJECT_ROOT / module
        if module_path.exists():
            print(f"✅ {module}/ exists")
        else:
            print(f"⚠️  {module}/ not found")
    
    return all_present

def check_python_version():
    """Check Python version compatibility."""
    print("\n" + "=" * 60)
    print("Python Environment")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python executable: {sys.executable}")
    
    # Check minimum version (3.9+)
    if version.major == 3 and version.minor >= 9:
        print("✅ Python version is compatible (3.9+)")
        return True
    else:
        print("❌ Python version should be 3.9 or higher")
        return False

def check_requirements_file():
    """Check if requirements.txt exists."""
    print("\n" + "=" * 60)
    print("Requirements File")
    print("=" * 60)
    
    req_file = PROJECT_ROOT / 'requirements.txt'
    if req_file.exists():
        print(f"✅ requirements.txt found at: {req_file}")
        with open(req_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"   Contains {len(lines)} package specifications")
        return True
    else:
        print(f"❌ requirements.txt not found at: {req_file}")
        return False

def main():
    """Main function to run all checks."""
    print("=" * 60)
    print("XCT Thermomagnetic Analysis Framework")
    print("Environment Check Script")
    print("=" * 60)
    print()
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check required packages
    print("\n" + "=" * 60)
    print("Required Packages")
    print("=" * 60)
    
    required_ok = True
    for package_name, import_name in REQUIRED_PACKAGES.items():
        if not check_package(package_name, import_name, required=True):
            required_ok = False
    
    # Check optional packages
    print("\n" + "=" * 60)
    print("Optional Packages (for development/testing)")
    print("=" * 60)
    
    for package_name, import_name in OPTIONAL_PACKAGES.items():
        check_package(package_name, import_name, required=False)
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check requirements file
    req_file_ok = check_requirements_file()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_ok = python_ok and required_ok and structure_ok and req_file_ok
    
    if all_ok:
        print("✅ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run tests: pytest tests/")
        print("2. Check documentation: docs/README.md")
        print("3. Try the notebooks: notebooks/")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nTo install all requirements:")
        print(f"  cd {PROJECT_ROOT}")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

