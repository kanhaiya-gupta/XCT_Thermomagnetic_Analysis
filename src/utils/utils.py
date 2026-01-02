"""
Utility Functions for XCT Thermomagnetic Analysis

Common utility functions for file I/O, data conversion, coordinate transformations,
unit conversions, and helper functions used across all analysis modules.
"""

import numpy as np
import os
import re
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
from enum import Enum
import logging

try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


class Unit(Enum):
    """Supported unit systems."""

    MICROMETER = "um"
    MICROMETRE = "µm"
    MILLIMETER = "mm"
    CENTIMETER = "cm"
    METER = "m"
    PIXEL = "pixel"
    VOXEL = "voxel"


# Unit conversion factors to millimeters (base unit)
UNIT_TO_MM = {
    "um": 0.001,
    "µm": 0.001,
    "micrometer": 0.001,
    "micrometre": 0.001,
    "micrometers": 0.001,
    "micrometres": 0.001,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "millimeters": 1.0,
    "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimetre": 10.0,
    "centimeters": 10.0,
    "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "metre": 1000.0,
    "meters": 1000.0,
    "metres": 1000.0,
    "pixel": 1.0,  # Default: assume 1 pixel = 1 mm if not specified
    "voxel": 1.0,  # Default: assume 1 voxel = 1 mm if not specified
}


def detect_unit_from_string(unit_str: str) -> Optional[str]:
    """
    Detect unit from string (e.g., "0.1 mm", "100 um", "1.5 cm").

    Args:
        unit_str: String containing unit information

    Returns:
        Detected unit abbreviation (mm, um, cm, etc.) or None
    """
    if unit_str is None:
        return None

    unit_str = str(unit_str).lower().strip()

    # Remove common prefixes/suffixes
    unit_str = re.sub(r"^per\s+", "", unit_str)
    unit_str = re.sub(r"^/voxel$", "", unit_str)
    unit_str = re.sub(r"^/pixel$", "", unit_str)

    # Check for explicit unit mentions
    for unit_key, _ in UNIT_TO_MM.items():
        if unit_key in unit_str:
            return unit_key

    # Check for common patterns
    patterns = [
        (r"(\d+\.?\d*)\s*(um|µm|micrometer|micrometre)", "um"),
        (r"(\d+\.?\d*)\s*(mm|millimeter|millimetre)", "mm"),
        (r"(\d+\.?\d*)\s*(cm|centimeter|centimetre)", "cm"),
        (r"(\d+\.?\d*)\s*(m|meter|metre)(?!m)", "m"),  # 'm' but not 'mm' or 'um'
    ]

    for pattern, unit in patterns:
        if re.search(pattern, unit_str):
            return unit

    return None


def convert_to_mm(
    value: Union[float, Tuple[float, ...], np.ndarray], from_unit: str
) -> Union[float, Tuple[float, ...], np.ndarray]:
    """
    Convert value(s) from specified unit to millimeters.

    Args:
        value: Value(s) to convert (scalar, tuple, or array)
        from_unit: Source unit (e.g., 'um', 'mm', 'cm', 'm')

    Returns:
        Converted value(s) in millimeters
    """
    from_unit = from_unit.lower().strip()

    if from_unit not in UNIT_TO_MM:
        logger.warning(f"Unknown unit '{from_unit}', assuming millimeters")
        conversion_factor = 1.0
    else:
        conversion_factor = UNIT_TO_MM[from_unit]

    if isinstance(value, (tuple, list)):
        return tuple(v * conversion_factor for v in value)
    elif isinstance(value, np.ndarray):
        return value * conversion_factor
    else:
        return value * conversion_factor


def convert_from_mm(
    value: Union[float, Tuple[float, ...], np.ndarray], to_unit: str
) -> Union[float, Tuple[float, ...], np.ndarray]:
    """
    Convert value(s) from millimeters to specified unit.

    Args:
        value: Value(s) in millimeters (scalar, tuple, or array)
        to_unit: Target unit (e.g., 'um', 'mm', 'cm', 'm')

    Returns:
        Converted value(s) in target unit
    """
    to_unit = to_unit.lower().strip()

    if to_unit not in UNIT_TO_MM:
        logger.warning(f"Unknown unit '{to_unit}', assuming millimeters")
        conversion_factor = 1.0
    else:
        conversion_factor = 1.0 / UNIT_TO_MM[to_unit]

    if isinstance(value, (tuple, list)):
        return tuple(v * conversion_factor for v in value)
    elif isinstance(value, np.ndarray):
        return value * conversion_factor
    else:
        return value * conversion_factor


def normalize_units(
    metadata: Union[Dict[str, Any], float],
    default_unit: str = "mm",
    target_unit: str = "mm",
) -> Union[Dict[str, Any], float]:
    """
    Normalize units in metadata to target unit (default: mm).

    Can also be used to convert a single value from one unit to another.

    Args:
        metadata: Metadata dictionary (may contain spacing, origin, etc.) OR a single float value
        default_unit: Default unit if not specified in metadata (or source unit for single value)
        target_unit: Target unit for normalization (default: 'mm')

    Returns:
        Updated metadata with normalized units, or converted float value
    """
    # Handle single value conversion
    if isinstance(metadata, (int, float)):
        if default_unit == target_unit:
            return float(metadata)
        # Convert: value -> mm -> target_unit
        value_mm = convert_to_mm(metadata, default_unit)
        if target_unit != "mm":
            return float(convert_from_mm(value_mm, target_unit))
        return float(value_mm)

    # Handle metadata dictionary
    normalized_metadata = metadata.copy()

    # Detect unit from metadata
    detected_unit = None
    if "unit" in metadata:
        detected_unit = metadata["unit"].lower().strip()
    elif "units" in metadata:
        detected_unit = metadata["units"].lower().strip()
    elif "spacing_unit" in metadata:
        detected_unit = metadata["spacing_unit"].lower().strip()

    # Try to detect from spacing string if it's a string
    if detected_unit is None and "spacing" in metadata:
        spacing = metadata["spacing"]
        if isinstance(spacing, str):
            detected_unit = detect_unit_from_string(spacing)

    # Use detected unit or default
    source_unit = detected_unit or default_unit

    # Convert spacing if present
    if "spacing" in normalized_metadata:
        spacing = normalized_metadata["spacing"]
        if isinstance(spacing, (tuple, list, np.ndarray)):
            if source_unit != target_unit:
                normalized_metadata["spacing"] = convert_to_mm(spacing, source_unit)
                if target_unit != "mm":
                    normalized_metadata["spacing"] = convert_from_mm(
                        normalized_metadata["spacing"], target_unit
                    )
        normalized_metadata["spacing_unit"] = target_unit

    # Convert origin if present
    if "origin" in normalized_metadata:
        origin = normalized_metadata["origin"]
        if isinstance(origin, (tuple, list, np.ndarray)):
            if source_unit != target_unit:
                normalized_metadata["origin"] = convert_to_mm(origin, source_unit)
                if target_unit != "mm":
                    normalized_metadata["origin"] = convert_from_mm(
                        normalized_metadata["origin"], target_unit
                    )

    # Store original unit for reference
    if source_unit != target_unit:
        normalized_metadata["original_unit"] = source_unit
        normalized_metadata["unit"] = target_unit

    return normalized_metadata


def parse_voxel_size_with_unit(
    voxel_size_str: Union[str, float, Tuple], default_unit: str = "mm"
) -> Union[Tuple[float, str], Tuple[float, float, float, str]]:
    """
    Parse voxel size string that may include unit information.

    Args:
        voxel_size_str: Voxel size as string (e.g., "0.1 mm", "100 um") or numeric
        default_unit: Default unit if not specified

    Returns:
        For single value strings: Tuple of (size, unit)
        For tuple/array inputs: Tuple of (dx, dy, dz, unit) in millimeters
    """
    if isinstance(voxel_size_str, (tuple, list, np.ndarray)):
        if len(voxel_size_str) == 3:
            # Check if last element is a unit string
            if isinstance(voxel_size_str[2], str):
                unit = detect_unit_from_string(voxel_size_str[2]) or default_unit
                spacing = tuple(voxel_size_str[:2])
            else:
                unit = default_unit
                spacing = tuple(voxel_size_str)
        else:
            unit = default_unit
            spacing = (
                tuple(voxel_size_str[:3])
                if len(voxel_size_str) >= 3
                else (1.0, 1.0, 1.0)
            )

        # Convert to millimeters and return (dx, dy, dz, unit)
        spacing_mm = convert_to_mm(spacing, unit)
        return spacing_mm[0], spacing_mm[1], spacing_mm[2], "mm"
    elif isinstance(voxel_size_str, str):
        # Try to extract unit and value
        unit = detect_unit_from_string(voxel_size_str) or default_unit
        # Extract numeric values
        numbers = re.findall(r"\d+\.?\d*", voxel_size_str)
        if len(numbers) >= 3:
            # Multiple values: return (dx, dy, dz, unit) - convert to mm for consistency
            spacing = tuple(float(n) for n in numbers[:3])
            spacing_mm = convert_to_mm(spacing, unit)
            return spacing_mm[0], spacing_mm[1], spacing_mm[2], "mm"
        elif len(numbers) == 1:
            # Single value: return (size, unit) - keep original value and unit
            size = float(numbers[0])
            return size, unit
        else:
            # No numbers found, return default
            return (1.0, default_unit)
    else:
        # Numeric value - return as single value with default unit
        unit = default_unit
        size = (
            float(voxel_size_str) if isinstance(voxel_size_str, (int, float)) else 1.0
        )
        return size, unit


def load_volume(
    file_path: Union[str, Path],
    dimensions: Optional[Tuple[int, int, int]] = None,
    dtype: Optional[np.dtype] = None,
    spacing: Optional[Union[Tuple[float, float, float], str]] = None,
    spacing_unit: Optional[str] = None,
    normalize_units_to: str = "mm",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load XCT volume from various file formats.

    Args:
        file_path: Path to the volume file
        dimensions: (width, height, depth) for RAW files
        dtype: Data type for RAW files (e.g., np.uint16)
        spacing: Voxel spacing (dx, dy, dz) - can be tuple, string with unit, or None
        spacing_unit: Unit of spacing if spacing is numeric (e.g., 'mm', 'um', 'cm')
        normalize_units_to: Target unit for normalization (default: 'mm')

    Returns:
        volume: 3D numpy array
        metadata: Dictionary with metadata (spacing in normalized units, origin, etc.)

    Supported formats:
        - DICOM (.dcm, .dicom)
        - TIFF (.tif, .tiff)
        - RAW (.raw)
        - NIfTI (.nii, .nii.gz)
        - MHD/MHA (.mhd, .mha)
        - NumPy (.npy, .npz)
        - CSV/Excel (.csv, .xlsx, .xls)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # Parse spacing with unit handling
    if spacing is not None:
        if isinstance(spacing, str):
            result = parse_voxel_size_with_unit(
                spacing, default_unit=spacing_unit or "mm"
            )
            # Handle both (size, unit) and (dx, dy, dz, unit) returns
            if len(result) == 2:
                # Single value: (size, unit) - convert to mm
                size, unit = result
                spacing_mm = convert_to_mm((size, size, size), unit)
            else:
                # Multiple values: (dx, dy, dz, unit) - values already in mm if unit is 'mm'
                dx, dy, dz, unit = result
                if unit == "mm":
                    spacing_mm = (dx, dy, dz)
                else:
                    spacing_mm = convert_to_mm((dx, dy, dz), unit)
        elif spacing_unit:
            spacing_mm = convert_to_mm(spacing, spacing_unit)
        else:
            spacing_mm = spacing  # Assume mm if no unit specified
    else:
        spacing_mm = (1.0, 1.0, 1.0)

    metadata = {
        "spacing": spacing_mm,
        "origin": (0.0, 0.0, 0.0),
        "unit": normalize_units_to,
    }

    # DICOM
    if suffix in [".dcm", ".dicom"]:
        if HAS_SITK:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(file_path))
            image = reader.Execute()
            volume = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
            origin = image.GetOrigin()
            # ITK spacing is typically in mm, but check metadata
            metadata["spacing"] = tuple(
                convert_to_mm(spacing[::-1], "mm")
            )  # ITK uses (x,y,z), numpy uses (z,y,x)
            metadata["origin"] = tuple(convert_to_mm(origin[::-1], "mm"))
            metadata["unit"] = normalize_units_to
        elif HAS_PYDICOM:
            ds = pydicom.dcmread(file_path)
            volume = ds.pixel_array
            if hasattr(ds, "SliceThickness") and hasattr(ds, "PixelSpacing"):
                # DICOM spacing is typically in mm
                spacing_dicom = (
                    ds.SliceThickness,
                    ds.PixelSpacing[0],
                    ds.PixelSpacing[1],
                )
                metadata["spacing"] = tuple(convert_to_mm(spacing_dicom, "mm"))
                metadata["unit"] = normalize_units_to
        else:
            raise ImportError("SimpleITK or pydicom required for DICOM support")

    # TIFF
    elif suffix in [".tif", ".tiff"]:
        try:
            from PIL import Image
            import imageio

            volume = imageio.volread(str(file_path))
        except ImportError:
            raise ImportError("imageio or PIL required for TIFF support")

    # RAW binary
    elif suffix == ".raw":
        if dimensions is None or dtype is None:
            raise ValueError("dimensions and dtype required for RAW files")
        volume = np.fromfile(file_path, dtype=dtype)
        volume = volume.reshape(dimensions)

    # NIfTI
    elif suffix in [".nii", ".gz"]:
        try:
            import nibabel as nib

            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
            header = nii.header
            if hasattr(header, "get_zooms"):
                spacing_nifti = header.get_zooms()[:3]
                # NIfTI spacing is typically in mm
                metadata["spacing"] = tuple(convert_to_mm(spacing_nifti, "mm"))
                metadata["unit"] = normalize_units_to
        except ImportError:
            raise ImportError("nibabel required for NIfTI support")

    # MHD/MHA
    elif suffix in [".mhd", ".mha"]:
        if HAS_SITK:
            image = sitk.ReadImage(str(file_path))
            volume = sitk.GetArrayFromImage(image)
            spacing_mhd = image.GetSpacing()[::-1]
            origin_mhd = image.GetOrigin()[::-1]
            # MHD spacing is typically in mm
            metadata["spacing"] = tuple(convert_to_mm(spacing_mhd, "mm"))
            metadata["origin"] = tuple(convert_to_mm(origin_mhd, "mm"))
            metadata["unit"] = normalize_units_to
        else:
            raise ImportError("SimpleITK required for MHD/MHA support")

    # NumPy
    elif suffix in [".npy", ".npz"]:
        if suffix == ".npy":
            volume = np.load(file_path)
        else:
            data = np.load(file_path)
            volume = data["volume"] if "volume" in data else data[list(data.keys())[0]]
            # Load all metadata from npz file
            for key in data.keys():
                if key != "volume" and key not in metadata:
                    try:
                        metadata[key] = (
                            data[key].item()
                            if hasattr(data[key], "item")
                            else data[key]
                        )
                    except:
                        metadata[key] = data[key]

            # Handle spacing/voxel_size
            if "spacing" in data:
                spacing_npz = tuple(data["spacing"])
                # Check if unit info is in metadata
                unit = "mm"  # Default
                if "unit" in data:
                    unit = str(data["unit"]).lower()
                elif "spacing_unit" in data:
                    unit = str(data["spacing_unit"]).lower()
                metadata["spacing"] = tuple(convert_to_mm(spacing_npz, unit))
                metadata["unit"] = normalize_units_to
            elif "voxel_size" in data:
                # Also check for voxel_size key (alias for spacing)
                voxel_size_npz = tuple(data["voxel_size"])
                unit = "mm"  # Default
                if "unit" in data:
                    unit = str(data["unit"]).lower()
                elif "spacing_unit" in data:
                    unit = str(data["spacing_unit"]).lower()
                metadata["spacing"] = tuple(convert_to_mm(voxel_size_npz, unit))
                metadata["voxel_size"] = metadata[
                    "spacing"
                ]  # Keep both for compatibility
                metadata["unit"] = normalize_units_to

    # CSV/Excel - Tabular data with spatial coordinates
    elif suffix in [".csv", ".xlsx", ".xls"]:
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV/Excel support")

        # For CSV/Excel, spacing might be in different units
        spacing_for_tab = spacing_mm if spacing is not None else (1.0, 1.0, 1.0)
        volume, tab_metadata = _load_tabular_data(
            file_path, suffix, dimensions, spacing_for_tab
        )
        metadata.update(tab_metadata)
        metadata["unit"] = normalize_units_to
        # Normalize units in tabular metadata
        metadata = normalize_units(
            metadata, default_unit=spacing_unit or "mm", target_unit=normalize_units_to
        )
        return volume, metadata

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Ensure 3D
    if volume.ndim == 2:
        volume = volume[np.newaxis, :, :]
    elif volume.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume, got {volume.ndim}D")

    # Normalize all units to target unit
    metadata = normalize_units(
        metadata, default_unit="mm", target_unit=normalize_units_to
    )

    logger.info(
        f"Loaded volume: shape={volume.shape}, dtype={volume.dtype}, "
        f"spacing={metadata['spacing']} {metadata.get('unit', normalize_units_to)}"
    )
    return volume, metadata


def save_volume(
    volume: np.ndarray,
    file_path: Union[str, Path],
    spacing: Optional[Tuple[float, float, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save volume to file.

    Args:
        volume: 3D numpy array
        file_path: Output file path
        spacing: Voxel spacing in mm
        metadata: Additional metadata
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        np.save(file_path, volume)
    elif suffix == ".npz":
        save_dict = {"volume": volume}
        if spacing:
            save_dict["spacing"] = np.array(spacing)
        if metadata:
            save_dict.update(metadata)
        np.savez(file_path, **save_dict)
    elif suffix in [".tif", ".tiff"]:
        try:
            import imageio

            imageio.volwrite(str(file_path), volume)
        except ImportError:
            raise ImportError("imageio required for TIFF support")
    elif suffix in [".mhd", ".mha"]:
        if HAS_SITK:
            image = sitk.GetImageFromArray(volume)
            if spacing:
                image.SetSpacing(spacing[::-1])
            sitk.WriteImage(image, str(file_path))
        else:
            raise ImportError("SimpleITK required for MHD/MHA support")
    elif suffix in [".csv", ".xlsx", ".xls"]:
        # Use save_segmented_data for CSV/Excel
        save_segmented_data(
            volume, file_path, voxel_size=spacing, save_only_nonzero=True
        )
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def normalize_volume(
    volume: np.ndarray,
    method: str = "minmax",
    percentile: Tuple[float, float] = (1, 99),
) -> np.ndarray:
    """
    Normalize volume intensity values.

    Args:
        volume: Input volume
        method: Normalization method ('minmax', 'zscore', 'percentile')
        percentile: Percentile range for percentile normalization

    Returns:
        Normalized volume
    """
    if method == "minmax":
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            return (volume - vmin) / (vmax - vmin)
        return volume

    elif method == "zscore":
        mean = volume.mean()
        std = volume.std()
        if std > 0:
            return (volume - mean) / std
        return volume

    elif method == "percentile":
        pmin, pmax = np.percentile(volume, percentile)
        if pmax > pmin:
            return np.clip((volume - pmin) / (pmax - pmin), 0, 1)
        return volume

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_voxel_volume(voxel_size: Tuple[float, float, float]) -> float:
    """
    Calculate volume of a single voxel.

    Args:
        voxel_size: (dx, dy, dz) in mm

    Returns:
        Voxel volume in mm³
    """
    return np.prod(voxel_size)


def convert_to_physical_coords(
    voxel_coords: np.ndarray,
    voxel_size: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert voxel coordinates to physical coordinates.

    Args:
        voxel_coords: Voxel coordinates (N, 3) or (3,)
        voxel_size: Voxel spacing (dx, dy, dz)
        origin: Origin offset (x, y, z)

    Returns:
        Physical coordinates in mm
    """
    voxel_coords = np.asarray(voxel_coords)
    if voxel_coords.ndim == 1:
        return np.array(origin) + voxel_coords * np.array(voxel_size)
    return np.array(origin) + voxel_coords * np.array(voxel_size)


def convert_to_voxel_coords(
    physical_coords: np.ndarray,
    voxel_size: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert physical coordinates to voxel coordinates.

    Args:
        physical_coords: Physical coordinates in mm (N, 3) or (3,)
        voxel_size: Voxel spacing (dx, dy, dz)
        origin: Origin offset (x, y, z)

    Returns:
        Voxel coordinates
    """
    physical_coords = np.asarray(physical_coords)
    if physical_coords.ndim == 1:
        return ((physical_coords - np.array(origin)) / np.array(voxel_size)).astype(int)
    return ((physical_coords - np.array(origin)) / np.array(voxel_size)).astype(int)


def ensure_binary(volume: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    Ensure volume is binary (0 and 1).

    Args:
        volume: Input volume
        threshold: Threshold value (if None, uses > 0)

    Returns:
        Binary volume (0 and 1)
    """
    if threshold is None:
        return (volume > 0).astype(np.uint8)
    return (volume > threshold).astype(np.uint8)


def get_volume_info(volume: np.ndarray) -> Dict[str, Any]:
    """
    Get information about a volume.

    Args:
        volume: Input volume

    Returns:
        Dictionary with volume information
    """
    return {
        "shape": volume.shape,
        "dtype": str(volume.dtype),
        "size": volume.size,
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std()),
        "memory_mb": volume.nbytes / (1024 * 1024),
    }


def _load_tabular_data(
    file_path: Union[str, Path],
    suffix: str,
    dimensions: Optional[Tuple[int, int, int]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load segmented data from CSV or Excel files.

    Expected column formats:
    - Option 1: x, y, z, value (or intensity, segmented, etc.)
    - Option 2: row, col, slice, value
    - Option 3: Already in 3D format with index columns

    Args:
        file_path: Path to CSV/Excel file
        suffix: File extension
        dimensions: Expected dimensions (if known)
        voxel_size: Voxel spacing (if known)

    Returns:
        volume: 3D numpy array
        metadata: Dictionary with metadata
    """
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")

    logger.info(f"Loaded tabular data: {len(df)} rows, columns: {list(df.columns)}")

    # Try to identify coordinate columns
    coord_cols = []
    value_col = None

    # Common column name patterns
    coord_patterns = {
        "x": ["x", "X", "x_coord", "x_coordinate", "column", "col", "j"],
        "y": ["y", "Y", "y_coord", "y_coordinate", "row", "i"],
        "z": ["z", "Z", "z_coord", "z_coordinate", "slice", "layer", "k"],
    }

    value_patterns = [
        "value",
        "intensity",
        "segmented",
        "binary",
        "label",
        "data",
        "voxel_value",
    ]

    # Find coordinate columns
    for coord_name, patterns in coord_patterns.items():
        for col in df.columns:
            if col.lower() in patterns:
                coord_cols.append((coord_name, col))
                break

    # Find value column
    for col in df.columns:
        if col.lower() in value_patterns or col.lower() not in [
            c[1] for c in coord_cols
        ]:
            if value_col is None:
                value_col = col
            break

    if not coord_cols or value_col is None:
        # Try to infer from data structure
        # If we have exactly 3 numeric columns + 1 value column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 4:
            coord_cols = [
                (name, col) for name, col in zip(["x", "y", "z"], numeric_cols[:3])
            ]
            value_col = numeric_cols[3] if len(numeric_cols) > 3 else df.columns[-1]
        else:
            raise ValueError(
                f"Could not identify coordinate and value columns. "
                f"Found columns: {list(df.columns)}. "
                f"Expected columns like: x, y, z, value"
            )

    logger.info(
        f"Identified coordinates: {[c[1] for c in coord_cols]}, value column: {value_col}"
    )

    # Extract coordinates and values
    coords = df[[c[1] for c in coord_cols]].values.astype(int)
    values = df[value_col].values

    # Determine volume dimensions
    if dimensions is None:
        dims = []
        for i, (coord_name, col) in enumerate(coord_cols):
            dim = int(coords[:, i].max() - coords[:, i].min() + 1)
            dims.append(dim)
        dimensions = tuple(dims)

    # Create 3D volume
    volume = np.zeros(dimensions, dtype=values.dtype)

    # Adjust coordinates to start from 0
    coords_adjusted = coords - coords.min(axis=0)

    # Fill volume
    for coord, value in zip(coords_adjusted, values):
        if all(0 <= coord[i] < dimensions[i] for i in range(3)):
            volume[tuple(coord)] = value

    # Convert voxel_size to mm if provided
    if voxel_size is not None:
        spacing_mm = (
            convert_to_mm(voxel_size, "mm")
            if isinstance(voxel_size, (tuple, list, np.ndarray))
            else (1.0, 1.0, 1.0)
        )
    else:
        spacing_mm = (1.0, 1.0, 1.0)

    metadata = {
        "spacing": spacing_mm,
        "origin": (0.0, 0.0, 0.0),
        "source_format": "tabular",
        "columns": list(df.columns),
        "n_points": len(df),
        "unit": "mm",
    }

    logger.info(
        f"Reconstructed volume: shape={volume.shape}, "
        f"filled {np.sum(volume != 0)} voxels from {len(df)} data points"
    )

    return volume, metadata


def load_segmented_data(
    file_path: Union[str, Path],
    coordinate_columns: Optional[list] = None,
    value_column: Optional[str] = None,
    dimensions: Optional[Tuple[int, int, int]] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load segmented data from CSV or Excel file with explicit column specification.

    Args:
        file_path: Path to CSV/Excel file
        coordinate_columns: List of column names for coordinates [x_col, y_col, z_col]
        value_column: Column name for values
        dimensions: Expected dimensions (if None, inferred from data)
        voxel_size: Voxel spacing in mm

    Returns:
        volume: 3D numpy array
        metadata: Dictionary with metadata

    Example:
        >>> volume, metadata = load_segmented_data(
        ...     'data.csv',
        ...     coordinate_columns=['x', 'y', 'z'],
        ...     value_column='segmented',
        ...     dimensions=(512, 512, 512),
        ...     voxel_size=(0.1, 0.1, 0.1)
        ... )
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for CSV/Excel support")

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Use specified columns or try to infer
    if coordinate_columns is None or value_column is None:
        return _load_tabular_data(file_path, suffix, dimensions, voxel_size)

    # Extract data
    coords = df[coordinate_columns].values.astype(int)
    values = df[value_column].values

    # Determine dimensions
    if dimensions is None:
        dims = []
        for i in range(3):
            dim = int(coords[:, i].max() - coords[:, i].min() + 1)
            dims.append(dim)
        dimensions = tuple(dims)

    # Create volume
    volume = np.zeros(dimensions, dtype=values.dtype)
    coords_adjusted = coords - coords.min(axis=0)

    for coord, value in zip(coords_adjusted, values):
        if all(0 <= coord[i] < dimensions[i] for i in range(3)):
            volume[tuple(coord)] = value

    metadata = {
        "spacing": voxel_size or (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "source_format": "tabular",
        "coordinate_columns": coordinate_columns,
        "value_column": value_column,
        "n_points": len(df),
    }

    return volume, metadata


def save_segmented_data(
    volume: np.ndarray,
    file_path: Union[str, Path],
    voxel_size: Optional[Tuple[float, float, float]] = None,
    coordinate_names: Tuple[str, str, str] = ("x", "y", "z"),
    value_name: str = "value",
    save_only_nonzero: bool = True,
) -> None:
    """
    Save segmented volume to CSV or Excel file.

    Args:
        volume: 3D volume to save
        file_path: Output file path (.csv, .xlsx, or .xls)
        voxel_size: Voxel spacing (for physical coordinates)
        coordinate_names: Names for coordinate columns
        value_name: Name for value column
        save_only_nonzero: If True, only save non-zero voxels

    Example:
        >>> save_segmented_data(
        ...     volume,
        ...     'output.csv',
        ...     voxel_size=(0.1, 0.1, 0.1),
        ...     coordinate_names=('x', 'y', 'z'),
        ...     value_name='segmented'
        ... )
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for CSV/Excel support")

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Get coordinates of all voxels (or only non-zero)
    if save_only_nonzero:
        coords = np.argwhere(volume != 0)
        values = volume[volume != 0]
    else:
        coords = np.argwhere(np.ones_like(volume))
        values = volume.flatten()

    # Convert to physical coordinates if voxel_size provided
    if voxel_size:
        coords_physical = coords * np.array(voxel_size)
        df = pd.DataFrame(
            {
                coordinate_names[0]: coords_physical[:, 0],
                coordinate_names[1]: coords_physical[:, 1],
                coordinate_names[2]: coords_physical[:, 2],
                value_name: values,
            }
        )
    else:
        df = pd.DataFrame(
            {
                coordinate_names[0]: coords[:, 0],
                coordinate_names[1]: coords[:, 1],
                coordinate_names[2]: coords[:, 2],
                value_name: values,
            }
        )

    # Save
    if suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif suffix in [".xlsx", ".xls"]:
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    logger.info(f"Saved segmented data to {file_path}: {len(df)} points")


def normalize_path(
    file_path: Union[str, Path],
    base_path: Optional[Union[str, Path]] = None,
    check_data_dir: bool = True,
    data_dir_name: str = "data",
) -> Path:
    """
    Normalize file path for cross-platform compatibility (Windows, Linux, macOS, WSL).

    Handles:
    - WSL paths: /mnt/c/... or /c/... -> C:\... (on Windows)
    - Windows paths: C:\... -> /c/... (on Linux/WSL)
    - Relative paths: resolved against base_path or current working directory
    - Mixed separators: normalized to platform-appropriate separators
    - Data directory fallback: automatically checks data/ subdirectory if file not found

    Args:
        file_path: Path to normalize (can be absolute or relative)
        base_path: Optional base path for resolving relative paths
        check_data_dir: If True and file not found, check in data_dir_name subdirectory
        data_dir_name: Name of data subdirectory to check (default: 'data')

    Returns:
        Normalized Path object

    Examples:
        >>> normalize_path("/mnt/c/Users/data/file.dcm")  # WSL -> Windows
        WindowsPath('C:/Users/data/file.dcm')

        >>> normalize_path("C:\\Users\\data\\file.dcm")  # Windows -> Linux
        PosixPath('/mnt/c/Users/data/file.dcm')  # On Linux/WSL

        >>> normalize_path("../data/file.dcm", base_path="/home/user/project")
        PosixPath('/home/user/data/file.dcm')

        >>> normalize_path("file.dcm", base_path="/project", check_data_dir=True)
        # Checks /project/data/file.dcm if /project/file.dcm doesn't exist
    """
    import platform
    import os

    original_path = str(file_path)
    file_path = str(file_path)

    # Handle WSL paths on Windows
    if platform.system() == "Windows":
        # Convert /mnt/c/... or /c/... to C:\...
        if file_path.startswith("/mnt/"):
            # /mnt/c/Users/... -> C:\Users\...
            parts = file_path.split("/")
            if len(parts) >= 3 and parts[2]:
                drive_letter = parts[2].upper()
                rest_path = "/".join(parts[3:])
                # Replace forward slashes with backslashes
                rest_path_win = rest_path.replace("/", "\\")
                file_path = f"{drive_letter}:\\{rest_path_win}"
        elif (
            file_path.startswith("/") and len(file_path) > 1 and file_path[1].isalpha()
        ):
            # /c/Users/... -> C:\Users\...
            drive_letter = file_path[1].upper()
            rest_path = file_path[2:]
            # Replace forward slashes with backslashes
            rest_path_win = rest_path.replace("/", "\\")
            file_path = f"{drive_letter}:\\{rest_path_win}"

    # Handle Windows paths on Linux/WSL
    elif platform.system() in ["Linux", "Darwin"]:
        # Convert C:\... to /mnt/c/... or /c/...
        if len(file_path) >= 2 and file_path[1] == ":" and file_path[0].isalpha():
            drive_letter = file_path[0].lower()
            rest_path = file_path[2:].replace("\\", "/")
            # Check if we're in WSL (has /mnt)
            if os.path.exists("/mnt"):
                file_path = f"/mnt/{drive_letter}{rest_path}"
            else:
                # Regular Linux/Mac - just use forward slashes
                file_path = f"/{drive_letter}{rest_path}"

    # Convert to Path object
    path_obj = Path(file_path)

    # Handle relative paths
    if not path_obj.is_absolute():
        if base_path:
            base = (
                normalize_path(base_path)
                if isinstance(base_path, (str, Path))
                else Path(base_path)
            )
            if base.is_absolute():
                path_obj = base / path_obj
            else:
                # Both relative - resolve from current working directory
                path_obj = (Path.cwd() / base / path_obj).resolve()
        else:
            # Resolve relative to current working directory
            path_obj = (Path.cwd() / path_obj).resolve()

    # Normalize the path (remove redundant separators, resolve .. and .)
    try:
        path_obj = path_obj.resolve()
    except (OSError, RuntimeError):
        # If path doesn't exist yet, just normalize the string representation
        # This handles cases where we're creating new files
        pass

    # Check if file exists, if not and check_data_dir is True, try data directory
    if check_data_dir and not path_obj.exists() and base_path:
        base = (
            normalize_path(base_path)
            if isinstance(base_path, (str, Path))
            else Path(base_path)
        )
        data_dir_path = base / data_dir_name / original_path
        # Normalize the data directory path too
        try:
            data_dir_path = normalize_path(data_dir_path, check_data_dir=False)
            if data_dir_path.exists():
                return data_dir_path
        except:
            pass

    return path_obj


def create_output_directory(output_path: Union[str, Path]) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        output_path: Path to output directory

    Returns:
        Path object
    """
    output_path = normalize_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
