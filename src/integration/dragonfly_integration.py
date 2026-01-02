"""
DragonFly Integration Module

Integrate with DragonFly software for XCT image analysis:
- Import/export volumes to/from DragonFly formats
- Import/export segmentation results
- Import/export analysis results
- Work with DragonFly project files
- Use DragonFly's analysis capabilities (if API available)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# DragonFly file formats
DRAGONFLY_VOLUME_FORMATS = [".raw", ".tif", ".tiff", ".mhd", ".mha", ".dcm", ".dicom"]
DRAGONFLY_RESULT_FORMATS = [".csv", ".json", ".txt"]
DRAGONFLY_PROJECT_FORMAT = ".dragonfly"


def import_dragonfly_volume(
    file_path: Union[str, Path],
    dimensions: Optional[Tuple[int, int, int]] = None,
    data_type: str = "uint16",
    byte_order: str = "little",
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Import volume from DragonFly-compatible format.

    DragonFly commonly uses:
    - RAW files (.raw) with metadata
    - TIFF stacks (.tif, .tiff)
    - MetaImage format (.mhd, .mha)
    - DICOM (.dcm, .dicom)

    Args:
        file_path: Path to DragonFly volume file
        dimensions: Optional (z, y, x) dimensions for RAW files
        data_type: Data type for RAW files (default: uint16)
        byte_order: Byte order for RAW files ('little' or 'big')
        voxel_size: Optional voxel spacing (mm)

    Returns:
        Tuple of (volume array, metadata dictionary)
    """
    file_path = Path(file_path)
    metadata = {
        "source": "dragonfly",
        "file_path": str(file_path),
        "file_format": file_path.suffix.lower(),
    }

    # Use existing load_volume function for supported formats
    from ..utils.utils import load_volume

    try:
        if file_path.suffix.lower() in [".raw"]:
            # RAW file - need dimensions
            if dimensions is None:
                raise ValueError("Dimensions required for RAW files")

            volume = load_volume(
                str(file_path),
                file_format="raw",
                dimensions=dimensions,
                data_type=data_type,
                byte_order=byte_order,
            )

            metadata["dimensions"] = dimensions
            metadata["data_type"] = data_type
            metadata["byte_order"] = byte_order

        elif file_path.suffix.lower() in [".tif", ".tiff"]:
            # TIFF stack
            volume = load_volume(str(file_path), file_format="tiff")

        elif file_path.suffix.lower() in [".mhd", ".mha"]:
            # MetaImage format
            volume = load_volume(str(file_path), file_format="mhd")

        elif file_path.suffix.lower() in [".dcm", ".dicom"]:
            # DICOM
            volume = load_volume(str(file_path), file_format="dicom")

        else:
            raise ValueError(f"Unsupported DragonFly format: {file_path.suffix}")

        # Add voxel size if provided
        if voxel_size:
            metadata["voxel_size"] = voxel_size
            metadata["voxel_size_unit"] = "mm"

        logger.info(f"Imported DragonFly volume: {volume.shape}, dtype={volume.dtype}")

        return volume, metadata

    except Exception as e:
        logger.error(f"Failed to import DragonFly volume: {e}")
        raise


def export_to_dragonfly_volume(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Optional[Tuple[float, float, float]] = None,
    format: str = "raw",
    metadata_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Export volume to DragonFly-compatible format.

    Args:
        volume: Volume array to export
        output_path: Output file path
        voxel_size: Optional voxel spacing (mm)
        format: Export format ('raw', 'tiff', 'mhd', 'dicom')
        metadata_file: Optional path to save metadata file

    Returns:
        Dictionary with export information
    """
    output_path = Path(output_path)

    # Use existing save_volume function
    from ..utils.utils import save_volume

    try:
        if format.lower() == "raw":
            # For RAW format, save binary data directly
            # Ensure output path has .raw extension
            if output_path.suffix.lower() != ".raw":
                output_path = output_path.with_suffix(".raw")

            # Save raw binary data
            volume.tofile(str(output_path))

            # Create metadata file for RAW
            if metadata_file is None:
                metadata_file = output_path.with_suffix(".mhd")

            # Create MetaImage header for RAW file
            _create_mhd_header(
                metadata_file,
                output_path.name,
                volume.shape,
                voxel_size or (1.0, 1.0, 1.0),
                volume.dtype,
            )

        elif format.lower() in ["tif", "tiff"]:
            # Ensure correct extension
            if output_path.suffix.lower() not in [".tif", ".tiff"]:
                output_path = output_path.with_suffix(".tiff")
            save_volume(volume, str(output_path), spacing=voxel_size)

        elif format.lower() in ["mhd", "mha"]:
            # Ensure correct extension
            if output_path.suffix.lower() not in [".mhd", ".mha"]:
                output_path = output_path.with_suffix(".mhd")
            save_volume(volume, str(output_path), spacing=voxel_size)

        elif format.lower() in ["dcm", "dicom"]:
            # Ensure correct extension
            if output_path.suffix.lower() not in [".dcm", ".dicom"]:
                output_path = output_path.with_suffix(".dcm")
            save_volume(volume, str(output_path), spacing=voxel_size)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        metadata = {
            "exported_to": "dragonfly",
            "output_path": str(output_path),
            "format": format,
            "dimensions": volume.shape,
            "dtype": str(volume.dtype),
        }

        if voxel_size:
            metadata["voxel_size"] = voxel_size
            metadata["voxel_size_unit"] = "mm"

        logger.info(f"Exported volume to DragonFly format: {output_path}")

        return metadata

    except Exception as e:
        logger.error(f"Failed to export to DragonFly format: {e}")
        raise


def _create_mhd_header(
    header_path: Path,
    data_filename: str,
    dimensions: Tuple[int, int, int],
    voxel_size: Tuple[float, float, float],
    dtype: np.dtype,
) -> None:
    """Create MetaImage header file for RAW data."""
    # Map numpy dtypes to MetaImage types
    dtype_map = {
        "uint8": "MET_UCHAR",
        "uint16": "MET_USHORT",
        "uint32": "MET_UINT",
        "int8": "MET_CHAR",
        "int16": "MET_SHORT",
        "int32": "MET_INT",
        "float32": "MET_FLOAT",
        "float64": "MET_DOUBLE",
    }

    element_type = dtype_map.get(str(dtype), "MET_USHORT")

    # MetaImage format: (x, y, z) but we store (z, y, x)
    # So we need to reverse dimensions
    dim_x, dim_y, dim_z = dimensions[2], dimensions[1], dimensions[0]
    spacing_x, spacing_y, spacing_z = voxel_size[2], voxel_size[1], voxel_size[0]

    header_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = {spacing_x} {spacing_y} {spacing_z}
DimSize = {dim_x} {dim_y} {dim_z}
ElementType = {element_type}
ElementDataFile = {data_filename}
"""

    with open(header_path, "w") as f:
        f.write(header_content)

    logger.info(f"Created MetaImage header: {header_path}")


def import_dragonfly_segmentation(
    file_path: Union[str, Path], format: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Import segmentation from DragonFly.

    DragonFly can export segmentation as:
    - Labeled volume (RAW, TIFF, MHD)
    - CSV with object properties
    - JSON with segmentation data

    Args:
        file_path: Path to DragonFly segmentation file
        format: File format ('auto', 'raw', 'tiff', 'csv', 'json')

    Returns:
        Tuple of (segmented volume, metadata)
    """
    file_path = Path(file_path)

    if format == "auto":
        format = file_path.suffix.lower().lstrip(".")

    metadata = {"source": "dragonfly", "file_path": str(file_path), "format": format}

    try:
        if format in ["raw", "tiff", "tif", "mhd", "mha"]:
            # Volume format
            volume, vol_metadata = import_dragonfly_volume(file_path)
            metadata.update(vol_metadata)
            return volume, metadata

        elif format == "csv":
            # CSV with object properties
            # Try to reconstruct volume from CSV
            df = pd.read_csv(file_path)

            # Check if CSV has spatial coordinates
            coord_cols = ["x", "y", "z", "X", "Y", "Z", "x_coord", "y_coord", "z_coord"]
            value_cols = ["label", "Label", "object_id", "segmented", "value"]

            has_coords = any(col in df.columns for col in coord_cols)
            has_values = any(col in df.columns for col in value_cols)

            if has_coords and has_values:
                # Can reconstruct volume
                from ..utils.utils import load_segmented_data

                volume, meta = load_segmented_data(str(file_path))
                metadata.update(meta)
                return volume, metadata
            else:
                # Just object properties, return as metadata
                metadata["object_properties"] = df.to_dict("records")
                return None, metadata

        elif format == "json":
            # JSON format
            with open(file_path, "r") as f:
                data = json.load(f)

            metadata.update(data.get("metadata", {}))

            # Check if JSON contains volume data
            if "volume" in data or "segmentation" in data:
                # Would need to reconstruct volume from JSON
                logger.warning("JSON volume reconstruction not fully implemented")
                return None, metadata
            else:
                return None, metadata

        else:
            raise ValueError(f"Unsupported segmentation format: {format}")

    except Exception as e:
        logger.error(f"Failed to import DragonFly segmentation: {e}")
        raise


def export_segmentation_to_dragonfly(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Optional[Tuple[float, float, float]] = None,
    format: str = "raw",
    object_properties: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Export segmentation to DragonFly-compatible format.

    Args:
        volume: Segmented volume (binary or labeled)
        output_path: Output file path
        voxel_size: Optional voxel spacing
        format: Export format ('raw', 'tiff', 'mhd')
        object_properties: Optional DataFrame with object properties

    Returns:
        Dictionary with export information
    """
    # Export volume
    metadata = export_to_dragonfly_volume(volume, output_path, voxel_size, format)

    # Export object properties if provided
    if object_properties is not None:
        props_path = Path(output_path).with_suffix(".csv")
        object_properties.to_csv(props_path, index=False)
        metadata["object_properties_file"] = str(props_path)
        logger.info(f"Exported object properties to: {props_path}")

    return metadata


def import_dragonfly_results(
    file_path: Union[str, Path], format: str = "auto"
) -> Dict[str, Any]:
    """
    Import analysis results from DragonFly.

    DragonFly exports results as:
    - CSV files with metrics
    - JSON files with structured data
    - Text files with reports

    Args:
        file_path: Path to DragonFly results file
        format: File format ('auto', 'csv', 'json', 'txt')

    Returns:
        Dictionary with DragonFly results
    """
    file_path = Path(file_path)

    if format == "auto":
        format = file_path.suffix.lower().lstrip(".")

    try:
        if format == "csv":
            df = pd.read_csv(file_path)
            # Convert to dictionary (first row if single sample, or all rows)
            if len(df) == 1:
                results = df.iloc[0].to_dict()
            else:
                results = df.to_dict("records")
            return results

        elif format == "json":
            with open(file_path, "r") as f:
                results = json.load(f)
            return results

        elif format == "txt":
            # Text report - parse if possible
            with open(file_path, "r") as f:
                content = f.read()

            # Simple parsing (would need more sophisticated parsing for full reports)
            results = {"raw_text": content}
            return results

        else:
            raise ValueError(f"Unsupported results format: {format}")

    except Exception as e:
        logger.error(f"Failed to import DragonFly results: {e}")
        raise


def export_results_to_dragonfly(
    results: Dict[str, Any], output_path: Union[str, Path], format: str = "csv"
) -> Dict[str, Any]:
    """
    Export analysis results to DragonFly-compatible format.

    Args:
        results: Results dictionary
        output_path: Output file path
        format: Export format ('csv', 'json')

    Returns:
        Dictionary with export information
    """
    output_path = Path(output_path)

    try:
        if format == "csv":
            # Convert to DataFrame
            if isinstance(results, dict):
                df = pd.DataFrame([results])
            elif isinstance(results, list):
                df = pd.DataFrame(results)
            else:
                raise ValueError("Results must be dict or list of dicts")

            df.to_csv(output_path, index=False)

        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported results to DragonFly format: {output_path}")

        return {
            "exported_to": "dragonfly",
            "output_path": str(output_path),
            "format": format,
        }

    except Exception as e:
        logger.error(f"Failed to export results to DragonFly: {e}")
        raise


def create_dragonfly_project_file(
    project_path: Union[str, Path],
    volumes: List[Dict[str, Any]],
    analyses: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a DragonFly project file (if format is known).

    Note: DragonFly project format is proprietary. This creates a JSON
    representation that can be used for reference or converted.

    Args:
        project_path: Path to save project file
        volumes: List of volume dictionaries with paths and metadata
        analyses: Optional list of analysis configurations

    Returns:
        Dictionary with project information
    """
    project_path = Path(project_path)

    # Create JSON representation (DragonFly format is proprietary)
    project_data = {
        "dragonfly_project": {
            "version": "1.0",
            "volumes": volumes,
            "analyses": analyses or [],
        }
    }

    # Save as JSON (can be converted to DragonFly format if needed)
    json_path = project_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(project_data, f, indent=2)

    logger.info(f"Created DragonFly project file: {json_path}")

    return {
        "project_path": str(json_path),
        "n_volumes": len(volumes),
        "n_analyses": len(analyses) if analyses else 0,
    }


def convert_to_dragonfly_workflow(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    output_dir: Union[str, Path],
    sample_name: str = "sample",
) -> Dict[str, Any]:
    """
    Convert our analysis workflow to DragonFly-compatible files.

    Creates a complete set of files that can be opened in DragonFly:
    - Volume file (RAW + MHD header)
    - Segmentation file (if segmented)
    - Results file (CSV)

    Args:
        volume: Volume array
        voxel_size: Voxel spacing (mm)
        output_dir: Output directory
        sample_name: Sample name for file naming

    Returns:
        Dictionary with created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = {}

    # Export volume
    volume_path = output_dir / f"{sample_name}_volume.raw"
    mhd_path = output_dir / f"{sample_name}_volume.mhd"

    export_to_dragonfly_volume(
        volume, volume_path, voxel_size, format="raw", metadata_file=mhd_path
    )

    created_files["volume"] = str(volume_path)
    created_files["volume_header"] = str(mhd_path)

    # If volume is binary/segmented, export segmentation
    if volume.dtype == bool or np.unique(volume).size <= 256:
        seg_path = output_dir / f"{sample_name}_segmentation.raw"
        seg_mhd_path = output_dir / f"{sample_name}_segmentation.mhd"

        export_segmentation_to_dragonfly(
            volume.astype(np.uint8), seg_path, voxel_size, format="raw"
        )

        created_files["segmentation"] = str(seg_path)
        created_files["segmentation_header"] = str(seg_mhd_path)

    logger.info(f"Created DragonFly workflow files in: {output_dir}")

    return {
        "output_directory": str(output_dir),
        "created_files": created_files,
        "sample_name": sample_name,
    }


def comprehensive_dragonfly_integration(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float],
    analysis_results: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "dragonfly_export",
    sample_name: str = "sample",
) -> Dict[str, Any]:
    """
    Comprehensive DragonFly integration: export everything needed.

    Args:
        volume: Volume array
        voxel_size: Voxel spacing
        analysis_results: Optional analysis results to export
        output_dir: Output directory
        sample_name: Sample name

    Returns:
        Dictionary with all exported files and integration status
    """
    output_dir = Path(output_dir)

    # Create workflow files
    workflow = convert_to_dragonfly_workflow(
        volume, voxel_size, output_dir, sample_name
    )

    # Export results if provided
    if analysis_results:
        results_path = output_dir / f"{sample_name}_results.csv"
        export_results_to_dragonfly(analysis_results, results_path)
        workflow["results_file"] = str(results_path)

    # Summary
    summary = {
        "integration_complete": True,
        "output_directory": str(output_dir),
        "sample_name": sample_name,
        "files_created": workflow.get("created_files", {}),
        "dragonfly_ready": True,
        "instructions": [
            f"1. Open DragonFly software",
            f"2. Load volume: {workflow.get('created_files', {}).get('volume_header', 'N/A')}",
            f"3. Load segmentation (if available): {workflow.get('created_files', {}).get('segmentation_header', 'N/A')}",
            f"4. Compare with results: {workflow.get('results_file', 'N/A')}",
        ],
    }

    return summary
