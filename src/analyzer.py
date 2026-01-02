"""
Main XCT Analyzer Class

Integrates all analysis modules into a unified workflow for comprehensive
XCT image analysis of 3D-printed thermomagnetic elements.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import logging

from .utils.utils import (
    load_volume,
    save_volume,
    normalize_volume,
    get_voxel_volume,
    convert_to_mm,
    convert_from_mm,
    normalize_units,
    parse_voxel_size_with_unit,
)
from .core.segmentation import segment_volume, otsu_threshold
from .core.morphology import (
    apply_morphological_operations,
    remove_small_objects,
    fill_holes,
)
from .core.metrics import compute_all_metrics
from .core.filament_analysis import estimate_filament_diameter, estimate_channel_width
from .core.porosity import analyze_porosity_distribution
from .core.slice_analysis import (
    analyze_slice_along_flow,
    analyze_slice_perpendicular_flow,
)
from .core.visualization import (
    visualize_3d_volume,
    visualize_slice,
    plot_porosity_profile,
    plot_metrics_comparison,
    create_analysis_report,
)

logger = logging.getLogger(__name__)


class XCTAnalyzer:
    """
    Main analyzer class for XCT image analysis.

    Integrates segmentation, morphological operations, metrics computation,
    filament analysis, porosity analysis, and slice analysis.
    """

    def __init__(
        self,
        voxel_size: Union[Tuple[float, float, float], str] = (0.1, 0.1, 0.1),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size_unit: str = "mm",
        target_unit: str = "mm",
    ):
        """
        Initialize XCT analyzer.

        Args:
            voxel_size: Voxel spacing (dx, dy, dz) - can be tuple or string with unit (e.g., "0.1 mm", "100 um")
            origin: Origin coordinates (x, y, z) in target_unit
            voxel_size_unit: Unit of voxel_size if numeric (default: 'mm')
            target_unit: Target unit for all analysis (default: 'mm')
        """
        # Parse and convert voxel_size to target unit
        if isinstance(voxel_size, str):
            spacing_mm, _, _, _ = parse_voxel_size_with_unit(
                voxel_size, default_unit=voxel_size_unit
            )
        else:
            spacing_mm = convert_to_mm(voxel_size, voxel_size_unit)
            if target_unit != "mm":
                spacing_mm = convert_from_mm(spacing_mm, target_unit)

        self.voxel_size = (
            spacing_mm if isinstance(spacing_mm, tuple) else tuple(spacing_mm)
        )
        self.origin = origin
        self.target_unit = target_unit
        self.volume = None
        self.original_volume = None
        self.segmented_volume = None
        self.metadata = {}
        self.metrics = {}
        self.filament_results = {}
        self.porosity_results = {}
        self.slice_results = {}

        logger.info(f"XCTAnalyzer initialized with voxel_size={voxel_size} mm")

    def load_volume(
        self,
        file_path: Union[str, Path],
        dimensions: Optional[Tuple[int, int, int]] = None,
        dtype: Optional[np.dtype] = None,
        normalize: bool = False,
        normalize_method: str = "minmax",
        coordinate_columns: Optional[list] = None,
        value_column: Optional[str] = None,
        spacing_unit: Optional[str] = None,
    ) -> None:
        """
        Load XCT volume from file.

        Args:
            file_path: Path to volume file
            dimensions: Dimensions for RAW files
            dtype: Data type for RAW files
            normalize: Whether to normalize intensity values
            normalize_method: Normalization method ('minmax', 'zscore', 'percentile')
            coordinate_columns: For CSV/Excel files, list of coordinate column names [x_col, y_col, z_col]
            value_column: For CSV/Excel files, name of value column
            spacing_unit: Unit of spacing if not specified in file (e.g., 'mm', 'um', 'cm')
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        # Handle CSV/Excel files with explicit column specification
        if suffix in [".csv", ".xlsx", ".xls"] and (
            coordinate_columns is not None or value_column is not None
        ):
            from .utils.utils import load_segmented_data

            self.volume, self.metadata = load_segmented_data(
                file_path,
                coordinate_columns=coordinate_columns,
                value_column=value_column,
                dimensions=dimensions,
                voxel_size=self.voxel_size,
            )
        else:
            # Use voxel_size directly (already in target_unit)
            self.volume, self.metadata = load_volume(
                file_path,
                dimensions=dimensions,
                dtype=dtype,
                spacing=self.voxel_size,
                spacing_unit=spacing_unit or self.target_unit,
                normalize_units_to=self.target_unit,
            )

        # Update voxel size from metadata (now in target_unit)
        if "spacing" in self.metadata:
            self.voxel_size = self.metadata["spacing"]

        # Store original volume
        self.original_volume = self.volume.copy()

        # Normalize if requested
        if normalize:
            self.volume = normalize_volume(self.volume, method=normalize_method)

        logger.info(
            f"Loaded volume: shape={self.volume.shape}, "
            f"dtype={self.volume.dtype}, voxel_size={self.voxel_size}"
        )

    def segment(
        self,
        volume: Optional[np.ndarray] = None,
        method: str = "otsu",
        refine: bool = True,
        min_object_size: int = 100,
        fill_holes_flag: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Segment volume using specified method.

        Args:
            volume: Optional volume to segment (if None, uses self.volume)
            method: Segmentation method ('otsu', 'multi', 'adaptive', 'manual')
            refine: Apply morphological refinement
            min_object_size: Minimum object size to keep
            fill_holes_flag: Fill holes in segmented volume
            **kwargs: Additional arguments for segmentation method

        Returns:
            Segmented binary volume
        """
        # Use provided volume or self.volume
        volume_to_segment = volume if volume is not None else self.volume

        if volume_to_segment is None:
            raise ValueError(
                "No volume provided. Either pass volume parameter or call load_volume() first."
            )

        # Segment
        self.segmented_volume = segment_volume(
            volume_to_segment, method=method, **kwargs
        )

        # Refine if requested
        if refine:
            if min_object_size > 0:
                self.segmented_volume = remove_small_objects(
                    self.segmented_volume, min_size=min_object_size
                )
            if fill_holes_flag:
                self.segmented_volume = fill_holes(self.segmented_volume)

        logger.info(
            f"Segmentation complete: method={method}, "
            f"material_fraction={np.mean(self.segmented_volume > 0):.2%}"
        )

        return self.segmented_volume

    def compute_morphology(self, operations: Optional[list] = None) -> np.ndarray:
        """
        Apply morphological operations to segmented volume.

        Args:
            operations: List of operations (if None, uses default cleaning)

        Returns:
            Morphologically processed volume
        """
        if self.segmented_volume is None:
            raise ValueError("No segmented volume. Call segment() first.")

        if operations is None:
            # Default: opening to remove small objects, then closing to fill gaps
            operations = [
                {"op": "open", "kernel_size": 3},
                {"op": "close", "kernel_size": 5},
            ]

        self.segmented_volume = apply_morphological_operations(
            self.segmented_volume, operations
        )

        logger.info("Morphological operations applied")
        return self.segmented_volume

    def compute_metrics(
        self,
        segmented_volume: Optional[np.ndarray] = None,
        include_surface_area: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute all scalar metrics.

        Args:
            segmented_volume: Optional segmented volume (if None, uses self.segmented_volume)
            include_surface_area: Whether to compute surface area (can be slow)

        Returns:
            Dictionary with all metrics
        """
        # Use provided segmented volume or self.segmented_volume
        volume_to_analyze = (
            segmented_volume if segmented_volume is not None else self.segmented_volume
        )

        if volume_to_analyze is None:
            raise ValueError(
                "No segmented volume provided. Either pass segmented_volume parameter or call segment() first."
            )

        self.metrics = compute_all_metrics(
            volume_to_analyze,
            self.voxel_size,
            include_surface_area=include_surface_area,
        )

        logger.info("Metrics computed successfully")
        return self.metrics

    def analyze_filaments(
        self, direction: str = "z", method: str = "distance_transform"
    ) -> Dict[str, Any]:
        """
        Analyze filament dimensions.

        Args:
            direction: Direction along which filaments extend ('x', 'y', 'z')
            method: Estimation method ('distance_transform', 'cross_section')

        Returns:
            Dictionary with filament analysis results
        """
        if self.segmented_volume is None:
            raise ValueError("No segmented volume. Call segment() first.")

        filament_results = estimate_filament_diameter(
            self.segmented_volume, self.voxel_size, direction=direction, method=method
        )

        channel_results = estimate_channel_width(
            self.segmented_volume, self.voxel_size, direction=direction, method=method
        )

        self.filament_results = {
            "filament_diameter": filament_results,
            "channel_width": channel_results,
            "direction": direction,
            "method": method,
        }

        logger.info(
            f"Filament analysis complete: "
            f"mean_diameter={filament_results['mean_diameter']:.3f} mm, "
            f"mean_channel_width={channel_results['mean_diameter']:.3f} mm"
        )

        return self.filament_results

    def analyze_porosity(
        self, printing_direction: str = "z", fit_distributions: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze porosity distribution.

        Args:
            printing_direction: Direction along which printing occurred
            fit_distributions: Whether to fit statistical distributions to pore sizes

        Returns:
            Dictionary with porosity analysis results
        """
        if self.segmented_volume is None:
            raise ValueError("No segmented volume. Call segment() first.")

        self.porosity_results = analyze_porosity_distribution(
            self.segmented_volume,
            self.voxel_size,
            printing_direction=printing_direction,
            fit_distributions=fit_distributions,
        )

        logger.info("Porosity analysis complete")
        return self.porosity_results

    def analyze_slices(
        self,
        flow_direction: str = "z",
        n_slices_along: Optional[int] = None,
        positions_perpendicular: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Analyze slices along and perpendicular to flow direction.

        Args:
            flow_direction: Direction of water flow ('x', 'y', 'z')
            n_slices_along: Number of slices along flow (if None, uses all)
            positions_perpendicular: Specific positions for perpendicular slices

        Returns:
            Dictionary with slice analysis results
        """
        if self.segmented_volume is None:
            raise ValueError("No segmented volume. Call segment() first.")

        # Analyze along flow
        along_flow = analyze_slice_along_flow(
            self.segmented_volume,
            flow_direction=flow_direction,
            n_slices=n_slices_along,
            voxel_size=self.voxel_size,
        )

        # Analyze perpendicular to flow
        perpendicular_flow = analyze_slice_perpendicular_flow(
            self.segmented_volume,
            flow_direction=flow_direction,
            positions=positions_perpendicular,
            voxel_size=self.voxel_size,
        )

        self.slice_results = {
            "along_flow": along_flow,
            "perpendicular_flow": perpendicular_flow,
            "flow_direction": flow_direction,
        }

        logger.info("Slice analysis complete")
        return self.slice_results

    def visualize(
        self, output_dir: Optional[Union[str, Path]] = None, sample_name: str = "sample"
    ) -> None:
        """
        Generate visualizations.

        Args:
            output_dir: Output directory for saving images
            sample_name: Sample name for file naming
        """
        if output_dir:
            from .utils.utils import create_output_directory

            output_dir = create_output_directory(output_dir)

        # 3D visualization
        if self.segmented_volume is not None:
            try:
                visualize_3d_volume(
                    self.segmented_volume,
                    voxel_size=self.voxel_size,
                    title=f"{sample_name} - Segmented Volume",
                )
            except Exception as e:
                logger.warning(f"3D visualization failed: {e}")

        # Porosity profile
        if self.porosity_results and "porosity_profile" in self.porosity_results:
            save_path = None
            if output_dir:
                save_path = str(output_dir / f"{sample_name}_porosity_profile.png")
            plot_porosity_profile(
                self.porosity_results["porosity_profile"], save_path=save_path
            )

        logger.info("Visualizations generated")

    def generate_report(
        self, output_path: Union[str, Path], sample_name: str = "sample"
    ) -> None:
        """
        Generate comprehensive analysis report.

        Args:
            output_path: Output directory path
            sample_name: Sample name
        """
        from .utils.utils import create_output_directory

        output_dir = create_output_directory(output_path)

        # Generate visualizations
        self.visualize(output_dir, sample_name)

        # Create HTML report
        create_analysis_report(self, str(output_dir), sample_name)

        # Save metrics as JSON
        import json

        metrics_path = output_dir / f"{sample_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "metrics": self.metrics,
                    "filament_results": self.filament_results,
                    "porosity_results": {
                        k: v
                        for k, v in self.porosity_results.items()
                        if k != "local_porosity_map"  # Exclude large arrays
                    },
                    "slice_results": self.slice_results,
                    "voxel_size": self.voxel_size,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Analysis report generated in {output_dir}")

    def save_segmented_volume(self, file_path: Union[str, Path]) -> None:
        """
        Save segmented volume to file.

        Args:
            file_path: Output file path
        """
        if self.segmented_volume is None:
            raise ValueError("No segmented volume to save.")

        save_volume(self.segmented_volume, file_path, spacing=self.voxel_size)
        logger.info(f"Segmented volume saved to {file_path}")
