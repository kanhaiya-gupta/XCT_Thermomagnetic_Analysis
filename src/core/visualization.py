"""
Visualization Utilities Module

3D visualization and plotting utilities for XCT analysis results.
Includes publication-quality plotting functions for journals.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from typing import Tuple, Dict, Any, Optional, List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    logger.warning("PyVista not available, 3D visualization will be limited")


def visualize_3d_volume(
    volume: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    colormap: str = "gray",
    opacity: float = 1.0,
    show_edges: bool = False,
    title: str = "3D Volume",
) -> None:
    """
    Visualize 3D volume using PyVista.

    Args:
        volume: 3D volume (binary or grayscale)
        voxel_size: Voxel spacing in mm
        colormap: Colormap name
        opacity: Opacity (0.0 to 1.0)
        show_edges: Show edges
        title: Plot title
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for 3D visualization")
        return

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = voxel_size
    grid.origin = (0, 0, 0)

    # Set volume data
    grid["values"] = volume.flatten(order="F")

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap=colormap, opacity=opacity, show_edges=show_edges)
    plotter.add_text(title, font_size=12)
    plotter.show()


def visualize_slice(
    slice_2d: np.ndarray,
    title: str = "",
    cmap: str = "gray",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """
    Visualize 2D slice.

    Args:
        slice_2d: 2D slice
        title: Plot title
        cmap: Colormap
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(slice_2d, cmap=cmap, origin="lower")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved slice visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Publication-Quality Visualization Functions
# ============================================================================

# Journal style presets
JOURNAL_STYLES = {
    "nature": {
        "figsize_single": (3.5, 2.625),  # inches (Nature single column)
        "figsize_double": (7.0, 5.25),  # inches (Nature double column)
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "science": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "ieee": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 9,
        "fontsize_title": 11,
        "fontsize_label": 10,
        "fontsize_legend": 8,
        "linewidth": 1.2,
        "markersize": 5,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "default": {
        "figsize_single": (4, 3),
        "figsize_double": (8, 6),
        "dpi": 300,
        "fontsize": 10,
        "fontsize_title": 12,
        "fontsize_label": 11,
        "fontsize_legend": 9,
        "linewidth": 1.5,
        "markersize": 6,
        "colors": plt.cm.tab10.colors,
        "grid_alpha": 0.3,
        "spine_width": 1.0,
    },
}


def apply_publication_style(
    fig: plt.Figure, style: str = "nature", column_width: str = "single"
) -> None:
    """
    Apply publication-quality styling to figure.

    Args:
        fig: Matplotlib figure
        style: Journal style ('nature', 'science', 'ieee', 'default')
        column_width: 'single' or 'double' column
    """
    if style not in JOURNAL_STYLES:
        style = "default"
        logger.warning(f"Unknown style '{style}', using 'default'")

    style_params = JOURNAL_STYLES[style]

    # Apply to all axes
    for ax in fig.get_axes():
        # Set font sizes
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            if isinstance(item, plt.Text):
                if item == ax.title:
                    item.set_fontsize(style_params["fontsize_title"])
                elif item in [ax.xaxis.label, ax.yaxis.label]:
                    item.set_fontsize(style_params["fontsize_label"])
                else:
                    item.set_fontsize(style_params["fontsize"])

        # Set line widths
        for line in ax.get_lines():
            line.set_linewidth(style_params["linewidth"])
            if hasattr(line, "get_markersize"):
                line.set_markersize(style_params["markersize"])

        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(style_params["spine_width"])

        # Grid
        ax.grid(True, alpha=style_params["grid_alpha"], linewidth=0.5)

    # Legend font size
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(style_params["fontsize_legend"])


def publication_quality_plot(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    style: str = "nature",
    formats: List[str] = ["png", "pdf"],
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure in publication-quality format.

    Args:
        fig: Matplotlib figure
        output_path: Output file path (without extension)
        dpi: Resolution (default 300 for publications)
        style: Journal style
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
        bbox_inches: Bounding box ('tight' or 'standard')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style
    apply_publication_style(fig, style)

    # Save in all requested formats
    for fmt in formats:
        if fmt.lower() == "png":
            fig.savefig(
                f"{output_path}.png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "pdf":
            fig.savefig(
                f"{output_path}.pdf",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "svg":
            fig.savefig(
                f"{output_path}.svg",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "eps":
            fig.savefig(
                f"{output_path}.eps",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )

    logger.info(
        f"Publication-quality figure saved to {output_path} ({', '.join(formats)})"
    )


def multi_panel_figure(
    panels: List[Dict[str, Any]],
    layout: Tuple[int, int],
    figsize: Optional[Tuple[float, float]] = None,
    style: str = "nature",
    column_width: str = "double",
    suptitle: Optional[str] = None,
    panel_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create multi-panel figure for publications.

    Args:
        panels: List of panel dictionaries with 'plot_func' and 'kwargs'
        layout: (nrows, ncols) layout
        figsize: Optional figure size (inches)
        style: Journal style
        column_width: 'single' or 'double' column
        suptitle: Optional overall title
        panel_labels: Optional list of panel labels (e.g., ['a', 'b', 'c', 'd'])

    Returns:
        Matplotlib figure
    """
    if figsize is None:
        style_params = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
        if column_width == "single":
            base_size = style_params["figsize_single"]
        else:
            base_size = style_params["figsize_double"]

        # Scale by layout
        figsize = (base_size[0] * layout[1], base_size[1] * layout[0])

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    # Flatten axes if needed
    if layout[0] == 1:
        axes = axes.reshape(1, -1) if layout[1] > 1 else [axes]
    elif layout[1] == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()

    # Plot each panel
    for i, panel in enumerate(panels):
        if i >= len(axes):
            break

        ax = axes[i] if isinstance(axes, np.ndarray) else axes

        # Add panel label
        if panel_labels and i < len(panel_labels):
            ax.text(
                0.02,
                0.98,
                panel_labels[i],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Call plot function
        plot_func = panel.get("plot_func")
        if plot_func:
            plot_kwargs = panel.get("kwargs", {})
            plot_func(ax, **plot_kwargs)

    # Hide unused axes
    if isinstance(axes, np.ndarray):
        for i in range(len(panels), len(axes)):
            axes[i].axis("off")

    # Overall title
    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])[
                "fontsize_title"
            ]
            + 2,
        )

    # Apply style
    apply_publication_style(fig, style, column_width)

    plt.tight_layout()

    return fig


def export_3d_for_publication(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dpi: int = 300,
    colormap: str = "gray",
    opacity: float = 0.8,
    camera_position: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Export 3D volume rendering for publication.

    Args:
        volume: 3D volume
        output_path: Output file path
        voxel_size: Voxel spacing
        dpi: Resolution
        colormap: Colormap name
        opacity: Opacity (0.0 to 1.0)
        camera_position: Optional camera position
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for 3D export")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = voxel_size
    grid.origin = (0, 0, 0)
    grid["values"] = volume.flatten(order="F")

    # Create plotter with high resolution
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, cmap=colormap, opacity=opacity)

    # Set camera
    if camera_position:
        plotter.camera_position = camera_position

    # Export
    plotter.screenshot(
        str(output_path), resolution=(dpi * 8, dpi * 8)
    )  # High resolution
    plotter.close()

    logger.info(f"3D rendering exported to {output_path} at {dpi} DPI")


def create_figure_caption(
    analysis_type: str, metrics: Dict[str, Any], additional_info: Optional[str] = None
) -> str:
    """
    Generate figure caption for publication.

    Args:
        analysis_type: Type of analysis ('porosity', 'filament', 'dimensional', etc.)
        metrics: Dictionary of relevant metrics
        additional_info: Optional additional information

    Returns:
        Formatted caption string
    """
    captions = {
        "porosity": f"Porosity distribution analysis. Mean porosity: {metrics.get('mean_porosity', 'N/A'):.2%}, "
        f"std: {metrics.get('std_porosity', 'N/A'):.2%}.",
        "filament": f"Filament diameter analysis. Mean diameter: {metrics.get('mean_diameter', 'N/A'):.3f} mm, "
        f"std: {metrics.get('std_diameter', 'N/A'):.3f} mm.",
        "dimensional": f"Dimensional accuracy analysis. RMS deviation: {metrics.get('rms_deviation', 'N/A'):.4f} mm, "
        f"dimensional accuracy: {metrics.get('dimensional_accuracy', 'N/A'):.2f}%.",
        "slice": f"Slice analysis. Material fraction: {metrics.get('material_fraction', 'N/A'):.2%}, "
        f"void fraction: {metrics.get('void_fraction', 'N/A'):.2%}.",
        "comparison": f"Comparative analysis across {metrics.get('n_samples', 'N/A')} samples.",
    }

    base_caption = captions.get(analysis_type, f"{analysis_type} analysis results.")

    if additional_info:
        base_caption += f" {additional_info}"

    return base_caption


def plot_porosity_profile(
    porosity_data: Dict[str, Any], direction: str = "z", save_path: Optional[str] = None
) -> None:
    """
    Plot porosity profile along direction.

    Args:
        porosity_data: Dictionary from porosity_along_direction()
        direction: Direction label
        save_path: Path to save figure (optional)
    """
    positions = porosity_data["positions"]
    porosity = porosity_data["porosity"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, porosity, "b-", linewidth=2, label="Porosity")
    ax.axhline(
        y=porosity_data["mean_porosity"],
        color="r",
        linestyle="--",
        label=f"Mean: {porosity_data['mean_porosity']:.2%}",
    )
    ax.fill_between(
        positions,
        porosity_data["mean_porosity"] - porosity_data["std_porosity"],
        porosity_data["mean_porosity"] + porosity_data["std_porosity"],
        alpha=0.2,
        color="red",
        label="±1 std",
    )

    ax.set_xlabel(f"Position along {direction.upper()} axis")
    ax.set_ylabel("Porosity (fraction)")
    ax.set_title(f"Porosity Profile along {direction.upper()} Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved porosity profile to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Publication-Quality Visualization Functions
# ============================================================================

# Journal style presets
JOURNAL_STYLES = {
    "nature": {
        "figsize_single": (3.5, 2.625),  # inches (Nature single column)
        "figsize_double": (7.0, 5.25),  # inches (Nature double column)
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "science": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "ieee": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 9,
        "fontsize_title": 11,
        "fontsize_label": 10,
        "fontsize_legend": 8,
        "linewidth": 1.2,
        "markersize": 5,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "default": {
        "figsize_single": (4, 3),
        "figsize_double": (8, 6),
        "dpi": 300,
        "fontsize": 10,
        "fontsize_title": 12,
        "fontsize_label": 11,
        "fontsize_legend": 9,
        "linewidth": 1.5,
        "markersize": 6,
        "colors": plt.cm.tab10.colors,
        "grid_alpha": 0.3,
        "spine_width": 1.0,
    },
}


def apply_publication_style(
    fig: plt.Figure, style: str = "nature", column_width: str = "single"
) -> None:
    """
    Apply publication-quality styling to figure.

    Args:
        fig: Matplotlib figure
        style: Journal style ('nature', 'science', 'ieee', 'default')
        column_width: 'single' or 'double' column
    """
    if style not in JOURNAL_STYLES:
        style = "default"
        logger.warning(f"Unknown style '{style}', using 'default'")

    style_params = JOURNAL_STYLES[style]

    # Apply to all axes
    for ax in fig.get_axes():
        # Set font sizes
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            if isinstance(item, plt.Text):
                if item == ax.title:
                    item.set_fontsize(style_params["fontsize_title"])
                elif item in [ax.xaxis.label, ax.yaxis.label]:
                    item.set_fontsize(style_params["fontsize_label"])
                else:
                    item.set_fontsize(style_params["fontsize"])

        # Set line widths
        for line in ax.get_lines():
            line.set_linewidth(style_params["linewidth"])
            if hasattr(line, "get_markersize"):
                line.set_markersize(style_params["markersize"])

        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(style_params["spine_width"])

        # Grid
        ax.grid(True, alpha=style_params["grid_alpha"], linewidth=0.5)

    # Legend font size
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(style_params["fontsize_legend"])


def publication_quality_plot(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    style: str = "nature",
    formats: List[str] = ["png", "pdf"],
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure in publication-quality format.

    Args:
        fig: Matplotlib figure
        output_path: Output file path (without extension)
        dpi: Resolution (default 300 for publications)
        style: Journal style
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
        bbox_inches: Bounding box ('tight' or 'standard')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style
    apply_publication_style(fig, style)

    # Save in all requested formats
    for fmt in formats:
        if fmt.lower() == "png":
            fig.savefig(
                f"{output_path}.png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "pdf":
            fig.savefig(
                f"{output_path}.pdf",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "svg":
            fig.savefig(
                f"{output_path}.svg",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "eps":
            fig.savefig(
                f"{output_path}.eps",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )

    logger.info(
        f"Publication-quality figure saved to {output_path} ({', '.join(formats)})"
    )


def multi_panel_figure(
    panels: List[Dict[str, Any]],
    layout: Tuple[int, int],
    figsize: Optional[Tuple[float, float]] = None,
    style: str = "nature",
    column_width: str = "double",
    suptitle: Optional[str] = None,
    panel_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create multi-panel figure for publications.

    Args:
        panels: List of panel dictionaries with 'plot_func' and 'kwargs'
        layout: (nrows, ncols) layout
        figsize: Optional figure size (inches)
        style: Journal style
        column_width: 'single' or 'double' column
        suptitle: Optional overall title
        panel_labels: Optional list of panel labels (e.g., ['a', 'b', 'c', 'd'])

    Returns:
        Matplotlib figure
    """
    if figsize is None:
        style_params = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
        if column_width == "single":
            base_size = style_params["figsize_single"]
        else:
            base_size = style_params["figsize_double"]

        # Scale by layout
        figsize = (base_size[0] * layout[1], base_size[1] * layout[0])

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    # Flatten axes if needed
    if layout[0] == 1:
        axes = axes.reshape(1, -1) if layout[1] > 1 else [axes]
    elif layout[1] == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()

    # Plot each panel
    for i, panel in enumerate(panels):
        if i >= len(axes):
            break

        ax = axes[i] if isinstance(axes, np.ndarray) else axes

        # Add panel label
        if panel_labels and i < len(panel_labels):
            ax.text(
                0.02,
                0.98,
                panel_labels[i],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Call plot function
        plot_func = panel.get("plot_func")
        if plot_func:
            plot_kwargs = panel.get("kwargs", {})
            plot_func(ax, **plot_kwargs)

    # Hide unused axes
    if isinstance(axes, np.ndarray):
        for i in range(len(panels), len(axes)):
            axes[i].axis("off")

    # Overall title
    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])[
                "fontsize_title"
            ]
            + 2,
        )

    # Apply style
    apply_publication_style(fig, style, column_width)

    plt.tight_layout()

    return fig


def export_3d_for_publication(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dpi: int = 300,
    colormap: str = "gray",
    opacity: float = 0.8,
    camera_position: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Export 3D volume rendering for publication.

    Args:
        volume: 3D volume
        output_path: Output file path
        voxel_size: Voxel spacing
        dpi: Resolution
        colormap: Colormap name
        opacity: Opacity (0.0 to 1.0)
        camera_position: Optional camera position
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for 3D export")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = voxel_size
    grid.origin = (0, 0, 0)
    grid["values"] = volume.flatten(order="F")

    # Create plotter with high resolution
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, cmap=colormap, opacity=opacity)

    # Set camera
    if camera_position:
        plotter.camera_position = camera_position

    # Export
    plotter.screenshot(
        str(output_path), resolution=(dpi * 8, dpi * 8)
    )  # High resolution
    plotter.close()

    logger.info(f"3D rendering exported to {output_path} at {dpi} DPI")


def create_figure_caption(
    analysis_type: str, metrics: Dict[str, Any], additional_info: Optional[str] = None
) -> str:
    """
    Generate figure caption for publication.

    Args:
        analysis_type: Type of analysis ('porosity', 'filament', 'dimensional', etc.)
        metrics: Dictionary of relevant metrics
        additional_info: Optional additional information

    Returns:
        Formatted caption string
    """
    captions = {
        "porosity": f"Porosity distribution analysis. Mean porosity: {metrics.get('mean_porosity', 'N/A'):.2%}, "
        f"std: {metrics.get('std_porosity', 'N/A'):.2%}.",
        "filament": f"Filament diameter analysis. Mean diameter: {metrics.get('mean_diameter', 'N/A'):.3f} mm, "
        f"std: {metrics.get('std_diameter', 'N/A'):.3f} mm.",
        "dimensional": f"Dimensional accuracy analysis. RMS deviation: {metrics.get('rms_deviation', 'N/A'):.4f} mm, "
        f"dimensional accuracy: {metrics.get('dimensional_accuracy', 'N/A'):.2f}%.",
        "slice": f"Slice analysis. Material fraction: {metrics.get('material_fraction', 'N/A'):.2%}, "
        f"void fraction: {metrics.get('void_fraction', 'N/A'):.2%}.",
        "comparison": f"Comparative analysis across {metrics.get('n_samples', 'N/A')} samples.",
    }

    base_caption = captions.get(analysis_type, f"{analysis_type} analysis results.")

    if additional_info:
        base_caption += f" {additional_info}"

    return base_caption


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
) -> None:
    """
    Compare metrics across multiple samples.

    Args:
        metrics_dict: Dictionary with sample names as keys and metrics as values
        save_path: Path to save figure (optional)
    """
    samples = list(metrics_dict.keys())

    # Extract metrics
    volumes = [metrics_dict[s].get("volume", 0) for s in samples]
    void_fractions = [metrics_dict[s].get("void_fraction", 0) for s in samples]
    surface_areas = [metrics_dict[s].get("surface_area", 0) for s in samples]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Volume
    axes[0].bar(samples, volumes, color="skyblue")
    axes[0].set_ylabel("Volume (mm³)")
    axes[0].set_title("Volume Comparison")
    axes[0].tick_params(axis="x", rotation=45)

    # Void fraction
    axes[1].bar(samples, void_fractions, color="lightcoral")
    axes[1].set_ylabel("Void Fraction")
    axes[1].set_title("Void Fraction Comparison")
    axes[1].tick_params(axis="x", rotation=45)

    # Surface area
    axes[2].bar(samples, surface_areas, color="lightgreen")
    axes[2].set_ylabel("Surface Area (mm²)")
    axes[2].set_title("Surface Area Comparison")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metrics comparison to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Publication-Quality Visualization Functions
# ============================================================================

# Journal style presets
JOURNAL_STYLES = {
    "nature": {
        "figsize_single": (3.5, 2.625),  # inches (Nature single column)
        "figsize_double": (7.0, 5.25),  # inches (Nature double column)
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "science": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "ieee": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 9,
        "fontsize_title": 11,
        "fontsize_label": 10,
        "fontsize_legend": 8,
        "linewidth": 1.2,
        "markersize": 5,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "default": {
        "figsize_single": (4, 3),
        "figsize_double": (8, 6),
        "dpi": 300,
        "fontsize": 10,
        "fontsize_title": 12,
        "fontsize_label": 11,
        "fontsize_legend": 9,
        "linewidth": 1.5,
        "markersize": 6,
        "colors": plt.cm.tab10.colors,
        "grid_alpha": 0.3,
        "spine_width": 1.0,
    },
}


def apply_publication_style(
    fig: plt.Figure, style: str = "nature", column_width: str = "single"
) -> None:
    """
    Apply publication-quality styling to figure.

    Args:
        fig: Matplotlib figure
        style: Journal style ('nature', 'science', 'ieee', 'default')
        column_width: 'single' or 'double' column
    """
    if style not in JOURNAL_STYLES:
        style = "default"
        logger.warning(f"Unknown style '{style}', using 'default'")

    style_params = JOURNAL_STYLES[style]

    # Apply to all axes
    for ax in fig.get_axes():
        # Set font sizes
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            if isinstance(item, plt.Text):
                if item == ax.title:
                    item.set_fontsize(style_params["fontsize_title"])
                elif item in [ax.xaxis.label, ax.yaxis.label]:
                    item.set_fontsize(style_params["fontsize_label"])
                else:
                    item.set_fontsize(style_params["fontsize"])

        # Set line widths
        for line in ax.get_lines():
            line.set_linewidth(style_params["linewidth"])
            if hasattr(line, "get_markersize"):
                line.set_markersize(style_params["markersize"])

        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(style_params["spine_width"])

        # Grid
        ax.grid(True, alpha=style_params["grid_alpha"], linewidth=0.5)

    # Legend font size
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(style_params["fontsize_legend"])


def publication_quality_plot(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    style: str = "nature",
    formats: List[str] = ["png", "pdf"],
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure in publication-quality format.

    Args:
        fig: Matplotlib figure
        output_path: Output file path (without extension)
        dpi: Resolution (default 300 for publications)
        style: Journal style
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
        bbox_inches: Bounding box ('tight' or 'standard')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style
    apply_publication_style(fig, style)

    # Save in all requested formats
    for fmt in formats:
        if fmt.lower() == "png":
            fig.savefig(
                f"{output_path}.png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "pdf":
            fig.savefig(
                f"{output_path}.pdf",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "svg":
            fig.savefig(
                f"{output_path}.svg",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "eps":
            fig.savefig(
                f"{output_path}.eps",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )

    logger.info(
        f"Publication-quality figure saved to {output_path} ({', '.join(formats)})"
    )


def multi_panel_figure(
    panels: List[Dict[str, Any]],
    layout: Tuple[int, int],
    figsize: Optional[Tuple[float, float]] = None,
    style: str = "nature",
    column_width: str = "double",
    suptitle: Optional[str] = None,
    panel_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create multi-panel figure for publications.

    Args:
        panels: List of panel dictionaries with 'plot_func' and 'kwargs'
        layout: (nrows, ncols) layout
        figsize: Optional figure size (inches)
        style: Journal style
        column_width: 'single' or 'double' column
        suptitle: Optional overall title
        panel_labels: Optional list of panel labels (e.g., ['a', 'b', 'c', 'd'])

    Returns:
        Matplotlib figure
    """
    if figsize is None:
        style_params = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
        if column_width == "single":
            base_size = style_params["figsize_single"]
        else:
            base_size = style_params["figsize_double"]

        # Scale by layout
        figsize = (base_size[0] * layout[1], base_size[1] * layout[0])

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    # Flatten axes if needed
    if layout[0] == 1:
        axes = axes.reshape(1, -1) if layout[1] > 1 else [axes]
    elif layout[1] == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()

    # Plot each panel
    for i, panel in enumerate(panels):
        if i >= len(axes):
            break

        ax = axes[i] if isinstance(axes, np.ndarray) else axes

        # Add panel label
        if panel_labels and i < len(panel_labels):
            ax.text(
                0.02,
                0.98,
                panel_labels[i],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Call plot function
        plot_func = panel.get("plot_func")
        if plot_func:
            plot_kwargs = panel.get("kwargs", {})
            plot_func(ax, **plot_kwargs)

    # Hide unused axes
    if isinstance(axes, np.ndarray):
        for i in range(len(panels), len(axes)):
            axes[i].axis("off")

    # Overall title
    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])[
                "fontsize_title"
            ]
            + 2,
        )

    # Apply style
    apply_publication_style(fig, style, column_width)

    plt.tight_layout()

    return fig


def export_3d_for_publication(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dpi: int = 300,
    colormap: str = "gray",
    opacity: float = 0.8,
    camera_position: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Export 3D volume rendering for publication.

    Args:
        volume: 3D volume
        output_path: Output file path
        voxel_size: Voxel spacing
        dpi: Resolution
        colormap: Colormap name
        opacity: Opacity (0.0 to 1.0)
        camera_position: Optional camera position
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for 3D export")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = voxel_size
    grid.origin = (0, 0, 0)
    grid["values"] = volume.flatten(order="F")

    # Create plotter with high resolution
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, cmap=colormap, opacity=opacity)

    # Set camera
    if camera_position:
        plotter.camera_position = camera_position

    # Export
    plotter.screenshot(
        str(output_path), resolution=(dpi * 8, dpi * 8)
    )  # High resolution
    plotter.close()

    logger.info(f"3D rendering exported to {output_path} at {dpi} DPI")


def create_figure_caption(
    analysis_type: str, metrics: Dict[str, Any], additional_info: Optional[str] = None
) -> str:
    """
    Generate figure caption for publication.

    Args:
        analysis_type: Type of analysis ('porosity', 'filament', 'dimensional', etc.)
        metrics: Dictionary of relevant metrics
        additional_info: Optional additional information

    Returns:
        Formatted caption string
    """
    captions = {
        "porosity": f"Porosity distribution analysis. Mean porosity: {metrics.get('mean_porosity', 'N/A'):.2%}, "
        f"std: {metrics.get('std_porosity', 'N/A'):.2%}.",
        "filament": f"Filament diameter analysis. Mean diameter: {metrics.get('mean_diameter', 'N/A'):.3f} mm, "
        f"std: {metrics.get('std_diameter', 'N/A'):.3f} mm.",
        "dimensional": f"Dimensional accuracy analysis. RMS deviation: {metrics.get('rms_deviation', 'N/A'):.4f} mm, "
        f"dimensional accuracy: {metrics.get('dimensional_accuracy', 'N/A'):.2f}%.",
        "slice": f"Slice analysis. Material fraction: {metrics.get('material_fraction', 'N/A'):.2%}, "
        f"void fraction: {metrics.get('void_fraction', 'N/A'):.2%}.",
        "comparison": f"Comparative analysis across {metrics.get('n_samples', 'N/A')} samples.",
    }

    base_caption = captions.get(analysis_type, f"{analysis_type} analysis results.")

    if additional_info:
        base_caption += f" {additional_info}"

    return base_caption


def plot_diameter_distribution(
    distribution_data: Dict[str, Any], save_path: Optional[str] = None
) -> None:
    """
    Plot filament diameter distribution.

    Args:
        distribution_data: Dictionary from compute_diameter_distribution()
        save_path: Path to save figure (optional)
    """
    if distribution_data["histogram"] is None:
        logger.warning("No diameter data to plot")
        return

    hist = np.array(distribution_data["histogram"])
    bin_edges = np.array(distribution_data["bin_edges"])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        bin_centers,
        hist,
        width=bin_edges[1] - bin_edges[0],
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax.axvline(
        distribution_data["mean"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {distribution_data['mean']:.3f} mm",
    )
    ax.axvline(
        distribution_data["median"],
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {distribution_data['median']:.3f} mm",
    )

    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Filament Diameter Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved diameter distribution to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Publication-Quality Visualization Functions
# ============================================================================

# Journal style presets
JOURNAL_STYLES = {
    "nature": {
        "figsize_single": (3.5, 2.625),  # inches (Nature single column)
        "figsize_double": (7.0, 5.25),  # inches (Nature double column)
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "science": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "ieee": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 9,
        "fontsize_title": 11,
        "fontsize_label": 10,
        "fontsize_legend": 8,
        "linewidth": 1.2,
        "markersize": 5,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "default": {
        "figsize_single": (4, 3),
        "figsize_double": (8, 6),
        "dpi": 300,
        "fontsize": 10,
        "fontsize_title": 12,
        "fontsize_label": 11,
        "fontsize_legend": 9,
        "linewidth": 1.5,
        "markersize": 6,
        "colors": plt.cm.tab10.colors,
        "grid_alpha": 0.3,
        "spine_width": 1.0,
    },
}


def apply_publication_style(
    fig: plt.Figure, style: str = "nature", column_width: str = "single"
) -> None:
    """
    Apply publication-quality styling to figure.

    Args:
        fig: Matplotlib figure
        style: Journal style ('nature', 'science', 'ieee', 'default')
        column_width: 'single' or 'double' column
    """
    if style not in JOURNAL_STYLES:
        style = "default"
        logger.warning(f"Unknown style '{style}', using 'default'")

    style_params = JOURNAL_STYLES[style]

    # Apply to all axes
    for ax in fig.get_axes():
        # Set font sizes
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            if isinstance(item, plt.Text):
                if item == ax.title:
                    item.set_fontsize(style_params["fontsize_title"])
                elif item in [ax.xaxis.label, ax.yaxis.label]:
                    item.set_fontsize(style_params["fontsize_label"])
                else:
                    item.set_fontsize(style_params["fontsize"])

        # Set line widths
        for line in ax.get_lines():
            line.set_linewidth(style_params["linewidth"])
            if hasattr(line, "get_markersize"):
                line.set_markersize(style_params["markersize"])

        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(style_params["spine_width"])

        # Grid
        ax.grid(True, alpha=style_params["grid_alpha"], linewidth=0.5)

    # Legend font size
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(style_params["fontsize_legend"])


def publication_quality_plot(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    style: str = "nature",
    formats: List[str] = ["png", "pdf"],
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure in publication-quality format.

    Args:
        fig: Matplotlib figure
        output_path: Output file path (without extension)
        dpi: Resolution (default 300 for publications)
        style: Journal style
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
        bbox_inches: Bounding box ('tight' or 'standard')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style
    apply_publication_style(fig, style)

    # Save in all requested formats
    for fmt in formats:
        if fmt.lower() == "png":
            fig.savefig(
                f"{output_path}.png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "pdf":
            fig.savefig(
                f"{output_path}.pdf",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "svg":
            fig.savefig(
                f"{output_path}.svg",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "eps":
            fig.savefig(
                f"{output_path}.eps",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )

    logger.info(
        f"Publication-quality figure saved to {output_path} ({', '.join(formats)})"
    )


def multi_panel_figure(
    panels: List[Dict[str, Any]],
    layout: Tuple[int, int],
    figsize: Optional[Tuple[float, float]] = None,
    style: str = "nature",
    column_width: str = "double",
    suptitle: Optional[str] = None,
    panel_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create multi-panel figure for publications.

    Args:
        panels: List of panel dictionaries with 'plot_func' and 'kwargs'
        layout: (nrows, ncols) layout
        figsize: Optional figure size (inches)
        style: Journal style
        column_width: 'single' or 'double' column
        suptitle: Optional overall title
        panel_labels: Optional list of panel labels (e.g., ['a', 'b', 'c', 'd'])

    Returns:
        Matplotlib figure
    """
    if figsize is None:
        style_params = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
        if column_width == "single":
            base_size = style_params["figsize_single"]
        else:
            base_size = style_params["figsize_double"]

        # Scale by layout
        figsize = (base_size[0] * layout[1], base_size[1] * layout[0])

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    # Flatten axes if needed
    if layout[0] == 1:
        axes = axes.reshape(1, -1) if layout[1] > 1 else [axes]
    elif layout[1] == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()

    # Plot each panel
    for i, panel in enumerate(panels):
        if i >= len(axes):
            break

        ax = axes[i] if isinstance(axes, np.ndarray) else axes

        # Add panel label
        if panel_labels and i < len(panel_labels):
            ax.text(
                0.02,
                0.98,
                panel_labels[i],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Call plot function
        plot_func = panel.get("plot_func")
        if plot_func:
            plot_kwargs = panel.get("kwargs", {})
            plot_func(ax, **plot_kwargs)

    # Hide unused axes
    if isinstance(axes, np.ndarray):
        for i in range(len(panels), len(axes)):
            axes[i].axis("off")

    # Overall title
    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])[
                "fontsize_title"
            ]
            + 2,
        )

    # Apply style
    apply_publication_style(fig, style, column_width)

    plt.tight_layout()

    return fig


def export_3d_for_publication(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dpi: int = 300,
    colormap: str = "gray",
    opacity: float = 0.8,
    camera_position: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Export 3D volume rendering for publication.

    Args:
        volume: 3D volume
        output_path: Output file path
        voxel_size: Voxel spacing
        dpi: Resolution
        colormap: Colormap name
        opacity: Opacity (0.0 to 1.0)
        camera_position: Optional camera position
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for 3D export")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = voxel_size
    grid.origin = (0, 0, 0)
    grid["values"] = volume.flatten(order="F")

    # Create plotter with high resolution
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, cmap=colormap, opacity=opacity)

    # Set camera
    if camera_position:
        plotter.camera_position = camera_position

    # Export
    plotter.screenshot(
        str(output_path), resolution=(dpi * 8, dpi * 8)
    )  # High resolution
    plotter.close()

    logger.info(f"3D rendering exported to {output_path} at {dpi} DPI")


def create_figure_caption(
    analysis_type: str, metrics: Dict[str, Any], additional_info: Optional[str] = None
) -> str:
    """
    Generate figure caption for publication.

    Args:
        analysis_type: Type of analysis ('porosity', 'filament', 'dimensional', etc.)
        metrics: Dictionary of relevant metrics
        additional_info: Optional additional information

    Returns:
        Formatted caption string
    """
    captions = {
        "porosity": f"Porosity distribution analysis. Mean porosity: {metrics.get('mean_porosity', 'N/A'):.2%}, "
        f"std: {metrics.get('std_porosity', 'N/A'):.2%}.",
        "filament": f"Filament diameter analysis. Mean diameter: {metrics.get('mean_diameter', 'N/A'):.3f} mm, "
        f"std: {metrics.get('std_diameter', 'N/A'):.3f} mm.",
        "dimensional": f"Dimensional accuracy analysis. RMS deviation: {metrics.get('rms_deviation', 'N/A'):.4f} mm, "
        f"dimensional accuracy: {metrics.get('dimensional_accuracy', 'N/A'):.2f}%.",
        "slice": f"Slice analysis. Material fraction: {metrics.get('material_fraction', 'N/A'):.2%}, "
        f"void fraction: {metrics.get('void_fraction', 'N/A'):.2%}.",
        "comparison": f"Comparative analysis across {metrics.get('n_samples', 'N/A')} samples.",
    }

    base_caption = captions.get(analysis_type, f"{analysis_type} analysis results.")

    if additional_info:
        base_caption += f" {additional_info}"

    return base_caption


def create_analysis_report(
    analyzer: Any, output_path: str, sample_name: str = "Sample"
) -> None:
    """
    Generate comprehensive analysis report with visualizations.

    Args:
        analyzer: XCTAnalyzer instance
        output_path: Output directory path
        sample_name: Sample name for report
    """
    from pathlib import Path

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create report HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XCT Analysis Report - {sample_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .metric {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>XCT Analysis Report: {sample_name}</h1>
        <p>Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Scalar Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """

    # Add metrics if available
    if hasattr(analyzer, "metrics") and analyzer.metrics:
        for key, value in analyzer.metrics.items():
            if value is not None:
                if isinstance(value, float):
                    html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>\n"
                else:
                    html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>\n"

    html_content += """
        </table>
        
        <h2>Analysis Results</h2>
        <p>See output images and data files for detailed results.</p>
    </body>
    </html>
    """

    report_path = output_dir / f"{sample_name}_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)

    logger.info(f"Analysis report saved to {report_path}")


def interactive_3d_viewer(
    volume: np.ndarray, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Create interactive 3D viewer (requires PyVista).

    Args:
        volume: 3D volume
        voxel_size: Voxel spacing
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for interactive 3D viewer")
        return

    visualize_3d_volume(volume, voxel_size, opacity=0.5)


def plot_slice_comparison(
    slices: List[np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot multiple slices side by side for comparison.

    Args:
        slices: List of 2D slices
        titles: List of titles (optional)
        save_path: Path to save figure (optional)
    """
    n_slices = len(slices)
    if titles is None:
        titles = [f"Slice {i+1}" for i in range(n_slices)]

    cols = min(4, n_slices)
    rows = (n_slices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_slices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (slice_2d, title) in enumerate(zip(slices, titles)):
        if i < len(axes):
            ax = axes[i]
            im = ax.imshow(slice_2d, cmap="gray", origin="lower")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax)

    # Hide unused subplots
    for i in range(n_slices, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved slice comparison to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Publication-Quality Visualization Functions
# ============================================================================

# Journal style presets
JOURNAL_STYLES = {
    "nature": {
        "figsize_single": (3.5, 2.625),  # inches (Nature single column)
        "figsize_double": (7.0, 5.25),  # inches (Nature double column)
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "science": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 8,
        "fontsize_title": 10,
        "fontsize_label": 9,
        "fontsize_legend": 7,
        "linewidth": 1.0,
        "markersize": 4,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "ieee": {
        "figsize_single": (3.5, 2.625),
        "figsize_double": (7.0, 5.25),
        "dpi": 300,
        "fontsize": 9,
        "fontsize_title": 11,
        "fontsize_label": 10,
        "fontsize_legend": 8,
        "linewidth": 1.2,
        "markersize": 5,
        "colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959"],
        "grid_alpha": 0.3,
        "spine_width": 0.5,
    },
    "default": {
        "figsize_single": (4, 3),
        "figsize_double": (8, 6),
        "dpi": 300,
        "fontsize": 10,
        "fontsize_title": 12,
        "fontsize_label": 11,
        "fontsize_legend": 9,
        "linewidth": 1.5,
        "markersize": 6,
        "colors": plt.cm.tab10.colors,
        "grid_alpha": 0.3,
        "spine_width": 1.0,
    },
}


def apply_publication_style(
    fig: plt.Figure, style: str = "nature", column_width: str = "single"
) -> None:
    """
    Apply publication-quality styling to figure.

    Args:
        fig: Matplotlib figure
        style: Journal style ('nature', 'science', 'ieee', 'default')
        column_width: 'single' or 'double' column
    """
    if style not in JOURNAL_STYLES:
        style = "default"
        logger.warning(f"Unknown style '{style}', using 'default'")

    style_params = JOURNAL_STYLES[style]

    # Apply to all axes
    for ax in fig.get_axes():
        # Set font sizes
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            if isinstance(item, plt.Text):
                if item == ax.title:
                    item.set_fontsize(style_params["fontsize_title"])
                elif item in [ax.xaxis.label, ax.yaxis.label]:
                    item.set_fontsize(style_params["fontsize_label"])
                else:
                    item.set_fontsize(style_params["fontsize"])

        # Set line widths
        for line in ax.get_lines():
            line.set_linewidth(style_params["linewidth"])
            if hasattr(line, "get_markersize"):
                line.set_markersize(style_params["markersize"])

        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(style_params["spine_width"])

        # Grid
        ax.grid(True, alpha=style_params["grid_alpha"], linewidth=0.5)

    # Legend font size
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(style_params["fontsize_legend"])


def publication_quality_plot(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    style: str = "nature",
    formats: List[str] = ["png", "pdf"],
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure in publication-quality format.

    Args:
        fig: Matplotlib figure
        output_path: Output file path (without extension)
        dpi: Resolution (default 300 for publications)
        style: Journal style
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
        bbox_inches: Bounding box ('tight' or 'standard')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style
    apply_publication_style(fig, style)

    # Save in all requested formats
    for fmt in formats:
        if fmt.lower() == "png":
            fig.savefig(
                f"{output_path}.png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "pdf":
            fig.savefig(
                f"{output_path}.pdf",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "svg":
            fig.savefig(
                f"{output_path}.svg",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )
        elif fmt.lower() == "eps":
            fig.savefig(
                f"{output_path}.eps",
                bbox_inches=bbox_inches,
                facecolor="white",
                edgecolor="none",
            )

    logger.info(
        f"Publication-quality figure saved to {output_path} ({', '.join(formats)})"
    )


def multi_panel_figure(
    panels: List[Dict[str, Any]],
    layout: Tuple[int, int],
    figsize: Optional[Tuple[float, float]] = None,
    style: str = "nature",
    column_width: str = "double",
    suptitle: Optional[str] = None,
    panel_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create multi-panel figure for publications.

    Args:
        panels: List of panel dictionaries with 'plot_func' and 'kwargs'
        layout: (nrows, ncols) layout
        figsize: Optional figure size (inches)
        style: Journal style
        column_width: 'single' or 'double' column
        suptitle: Optional overall title
        panel_labels: Optional list of panel labels (e.g., ['a', 'b', 'c', 'd'])

    Returns:
        Matplotlib figure
    """
    if figsize is None:
        style_params = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
        if column_width == "single":
            base_size = style_params["figsize_single"]
        else:
            base_size = style_params["figsize_double"]

        # Scale by layout
        figsize = (base_size[0] * layout[1], base_size[1] * layout[0])

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    # Flatten axes if needed
    if layout[0] == 1:
        axes = axes.reshape(1, -1) if layout[1] > 1 else [axes]
    elif layout[1] == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()

    # Plot each panel
    for i, panel in enumerate(panels):
        if i >= len(axes):
            break

        ax = axes[i] if isinstance(axes, np.ndarray) else axes

        # Add panel label
        if panel_labels and i < len(panel_labels):
            ax.text(
                0.02,
                0.98,
                panel_labels[i],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Call plot function
        plot_func = panel.get("plot_func")
        if plot_func:
            plot_kwargs = panel.get("kwargs", {})
            plot_func(ax, **plot_kwargs)

    # Hide unused axes
    if isinstance(axes, np.ndarray):
        for i in range(len(panels), len(axes)):
            axes[i].axis("off")

    # Overall title
    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])[
                "fontsize_title"
            ]
            + 2,
        )

    # Apply style
    apply_publication_style(fig, style, column_width)

    plt.tight_layout()

    return fig


def export_3d_for_publication(
    volume: np.ndarray,
    output_path: Union[str, Path],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dpi: int = 300,
    colormap: str = "gray",
    opacity: float = 0.8,
    camera_position: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Export 3D volume rendering for publication.

    Args:
        volume: 3D volume
        output_path: Output file path
        voxel_size: Voxel spacing
        dpi: Resolution
        colormap: Colormap name
        opacity: Opacity (0.0 to 1.0)
        camera_position: Optional camera position
    """
    if not HAS_PYVISTA:
        logger.error("PyVista required for 3D export")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = voxel_size
    grid.origin = (0, 0, 0)
    grid["values"] = volume.flatten(order="F")

    # Create plotter with high resolution
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, cmap=colormap, opacity=opacity)

    # Set camera
    if camera_position:
        plotter.camera_position = camera_position

    # Export
    plotter.screenshot(
        str(output_path), resolution=(dpi * 8, dpi * 8)
    )  # High resolution
    plotter.close()

    logger.info(f"3D rendering exported to {output_path} at {dpi} DPI")


def create_figure_caption(
    analysis_type: str, metrics: Dict[str, Any], additional_info: Optional[str] = None
) -> str:
    """
    Generate figure caption for publication.

    Args:
        analysis_type: Type of analysis ('porosity', 'filament', 'dimensional', etc.)
        metrics: Dictionary of relevant metrics
        additional_info: Optional additional information

    Returns:
        Formatted caption string
    """
    captions = {
        "porosity": f"Porosity distribution analysis. Mean porosity: {metrics.get('mean_porosity', 'N/A'):.2%}, "
        f"std: {metrics.get('std_porosity', 'N/A'):.2%}.",
        "filament": f"Filament diameter analysis. Mean diameter: {metrics.get('mean_diameter', 'N/A'):.3f} mm, "
        f"std: {metrics.get('std_diameter', 'N/A'):.3f} mm.",
        "dimensional": f"Dimensional accuracy analysis. RMS deviation: {metrics.get('rms_deviation', 'N/A'):.4f} mm, "
        f"dimensional accuracy: {metrics.get('dimensional_accuracy', 'N/A'):.2f}%.",
        "slice": f"Slice analysis. Material fraction: {metrics.get('material_fraction', 'N/A'):.2%}, "
        f"void fraction: {metrics.get('void_fraction', 'N/A'):.2%}.",
        "comparison": f"Comparative analysis across {metrics.get('n_samples', 'N/A')} samples.",
    }

    base_caption = captions.get(analysis_type, f"{analysis_type} analysis results.")

    if additional_info:
        base_caption += f" {additional_info}"

    return base_caption
