"""
Generate Segmented Data CSV Files

Creates 10 sample segmented data CSV files for testing and demonstration.
Each file contains voxel-level data with coordinates and segmented values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple

# Add XCT_Thermomagnetic_Analysis root to path
script_dir = Path(__file__).parent
analysis_root = script_dir.parent
sys.path.insert(0, str(analysis_root))
sys.path.insert(0, str(analysis_root / 'src'))

# Import from src package (new organized structure)
from src.core.segmentation import otsu_threshold
from src.preprocessing.preprocessing import analyze_object_properties
from src.utils.utils import save_segmented_data


def create_synthetic_volume(shape: Tuple[int, int, int], 
                           seed: int = None,
                           porosity_level: float = 0.3) -> np.ndarray:
    """
    Create synthetic 3D volume with realistic structure.
    
    Args:
        shape: Volume shape (z, y, x)
        seed: Random seed
        porosity_level: Target porosity level (0.0 to 1.0)
    
    Returns:
        3D numpy array
    """
    if seed is not None:
        np.random.seed(seed)
    
    volume = np.random.rand(*shape)
    
    # Add structured features (simulating filaments/channels)
    z, y, x = shape
    
    # Create filament-like structures
    n_filaments = int(z * 0.1)  # ~10% of depth
    for _ in range(n_filaments):
        # Random filament position
        center_y = np.random.randint(10, y - 10)
        center_x = np.random.randint(10, x - 10)
        radius = np.random.uniform(2, 5)
        
        # Create filament along z-axis
        for zi in range(z):
            yy, xx = np.ogrid[:y, :x]
            dist = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)
            volume[zi, dist < radius] += 0.5
    
    # Add some porosity (voids)
    n_pores = int(np.prod(shape) * porosity_level / 100)  # Approximate
    for _ in range(n_pores):
        zi = np.random.randint(0, z)
        yi = np.random.randint(0, y)
        xi = np.random.randint(0, x)
        pore_size = np.random.randint(1, 4)
        volume[max(0, zi-pore_size):min(z, zi+pore_size),
               max(0, yi-pore_size):min(y, yi+pore_size),
               max(0, xi-pore_size):min(x, xi+pore_size)] *= 0.3
    
    # Normalize
    volume = (volume / volume.max()).astype(np.float32)
    
    return volume


def generate_segmented_csv(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float],
                          output_path: Path,
                          sample_name: str,
                          include_object_properties: bool = True) -> None:
    """
    Generate segmented data CSV file with all required columns.
    
    Args:
        volume: 3D volume
        voxel_size: Voxel spacing in mm
        output_path: Output file path
        sample_name: Name of the sample
        include_object_properties: Whether to include object-level properties
    """
    # Segment the volume
    segmented = otsu_threshold(volume)
    
    # Get voxel coordinates and values
    coords = np.argwhere(segmented > 0)
    values = segmented[segmented > 0]
    
    # Convert to physical coordinates
    coords_physical = coords * np.array(voxel_size)
    
    # Create DataFrame with voxel-level data
    df_voxels = pd.DataFrame({
        'x': coords_physical[:, 2],  # Note: numpy uses (z, y, x)
        'y': coords_physical[:, 1],
        'z': coords_physical[:, 0],
        'x_voxel': coords[:, 2],  # Voxel indices
        'y_voxel': coords[:, 1],
        'z_voxel': coords[:, 0],
        'value': values.astype(int),
        'segmented': values.astype(int),  # Alias
        'sample_name': sample_name
    })
    
    # If including object properties, analyze objects
    if include_object_properties:
        try:
            object_props = analyze_object_properties(segmented, voxel_size)
            
            # Create object properties DataFrame
            df_objects = pd.DataFrame(object_props)
            df_objects['sample_name'] = sample_name
            
            # Merge object properties with voxel data (assign object ID to each voxel)
            from scipy import ndimage
            labeled, _ = ndimage.label(segmented)
            
            # Get label for each voxel
            voxel_labels = []
            for coord in coords:
                voxel_labels.append(int(labeled[tuple(coord)]))
            
            df_voxels['object_id'] = voxel_labels
            
            # Merge with object properties
            df_objects_merge = df_objects[['label_id', 'volume_mm3', 'sphericity', 
                                          'max_aspect_ratio', 'on_edge']].copy()
            df_objects_merge.columns = ['object_id', 'object_volume_mm3', 
                                       'object_sphericity', 'object_max_aspect_ratio', 'object_on_edge']
            
            df_voxels = df_voxels.merge(df_objects_merge, on='object_id', how='left')
            
            # Save object properties separately
            objects_file = output_path.parent / f"{output_path.stem}_objects.csv"
            df_objects.to_csv(objects_file, index=False)
            print(f"   📊 Object properties saved to: {objects_file.name}")
            
        except Exception as e:
            print(f"   ⚠️  Could not compute object properties: {e}")
    
    # Save voxel-level data
    df_voxels.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved: {output_path.name} ({len(df_voxels):,} voxels)")


def main():
    """Generate 10 segmented data CSV files"""
    
    # Setup - save to data folder
    data_dir = Path(__file__).parent.parent / 'data' / 'segmented_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir
    
    print("📦 Generating Segmented Data CSV Files")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    # Parameters
    n_samples = 10
    volume_shape = (100, 100, 100)  # (z, y, x)
    voxel_size = (0.1, 0.1, 0.1)  # mm
    
    # Generate different porosity levels for variety
    porosity_levels = np.linspace(0.2, 0.5, n_samples)
    
    for i in range(n_samples):
        sample_name = f"Sample_{i+1:02d}"
        print(f"[{i+1}/{n_samples}] Generating {sample_name}...")
        
        # Create synthetic volume
        volume = create_synthetic_volume(
            volume_shape,
            seed=42 + i,  # Different seed for each sample
            porosity_level=porosity_levels[i]
        )
        
        # Generate CSV
        output_file = output_dir / f"{sample_name}_segmented.csv"
        generate_segmented_csv(
            volume,
            voxel_size,
            output_file,
            sample_name,
            include_object_properties=True
        )
        
        print()
    
    print("=" * 60)
    print(f"✅ Generated {n_samples} segmented data CSV files")
    print(f"   Location: {output_dir}")
    print()
    print("📋 File Format:")
    print("   - Voxel-level data: x, y, z (mm), x_voxel, y_voxel, z_voxel, value, segmented")
    print("   - Object properties: object_id, object_volume_mm3, object_sphericity, etc.")
    print("   - Object-level files: *_objects.csv with full object properties")
    print()
    print("💡 Usage:")
    print("   from src.utils.utils import load_segmented_data")
    print("   volume, metadata = load_segmented_data('data/segmented_data/Sample_01_segmented.csv')")


if __name__ == '__main__':
    main()

