# ParaView Segmentation Fault Fix

## Problem
The original `postprocess_fenicsx.py` script using `pvpython` was experiencing segmentation faults (exit code 139) when trying to load and visualize FEniCSx simulation results.

## Root Cause
The segmentation fault was caused by ParaView 5.13.3's `XDMFReader` having issues with:
1. Temporal grid collections in the XDMF files
2. Missing or improper `UpdatePipeline()` calls between filter operations
3. Memory management issues when creating multiple visualizations in sequence

## Solution
Created a new PyVista-based postprocessing script (`postprocess_pyvista.py`) that:

### Advantages over ParaView pvpython:
- **More stable**: PyVista is designed for programmatic visualization
- **Simpler**: Direct HDF5 file reading without XDMF temporal collection issues
- **Better resource management**: Explicit cleanup and fresh plotter for each visualization
- **Easier debugging**: Pure Python without ParaView server/client architecture

### Key Implementation Details:
1. **Direct HDF5 reading**: Loads mesh and field data directly from HDF5 files
2. **Proper field naming**: 
   - Displacement: `u`
   - Von Mises stress: `sigma_vm` (cell data)
   - Stress tensor: `sigma` (cell data)
   - Strain tensor: `epsilon` (cell data, not `eps`)
3. **Cell-to-point conversion**: Applies `cell_data_to_point_data()` for smooth visualization
4. **Fresh plotters**: Creates new plotter for each visualization to avoid state conflicts

## Usage

### Install dependencies:
```bash
pip install pyvista h5py
```

### Run postprocessing:
```bash
# All fields
python postprocess_pyvista.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full

# Specific field
python postprocess_pyvista.py --job-dir PATH --field displacement
python postprocess_pyvista.py --job-dir PATH --field vonmises
python postprocess_pyvista.py --job-dir PATH --field stress
python postprocess_pyvista.py --job-dir PATH --field strain
python postprocess_pyvista.py --job-dir PATH --field mesh

# Custom resolution
python postprocess_pyvista.py --job-dir PATH --resolution 3840 2160
```

## Output Files
The script generates the following visualization files:

1. `displacement_magnitude.png` - Displacement field with magnitude coloring
2. `von_mises_stress.png` - Von Mises stress distribution
3. `von_mises_stress_with_mesh.png` - Von Mises stress with mesh overlay
4. `stress_sigma_xx.png`, `stress_sigma_yy.png`, `stress_sigma_xy.png` - Stress components
5. `strain_epsilon_xx.png`, `strain_epsilon_yy.png`, `strain_epsilon_xy.png` - Strain components
6. `mesh_structure.png` - Wireframe mesh visualization

All images are saved to `<job-dir>/postprocess/` by default.

## Original ParaView Script Status
The original `postprocess_fenicsx.py` has been updated with:
- Proper `UpdatePipeline()` calls after each filter
- Better resource cleanup with `Hide()` and `Delete()`
- Explicit memory management

However, **the PyVista version is recommended** due to better stability and simpler usage.

## Files
- `postprocess_pyvista.py` - **Recommended**: Stable PyVista-based implementation
- `postprocess_fenicsx.py` - Updated ParaView version (may still have stability issues)
- `postprocess_fenicsx_simple.py` - Simplified ParaView attempt (experimental; file no longer exists)
