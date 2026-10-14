# FEniCSx ParaView Post-Processing Scripts

> **DEPRECATED.** This documents the ParaView/pvpython pipeline
> (`postprocess_fenicsx.py`), which loads 0 points from DOLFINx XDMF
> (`GridStatus` name mismatch) and reports hardcoded success. Use
> `postprocess_pyvista.py` (XDMF/H5) or `postprocess_vtu.py` (VTU) instead —
> see `INDEX.md` and `QUICK_START.md`.

This directory contains scripts for post-processing FEniCSx elasticity simulation results using ParaView's Python API (pvpython).

## Requirements

- **ParaView** (5.11 or later) with pvpython
  - Tested with ParaView 5.11 and 5.12
  - Automatically handles API differences between versions
- FEniCSx output files in XDMF format

## Installation

1. Install ParaView from https://www.paraview.org/download/
2. Locate the `pvpython` executable (usually in ParaView.app/Contents/bin/ on macOS)

## Usage

### Basic Usage

```bash
pvpython postprocess_fenicsx.py --job-dir /path/to/job/directory
```

This will:
- Load displacement and stress fields from the job's `output/` directory
- Generate high-quality visualizations
- Save images to `job_directory/postprocess/`

### Advanced Usage

```bash
# Specify custom output directory
pvpython postprocess_fenicsx.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full \
    --output-dir ./my_results

# High-resolution output for publications
pvpython postprocess_fenicsx.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full \
    --resolution 3840 2160 --dpi 600

# Generate animations (note: may be slow)
pvpython postprocess_fenicsx.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full \
    --animations

# Process only specific fields
pvpython postprocess_fenicsx.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full \
    --field von_mises
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--job-dir`, `-j` | Path to job directory containing output/ | Required |
| `--output-dir`, `-o` | Output directory for images | `job-dir/postprocess/` |
| `--resolution`, `-r` | Image resolution (width height) | `1920 1080` |
| `--dpi` | DPI for image export | `300` |
| `--animations`, `-a` | Generate rotation animations | `False` |
| `--field` | Which fields to visualize (displacement, von_mises, stress, strain, all) | `all` |

## Output Files

The script generates the following visualizations:

### Core Visualizations
- `displacement_magnitude.png` - Displacement field with deformed mesh
- `von_mises_stress.png` - Von Mises stress distribution
- `von_mises_stress_with_mesh.png` - Von Mises stress with mesh overlay
- `mesh_structure.png` - Mesh structure (wireframe)

### Stress Components
- `stress_sigma_xx.png` - Normal stress in x-direction
- `stress_sigma_yy.png` - Normal stress in y-direction
- `stress_sigma_xy.png` - Shear stress

### Strain Components
- `strain_epsilon_xx.png` - Normal strain in x-direction
- `strain_epsilon_yy.png` - Normal strain in y-direction
- `strain_epsilon_xy.png` - Shear strain

### Summary
- `summary.txt` - Statistics and metadata

## Examples

### Example 1: Quick Visualization

```bash
# Process latest simulation
pvpython postprocess_fenicsx.py \
    --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full
```

Output:
```
=== Creating Displacement Plot ===
Saved: /path/to/postprocess/displacement_magnitude.png

=== Creating Von Mises Stress Plot ===
Saved: /path/to/postprocess/von_mises_stress.png
...

Post-processing completed successfully!
Output saved to: /path/to/postprocess/
```

### Example 2: Publication-Quality Figures

```bash
# 4K resolution with high DPI
pvpython postprocess_fenicsx.py \
    --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full \
    --resolution 3840 2160 \
    --dpi 600 \
    --output-dir ./publication_figures
```

### Example 3: Batch Processing Multiple Jobs

```bash
# Process all jobs in a loop
for job in ../../runs/jobs/session-atomagent-fem_demo0*/; do
    echo "Processing $job"
    pvpython postprocess_fenicsx.py --job-dir "$job"
done
```

## Customization

To customize visualizations, modify the `FEniCSXPostProcessor` class in `postprocess_fenicsx.py`:

### Change Color Maps

```python
# In create_vonmises_plot()
lut.ApplyPreset('Jet', True)  # Change to 'Viridis', 'Plasma', etc.
```

### Adjust Camera Views

```python
# Add custom camera positioning
camera = GetActiveCamera()
camera.SetPosition(x, y, z)
camera.SetFocalPoint(fx, fy, fz)
```

### Add Custom Filters

```python
# Example: Add contour lines
contour = Contour(Input=vm_field)
contour.ContourBy = ['POINTS', 'f_16']
contour.Isosurfaces = [1e6, 2e6, 3e6]  # Stress levels
```

## Troubleshooting

### Issue: "pvpython: command not found"

**Solution**: Add ParaView to your PATH or use full path:
```bash
# macOS
/Applications/ParaView-5.11.1.app/Contents/bin/pvpython postprocess_fenicsx.py ...

# Linux
/usr/local/paraview-5.11/bin/pvpython postprocess_fenicsx.py ...
```

### Issue: "Missing required files"

**Solution**: Ensure your job directory has the correct structure:
```
job_directory/
├── output/
│   ├── solution.xdmf
│   ├── solution.h5
│   └── derived/
│       ├── field_sigma_vm.xdmf
│       ├── field_sigma.xdmf
│       └── field_eps.xdmf
```

### Issue: Field names don't match (f_15, f_16, etc.)

**Solution**: Check your XDMF files for actual field names:
```bash
grep 'Attribute Name' output/solution.xdmf
```

Then update the script to use correct field names.

### Issue: Animation export doesn't work

**Note**: Animation export using `SaveAnimation()` typically requires ParaView GUI mode. For pvpython, consider:
1. Export individual frames in a loop
2. Use ffmpeg to combine frames into video
3. Run the script in ParaView's Python shell (GUI mode)

## Advanced: Integration with Python Scripts

You can also import and use the postprocessor in your own Python scripts:

```python
from postprocess_fenicsx import FEniCSXPostProcessor

# Create processor
processor = FEniCSXPostProcessor(
    job_dir='../../runs/jobs/my_simulation',
    resolution=(2560, 1440),
    dpi=300
)

# Generate specific visualizations
processor.create_vonmises_plot(show_mesh=True)
processor.create_displacement_plot(warp_scale=2.0)

# Or run everything
processor.run_all(include_animations=False)
```

## Performance Notes

- **Image generation**: ~1-5 seconds per image (1920x1080)
- **Animations**: ~30-60 seconds per field (5s @ 30fps)
- **Memory**: Scales with mesh size; fine meshes may require 4-8 GB RAM

## License

This script is part of the AtomAgent project. Use freely for research and education.

## Contact

For issues or questions, please refer to the main AtomAgent repository.
