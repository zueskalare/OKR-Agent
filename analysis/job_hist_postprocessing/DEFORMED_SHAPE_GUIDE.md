# Deformed Shape Visualization Guide

## Overview
The PyVista postprocessing script now supports displaying stress and strain results on the deformed mesh geometry, allowing you to visualize how the structure deforms under load while showing the field values.

## Usage

### Command-Line Options

#### `--show-deformed`
Enable deformed shape visualization for stress and strain plots.

```bash
python postprocess_pyvista.py --job-dir ../../runs/jobs/your_job --show-deformed
```

#### `--warp-scale FACTOR`
Control the deformation magnification factor (default: 1.0).

```bash
python postprocess_pyvista.py --job-dir ../../runs/jobs/your_job --show-deformed --warp-scale 2.0
```

### Output File Naming

When `--show-deformed` is enabled:
- Deformed versions have `_deformed` suffix: `stress_sigma_xx_deformed.png`
- Undeformed versions (if generated separately): `stress_sigma_xx.png`

## Examples

### Example 1: Standard Postprocessing (Undeformed)
```bash
python postprocess_pyvista.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full
```
**Output files:**
- `von_mises_stress.png` - von Mises on original geometry
- `stress_sigma_xx.png` - σ_xx on original geometry
- `strain_epsilon_xx.png` - ε_xx on original geometry
- etc.

### Example 2: Deformed Shape Visualization
```bash
python postprocess_pyvista.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0_3holes_full --show-deformed
```
**Output files:**
- `von_mises_stress_deformed.png` - von Mises on deformed geometry
- `stress_sigma_xx_deformed.png` - σ_xx on deformed geometry
- `strain_epsilon_xx_deformed.png` - ε_xx on deformed geometry
- etc.

### Example 3: Exaggerated Deformation
For small displacements that are hard to see at 1:1 scale:
```bash
python postprocess_pyvista.py --job-dir ../../runs/jobs/your_job --show-deformed --warp-scale 5.0
```

### Example 4: Generate Both Versions
Run twice with different settings:
```bash
# First run: undeformed
python postprocess_pyvista.py --job-dir ../../runs/jobs/your_job

# Second run: deformed
python postprocess_pyvista.py --job-dir ../../runs/jobs/your_job --show-deformed --warp-scale 2.0
```

## What Gets Deformed

### Fields Affected by `--show-deformed`:
- ✅ Von Mises stress (`von_mises_stress_deformed.png`)
- ✅ Stress components σ_xx, σ_yy, σ_xy (`stress_sigma_*_deformed.png`)
- ✅ Strain components ε_xx, ε_yy, ε_xy (`strain_epsilon_*_deformed.png`)

### Fields NOT Affected (Always Deformed):
- Displacement magnitude (`displacement_magnitude.png`) - always shows deformation
- Mesh structure (`mesh_structure.png`) - shows undeformed mesh wireframe

## Technical Details

### How It Works
1. The displacement field (u) is loaded from `solution.h5`
2. The field grid (stress/strain) is loaded from derived HDF5 files
3. PyVista's `warp_by_vector()` method applies the displacement to deform the grid
4. The field values are displayed on the deformed geometry

### Warp Scale Factor
- `warp_scale = 1.0`: True 1:1 deformation (actual displacements)
- `warp_scale > 1.0`: Exaggerated deformation (useful for small displacements)
- `warp_scale < 1.0`: Reduced deformation (rarely used)

### Physical Interpretation
- **Undeformed view**: Shows field values at original mesh positions (Lagrangian reference)
- **Deformed view**: Shows field values at displaced mesh positions (updated Lagrangian)

## Comparison with Other Tools

| Feature | Undeformed (`--show-deformed` off) | Deformed (`--show-deformed` on) |
|---------|-----------------------------------|--------------------------------|
| Geometry | Original mesh positions | Warped by displacement field |
| File suffix | None | `_deformed` |
| Use case | Standard FEA visualization | See actual shape change |
| Warp scale | N/A | Controlled by `--warp-scale` |

## Best Practices

1. **Small Displacements**: Use `--warp-scale 5.0` or higher to make deformation visible
2. **Large Displacements**: Use `--warp-scale 1.0` for realistic visualization
3. **Comparison**: Generate both deformed and undeformed versions for side-by-side comparison
4. **Documentation**: Note the warp scale in figure captions for publications

## Related Documentation
- [COLORBAR_CONFIG.md](COLORBAR_CONFIG.md) - Colorbar customization guide
- [FIX_NOTES.md](FIX_NOTES.md) - ParaView segfault fix and PyVista migration
- [QUICK_START.md](QUICK_START.md) - General postprocessing guide
