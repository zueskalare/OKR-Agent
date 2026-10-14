# Job History Post-Processing Tools

Tools for analyzing and visualizing results from completed FEniCSx and LAMMPS
simulation sessions under `jobs/`.

## Which script do I use?

| Your output format | Script | Status |
|---|---|---|
| FEniCSx XDMF/H5 (`solution.xdmf` + `.h5`) | `postprocess_pyvista.py` | ✅ **recommended** |
| VTU files (FEniCSx `VTKFile`, OpenFOAM, …) | `postprocess_vtu.py` | ✅ newest |
| — | `postprocess_fenicsx.py` (ParaView/pvpython) | ❌ **deprecated** |

Why the ParaView script is deprecated: its XDMF reader is configured with
`GridStatus=["Grid"]`, which never matches the grid name DOLFINx writes, so it
loads an empty dataset (0 points) and then prints a hardcoded success summary
listing images it never produced. It now exits with a pointer here unless
`ATOMAGENTS_FORCE_PARAVIEW=1` is set.

## Quick usage

```bash
# XDMF/H5 job directory
python postprocess_pyvista.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0

# VTU file(s)
python postprocess_vtu.py --job-dir ../../runs/jobs/<session>     # auto-discovers VTU
python postprocess_vtu.py --vtu-file results.vtu

# Batch over multiple jobs
./batch_postprocess.sh
```

Requirements: `pip install pyvista meshio numpy h5py`.

See `QUICK_START.md` for a walkthrough and `README_postprocess.md` for full
option reference. Feature guides: `DEFORMED_SHAPE_GUIDE.md`,
`COLORBAR_CONFIG.md`, `FONT_SCALING_GUIDE.md`, `COMPATIBILITY_FIX.md`,
`FIX_NOTES.md`.

## Notebooks & other tools

- `fem_post_processing.ipynb` — interactive FEM result exploration
- `MD_postprocessing.ipynb` — LAMMPS/MD analysis
- `visualize.dump.ipynb` — LAMMPS dump file visualization
- `scatter_plot.ipynb` — stress/strain scatter plots (uses `beaut4mpl`)

Plot styling uses the `beaut4mpl` library (v2 API: `from beaut4mpl.layout import ...`).
