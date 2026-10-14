# FEniCSx Post-Processing — Quick Start

## Overview

Generate publication-quality visualizations (displacement, von Mises stress,
stress/strain components, deformed shapes) from completed FEniCSx jobs.

Two supported scripts, both pure-Python (no ParaView installation needed):

| File | Use for |
|------|---------|
| `postprocess_pyvista.py` | XDMF + H5 outputs (`solution.xdmf`) — **recommended** |
| `postprocess_vtu.py` | VTU outputs (also auto-discovers them in a job dir) |
| `batch_postprocess.sh` | Batch processing multiple jobs |
| `README_postprocess.md` | Full option reference |

> `postprocess_fenicsx.py` (ParaView/pvpython) is **deprecated** — it silently
> loaded 0 points from DOLFINx XDMF and reported hardcoded success. It exits
> with an explanation if run.

## 1. Install requirements

```bash
pip install pyvista meshio numpy h5py
```

## 2. Process a job

```bash
# XDMF/H5 job
python postprocess_pyvista.py --job-dir ../../runs/jobs/session-atomagent-fem_demo0

# VTU job (auto-discovers *.vtu under the job dir)
python postprocess_vtu.py --job-dir ../../runs/jobs/<session>

# Results land in <job-dir>/postprocess/
```

## 3. View results

```bash
open ../../runs/jobs/session-atomagent-fem_demo0/postprocess/von_mises_stress.png
```

## Common options

```bash
# Deformed shape with scale factor
python postprocess_pyvista.py --job-dir <dir> --deformed --scale-factor 100

# Custom resolution
python postprocess_pyvista.py --job-dir <dir> --resolution 1920 1080

# Batch over every session under jobs/
./batch_postprocess.sh
```

See `DEFORMED_SHAPE_GUIDE.md`, `COLORBAR_CONFIG.md`, and
`FONT_SCALING_GUIDE.md` for feature-specific configuration, and run any script
with `--help` for the authoritative option list.

## Sanity check your inputs first

If a run's solver log shows `PETSc reason=3` or a very high iteration count,
validate before trusting the fields you plot:

```bash
python -m AtomAgents.utils.run_validation --fem-log <job>/analysis/solver.log
```
