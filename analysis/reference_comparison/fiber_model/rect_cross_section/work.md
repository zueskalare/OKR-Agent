# Comparison of results from AtomAgents — Demo1 variant, rectangular cross-section fiber

## Status: NOT RUN

This directory is a placeholder. **No rectangular-cross-section model has ever
been built or run** — neither here nor by any agent session: in the archive,
`session-atomagent-demo0_square` and `demo0_third_square` were asked for the
square cross-section but silently re-reported the cylindrical results
(`output/stress.dat` md5-identical to the cylindrical run).

The previous content of this file was a copy-paste of the hex_cross_section
pointer and referred to data that does not exist for this case.

## What is needed to produce this comparison
1. Build a rectangular-cross-section CNT-bundle data file (same bead model,
   force field, and box conventions as
   `../hex_cross_section` / `runs/jobs/session-atomagent-demo1/input/model1.data`,
   with the bundle packed on a rectangular lattice instead of hexagonal).
2. Adapt `../hex_cross_section/run_tension.in` (only `read_data` path and the
   gage-region definition change).
3. Run, then extract `stress_strain_gage.dat` and validate with
   `python -m AtomAgents.utils.run_validation --lammps-log <log> --script <in> --stress-dat <dat>`.
4. Add a `compare_rect.py` overlay mirroring
   `analysis/reference_comparison/compare_demo1.py`.

Local `mpirun`/`lmp_mpi` are available, so step 3 can run on this machine once
the model file exists (step 1 is the open modeling task).
