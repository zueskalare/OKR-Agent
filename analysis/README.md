# analysis/ — post-hoc analysis tooling

Work *about* agent runs: figure post-processing and the human-built reference
models used for the paper's agent-vs-reference overlays. Framework code lives
in `AtomAgents/`; the benchmark harness in `benchmarks/`.

| Path | Contents |
|---|---|
| `job_hist_postprocessing/` | FEM/MD figure tooling. `postprocess_pyvista.py` (XDMF/H5) or `postprocess_vtu.py` (VTU) render deformed-shape and field figures from a session's output; `MD_postprocessing.ipynb` / `fem_post_processing.ipynb` / `scatter_plot.ipynb` are the notebooks the first-round figures came from. Start at `INDEX.md`. |
| `reference_comparison/` | Hand-built reference LAMMPS models for the two MD tasks (`fiber_model/` CNT bundle, hex and rectangular cross-sections; `multi_grain_model/` polycrystalline Fe) with their input decks, run logs and gauge stress–strain data, plus the overlay scripts `compare_demo1.py` / `compare_demo2.py` / `compare_rect.py` that read an agent session's `.dat`/log files, stamp validation verdicts and write `overlay_*_metrics.md`. |

**Where the data is.** These tools read session directories under `runs/`
(set `OKR_RUNS_ROOT` to point elsewhere; default `../../runs` relative to the
script). The sessions analysed in the paper are published at
<https://huggingface.co/datasets/BioDMX-UGA/OKRAgent_runs>; the audit and
figure code that produces every reported number lives there too
(`audit/scripts`, `manuscript-figures/`). Paths of the form `<repo>/…` in the
recorded logs and metric files denote the study checkout.

Conventions: never write into a session you are analysing; validate any
simulation output before plotting with `python -m AtomAgents.utils.run_validation`.
Notebooks are committed without outputs (`tests/test_release_hygiene.py` enforces it).
