# Installation

Three pieces: the **framework environment** (Python 3.11 + the scientific
stack), **LAMMPS** (a system binary the MCP tool service launches), and — only
if you run the CrewAI baseline — a second, **isolated CrewAI environment**.
The versions below are the ones recorded in the published run manifests.

## 1. Framework environment (conda + pip)

```bash
git clone https://github.com/zueskalare/OKR-Agent.git && cd OKR-Agent
conda env create -f environment.yml -p ./venv     # CPython 3.11.14, DOLFINx 0.9.0, PETSc 3.24, MPICH 4.3.2, VTK 9.5
./venv/bin/pip install -e .                       # the framework (editable; prompt templates are read from the tree)
./venv/bin/python -m pytest -q tests AtomAgents/tests   # hermetic — no services, no keys needed
```

`environment.yml` holds the compiled scientific stack from conda-forge;
`requirements.txt` is the pinned pip layer (langchain 1.3.14, langgraph 1.2.10,
openai 2.53.0, anthropic 0.120.2, mcp 1.29.0, numpy 2.3.4, gmsh 4.14.1,
ovito 3.13.1, …) that `environment.yml` installs on top. The full package list
of the study machine is kept as an audit record in `docs/env/pip-freeze-study-machine.txt`
(not installable as-is — it includes conda-only distributions).

The repo expects the environment at `./venv` (`Makefile`, `benchmarks/runner.py`);
put it elsewhere and set `OKR_PYTHON=/path/to/python` (or `make … PY=…`).

Optional extras: `pip install -e .[local-embeddings]` (on-device
sentence-transformers for `ATOMAGENTS_EMBED_PROVIDER=local`), `.[analysis]`
(pyvista/matplotlib tooling for `analysis/`), `.[dev]` (pytest, ruff, nbformat).

## 2. LAMMPS

The MCP tool `lammps_run` executes LAMMPS from the service's `PATH`. The study
used **LAMMPS 22 Jul 2025 – Update 3** (Homebrew build, Open MPI).

| Platform | Install | Launcher to pin |
|---|---|---|
| macOS (Homebrew) | `brew install lammps` (`lmp_mpi` + Open MPI `mpirun`) | `ATOMAGENTS_LAMMPS_CMD="/opt/homebrew/bin/mpirun -np 12 /opt/homebrew/bin/lmp_mpi"` |
| Linux (conda) | `conda install -p ./venv -c conda-forge lammps` (MPICH build, consistent with the env's `mpirun`) | `ATOMAGENTS_LAMMPS_CMD="$PWD/venv/bin/mpirun -np 12 $PWD/venv/bin/lmp"` |
| Linux (source) | build with the MPI you will launch with | the matching `mpirun … lmp_mpi` |

**MPI launcher hazard (reproduced).** Auto-detection runs
`mpirun -np $ATOMAGENTS_LAMMPS_NP lmp_mpi` with whatever `mpirun` is first on
`PATH`. The conda environment ships MPICH's `mpirun`; Homebrew's `lmp_mpi` is
linked against Open MPI. An MPICH launcher driving an Open MPI binary does not
fail — it starts N **independent serial copies** of the same run. Either pin the
full command with `ATOMAGENTS_LAMMPS_CMD` (recommended; it is recorded per run)
or start `make mcp` from a shell where the conda env is **not** activated.
Set `ATOMAGENTS_LAMMPS_NP` to your core count; the study ran `-np 12`.

Potential files: the Fe task names `Fe_Mendelev.eam.fs`; agents fetch
potentials themselves (LAMMPS ships a copy under `potentials/`; see the NIST
Interatomic Potentials Repository for provenance). Atomsk was not installed on
the study machine.

## 3. CrewAI baseline environment (optional)

`crewai 0.201.1` cannot share an environment with the framework's langchain 1.x
stack, so the hierarchical-crew arm runs in its own interpreter
(`benchmarks/crewai-env/`, CPython 3.12.9, 165 pinned distributions):

```bash
make crewai-env        # = uv venv --python 3.12.9 benchmarks/crewai-env/.venv && uv sync --locked --project benchmarks/crewai-env
```

Without `uv`: `python3.12 -m venv benchmarks/crewai-env/.venv && benchmarks/crewai-env/.venv/bin/pip install -r benchmarks/crewai-env/requirements.txt`.
A different location goes in `OKR_CREWAI_PYTHON`.

## 4. Services and keys

```bash
cp .env.example .env            # fill in: endpoints, keys, model ids, embeddings endpoint
make mcp                        # terminal 1 — MCP tool service, 127.0.0.1:9999; export TAVILY_API_KEY + LAMMPS vars in THIS shell
set -a; . ./.env; set +a
make backend                    # terminal 2 — OpenAI-compatible API, 127.0.0.1:8002
make mcp-check                  # the service exposes the pinned 17-tool surface (digest 357bd60c…)
```

Who reads which variable is tabulated in `docs/CONFIGURATION.md`. The MCP
service never reads `.env`; the benchmark runner parses `.env` itself and
passes it to the backend it starts.

## 5. Platform notes

* **macOS.** Child processes of the exec tools are confined with `sandbox-exec`
  (writes denied outside `ATOMAGENTS_TOOL_ROOT`); this is mandatory when a jail
  is set. Apple-silicon wheels exist for everything in `requirements.txt`.
* **Linux.** No per-process confinement — the service starts with a warning and
  relies on the jail/denylist layers. Run it in a container or under a
  throwaway user with `runs/jobs` bind-mounted as `ATOMAGENTS_TOOL_ROOT`, bound
  to `127.0.0.1`. OVITO rendering (`md_visualize`) needs an OpenGL-capable
  environment (`xvfb-run` works).
* **Windows.** Not supported (the tool service uses POSIX process groups).
