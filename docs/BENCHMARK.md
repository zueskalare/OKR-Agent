# Running the benchmark campaign

This is the procedure behind the thirty runs reported in the paper, written so
that the same bytes reach every arm you run. Scoring is *not* part of this
repository: the deterministic extraction, the fixed-method recomputation and
the adjudication live with the run records at
<https://huggingface.co/datasets/BioDMX-UGA/OKRAgent_runs> (`audit/scripts`,
`manuscript-figures/`).

## The design in one table

| | |
|---|---|
| Tasks | `demo1` CNT-bundle tension (LAMMPS) · `demo2` polycrystalline-Fe compression (LAMMPS) · `fem_demo0` plate with a circular hole (FEniCSx) |
| Arms | `gpt`, `opus` (OKR-Agent; 3 trials each) · `react`, `react_gpt`, `crewai`, `crewai_gpt` (baselines; 1 trial each) |
| Cells | 3 tasks × (2 × 3 + 4 × 1) = **30** |
| Brief | one per task, `benchmarks/briefs/<task>.txt`, SHA-256 in `briefs/briefs.lock.json`; the README `## Demos` curl blocks are the canonical text (`make briefs-check`) |
| Prompts | OKR-Agent: `AtomAgents/templates/prompts/` (MD) or `templates/fenicsx/` (FEM), every file hashed into the manifest · baselines: `benchmarks/prompts/baseline_{md,fem}.md` |
| Tools | the same 17 for every arm — 13 MCP tools + 4 bookkeeping tools — pinned in `benchmarks/tools.lock.json` (digest `357bd60c…`) and asserted before each run |
| Turns | brief, then at most three literal `Continue.` turns (`MAX_TURNS = 4`); early stop when a continuation adds no new deliverable files or bytes; 5,400 s per turn |
| Models | `gpt-5.6-sol` on the OpenAI route, `claude-opus-5` on the Anthropic route, reasoning effort `high` (see "Model identifiers" in the README) |

## Prerequisites

1. `docs/INSTALL.md` done: framework env, LAMMPS, and the CrewAI env if you run `crewai*`.
2. `.env` filled in from `.env.example`. The OKR-Agent arms need a working
   **embeddings** endpoint (`ATOMAGENTS_EMBED_BASE_URL` + key): a run with
   semantic recall off is aborted after turn 1 (`embedding_failure`).
3. The MCP tool service running in its own shell, with `TAVILY_API_KEY` and
   the LAMMPS variables exported there:
   ```bash
   make mcp
   make mcp-check        # must print: ok: 17 tools match tools.lock.json
   ```
   The runner refuses to start a cell unless this check passes (`mcp_preflight`
   in the manifest), and aborts after turn 1 if the backend bound anything other
   than 13 MCP tools (`mcp_tools_loaded` / `mcp_failure`). Do **not** start the
   backend yourself — the runner starts, restarts and stops it per run
   (`lsof -iTCP:8002`; anything listening there is terminated).
4. `make briefs-check` passes.

## Commands

```bash
# OKR-Agent, three trials each (9 + 9 runs)
for t in 1 2 3; do python -m benchmarks.runner --all --arms gpt  --trial $t --halt-on-failure; done
for t in 1 2 3; do python -m benchmarks.runner --all --arms opus --trial $t --halt-on-failure; done
# baselines, one trial each (12 runs)
python -m benchmarks.runner --all --arms react,react_gpt,crewai,crewai_gpt --trial 1 --halt-on-failure
```

Useful flags: `--skip-existing` resumes a campaign (cells with a
`manifest.json` are skipped); `--task demo1 --arm opus --trial 2` runs one
cell; `--dry-run` prints the manifest that *would* be written and launches
nothing; `--prefix ws1` namespaces the session directories
(`session-ws1-<task>-<arm>-t<n>`); `--halt-on-failure` stops the campaign on a
failed cell or on provider throttling (HTTP 429 / `auth_unavailable` storms)
instead of burning the remaining budget. `opus_std` (an ablation arm withheld
from the dataset) runs only when named explicitly.

Each run writes `runs/jobs/session-<task>-<arm>-t<n>/` (the agents' deliverables,
`memory.db`, `README.md` board, `manifest.json`; `graph_state.db` for OKR-Agent
arms — see `runs/README.md` for why that file is sensitive) and
`runs/logs/<run_id>.{backend.log,turnN.sse,arm.log}`.

Expect hours per cell: the study's wall-clock ranged from minutes (a baseline
that stopped early) to several hours (a full OKR-Agent MD run), and LAMMPS
production runs dominate.

## What a run records (`manifest.json`)

| Key | Meaning |
|---|---|
| `run_id`, `task`, `arm`, `trial`, `campaign` | cell identity; `campaign` is the `--prefix` (empty by default) |
| `model_id`, `provider`, `models` | backend model alias (`atomagent`, `fenicsagent[@anthropic]`), route, and the effective model identifiers (`openai`, `anthropic`, `bench_model`) |
| `env` | the per-arm pins (`ATOMAGENTS_SUPERVISOR_EXEC_TOOLS`, `…_ENABLE_TRIAGE`, `…_STAGE_GUARD`, `…_ENABLE_HITL`, `…_REASONING_EFFORT`, `…_EMBED_PROVIDER`; `BENCH_PROVIDER`/`BENCH_MODEL` for baselines) |
| `brief` | `sha256`, `chars`, `brief_file`, `session_dir` from the lockfile |
| `prompt_flavor`, `prompt_hashes` | prompt family and the SHA-256 (16 hex) of every template file loaded |
| `tools_lock` (baselines), `mcp_preflight` (all arms) | the tool-surface digest asserted before the run |
| `git`, `runner_sha256`, `runner_release` | code provenance (see below) |
| `toolchain` | dolfinx / gmsh / ufl / petsc4py / ovito / numpy versions, the LAMMPS build banner, and the `mpirun` that would launch it |
| `banner`, `embeddings`, `resolved_model_lines` | what the backend reported at startup: provider, resolved embedding backend and dimensionality, the model that served each call |
| `turns[]` | per turn: what was sent, seconds, transport status, and `session_stats` (messages, embedding vectors, artefacts, metrics, deliverable files and bytes) |
| `mcp_tools_loaded`, `stopped_early`, `embedding_failure`, `mcp_failure`, `cap_hit` | gates and stops that fired |
| `final`, `wall_clock_s`, `stage_guard_catches`, `provider_error_429_lines`, `provider_auth_503_lines` | end-of-run statistics |
| `arm_result`, `exit_code` (baselines) | the arm's own summary (model, tools bound, tool calls, caps) |

## Provenance and comparability with the published runs

* The published manifests record `git.head = 19ee9ba`, `dirty: true`. This
  release *is* that working tree. Every hash-locked input matches: brief
  SHA-256s `695c96ae…` (demo1), `b538184c…` (demo2), `ac24b04c…` (fem_demo0);
  tool-lock digest `357bd60c…`; all prompt-template hashes.
* `runner_sha256` identifies the runner that wrote a manifest. Eighteen study
  manifests carry `62716586fe8fb911`; twelve carry four earlier never-committed
  revisions. The released runner hashes differently — it adds the MCP preflight
  and the `mcp_tools_loaded` gate, configurable interpreters (`OKR_PYTHON`,
  `OKR_CREWAI_PYTHON`), an empty default `--prefix` (the study used `ws29`…`ws31`
  prefixes; the dataset stripped them), explicit model pinning with
  `manifest["models"]`, a corrected LAMMPS version probe (the study manifests
  record the `lmp -h` options header, not the build date) and a `mpirun` entry
  in `toolchain`. **None of these change what an agent receives**: briefs,
  prompts, tool surface and per-arm environment are untouched; the new keys are
  additive, and the new refusals fire only on a misconfigured host.
* The model identifiers were served through a gateway on the study machine
  and the arms contain a few gateway-specific shims (README, "Fidelity
  caveats"). Runs against the vendor APIs are the same experiment with a
  different transport; they are not byte-reproductions, and agent outputs vary
  by design.

## Lock files — the regenerate-deliberately rule

`benchmarks/briefs/briefs.lock.json` and `benchmarks/tools.lock.json` exist so
that "the arms received the same inputs" is a checked fact. Regenerate them
only when you *mean* to change the experiment, and say so in the commit:

```bash
python -m benchmarks.extract_briefs            # README demo blocks → briefs/*.txt + lockfile
python -m benchmarks.toolbind --write          # live tool surface → tools.lock.json
```

Any edit to a tool's name, docstring or argument schema in
`AtomAgents/mcp_service/mcp_service.py` changes the digest and makes new runs
incomparable with the dataset.

## After a campaign

`python -m benchmarks.compare_arms [--prefix …] [--trial N]` prints a quick
cross-arm table from manifests and the filesystem (wall-clock, messages, tool
calls, registered vs hash-verified artefacts, deliverables), and
`python -m benchmarks.export_transcript --all` rebuilds a normalised
`transcript.*.jsonl` per session. For the audited quantities of the paper use
the dataset's `audit/` pipeline on your `runs/jobs/` tree (`JOBS_ROOT`).
