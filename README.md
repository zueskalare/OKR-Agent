# OKR-Agent

OKR-Agent is a coordinated peer multi-agent framework that turns an informal
engineering request into executed, documented mechanical simulations. Three
agents — **Supervisor**, **Modeler** and **Simulator** — share an *Objectives
and Key Results* (OKR) board kept in one SQLite database per session, retrieve
each other's work through hybrid (embedding + BM25) recall, and act on the
world only through a fixed set of seventeen tools exposed over the Model
Context Protocol: file and shell access, Python execution, web and
documentation search, LAMMPS molecular dynamics, and FEniCSx finite-element
meshing, solving and visualization. This repository holds the framework
(Python package `AtomAgents`), the OpenAI-compatible backend that serves it,
the MCP tool service, and the benchmark harness used for the paper.

> **Paper.** J. Tian *et al.*, "Objectives and Key Results Driven Multi-Agent
> Framework for Mechanical Design and Simulation", *Advanced Intelligent
> Systems* (2026). Citation metadata in `CITATION.cff`.
>
> **Run records.** The complete records of the thirty audited benchmark runs —
> session logs, per-call tool traces, OKR board states, artefact registries with
> SHA-256 checksums, plus the audit and figure code — are published at
> <https://huggingface.co/datasets/BioDMX-UGA/OKRAgent_runs>.
>
> **License.** MIT (`LICENSE`).

## Repository layout

    AtomAgents/          Framework: agents, SQLite OKR/RAG memory, MCP tool service, prompt templates, config
      agents/              Supervisor / Modeler / Simulator (think → plan → act → report), tool node
      memory/              SQLiteRAGMemory: messages + FTS5 + embeddings; OKR / artefact / metric / attempt tables
      mcp_service/         FastMCP tool server — 13 tools: files, shell, Python, search, LAMMPS, FEniCSx, OVITO
      templates/           Prompt templates: prompts/ (MD tasks) and fenicsx/ (FEM task); hashed into every run manifest
      utils/               Board rendering, OKR tools, run validation, prompt loader, embeddings
      config.py            Every ATOMAGENTS_* setting (reference: docs/CONFIGURATION.md)
    backend/             OpenAI-compatible FastAPI server (Responses + Chat Completions on 127.0.0.1:8002/v1)
    benchmarks/          Campaign harness: runner, ReAct + CrewAI baseline arms, hash-locked briefs,
                         17-tool lock, baseline prompts, CrewAI environment spec (crewai-env/)
    tests/               Hermetic test suite (real-failure fixtures under tests/fixtures/)
    examples/            Client sample, notebook walkthroughs, small MCP / OKR / memory demos
    analysis/            Figure post-processing and the human-built reference models behind the paper's overlays
    docs/                INSTALL.md · BENCHMARK.md · CONFIGURATION.md · env/ (study-machine package record)
    runs/                Runtime data, gitignored: runs/jobs sessions, runs/logs, runs/archive

## Quick start

```bash
conda env create -f environment.yml -p ./venv      # Python 3.11 + DOLFINx / PETSc / VTK  (docs/INSTALL.md)
./venv/bin/pip install -e .                        # the framework, editable
cp .env.example .env                               # endpoint(s), keys, model ids — never committed
make mcp                                           # MCP tool service on 127.0.0.1:9999 (its own shell; needs LAMMPS)
set -a; . ./.env; set +a; make backend             # OpenAI-compatible API on 127.0.0.1:8002
curl -sN http://localhost:8002/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"atomagent","session_dir":"session-hello","input":"Explain LAMMPS basics.","stream":true}'
```

The three benchmark requests are under [Demos](#demos); the full campaign is
one command per cell — see the next section and `docs/BENCHMARK.md`.

## Reproducing the benchmark

The paper's benchmark is thirty runs: three tasks (CNT-bundle tension and
polycrystalline-Fe compression in LAMMPS, a plate with a circular hole in
FEniCSx) × six configurations, with three independent trials per OKR-Agent
configuration and one per baseline. Everything an arm receives — the brief,
the prompt templates, the tool surface, the per-arm environment — is
hash-locked and recorded in each run's `manifest.json`.

```bash
make mcp && make mcp-check       # tool service up; digest must match benchmarks/tools.lock.json
make briefs-check                # README briefs == benchmarks/briefs/*.txt (hash-locked)
python -m benchmarks.runner --all --arms gpt  --trial 1     # OKR · GPT     (repeat with --trial 2, 3)
python -m benchmarks.runner --all --arms opus --trial 1     # OKR · Claude  (repeat with --trial 2, 3)
python -m benchmarks.runner --all --arms react,react_gpt,crewai,crewai_gpt --trial 1   # the four baselines
```

### Arm names

| Paper / figures | Runner arm (`--arms`) | Dataset directory | Name in older internal records |
|---|---|---|---|
| OKR · GPT | `gpt` | `jobs/okr-gpt/` | `atomagents-gpt-standard` |
| OKR · Claude | `opus` | `jobs/okr-claude/` | `atomagents-opus-fulltools` |
| ReAct (Claude) | `react` | `jobs/react-claude/` | `baseline-react-claude` |
| ReAct (GPT) | `react_gpt` | `jobs/react-gpt/` | `baseline-react-gpt` |
| CrewAI (Claude) | `crewai` | `jobs/crewai-claude/` | `baseline-crewai-claude` |
| CrewAI (GPT) | `crewai_gpt` | `jobs/crewai-gpt/` | `baseline-crewai-gpt` |
| — (ablation, not reported) | `opus_std` | withheld | `atomagents-claude-standard` |

A fresh run lands in `runs/jobs/session-<task>-<arm>-t<n>/`; the dataset keeps
the same session directories under `jobs/<dataset arm>/<task>-t<n>/`.

### Provenance

All thirty published manifests record `git.head = 19ee9ba` with a dirty
working tree: the campaign ran on uncommitted changes to the briefs, the
runner, the two baseline arms and the tool-binding layer. This release contains
exactly those working-tree files; the hash-locked inputs (briefs, prompt
templates, tool lock, baseline prompts) are byte-identical to the manifests.
The runner hashes itself into every manifest (`runner_sha256`). Eighteen of the
thirty study manifests carry the final study-era hash `62716586fe8fb911`; the
other twelve carry four earlier, never-committed revisions of the same file
that cannot be recovered. None of those revisions changed what the agents
received. The released runner hashes differently by design — it gained the
preflight checks described in `docs/BENCHMARK.md` and no longer hardcodes
machine paths — and stamps `runner_release: "v1.0.0"` into the manifests it writes.

### Fidelity caveats

* **Model identifiers.** The runs record `gpt-5.6-sol` (OpenAI Responses API)
  and `claude-opus-5` (Anthropic Messages API). On the study machine both were
  served through an OpenAI-/Anthropic-compatible gateway; set the identifiers
  that exist on *your* endpoint (`ATOMAGENTS_OPENAI_MODEL`,
  `ATOMAGENTS_ANTHROPIC_MODEL`, `BENCH_MODEL`). The runner writes the effective
  values to `manifest["models"]`.
* **Gateway-specific code paths are left as-run.** `AtomAgents/llm_factory.py`
  normalises dict-shaped stream events, `benchmarks/toolbind.py` replaces
  image-shaped tool results on the OpenAI wire, and the CrewAI arm injects a
  system block for its route. They are inert against the vendor APIs but are
  part of the code the dataset was produced with, so they stay.
* **Toolchain.** LAMMPS 22 Jul 2025 (Update 3), DOLFINx 0.9.0, gmsh 4.14.1,
  petsc4py 3.24.0, NumPy 2.3.4 (`environment.yml`, `requirements.txt`). Atomsk
  was *not* installed on the study machine — the agents wrote their own
  polycrystal builders; installing it changes the task.
* **What re-running does and does not give you.** The runs consumed commercial
  LLM APIs and hours of simulation, and agent outputs vary by design (the
  briefs leave secondary modelling choices to the agents). The audit and figure
  code that turns run records into the paper's numbers lives in the dataset
  (`audit/scripts`, `manuscript-figures/`), not here.

## Features

- **Multi-agent orchestration:** Specialized agents for supervision, input processing, model building, simulation execution, and more.
- **Natural language to simulation:** Converts high-level problem descriptions (e.g. text instructions or molecular structures) into actionable simulations and reports.
- **Extensible tool & workflow system:** Easily add new tools or agents for specific research domains or computational tasks.
- **API access:** OpenAI-compatible backend (`/v1`) plus an MCP tool service exposing simulation tools to the agents.

## Getting started

### Requirements
- Python 3.11 — the framework environment is a conda env built from
  `environment.yml` (DOLFINx 0.9.0, PETSc, MPICH, VTK) plus the pinned pip layer
  in `requirements.txt`; `docs/INSTALL.md` walks through it on macOS and Linux.
- LAMMPS (`lmp_mpi` + `mpirun`, or serial `lmp`); pin the launcher with
  `ATOMAGENTS_LAMMPS_CMD` — see the MPI note in `docs/INSTALL.md`.
- For the CrewAI baseline only: a second, isolated environment
  (`make crewai-env`, needs `uv`).

### Installation

    conda env create -f environment.yml -p ./venv
    ./venv/bin/pip install -e .
    cp .env.example .env     # endpoints, keys, model ids — never committed

### Run the stack

    set -a; . ./.env; set +a      # the backend reads its endpoint/keys from the shell
    make backend                  # OpenAI-compatible API on 127.0.0.1:8002 (docs at /docs)

Start the MCP tool service (required for agent tool calls):

    make mcp            # 127.0.0.1:9999, jail = ./runs/jobs; needs TAVILY_API_KEY and the LAMMPS vars in ITS shell

The service **refuses to start** unless `ATOMAGENTS_TOOL_ROOT` points at an
existing directory — that is the write/exec jail root and the writable
boundary of the child sandbox (see Security below). Startup prints the active
jail and sandbox status.

See `AtomAgents/mcp_service/README.md` for the registered tool list. The MCP
server speaks the Model Context Protocol over streamable HTTP at `/mcp/` —
there is no REST API.

With both services up, try the ready-to-run benchmark prompts in
[Demos](#demos).

### Security (tool sandbox)

Agent-issued shell/code/file operations run under a layered sandbox
(`AtomAgents/mcp_service/tools.py`):

- **Process sandbox (macOS)** — every child spawned by `code_runner`,
  `bash_run` and `lammps_run` runs under `sandbox-exec` with a profile that
  denies **all filesystem writes outside `ATOMAGENTS_TOOL_ROOT`**. The
  confinement is inherited by grandchildren (`mpirun` → `lmp_mpi`), matches
  canonical paths (so symlink and `..` tricks fail), and cannot be widened
  by a nested `sandbox-exec`. **Reads and network stay open by design** —
  runs legitimately fetch potentials and reference data. On macOS with a
  jail set the confinement is mandatory: the exec tools refuse to run
  unconfined (`ATOMAGENTS_SANDBOX_EXEC=0` opts out explicitly). On
  Linux/containers it auto-disables — there the container is the boundary.
  This is the layer that actually contains a child process; everything
  below it validates tool *arguments*.
- **Write/exec jail** — every write, save, script, and working directory
  must resolve inside `ATOMAGENTS_TOOL_ROOT` (symlink- and traversal-safe).
  `bash_run`/`code_runner` REQUIRE a jailed `cwd`. It **fails closed**: with
  no valid root the service will not start and the guard refuses every
  write/exec (`ATOMAGENTS_ALLOW_UNJAILED=1` restores denylist-only behaviour
  for library use). **Reads are deliberately NOT jailed** — they keep a
  system-directory denylist (`/etc`, `/private/etc`, `/usr`, `/System`, …)
  instead.
- **Command deny** — token-based, applied to `bash_run` commands: every line
  is scanned, tokens are split on shell metacharacters, quotes and
  backslashes stripped, path prefixes dropped, and comparison is case-folded
  — so `/usr/bin/git`, `env git`, `` `git` ``, `g"i"t`, `GIT` and anything
  after a newline are all refused, while `legit_tool --sudo-mode` and
  `.gitignore` still pass. Covers `git` (unless `ATOMAGENTS_ALLOW_GIT=1`),
  `sudo`/`su`/`doas`, `ssh`/`scp`/`sftp`, `shutdown`/`reboot`/`halt`,
  `mkfs*`, and root-targeting `rm`. For `code_runner` the same denylist is
  applied to **shell-escape call sites only** (AST-extracted literal
  arguments of `os.system`/`subprocess.*`/`pty` etc.) — never to ordinary
  Python identifiers or comments, so domain code like `Su = 250e6` or
  `halt = False` runs. Treat it as a guardrail, not the boundary —
  obfuscated evasion (e.g. base64 piped to a shell) is out of scope by
  design; the process sandbox is the boundary.
- **Env scrub** — child processes get a minimal environment (PATH, LANG,
  LC_ALL, TERM); every API key is dropped, `HOME`/`TMPDIR` are redirected
  INTO the session workspace, and `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM`
  are pinned to `/dev/null` so no git identity can resolve (a planted
  `.home/.gitconfig` used to restore one). Applies to `lammps_run` too —
  LAMMPS input scripts can execute code, so that child no longer inherits
  the service environment.
- **LAMMPS input scan** — a script using the `shell` or `python` built-ins
  (arbitrary code execution from inside the simulation) is refused.
- **Budgets** — server-side timeout ceilings (`ATOMAGENTS_BASH_TIMEOUT_MAX`
  600s, `ATOMAGENTS_CODE_TIMEOUT_MAX` 1800s, `ATOMAGENTS_LAMMPS_TIMEOUT`
  3000s) clamp caller-supplied values, and a timeout kills the whole process
  group so backgrounded children can't outlive it; outputs are truncated
  server-side, binary reads capped (`ATOMAGENTS_TOOL_BINARY_MAX` 8 MB); the
  client adds its own output cap (`ATOMAGENTS_TOOL_OUTPUT_MAX`) so even
  foreign MCP servers are bounded.

**Known gaps (deliberate, not yet closed):** reads are unjailed and the
network is open, so a determined model can still read this repo's `.env` and
exfiltrate it; the jail root is shared across concurrent sessions rather than
per-session; `download_file` has no SSRF filter. Keep
secrets out of the service's environment (it never reads `.env`) and run it on
an isolated host when the model is untrusted.

**Dry runs:** `ATOMAGENTS_EMULATE_TOOLS="lammps_run,md_visualize"` swaps the
listed tools for LLM-backed emulators (same name/schema, fabricated output)
— full agent flows without real MD/FEM execution. Board/OKR tools always
stay real.

**Embeddings note:** semantic memory recall needs an embedding endpoint.
`ATOMAGENTS_EMBED_PROVIDER=remote` (the campaign setting) requires
`ATOMAGENTS_EMBED_BASE_URL` / `ATOMAGENTS_EMBED_API_KEY` to be set explicitly —
a chat-only gateway on `OPENAI_BASE_URL` has no `/v1/embeddings`, and `remote`
refuses to fall back silently. `local` uses sentence-transformers on-device
(`pip install -e .[local-embeddings]`); `none` keeps SQLite FTS5 lexical recall only.

### Tests & checks

    make check          # pytest (tests/ + AtomAgents/tests/) + ruff + briefs-check

Validate simulation outputs standalone (the same gates run automatically on
every `lammps_run` tool result):

    ./venv/bin/python -m AtomAgents.utils.run_validation --lammps-log <log> --script <in>
    ./venv/bin/python -m AtomAgents.utils.run_validation --fem-log <solver.log>

### Configuration

All knobs are env-overridable (full reference with defaults and the process that
reads each one: `docs/CONFIGURATION.md`; source of truth: `AtomAgents/config.py`): `ATOMAGENTS_MODEL`,
`ATOMAGENTS_REASONING_EFFORT`, `ATOMAGENTS_JOBS_DIR`, `ATOMAGENTS_MCP_URL`,
`ATOMAGENTS_LAMMPS_CMD`, `ATOMAGENTS_EMBED_MODEL`, timeouts, and more.
Notable groups:

- **Sandbox** — `ATOMAGENTS_TOOL_ROOT` (write/exec jail; **required**, the
  service refuses to start without it), `ATOMAGENTS_ALLOW_UNJAILED` (opt out,
  library use only), `ATOMAGENTS_SANDBOX_EXEC` (`0` disables the macOS child
  sandbox), `ATOMAGENTS_SANDBOX_EXTRA_WRITE` (extra writable roots, colon-
  separated), `ATOMAGENTS_ALLOW_GIT`, `ATOMAGENTS_BASH_TIMEOUT_MAX`,
  `ATOMAGENTS_CODE_TIMEOUT_MAX`, `ATOMAGENTS_LAMMPS_TIMEOUT`,
  `ATOMAGENTS_TOOL_BINARY_MAX`, `ATOMAGENTS_TOOL_OUTPUT_MAX`.
- **Context management** — `ATOMAGENTS_MAX_INPUT_TOKENS` (declared model
  window, 120k), `ATOMAGENTS_TOOL_CLEAR_TRIGGER` (clear old tool results
  past ~30k tokens; `_KEEP`=3 newest kept), `ATOMAGENTS_SUMMARIZE` (history
  summarization, OFF — prompt-content-changing), `ATOMAGENTS_RAG_SNIPPET_CHARS`.
- **Loop/cost budgets** — `ATOMAGENTS_MAX_ACT_CYCLES`, `ATOMAGENTS_MAX_HANDOFFS`,
  `ATOMAGENTS_LLM_RPS`, `ATOMAGENTS_ANTHROPIC_PROMPT_CACHE`.
- **Stage/tool honesty** — `ATOMAGENTS_SUPERVISOR_EXEC_TOOLS` (default off:
  the Supervisor's act stage binds coordination tools only — read/search/OKR
  — so domain work must be delegated; `1` restores the legacy all-tools
  Supervisor), `ATOMAGENTS_STAGE_GUARD` (default on: simulated tool-call
  markup and the invented "results" after it are stripped from prose-stage
  output before persisting).
- **Backend lifecycle** — `ATOMAGENTS_MAX_SESSIONS` (LRU cap on cached
  session graphs, 32; eviction never loses state — checkpoints rebuild),
  `ATOMAGENTS_SESSION_LOCK_TIMEOUT`, `ATOMAGENTS_EXECUTOR_WORKERS`,
  `ATOMAGENTS_TOKEN_STREAM` (live token deltas on `/v1/responses`).
- **Dry run** — `ATOMAGENTS_EMULATE_TOOLS` (comma-separated tool names to
  emulate with the LLM instead of executing).

### Multiple API backends (providers)

The framework routes LLM calls through a **provider registry**
(`AtomAgents/config.py: discover_providers`), auto-detected from standard env:

| Provider | Detected from | API |
|---|---|---|
| `openai` | `OPENAI_BASE_URL` / `OPENAI_API_KEY` (works with OpenAI-compatible proxies) | chat completions |
| `anthropic` | `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) | messages |
| `local` | `ATOMAGENTS_LOCAL_LLM_BASE_URL` (Ollama-style, keyless) | chat completions |
| custom | `ATOMAGENTS_PROVIDER_<NAME>_BASE_URL` (+ `_API_TYPE`, `_API_KEY_ENV`, `_MODEL`) | either |

Select a backend per request with a provider-qualified model id —
`"model": "atomagent@anthropic"`, `"fenicsagent@local"`, etc.; a bare id uses
`ATOMAGENTS_LLM_PROVIDER` (default: `openai` if configured). `/v1/models`
lists every pipeline×provider combination. Per-provider default models:
`ATOMAGENTS_OPENAI_MODEL`, `ATOMAGENTS_ANTHROPIC_MODEL`, `ATOMAGENTS_LOCAL_MODEL`.
The factory (`AtomAgents/llm_factory.py`) maps generation params per backend
(e.g. `reasoning_effort` is OpenAI-only and dropped elsewhere). Claude models
MUST go through the `anthropic` provider: routed through an OpenAI-compatible
surface they answer structured-output calls with fenced/prose JSON and every
triage/plan/report call fails — the factory now refuses that combination with
a `ValueError` naming the fix (`ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI=1`
downgrades the refusal to an error log for gateways that genuinely translate).

## Agent Architecture

The multi-agent system (MAS) is orchestrated by the `SimulationAgentSystem`, which coordinates a sequence of specialized agents via a LangGraph workflow. The primary agents and components are:

- **Supervisor**: Oversees the workflow, manages Objectives & Key Results (OKRs), and directs hand-offs between agents.
- **ModelBuildingAgent**: Builds and validates molecular models (topology, coordinates, force fields), tracks sub-task jobs, and logs progress.
- **SimulatorAgent**: Executes the MD simulation through the MCP LAMMPS runner, monitors simulation progress, performs post-run analysis (energy, structural checks), and generates visualization snapshots.
- **MCP Service**: Exposes tools (file_reader, write_file, download_file, code_runner, bash_run, web_search, md_visualize, lammps_run, FEM helpers) over the Model Context Protocol for use by agents. `lammps_run` results carry a physics validation verdict the Simulator must quote.
- **Memory (SQLite RAG)**: Stores all agent messages and metadata in `memory.db` for semantic retrieval and context continuity.

The LangGraph workflow for a typical MD task follows:
```
Supervisor ⇄ Modeler ⇄ Simulator → END
```
Each agent runs an internal think → plan → act(→tools) → report subgraph; the
outer graph is compiled with a per-session checkpointer (`graph_state.db`), so
conversation state survives across API turns.

This architecture supports modular, stepwise execution of complex molecular simulation workflows, with clear responsibility boundaries and centralized state tracking.
## Database Table (for Memory)

OKR-Agent uses a hybrid message storage and retrieval system in `memory/sqlite_rag_memory.py`: SQLite for persistent message storage, with cosine similarity over stored embeddings and an FTS5 BM25 index for retrieval. The core table for this memory system is:

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- Unique message ID
    msg_json TEXT,                           -- Serialized (JSON) message object
    msg_text TEXT,                           -- Raw text content for embedding/search
    job_id TEXT,                             -- Optional: job identifier if relevant
    objective_id INTEGER,                    -- Optional: task ID (objective) tag
    key_result_id INTEGER,                   -- Optional: key result or subtask ID tag
    embedding TEXT,                          -- JSON-encoded vector for semantic retrieval
    agent_name TEXT,                         -- Name of agent that stored this message
    created_at TIMESTAMP                     -- Insert timestamp
);
```

A `messages_fts` FTS5 index provides the lexical half of hybrid retrieval; it
is (re)built automatically when a session database is created or rebound.

**Note:** This table is automatically created in the SQLite database (`memory.db`) when the agent system starts. The structure allows storage of all agent messages, making them available for both semantic (vector) retrieval and conventional filters.
## Task/OKR-Oriented Database Tables

The OKR detail store lives in the same session `memory.db` and is created
automatically. **SQL holds the detail; the job README renders the overview**
(`## Tasks` / `## Artefacts` sections are re-rendered by the harness after
every mutation — treat them as read-only views). Agents write through the
structured tools `okr_update`, `register_artefact` (filesystem-verified),
`record_metric`, and `record_attempt` — never raw SQL:

```
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE,             -- Unique reference for this task/job
    description TEXT,               -- Task or objective description
    status TEXT,                    -- e.g., 'submitted', 'active', 'done'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subtasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,                -- FK to tasks.id
    description TEXT,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS key_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,                -- FK to tasks.id
    subtask_id INTEGER,             -- Optionally link to a specific subtask
    description TEXT,
    achieved BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(task_id) REFERENCES tasks(id),
    FOREIGN KEY(subtask_id) REFERENCES subtasks(id)
);
```

In addition, three detail tables support evidence-backed status:

```
artefacts(path PRIMARY KEY, kind, bytes, hash, kr_id, task_id, created_at)
validation_metrics(id, metric, value, unit, job_id, kr_id, created_at)
attempts(id, subtask_id, description, command_hash, result, error_excerpt, created_at)
```

`register_artefact` refuses missing/empty files — a key result can only be
marked done against an artefact that actually exists. `attempts` is the
failure ledger surfaced to every agent so retries see what was already tried.

### Helper Methods in SQLiteRAGMemory
The `SQLiteRAGMemory` class provides convenient helpers:
- `get_current_task(job_id=None)`: fetch the most recent task (optionally filtered by `job_id`).
- `get_finished_tasks(job_id=None)`: list tasks with status not equal to 'submitted'.
- `get_finished_subtasks()`: list subtasks with status 'done'.
- `get_finished_key_results()`: list key results with `achieved = 1`.
- `get_messages_by_agent(agent_name: str)`: list stored message texts by the specified agent.

### Worked examples
`tests/test_memory_schema.py` exercises the full memory workflow (schema,
message hygiene, hybrid search with own-agent exclusion, rebind + FTS
backfill, OKR CRUD, artefact registration) and doubles as usage reference.
  
## Agent Memory & Workflow Integration
Agents can leverage `SQLiteRAGMemory` to maintain task/context awareness and perform RAG-based retrieval.

1. Initialize memory in your agent (e.g. in `__init__`):
   ```python
   from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory
   self.memory = SQLiteRAGMemory(sqlite_path='memory.db', k=10)
   ```

2. Before thinking, fetch completed OKR items and recent messages:
   ```python
   # Retrieve all tasks marked complete (or for a specific job)
   finished_tasks = self.memory.get_finished_tasks(job_id=your_job_id)
   # Retrieve all done subtasks
   finished_subtasks = self.memory.get_finished_subtasks()
   # Retrieve all achieved key results
   finished_results = self.memory.get_finished_key_results()
   # Retrieve up to k semantically similar messages not authored by this agent
   history = self.memory.load_memory_variables({
       'input': user_input,
       'agent_name': self.agent_type
   })['history']
   ```

3. Incorporate these as inputs into your THINK prompt for richer context:
   ```text
   ### Finished Tasks:
   {finished_tasks}
   ### Finished Subtasks:
   {finished_subtasks}
   ### Achieved Key Results:
   {finished_results}
   ### Conversation History (others):
   {history}

   [Your THINK prompt template here]
   ```

4. After responding, record your own messages with your agent name:
   ```python
   self.memory.save_context(
       inputs={'input': user_input},
       outputs={'output': assistant_reply},
       job_id=your_job_id,
       task_id=current_task_id,
       key_result_id=current_result_id,
       agent_name=self.agent_type
   )
   ```

### Summary of New Memory Methods
- `get_current_task(job_id=None)`: Fetch latest task.
- `get_finished_tasks(job_id=None)`: List tasks with status ≠ 'submitted'.
- `get_finished_subtasks()`: List subtasks where `status='done'`.
- `get_finished_key_results()`: List key results with `achieved=True`.
- `load_memory_variables(inputs)`: Retrieves top-K messages, optionally excluding those by passing `agent_name` in inputs.
- `get_messages_by_agent(agent_name: str)`: List all messages authored by the given agent.
## Agent Pipeline Subgraph
`BaseAgent.__init__` compiles each agent's internal workflow as a LangGraph
subgraph: `think → plan → act (⇄ tool) → report`, exposed as `agent.SubGraph`.
Loop guards bound plan↔act cycles without progress and refuse a third
identical tool call; stage headers are bound by node code (LLM-emitted wrong
stage headers are stripped and logged).

## OpenAI-Compatible Backend (Streaming)

The backend serves the OpenAI **Responses API** (`POST /v1/responses`) as its
primary surface, with `POST /v1/chat/completions` kept as a legacy layer.

Start the backend (lazy builds the graph on first use):

```bash
uvicorn backend.openai_backend:app --host 127.0.0.1 --port 8002
```

### Responses API (primary)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

stream = client.responses.create(
    model="atomagent",
    input="Explain LAMMPS basics.",
    stream=True,
    extra_body={"session_dir": "session-demo"},  # AtomAgents continuity extension
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

```bash
curl -sN http://localhost:8002/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"atomagent","input":"Explain LAMMPS basics.","stream":true,
       "session_dir":"session-demo"}'
```

Contract highlights:

- **Per-token streaming.** Prose stages (think/act) stream REAL
  `response.output_text.delta` frames token-by-token, chronologically per
  stage; structured outputs (triage/plan/report) never stream tokens. The
  deltas are a *raw-LLM preview*; the authoritative text arrives in
  `output_text.done`/`output_item.done` and carries the `# Agent - stage`
  header — concatenated deltas ≠ done text, by design. Disable everything
  incremental with `ATOMAGENTS_TOKEN_STREAM=0` (wire reverts to whole-run
  batches). Tokens surface on `/v1/responses` only; the legacy chat surface
  stays whole-message.
- **One output item per agent message.** Each item carries an AtomAgents
  `metadata` extension `{atom_agent, atom_stage, atom_type}`, readable by SDK
  clients via `item.model_extra["metadata"]` — the chat surface never exposed
  agent identity through typed SDK models. Streaming uses the standard named
  events (`response.created`, `response.output_item.added`,
  `response.output_text.delta`, `response.output_item.done`,
  `response.completed`; no `[DONE]` sentinel) plus custom
  `response.atomagents.working` / `.awaiting_user` / `.cancelled` progress
  events (additive; safely ignored by generic clients).
- **Terminal statuses:** `completed`; **HITL pause** → `status:"incomplete"`
  with `metadata.atom_status="awaiting_user"` (the SDK's
  `incomplete_details.reason` is a closed enum, so ours rides in metadata —
  reply in the same `session_dir` to resume); cancel → `status:"cancelled"`;
  errors → `status:"failed"` + `error`.
- **Continuity is `session_dir`**, not server-side storage:
  `previous_response_id` is rejected with 400, `store` is ignored (echoed
  `false`), and there is no `GET /v1/responses/{id}`. Send only the newest
  user message per turn. `POST /v1/responses/{id}/cancel` and the legacy
  `POST /v1/chat/cancel` both cancel a running session.

### Chat Completions (legacy, still supported)

Stream responses with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

stream = client.chat.completions.create(
    model="atomagent",  # or "atomagent_local"
    messages=[{"role": "user", "content": "Explain LAMMPS basics."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Stream with curl (Server-Sent Events):

```bash
curl -sS -N http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent_local",
        "messages": [{"role":"user","content":"Explain LAMMPS basics."}],
        "stream": true
      }'
```

Note on trivial/knowledge-only prompts: the Supervisor now runs a **triage
gate** on the first hop of each user turn. Greetings, small talk, and
self-contained knowledge questions (like "Explain LAMMPS basics." above) get a
direct 1-call answer and end the turn — the full Supervisor→Modeler→Simulator
pipeline only engages for actionable simulation requests. Under-specified
requests get 1-3 clarifying questions instead of an invented project. Disable
with `ATOMAGENTS_ENABLE_TRIAGE=0`. Runs are also bounded by
`ATOMAGENTS_MAX_HANDOFFS` (default 12 agent runs per user turn), and agents may
pause for your input mid-task (`ATOMAGENTS_ENABLE_HITL`, streamed as an
`awaiting_user` event; your next message resumes the run).

Models:
- `atomagent`: default pipeline on the default provider (`ATOMAGENTS_LLM_PROVIDER`)
- `atomagent_local`: legacy alias for `atomagent@local` (OpenAI-compatible server at `http://localhost:11434/v1/`)
- `fenicsagent`: FEniCSx-flavoured prompts
- Any of the above with an `@provider` suffix, e.g. `atomagent@anthropic`,
  `fenicsagent@local` — see "Multiple API backends" above.

### Conversation Memory and Continuations

The backend maintains per-session memory bound to a filesystem directory so you can continue chats without resending full history.

- How it works: pass `session_dir` with each request. If it's a relative path, the server will use `runs/jobs/<session_dir>` (see `ATOMAGENTS_JOBS_DIR`); if absolute, it uses that path directly. Graph state is checkpointed per session — send only the newest user message per turn.
- Memory location: `memory.db` is created under the chosen `session_dir` and reused across calls and restarts.
- Recommended: use a stable convention like `session-<model>-<client_id>`.

Python continuation example (non-streaming):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

# First turn — establishes memory under runs/jobs/session-<model>-<id>
resp1 = client.chat.completions.create(
    model="atomagent_local",
    extra_body={"session_dir": "session-atomagent_local-session-42"},
    messages=[{"role": "user", "content": "Plan a short MD tutorial."}],
)
print(resp1.choices[0].message.content)

# Second turn — continue chat without resending prior messages
resp2 = client.chat.completions.create(
    model="atomagent_local",
    extra_body={"session_dir": "session-atomagent_local-session-42"},
    messages=[{"role": "user", "content": "Great, add a LAMMPS example."}],
)
print(resp2.choices[0].message.content)
```

curl continuation example (streaming):

```bash
# First turn
curl -sS -N http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent",
        "session_dir": "session-atomagent-tutorial",
        "messages": [{"role":"user","content":"Plan a short MD tutorial."}],
        "stream": true
      }'

# Later — same session_dir continues the conversation; send only the newest message
curl -sS -N http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent",
        "session_dir": "session-atomagent-tutorial",
        "messages": [{"role":"user","content":"Great, add a LAMMPS example."}],
        "stream": true
      }'
```

For full multi-turn benchmark transcripts, see [Demos](#demos).

Multimodal note: you can include images by using OpenAI-style `image_url` content parts with base64 data URLs. Images are saved under the session workspace and referenced in the prompt context.

```python
messages=[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "Analyze this plot"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_DATA>"}}
    ]
  }
]
```

## Demos

These are the benchmark prompts behind the paper's framework comparison
(agent-vs-reference overlays live in `analysis/reference_comparison/`). Each
demo is a single `curl` against the backend; everything the agents produce —
models, input scripts, run logs, plots, `memory.db` — lands in
`runs/jobs/<session_dir>/`.

Conventions shared by all demos:

- **Start the stack first** (see [Run the stack](#run-the-stack)): the MCP
  tool service (`make mcp`, on `:9999`) and the backend (`make backend`, on
  `:8002`).
- **`session_dir` is the conversation.** Follow-up turns post to the same
  `session_dir` with only the newest message — never resend history.
- **Both API surfaces work.** Kick-offs below use the primary `/v1/responses`
  surface (`"input"`); follow-ups show the legacy `/v1/chat/completions` form
  (`"messages"`). Both drive the same pipeline and share `session_dir`
  continuity, so they can be mixed freely within a session.
- **No LAMMPS/FEniCSx installed?** Export
  `ATOMAGENTS_EMULATE_TOOLS="lammps_run,md_visualize"` (see
  [Security](#security-tool-sandbox) → Dry runs) to exercise the full agent
  flow with emulated execution tools.

| Demo | Model | `session_dir` | Task |
|---|---|---|---|
| [MD demo 1](#md-demo-1--twisted-cnt-fiber-bundle-under-tension) | `atomagent` | `session-atomagent-demo1` | Coarse-grained twisted CNT fiber bundle, uniaxial tension in LAMMPS, stress–strain curve |
| [MD demo 2](#md-demo-2--polycrystalline-fe-under-compression) | `atomagent` | `session-atomagent-demo2` | Polycrystalline bcc Fe (EAM), uniaxial compression in LAMMPS, stress–strain curve |
| [FEM demo](#fem-demo--plate-with-a-circular-hole-fenicsx) | `fenicsagent` | `session-atomagent-fem_demo0` | FEniCSx plate-with-hole benchmark, mesh convergence + Kirsch validation |
| [Notebook demos](#notebook-demos-interactive-no-backend) | — | `runs/jobs/session-nb-{fem,md}-<timestamp>/` | Architecture walkthrough notebooks: build the graph by hand, then run the MD or FEM demo |

Every demo runs on **either LLM backend**. The backend demos take it from the
model id — a bare `atomagent`/`fenicsagent` uses the default provider
(`ATOMAGENTS_LLM_PROVIDER`, `openai` if configured), and `atomagent@anthropic`
/ `fenicsagent@anthropic` selects the Anthropic route (e.g. `claude-opus-5`
via `ATOMAGENTS_ANTHROPIC_MODEL`) — just swap the `"model"` field in the
kick-off payloads below. The notebook demos have an explicit `PROVIDER`
switch in their first code cell.

### MD demo 1 — twisted CNT fiber bundle under tension

Builds a coarse-grained model of 92 (5,5) SWCNTs in a helically twisted,
hexagonally packed bundle, runs uniaxial tension in LAMMPS with gripped ends,
and post-processes the engineering stress–strain curve. The optional
`image_dir` (an AtomAgents extension) attaches the reference figures in
`examples/demos/demo_img/demo5/` to the prompt — a path relative to the backend's
working directory (the repo root under `make backend`) or absolute; drop the
field to send text only. It is accepted on any turn, not just the kick-off. The
benchmark runner never sends it.

<details>
<summary>Kick-off request (click to expand)</summary>

```bash
curl -sS -N http://localhost:8002/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent",
        "session_dir": "session-atomagent-demo1",
        "input": "Generate a full molecular dynamics (MD) simulation of a twisted carbon nanotube (CNT) fiber bundle using LAMMPS.\n\nModel Generation:\n\t•\tBuild a coarse-grained model of 92 (5,5) SWCNTs arranged in a hexagonally packed cylindrical bundle, total fiber length 100 nm.\n\t•\tApply helical twist to the bundle with n = 1 twist lap over the fiber length, where each CNT follows a helical centerline:\n\t•\tUse the natural bundle radius R_b from hexagonal close packing at ~1.0 nm inter-tube spacing, and pitch L/n, to generate helical coordinates for each CNT.\n\t•\tUse a finite fiber: non-periodic boundary conditions in all three directions, with vacuum padding along z.\n\t•\tOutput the model as a LAMMPS data file (model1.data) with proper atom types, bonds, and box dimensions.\n\nSimulation Setup:\n\t•\tWrite a LAMMPS input script to:\n\t•\tLoad model1.data\n\t•\tUse appropriate CG force fields (e.g. lj/cut, bond_style harmonic)\n\t•\tPerform energy minimization and relaxation with NVT ensemble at T = 1K\n\t•\tApply uniaxial tension with grips: define ~1 nm grip regions at the top and bottom fiber ends; hold the bottom grip fixed and displace the top grip along +z at a constant engineering strain rate (do not use fix deform).\n\t•\tEnable output of thermo data, trajectory dumps, and per-atom stress and displacement data.\n\nPostprocessing:\n\t•\tCompute the engineering stress-strain curve using the LAMMPS outputs:\n\t•\tUse the zz-component of atomic stress and z-displacement or box length.\n\t•\tOutput stress vs strain data as plain text.\n\t•\tGenerate and save a plot of the stress-strain curve as a PNG.\n\nOutput Requirements:\n\t•\tmodel1.data – Coarse-grained model of the twisted CNT bundle.\n\t•\trun_tension.in – LAMMPS input script with tension simulation and data dumps.\n\t•\tstress.dat and strain.dat – Text files of computed stress and strain.\n\t•\tstress_strain_curve.png – Plot of the tensile response of the fiber.\n\t•\tdump.tension.lammpstrj – Full trajectory of the tension simulation for visualization.\n\nNotes:\n\t•\tUse appropriate coarse-grained parameters for (5,5) CNTs (mass, bead spacing, LJ σ/ε, bond force constants).\n\t•\tDump and log the simulation process at regular intervals for analysis.",
        "image_dir": "examples/demos/demo_img/demo5",
        "stream": true
      }'
```

</details>

Example follow-up turn (same `session_dir`, newest message only):

```bash
curl -sS -N http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent",
        "session_dir": "session-atomagent-demo1",
        "messages": [{"role":"user","content":"Continue"}],
        "stream": true
      }'
```

### MD demo 2 — polycrystalline Fe under compression

Builds a bcc iron nanocrystal with 4–8 randomly oriented grains (Atomsk-style
packing), relaxes it with a published EAM potential, applies uniaxial
compression in LAMMPS, and outputs the stress–strain curve plus trajectory.

<details>
<summary>Kick-off request (click to expand)</summary>

```bash
curl -sS -N http://localhost:8002/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent",
        "session_dir": "session-atomagent-demo2",
        "input": "Generate a simple molecular dynamics (MD) simulation of polycrystalline bcc iron (Fe) under uniaxial compression using LAMMPS.\n\n\n\nModel Generation:\n\n    • Build a bcc iron nanocrystal with 4–8 randomly oriented grains, cubic box ~12 nm per side.\n\n    • Use Atomsk (or similar tool) to generate and pack grains; export as LAMMPS data file (model_Fe.data).\n\n    • Apply periodic boundary conditions in all three directions.\n\n\n\nSimulation Setup:\n\n    • Write a LAMMPS input script:\n\n        • Load model_Fe.data\n\n        • Use an appropriate EAM interatomic potential for Fe (e.g., Fe_Mendelev.eam.fs)\n\n        • Minimize energy and relax structure at T = 10K (NVT ensemble)\n\n        • Apply uniaxial compression along the z-axis using fix deform with a small negative engineering strain rate.\n\n        • Output thermo data, trajectory dumps, and per-atom stress for postprocessing.\n\n\n\nPostprocessing:\n\n    • Compute engineering stress-strain curve from LAMMPS global stress (zz-component) and box length.\n\n    • Output stress.dat and strain.dat as plain text files.\n\n    • Generate a plot (stress_strain_Fe.png) of the stress-strain curve.\n\n    • Save the full trajectory as dump.compression.lammpstrj for visualization.\n\n\n\nOutput Requirements:\n\n    • model_Fe.data – Polycrystalline Fe model (coarse nanocrystal)\n\n    • run_compress.in – LAMMPS input script\n\n    • stress.dat, strain.dat – Computed stress/strain\n\n    • stress_strain_Fe.png – Stress-strain plot\n\n    • dump.compression.lammpstrj – Trajectory dump\n\n\n\nNotes:\n\n    • Use published EAM Fe potential, e.g., Mendelev.\n\n    • Log/dump data at regular simulation intervals for analysis.\n\n",
        "stream": true
      }'
```

</details>

Follow-up turn used to resume once the (long) MD run has finished:

```bash
curl -sS -N http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "atomagent",
        "session_dir": "session-atomagent-demo2",
        "messages": [{"role":"user","content":"Simulation is finished, continue to do the post analysis"}],
        "stream": true
      }'
```

### FEM demo — plate with a circular hole (FEniCSx)

A fully reproducible FEniCSx project for 2D linear elasticity of a plate with
a centered circular hole under uniaxial tension: Gmsh meshing with physical
tags, mesh-convergence study, plane stress vs plane strain, validation
against the Kirsch analytic solution, and publication-quality figures/CSVs.
Uses the `fenicsagent` model id (FEniCSx-flavoured prompts).

<details>
<summary>Kick-off request (click to expand)</summary>

```bash
curl -sS -N http://localhost:8002/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "fenicsagent",
        "session_dir": "session-atomagent-fem_demo0",
        "input": "# AI Agent Execution Brief — FEniCSx Plate with Circular Hole (Short)\n\nOBJECTIVE\n- Build a fully reproducible FEniCSx project for 2D isotropic elasticity of a plate with a centered circular hole under uniaxial tension.\n- Support plane stress/plane strain and full/quarter models; run mesh convergence; validate vs Kirsch; output publication-quality figures/CSVs.\n\nGEOMETRY & LOADS\n- R=0.05 m; W=6R=0.30 m; H=4R=0.20 m. Center at (0,0).\n- Variants: full plate ([-W/2,W/2]×[-H/2,H/2]) or quarter (x≥0,y≥0; default=quarter).\n- Right edge traction T = σ∞ = 100 MPa (uniform, +x). Hole: traction-free. θ from +x.\n\nMATERIAL\n- E=210 GPa, ν=0.30.\n- Plane strain: λ=Eν/((1+ν)(1−2ν)), μ=E/(2(1+ν)).\n- Plane stress:  λ=Eν/(1−ν²),            μ=E/(2(1+ν)).\n\nBCs\n- Full: left u_x=0; pin (-W/2,0) with u_y=0; right traction T; hole free.\n- Quarter (default): x=0 → u_x=0; y=0 → u_y=0; right traction T; top & hole free.\n\nMESH (Gmsh 2D)\n- Physical tags: 1=Left, 2=Right, 3=Bottom, 4=Top, 5=Hole, 6=Domain.\n- Hole resolution ≥128 segments. Size field: h_min=R/20, h_max=R/5.\n- Convergence h_min ∈ {R/30, R/20, R/15, R/10}.\n\nNUMERICS\n- Elements: vector CG2 (default), CG1 optional. Quadrature deg=4.\n- PETSc: CG (or GMRES), PC=GAMG (fallback ILU), rtol=1e-8, max_it=1000.\n- Post: ε, σ, σ_vm; project stresses to DG0/CG1 as needed.\n\nVALIDATION (Kirsch, large-domain case)\n- Kirsch is the infinite-plate solution; validate the solver on an auxiliary large-domain case W=H=40R (same R, material, load; quarter model).\n- On r=R: σ_θθ(R,θ)=σ∞(1−2cos2θ). Expect σ_θθ(R,90°)=3σ∞ and σ_θθ(R,0°)=−σ∞.\n- Sample θ ∈ {0°,45°,90°}; compute FEM vs Kirsch, relative error, and Kt=σ_θθ^max/σ∞.\n\nCONVERGENCE & ACCEPTANCE\n- Spec geometry (W=6R, H=4R): track σ_θθ(R,θ) and Kt vs h_min; report the converged Kt. Finite-width effects make Kt exceed the infinite-plate value 3 here — do NOT gate on Kt≈3 for this geometry. Accept if Kt changes ≤1% between the two finest meshes.\n- Large-domain case (W=H=40R, finest mesh): accept if |Kt−3|≤2% and rel_err(90°)≤3%.\n\nOUTPUTS\n- CSVs per mesh: θ, σ_θθ^FEM, σ_θθ^Kirsch, rel_err, Kt, h_min, ndofs, wall_time.\n- Figures (PNG+PDF, 300 dpi): σ_vm field (+ mesh), Kt vs h_min (log-x), θ-point comparison.\n- XDMF/H5 fields; README; metadata.json (package versions & run params).\n\nFOLDER STRUCTURE\nfenicsx_plate_with_hole/\n├─ src/{make_mesh.py, solve_elasticity.py, run_convergence.py, compare_ps_pe.py, make_figures.py}\n├─ outputs/{run_quarter_ps, conv_quarter_ps, compare_ps_pe, conv_kirsch_large}\n└─ README.md\n\nSCRIPT CONTRACTS\n- make_mesh.py: build geometry, tags (1–6), graded size; write .msh and .xdmf.\n- solve_elasticity.py: apply BCs, assemble/solve; save u, ε, σ, σ_vm; reactions & avg traction.\n- run_convergence.py: loop h_min; run quarter+plane_stress; sample θ points; compute Kirsch/rel_err/Kt; append CSV.\n- compare_ps_pe.py: same mesh, PS vs PE; compare stresses/displacements.\n- make_figures.py: generate all figures (consistent style).\n\nDEFAULTS\n- variant=quarter, model=plane_stress, element=P2,\n- R=0.05 m, T=100 MPa, h_min set above, h_max=R/5, segments_hole=128,\n- θ sampling tolerance ±0.25°.\n\nEXECUTE NOW (no prompts)\n1) Build meshes for all h_min. 2) Solve plane_stress on each; store fields.\n3) Sample σ_θθ at θ=0/45/90; compute Kirsch, rel_err, Kt; write CSV.\n4) Make figures (σ_vm, Kt vs h_min, θ comparison).\n5) Repeat on finest mesh for plane_strain; run PS vs PE comparison & plot.\n6) Run the large-domain Kirsch validation case (W=H=40R) and record Kt and rel_err.\n7) Write README and metadata.json.\nReturn all files under fenicsx_plate_with_hole/.\n",
        "stream": true
      }'
```

</details>

Example follow-up turn:

```bash
curl -sS -N http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "fenicsagent",
        "session_dir": "session-atomagent-fem_demo0",
        "messages": [{"role":"user","content":"Model have been built, Compose the simulation script and run the simulation and post analysis"}],
        "stream": true
      }'
```

### Notebook demos (interactive, no backend)

`examples/notebooks/` holds two Jupyter walkthroughs that build the
Supervisor → Modeler → Simulator graph **by hand, one architectural layer per
cell** — services, MCP tools, session + RAG memory, agents, graph wiring
(with rendered graph diagrams) — and then stream a real demo run node-by-node:

- `demo_atomagent.ipynb` — MD path, default LAMMPS templates. `RUN_MODE`
  cell selects the full benchmark (`DEMO_CHOICE = "cnt"` for MD demo 1 with
  its reference image, `"fe"` for MD demo 2) or a minutes-scale `quick`
  smoke task.
- `demo_fenicsx.ipynb` — FEM path, `templates/fenicsx` prompts, running the
  FEM demo prompt (or its own `quick` task).

Both notebooks run on **either LLM backend**: the `PROVIDER` switch in the
first code cell selects `"openai"` (model from `ATOMAGENTS_OPENAI_MODEL`,
e.g. `gpt-5.6-sol`) or `"anthropic"` (model from `ATOMAGENTS_ANTHROPIC_MODEL`,
e.g. `claude-opus-5`). Exporting `ATOMAGENTS_LLM_PROVIDER` before launching
Jupyter overrides the in-notebook value, so headless/CI runs switch backends
without editing the notebook. `OPENAI_API_KEY` is required either way (the
RAG memory's embedding client uses it); the anthropic backend additionally
needs `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`).

Prerequisites: `make mcp` running and `.env` filled in (the notebooks load it
themselves — no shell sourcing needed). Sessions land in
`runs/jobs/session-nb-{fem,md}-<timestamp>/`, inside the MCP jail. Launch with
`./venv/bin/jupyter lab examples/notebooks/`.

### Verifying demo outputs

Every `lammps_run` tool result already carries a physics validation verdict
the Simulator must quote; the same gates run standalone:

```bash
./venv/bin/python -m AtomAgents.utils.run_validation --lammps-log <log> --script <in>
./venv/bin/python -m AtomAgents.utils.run_validation --fem-log <solver.log>
```

Agent-vs-reference overlay plots for the MD demos live in
`analysis/reference_comparison/` (`compare_demo1.py`, `compare_demo2.py`,
`compare_rect.py`); each writes an overlay figure plus an
`overlay_*_metrics.md` carrying the verdict.
