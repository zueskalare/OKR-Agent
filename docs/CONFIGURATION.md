# Configuration reference

Every setting is an environment variable. Source of truth: `AtomAgents/config.py`
(framework), `backend/openai_backend.py` (`AgentConfig`), `AtomAgents/mcp_service/tools.py`
(tool service), `benchmarks/runner.py` and `benchmarks/arms/*` (harness).

**Who reads what.** The **backend** and the **baseline arms** take the LLM
endpoint, model and embedding settings; the **MCP tool service** takes only the
sandbox/LAMMPS/Tavily block and never reads `.env`; the **runner** parses `.env`
itself, adds the per-arm pins, and passes the result to the backend it starts
and to the arm subprocesses. `make backend` by hand needs `set -a; . ./.env; set +a`.

## LLM endpoints and models

| Variable | Default | Read by | Notes |
|---|---|---|---|
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | SDK defaults | backend, arms | OpenAI route; any OpenAI-compatible endpoint. Presence of either registers the `openai` provider. |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) | SDK defaults | backend, arms | Anthropic route (`/v1/messages`). |
| `ATOMAGENTS_OPENAI_MODEL` | `gpt-5` (runner pins `gpt-5.6-sol`) | backend | model for the `openai` provider |
| `ATOMAGENTS_ANTHROPIC_MODEL` (or `ANTHROPIC_MODEL`) | `claude-sonnet-5` (runner pins `claude-opus-5`) | backend | model for the `anthropic` provider |
| `ATOMAGENTS_LLM_PROVIDER` | first available of openai → anthropic → local | backend | provider for a bare model id (`atomagent`); `@provider` suffix overrides per request |
| `ATOMAGENTS_REASONING_EFFORT` | backend `minimal` (runner pins `high`) | backend, ReAct arm | OpenAI reasoning effort; Anthropic extended thinking budget mapped by the factory |
| `ATOMAGENTS_MAX_TOKENS` | `40960` | backend | max output tokens per call |
| `ATOMAGENTS_TEMPERATURE` | `1.0` | backend | OpenAI route only; omitted on the Anthropic route when thinking is enabled |
| `ATOMAGENTS_OPENAI_USE_RESPONSES` | `1` | backend | use the Responses API on the OpenAI route (the ReAct arm sets `0`: it uses chat completions) |
| `ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI` | `0` | backend | a Claude id on the openai provider is refused unless `1` |
| `ATOMAGENTS_ANTHROPIC_PROMPT_CACHE` | `1` | backend | cache breakpoint on the first system block |
| `ATOMAGENTS_LLM_RPS` | unset | backend | optional client-side rate limit (the study used none) |
| `ATOMAGENTS_LOCAL_LLM_BASE_URL`, `ATOMAGENTS_LOCAL_MODEL`, `ATOMAGENTS_LOCAL_API_KEY` | `http://localhost:11434/v1/`, qwen, `local` | backend | keyless local servers (`atomagent@local`) |
| `ATOMAGENTS_PROVIDER_<NAME>_BASE_URL` (+ `_API_TYPE`, `_API_KEY_ENV`, `_MODEL`) | — | backend | additional named providers |
| `BENCH_PROVIDER`, `BENCH_MODEL` | `anthropic`, `claude-opus-5` / `gpt-5.6-sol` | ReAct and CrewAI arms | route and model of a baseline arm; the runner sets them per arm |

## Embeddings (semantic recall, OKR-Agent arms)

| Variable | Default | Notes |
|---|---|---|
| `ATOMAGENTS_EMBED_PROVIDER` | `local` (runner pins `remote`) | `remote` = OpenAI-compatible `/v1/embeddings`, never falls back silently; `local` = sentence-transformers; `auto` = probe remote then local; `none` = lexical FTS5 only |
| `ATOMAGENTS_EMBED_BASE_URL`, `ATOMAGENTS_EMBED_API_KEY` | inherit `OPENAI_BASE_URL` / `OPENAI_API_KEY` | **set explicitly** whenever the chat endpoint is a gateway without embeddings |
| `ATOMAGENTS_EMBED_MODEL` | `text-embedding-3-large` | 3,072 dimensions in the study |
| `ATOMAGENTS_EMBED_LOCAL_MODEL`, `ATOMAGENTS_EMBED_DEVICE` | `sentence-transformers/all-MiniLM-L6-v2`, auto | local backend only |
| `ATOMAGENTS_RAG_SNIPPET_CHARS` | `1200` | per-snippet truncation of retrieved rows |

## Agent behaviour (backend)

| Variable | Default | Notes |
|---|---|---|
| `ATOMAGENTS_SUPERVISOR_EXEC_TOOLS` | `0` | `0`: the Supervisor's act stage binds the 8 coordination tools only; `1`: all 17. Runner: `0` for `gpt`, `1` for `opus`. |
| `ATOMAGENTS_ENABLE_TRIAGE` | `1` | first-hop triage of trivial / under-specified requests |
| `ATOMAGENTS_STAGE_GUARD` | `1` | strips simulated tool-call markup and invented results from prose stages |
| `ATOMAGENTS_ENABLE_HITL` | `1` (runner pins `0`) | agents may pause for human input |
| `ATOMAGENTS_MAX_ACT_CYCLES` | `15` | act cycles per agent invocation |
| `ATOMAGENTS_MAX_HANDOFFS` | `12` | inter-agent handoffs per user turn |
| `ATOMAGENTS_GRAPH_RECURSION_LIMIT` | `40` | LangGraph recursion limit |
| `ATOMAGENTS_MAX_INPUT_TOKENS` | `120000` | declared input budget |
| `ATOMAGENTS_TOOL_CLEAR_TRIGGER`, `ATOMAGENTS_TOOL_CLEAR_KEEP` | `30000`, `3` | clear old tool results past the trigger, keep the newest N |
| `ATOMAGENTS_SUMMARIZE` | `0` | history summarisation (off: it changes prompt content) |
| `ATOMAGENTS_TOOL_OUTPUT_MAX` | `65536` | client-side cap on a tool result, 75 % head / 25 % tail (also used by the baseline arms) |
| `ATOMAGENTS_MCP_URL` | `http://localhost:9999/mcp/` | tool service address (the baselines default to the same host) |
| `ATOMAGENTS_MCP_TIMEOUT`, `ATOMAGENTS_MCP_SSE_READ_TIMEOUT` | `120`, `3600` | MCP client timeouts (s) |
| `ATOMAGENTS_JOBS_DIR` | `runs/jobs` | where relative `session_dir`s are created |
| `ATOMAGENTS_MAX_SESSIONS` | `32` | LRU cap on cached session graphs |
| `ATOMAGENTS_SESSION_LOCK_TIMEOUT`, `ATOMAGENTS_EXECUTOR_WORKERS`, `ATOMAGENTS_TOKEN_STREAM`, `ATOMAGENTS_DURABILITY`, `ATOMAGENTS_NODE_MAX_ATTEMPTS` | see `config.py` | backend lifecycle and streaming |
| `ATOMAGENTS_EMULATE_TOOLS` | unset | comma-separated tool names to replace with LLM-backed emulators (dry runs without solvers) |

## MCP tool service (`make mcp` shell)

| Variable | Default | Notes |
|---|---|---|
| `ATOMAGENTS_TOOL_ROOT` | **required** (`Makefile` sets `runs/jobs`) | write/exec jail root; the service refuses to start without it |
| `ATOMAGENTS_ALLOW_UNJAILED` | `0` | `1` starts without a jail — library/testing only |
| `ATOMAGENTS_SANDBOX_EXEC` | `auto` | macOS `sandbox-exec` confinement of child processes (`0` opts out; auto-off elsewhere) |
| `ATOMAGENTS_SANDBOX_EXTRA_WRITE` | unset | extra writable roots, colon-separated |
| `ATOMAGENTS_ALLOW_GIT` | `0` | permit `git` in agent shells |
| `ATOMAGENTS_BASH_TIMEOUT_MAX`, `ATOMAGENTS_CODE_TIMEOUT_MAX`, `ATOMAGENTS_LAMMPS_TIMEOUT` | `600`, `1800`, `3000` | server-side timeout ceilings (s) |
| `ATOMAGENTS_TOOL_BINARY_MAX` | 8 MiB | cap on binary reads |
| `ATOMAGENTS_LAMMPS_CMD` | auto (`mpirun -np $ATOMAGENTS_LAMMPS_NP lmp_mpi`, then `lmp`) | **pin it** — see `docs/INSTALL.md`, MPI hazard |
| `ATOMAGENTS_LAMMPS_NP` | `12` | ranks for the auto-detected launcher |
| `ATOMAGENTS_VAL_*` | see `AtomAgents/utils/run_validation.py` | thresholds of the physics validation verdict attached to `lammps_run` results |
| `TAVILY_API_KEY` | unset | `web_search`, `lammps_doc_query`, `fenicsx_doc_query` |

Child processes receive a scrubbed environment (`PATH`, `LANG`, `LC_ALL`, `TERM`
only; `HOME`/`TMPDIR` redirected into the session), so none of the keys above
reach agent-authored code.

## Benchmark runner

| Variable | Default | Notes |
|---|---|---|
| `OKR_PYTHON` | `./venv/bin/python`, else the running interpreter | backend + ReAct arm interpreter |
| `OKR_CREWAI_PYTHON` | `benchmarks/crewai-env/.venv/bin/python` | CrewAI arm interpreter |
| `OKR_SKIP_MCP_PREFLIGHT` | unset | `1` skips the tool-surface preflight (never for a reported run) |
| `OKR_RUNS_ROOT` | `../../runs` relative to the script | `analysis/` scripts and notebooks: where the session tree lives |
| `CREWAI_TRACING_ENABLED` | runner sets `false` | prevents CrewAI's interactive tracing prompt |

The runner also pins, per arm: `ATOMAGENTS_LLM_PROVIDER`,
`ATOMAGENTS_SUPERVISOR_EXEC_TOOLS`, `ATOMAGENTS_ENABLE_TRIAGE=1`,
`ATOMAGENTS_STAGE_GUARD=1`, `ATOMAGENTS_ENABLE_HITL=0`,
`ATOMAGENTS_REASONING_EFFORT=high`, `ATOMAGENTS_EMBED_PROVIDER=remote`, and the
model identifiers (`PAPER_MODELS` unless `.env` sets them). These appear in
`manifest["env"]` / `manifest["models"]`.
