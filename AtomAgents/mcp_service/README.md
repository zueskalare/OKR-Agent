# AtomAgents MCP Service

FastMCP (Model Context Protocol) server exposing the AtomAgents tool surface
over streamable HTTP at `http://<host>:9999/mcp/`. There is **no REST API** —
clients connect via MCP (e.g. `langchain_mcp_adapters`; see
`AtomAgents/config.py: make_mcp_client()` for the configured client with
matching timeouts).

## Registered tools

| Tool | Purpose |
|---|---|
| `file_reader` | Read any text file (content-sniffed; head+tail windows with offset paging), PNG/JPG/PDF as multimodal parts, directory listings |
| `write_file` | Write file content verbatim (the safe replacement for shell heredocs) |
| `download_file` | Download with payload validation (rejects HTML-as-data) |
| `code_runner` | Run a Python script (scrubbed env, jailed `cwd`, seatbelt-confined writes on macOS); `cwd` is required |
| `bash_run` | Run a bash command (same containment); shell state does NOT persist between calls |
| `web_search` | Tavily search (requires `TAVILY_API_KEY`) |
| `lammps_run` | Run LAMMPS from script content; returns returncode, parsed ERROR/WARNING lines, log tail, and a physics `validation` verdict |
| `md_visualize` | OVITO render of a structure file → PNG (base64 part) |
| `fem_mesh_visualize` / `fem_poisson_solve_visualize` / `fem_elasticity_solve_visualize` | DOLFINx demo solves + renders |
| `fenicsx_doc_query` / `lammps_doc_query` | Tavily crawl of the FEniCSx / LAMMPS docs |

## Running locally

```bash
make mcp            # from the repo root — sets the jail to ./runs/jobs
```

or explicitly:

```bash
ATOMAGENTS_TOOL_ROOT="$PWD/runs/jobs" \
  python -m uvicorn AtomAgents.mcp_service.mcp_service:app \
  --host 127.0.0.1 --port 9999
```

**The service refuses to start without `ATOMAGENTS_TOOL_ROOT`** pointing at an
existing directory — it is the write/exec jail root for every agent-issued
file, script and shell command. Startup prints the active jail and the child
sandbox status.

Environment:
- `ATOMAGENTS_TOOL_ROOT` — **required**; write/exec jail root (and the
  writable boundary of the macOS child sandbox)
- `ATOMAGENTS_ALLOW_UNJAILED` — `1` starts without a jail (library/testing
  use only; never with a real model attached)
- `ATOMAGENTS_SANDBOX_EXEC` — `0` disables the macOS `sandbox-exec`
  confinement of child processes (auto-off on non-macOS)
- `ATOMAGENTS_SANDBOX_EXTRA_WRITE` — extra writable roots for the sandbox
  profile, colon-separated
- `ATOMAGENTS_ALLOW_GIT` — `1` permits `git` in agent shells (denied by default)
- `ATOMAGENTS_BASH_TIMEOUT_MAX` / `ATOMAGENTS_CODE_TIMEOUT_MAX` — server-side
  timeout ceilings (600s / 1800s)
- `ATOMAGENTS_LAMMPS_CMD` — LAMMPS launcher override (default: auto-detect
  `mpirun -np $ATOMAGENTS_LAMMPS_NP lmp_mpi`, falling back to serial `lmp`)
- `ATOMAGENTS_LAMMPS_TIMEOUT` — run budget ceiling in seconds (default 3000)
- `ATOMAGENTS_TOOL_OUTPUT_MAX` — per-result text cap in bytes (default 65536)
- `TAVILY_API_KEY` — for `web_search` and the doc-query tools
- FEM tools require `dolfinx`; `md_visualize` requires `ovito`

## Containment on Linux

The per-process confinement (`sandbox-exec`) exists on macOS only. On Linux
the service starts with a warning and the jail/denylist layers alone, so run
it inside a container or under a throwaway user with a bind-mounted
`ATOMAGENTS_TOOL_ROOT`, and bind it to `127.0.0.1` as `make mcp` does.
**Never expose this service on a public interface** — `code_runner`/`bash_run`
execute agent-authored code.
