# runs/ — runtime data (gitignored)

Everything in this directory is generated at runtime and never committed
(only this README is tracked).

| Path | Contents |
|---|---|
| `jobs/` | Live agent sessions, one directory per `session_dir`. Created by the backend (location overridable via `ATOMAGENTS_JOBS_DIR`) and used as the write/exec jail root of the MCP tool service (`make mcp`). |
| `logs/` | Benchmark-runner logs: `<run_id>.backend.log`, `<run_id>.turn<N>.sse` (raw streamed frames incl. token usage), `<run_id>.arm.log` for baseline arms. |
| `archive/superseded-cells/` | A benchmark cell relaunched into an existing session directory moves the previous contents here first — never deleted, never inherited. |

## What a session directory holds

| File | Written by | Notes |
|---|---|---|
| `memory.db` | framework (`SQLiteRAGMemory`) / baseline `OKRStore` | messages (+ embeddings, FTS5 index), OKR board tables, artefact registry with SHA-256, metrics, attempts |
| `graph_state.db` | framework (LangGraph `AsyncSqliteSaver`) | per-session checkpoint of the conversation graph — **contains the raw request/response payloads, i.e. everything the model saw, which can include values of environment variables an agent printed. Treat it as sensitive; the published dataset omits these files for that reason.** |
| `README.md` | framework (`readme_sync`) | rendered OKR board (Tasks / Artefacts sections) |
| `manifest.json` | `benchmarks.runner` | run provenance: brief hash, prompt hashes, tool-lock digest, env pins, toolchain versions, per-turn stats |
| `transcript.jsonl`, `arm_result.json` | baseline arms | normalised transcript and result summary |
| everything else | the agents | models, input decks, logs, plots — the deliverables |

The thirty sessions reported in the paper, in exactly this layout, are published
at <https://huggingface.co/datasets/BioDMX-UGA/OKRAgent_runs> (`jobs/<arm>/<task>-t<n>/session-…/`).
