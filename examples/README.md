# examples/ — usage samples

| Path | Contents |
|---|---|
| `openai_client_example.py` | Minimal OpenAI-client call against the backend (`/v1`). |
| `notebooks/demo_atomagent.ipynb` | MD walkthrough: builds the Supervisor → Modeler → Simulator graph by hand, one layer per cell, then streams a real run (MD demo 1 or 2, or a minutes-scale `quick` task). |
| `notebooks/demo_fenicsx.ipynb` | Same walkthrough for the FEM path (`templates/fenicsx` prompts, the plate-with-hole demo). |
| `demos/demo_simple_mcp_agent.py`, `demos/demo_langchain_mcp.py` | Smallest possible agent ↔ MCP tool-service round trips. |
| `demos/okr_demo.py`, `demos/sqlite_memory_demo.py` | The OKR board tools and the SQLite/RAG memory used standalone. |
| `demos/demo_img/` | Reference figures the README's MD demo 1 request can attach via `image_dir`. |

Both notebooks have a `PROVIDER` switch in their first code cell (`openai`, model
from `ATOMAGENTS_OPENAI_MODEL`; or `anthropic`, model from
`ATOMAGENTS_ANTHROPIC_MODEL`); `ATOMAGENTS_LLM_PROVIDER` exported before launching
Jupyter overrides it for headless runs. They load `.env` themselves. Sessions land
in `runs/jobs/session-nb-{md,fem}-<timestamp>/`, inside the MCP jail.

The three benchmark briefs (CNT tension, Fe compression, FEM plate-with-hole)
are the curl examples in the root `README.md` → *Demos*; `benchmarks/briefs/`
holds their hash-locked copies. Sessions created by any example land in
`runs/jobs/`.
