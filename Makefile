# Interpreter of the framework environment (conda env created from
# environment.yml, see docs/INSTALL.md). Override with OKR_PYTHON or PY=.
PY ?= $(or $(OKR_PYTHON),./venv/bin/python)
MCP_PORT ?= 9999
# The write/exec jail root for the MCP tool service. runs/jobs (not runs/)
# keeps runs/archive and runs/logs out of agent reach.
TOOL_ROOT ?= $(CURDIR)/runs/jobs

.PHONY: check test lint backend mcp mcp-check briefs-check crewai-env

# --- services ---------------------------------------------------------------

# Start this from a shell where the conda env is NOT activated when LAMMPS is a
# Homebrew (Open MPI) build — or pin ATOMAGENTS_LAMMPS_CMD. See docs/INSTALL.md.
mcp:
	@mkdir -p "$(TOOL_ROOT)"
	ATOMAGENTS_TOOL_ROOT="$(TOOL_ROOT)" \
	  $(PY) -m uvicorn AtomAgents.mcp_service.mcp_service:app \
	  --host 127.0.0.1 --port $(MCP_PORT)

backend:
	$(PY) -m backend.openai_backend --port 8002

# --- benchmark preflight ------------------------------------------------------

# Live check that the running MCP service exposes exactly the 17-tool surface
# pinned in benchmarks/tools.lock.json (the arms refuse to run otherwise).
mcp-check:
	$(PY) -m benchmarks.toolbind --url http://127.0.0.1:$(MCP_PORT)/mcp

# The README demo blocks are the canonical brief text; this verifies the
# hash-locked copies under benchmarks/briefs/ still match them byte for byte.
briefs-check:
	$(PY) -m benchmarks.extract_briefs --check

# Provision the isolated CrewAI baseline environment (needs `uv`).
crewai-env:
	uv venv --python 3.12.9 benchmarks/crewai-env/.venv
	uv sync --locked --project benchmarks/crewai-env

# --- quality ----------------------------------------------------------------

test:
	$(PY) -m pytest -q tests AtomAgents/tests

lint:
	$(PY) -m ruff check AtomAgents backend benchmarks tests || true

check: test lint briefs-check
