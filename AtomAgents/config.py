"""
Central configuration for AtomAgents.

Every value is overridable via an ``ATOMAGENTS_*`` environment variable so that
backends, agents, tools, and demo scripts share one source of truth instead of
scattered hardcoded literals.
"""

from __future__ import annotations

import os
from datetime import timedelta


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# --- LLM ---------------------------------------------------------------
LLM_MODEL = _env("ATOMAGENTS_MODEL", "gpt-5")
LLM_BASE_URL = _env("ATOMAGENTS_LLM_BASE_URL", "")  # empty = provider default
LLM_TEMPERATURE = _env_float("ATOMAGENTS_TEMPERATURE", 1.0)
LLM_MAX_TOKENS = _env_int("ATOMAGENTS_MAX_TOKENS", 16000)
LLM_REASONING_EFFORT = _env("ATOMAGENTS_REASONING_EFFORT", "medium")

LOCAL_LLM_MODEL = _env("ATOMAGENTS_LOCAL_MODEL", "qwen2.5-vl-72b-instruct")
LOCAL_LLM_BASE_URL = _env("ATOMAGENTS_LOCAL_LLM_BASE_URL", "http://localhost:11434/v1/")

# --- Embeddings (semantic memory recall) -------------------------------
# Configured independently of the chat LLM ON PURPOSE. Sharing OPENAI_BASE_URL
# is what broke recall: behind a chat-only OpenAI-compatible gateway that URL serves chat
# completions but no /v1/embeddings, so a bare OpenAIEmbeddings() inherited it
# and every recall 404'd — silently, because the embedder was constructed but
# never called. A whole benchmark campaign ran with recall off before anyone
# noticed. See AtomAgents/utils/embeddings.build_embeddings, which probes.
#
# Which backend recall uses:
#   local  — sentence-transformers on-device; no network after first download
#   remote — OpenAI-compatible endpoint; a dead one is a LOUD error, never a
#            silent fallback (use this when you want to know it broke)
#   auto   — probe remote, fall back to local
#   none   — disable semantic recall deliberately (FTS/keyword recall remains)
EMBED_PROVIDER = _env("ATOMAGENTS_EMBED_PROVIDER", "local").strip().lower()

# Remote backend (provider = remote | auto)
EMBED_MODEL = _env("ATOMAGENTS_EMBED_MODEL", "text-embedding-3-large")
# Empty means "inherit OPENAI_BASE_URL", which is the trap described above —
# set this explicitly whenever OPENAI_BASE_URL points at a proxy.
EMBED_BASE_URL = _env("ATOMAGENTS_EMBED_BASE_URL", "")
# Empty means "inherit OPENAI_API_KEY". Set explicitly when the embedding
# endpoint and the chat endpoint take different credentials — which is the
# normal case once a gateway is in the picture.
EMBED_API_KEY = _env("ATOMAGENTS_EMBED_API_KEY", "")

# Local backend (provider = local | auto fallback)
EMBED_LOCAL_MODEL = _env(
    "ATOMAGENTS_EMBED_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# Empty lets sentence-transformers pick (mps on Apple silicon, else cuda/cpu).
EMBED_DEVICE = _env("ATOMAGENTS_EMBED_DEVICE", "")

# --- Service ports -----------------------------------------------------
BACKEND_PORT = _env_int("ATOMAGENTS_BACKEND_PORT", 8002)
FRONTEND_PORT = _env_int("ATOMAGENTS_FRONTEND_PORT", 7862)

# --- Workspace layout ---------------------------------------------------
# All runtime session data lives under runs/ (gitignored); JOBS_DIR is the
# single source of truth for where sessions are created/resolved.
JOBS_DIR = _env("ATOMAGENTS_JOBS_DIR", "runs/jobs")

# --- MCP tool service --------------------------------------------------
MCP_URL = _env("ATOMAGENTS_MCP_URL", "http://localhost:9999/mcp/")
# HTTP timeout for individual MCP operations.
MCP_TIMEOUT_SEC = _env_float("ATOMAGENTS_MCP_TIMEOUT", 120.0)
# How long the client waits for the next SSE event. Must exceed the longest
# tool execution budget (LAMMPS runs are allowed LAMMPS_TIMEOUT_SEC below).
MCP_SSE_READ_TIMEOUT_SEC = _env_float("ATOMAGENTS_MCP_SSE_READ_TIMEOUT", 3600.0)

# --- Simulation tooling ------------------------------------------------
# Full launcher command, e.g. "mpirun -np 12 lmp_mpi". Empty string = auto-detect
# (mpirun+lmp_mpi if both are on PATH, else serial lmp).
LAMMPS_CMD = _env("ATOMAGENTS_LAMMPS_CMD", "")
LAMMPS_NPROC = _env_int("ATOMAGENTS_LAMMPS_NP", 12)
LAMMPS_TIMEOUT_SEC = _env_int("ATOMAGENTS_LAMMPS_TIMEOUT", 3000)

# Cap applied to every tool result returned to an LLM.
TOOL_OUTPUT_MAX_BYTES = _env_int("ATOMAGENTS_TOOL_OUTPUT_MAX", 64 * 1024)

# --- Agent loop budgets --------------------------------------------------
# NOTE: consumers must read these via module-attribute access
# (`config.MAX_ACT_CYCLES`), never `from .config import MAX_ACT_CYCLES` —
# value imports freeze the number at import time and break monkeypatching
# in tests.
#
# Per-run act cycles for one agent before tools are withheld and the run is
# forced to report.
MAX_ACT_CYCLES = _env_int("ATOMAGENTS_MAX_ACT_CYCLES", 15)
# Completed agent runs (report hand-offs) allowed for ONE user turn in the
# outer Supervisor/Modeler/Simulator graph before the turn is forced to END.
MAX_HANDOFFS = _env_int("ATOMAGENTS_MAX_HANDOFFS", 12)
# Outer-graph superstep ceiling — last-resort backstop behind MAX_HANDOFFS.
GRAPH_RECURSION_LIMIT = _env_int("ATOMAGENTS_GRAPH_RECURSION_LIMIT", 40)
# Supervisor triage gate on the first hop of each user turn: trivial or
# under-specified messages get a direct reply + END instead of the pipeline.
ENABLE_TRIAGE = _env_bool("ATOMAGENTS_ENABLE_TRIAGE", True)
# Human-in-the-loop: agents may route to a Human node that interrupts the
# graph and waits for the user's next message.
ENABLE_HITL = _env_bool("ATOMAGENTS_ENABLE_HITL", True)

# --- Stage/tool honesty guards (WS25) ------------------------------------
# A7: the Supervisor's act stage binds NO execution/write tools by default —
# the delegation prompts alone did not hold on claude-opus-5 (5/5 runs
# single-agent) because the coordinator could simply do everything itself.
# 1 restores the legacy "Supervisor holds every tool" shape.
SUPERVISOR_EXEC_TOOLS = _env_bool("ATOMAGENTS_SUPERVISOR_EXEC_TOOLS", False)
# A6: strip simulated tool-call markup (and the invented "results" that
# follow it) from prose-stage output before it is persisted or read by later
# stages. 0 disables the scrubber — diagnostics only.
STAGE_GUARD = _env_bool("ATOMAGENTS_STAGE_GUARD", True)
# Routing a claude-* model through an OpenAI-compatible provider makes every
# structured call fail (fenced/prose JSON instead of tool-enforced schema);
# the factory refuses to build that client. 1 downgrades the raise to an
# error log for gateways that genuinely translate.
ALLOW_CLAUDE_VIA_OPENAI = _env_bool("ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI", False)

# --- Graph robustness ----------------------------------------------------
# Checkpoint durability for outer-graph runs: "sync" persists each superstep
# before the next starts (true crash-resume for multi-hour simulation
# sessions); "async" (langgraph default) has a small loss window; "exit"
# persists only at the end.
GRAPH_DURABILITY = _env("ATOMAGENTS_DURABILITY", "sync")
# Per-node retry attempts for outer agent nodes (transient proxy/network
# failures). Interrupts (HITL) are exempt from retries — verified.
NODE_MAX_ATTEMPTS = _env_int("ATOMAGENTS_NODE_MAX_ATTEMPTS", 2)

# Master switch for WS19 streaming: incremental message publishing from agent
# runs AND per-token republication. Off = WS18 wire behavior (whole-run
# batches) on the async core — the live-rollback lever.
TOKEN_STREAM = _env_bool("ATOMAGENTS_TOKEN_STREAM", True)
# Same-session runs are serialized (agents are per-session singletons with
# mutable run state). 0 = wait indefinitely; >0 = seconds to wait before the
# backend answers 409 "session busy".
SESSION_LOCK_TIMEOUT = _env_float("ATOMAGENTS_SESSION_LOCK_TIMEOUT", 0.0)
# Default-executor threads for the async backend (sync inner nodes +
# to_thread epilogues both draw on it; the stdlib default starves under
# concurrent sessions).
EXECUTOR_WORKERS = _env_int("ATOMAGENTS_EXECUTOR_WORKERS", 64)

# --- Context management (WS20.2) -----------------------------------------
# Declared context window for the (proxy-named) models — they are absent
# from langchain's bundled model-profile tables, and fraction-based
# summarization triggers RAISE without a profile. Passed to the model
# constructors as profile={"max_input_tokens": ...}.
MAX_INPUT_TOKENS = _env_int("ATOMAGENTS_MAX_INPUT_TOKENS", 120_000)
# Clear old tool results from prompts once the (approximate) token count
# exceeds the trigger; the N most recent tool results are kept intact.
TOOL_CLEAR_TRIGGER = _env_int("ATOMAGENTS_TOOL_CLEAR_TRIGGER", 30_000)
TOOL_CLEAR_KEEP = _env_int("ATOMAGENTS_TOOL_CLEAR_KEEP", 3)
# Summarization of long histories (prompt-content-changing — OFF by default;
# do not enable before comparative runs).
SUMMARIZE = _env_bool("ATOMAGENTS_SUMMARIZE", False)
# Per-snippet cap for RAG history injected into prompts.
RAG_SNIPPET_CHARS = _env_int("ATOMAGENTS_RAG_SNIPPET_CHARS", 1200)
# Max concurrently-cached SESSIONFUL graphs in the backend; the least
# recently used beyond this is closed (continuity survives — the on-disk
# checkpoint rebuilds the thread on the next request). 0 = unbounded.
MAX_SESSIONS = _env_int("ATOMAGENTS_MAX_SESSIONS", 32)
# Dry-run emulation: comma-separated MCP tool names to replace with
# LLM-backed emulators (e.g. "lammps_run,md_visualize"). Empty = all real.
# Board/OKR tools are never emulated (merged after emulation).
EMULATE_TOOLS = os.environ.get("ATOMAGENTS_EMULATE_TOOLS", "")

# --- LLM cost controls ---------------------------------------------------
# Process-wide requests/second across ALL agents' LLM calls; 0 = disabled.
# One shared InMemoryRateLimiter instance is attached to every model the
# factory builds, so multi-agent fan-out cannot stampede the provider.
LLM_RPS = _env_float("ATOMAGENTS_LLM_RPS", 0.0)
# Stamp cache_control on the first (large, static) system block of every
# Anthropic call — repeated agent calls in a session then hit the provider
# prompt cache instead of re-paying for the full prefix.
ANTHROPIC_PROMPT_CACHE = _env_bool("ATOMAGENTS_ANTHROPIC_PROMPT_CACHE", True)


# --- LLM provider registry ---------------------------------------------
# Named API backends, auto-detected from standard env conventions (the same
# names an OpenAI-/Anthropic-compatible gateway client exports) plus explicit ATOMAGENTS_PROVIDER_*
# definitions. A provider is "available" only when its credentials/endpoint
# are actually present in the environment.
#
# Selection: OpenAI-compatible requests use `model@provider` ids (e.g.
# "atomagent@anthropic"); ATOMAGENTS_LLM_PROVIDER sets the default.


def discover_providers() -> dict:
    """Return {name: provider_config} for every configured backend.

    provider_config keys: api_type ("openai"|"anthropic"), base_url (str|None),
    api_key (str|None), model (default model id str).
    Evaluated lazily (not at import) so env changes/tests take effect.
    """
    providers: dict = {}

    # openai: standard OpenAI SDK env (often pointed at a proxy's
    # /v1/chat/completions route). use_responses_api routes LLM calls through
    # POST /v1/responses instead of chat completions (langchain handles the
    # payload differences); default ON here, OFF for local/custom endpoints
    # that usually don't implement the Responses API.
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        providers["openai"] = {
            "api_type": "openai",
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": _env("ATOMAGENTS_OPENAI_MODEL", LLM_MODEL),
            "use_responses_api": _env_bool("ATOMAGENTS_OPENAI_USE_RESPONSES", True),
        }

    # anthropic: Claude-Code-style env (ANTHROPIC_AUTH_TOKEN preferred; the
    # proxy serves /v1/messages on ANTHROPIC_BASE_URL).
    anthropic_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if anthropic_key:
        providers["anthropic"] = {
            "api_type": "anthropic",
            "base_url": os.environ.get("ANTHROPIC_BASE_URL") or None,
            "api_key": anthropic_key,
            "model": _env(
                "ATOMAGENTS_ANTHROPIC_MODEL",
                os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            ),
        }

    # local: OpenAI-compatible server (Ollama/LM Studio style), keyless.
    # Ollama has no /v1/responses — Responses API stays off unless forced.
    providers["local"] = {
        "api_type": "openai",
        "base_url": LOCAL_LLM_BASE_URL,
        "api_key": os.environ.get("ATOMAGENTS_LOCAL_API_KEY", "local"),
        "model": LOCAL_LLM_MODEL,
        "use_responses_api": _env_bool("ATOMAGENTS_LOCAL_USE_RESPONSES", False),
    }

    # custom: ATOMAGENTS_PROVIDER_<NAME>_BASE_URL (+ _API_TYPE, _API_KEY_ENV,
    # _MODEL). The key is referenced by env-var NAME so secrets stay in env.
    prefix = "ATOMAGENTS_PROVIDER_"
    for var in os.environ:
        if var.startswith(prefix) and var.endswith("_BASE_URL"):
            name = var[len(prefix) : -len("_BASE_URL")].lower()
            up = var[: -len("_BASE_URL")]
            key_env = os.environ.get(f"{up}_API_KEY_ENV", "")
            providers[name] = {
                "api_type": os.environ.get(f"{up}_API_TYPE", "openai"),
                "base_url": os.environ[var],
                "api_key": os.environ.get(key_env) if key_env else None,
                "model": os.environ.get(f"{up}_MODEL", LLM_MODEL),
                "use_responses_api": _env_bool(f"{up}_USE_RESPONSES", False),
            }

    return providers


def default_provider() -> str:
    """The provider used when a model id carries no @provider suffix."""
    explicit = os.environ.get("ATOMAGENTS_LLM_PROVIDER")
    available = discover_providers()
    if explicit:
        if explicit in available:
            return explicit
        raise ValueError(
            f"ATOMAGENTS_LLM_PROVIDER={explicit!r} is not configured; "
            f"available: {sorted(available)}"
        )
    for candidate in ("openai", "anthropic", "local"):
        if candidate in available:
            return candidate
    return next(iter(available))


def mcp_server_config() -> dict:
    """Connection map for MultiServerMCPClient, with timeouts wired in."""
    return {
        "AtomAgent_Tools": {
            "url": MCP_URL,
            "transport": "streamable_http",
            "timeout": timedelta(seconds=MCP_TIMEOUT_SEC),
            "sse_read_timeout": timedelta(seconds=MCP_SSE_READ_TIMEOUT_SEC),
        }
    }


def make_mcp_client():
    """Build the shared MCP client. Import is deferred so this module stays
    importable in environments without langchain_mcp_adapters."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient(mcp_server_config())
