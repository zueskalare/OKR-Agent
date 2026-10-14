#!/usr/bin/env python
"""
OpenAI v1 API Compatible Backend for AtomAgents

This module provides an OpenAI ChatCompletion API v1 compatible wrapper around
the AtomAgents multi-agent workflow. It supports:

- OpenAI ChatCompletion API v1 format
- Streaming and non-streaming responses
- Image uploads via base64 encoding
- Multiple message history
- Compatible with OpenAI client libraries

Usage:
    # Start the server
    python openai_backend.py

    # Use with OpenAI client
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

    response = client.chat.completions.create(
        model="atomagent",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True
    )
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re
import signal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from functools import partial

from AtomAgents import config as atom_config
from AtomAgents.agents.human_interput import human_node
from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory
from AtomAgents.utils.prompt_loader import PromptLoader
from AtomAgents.utils.routing import handoff_router
from AtomAgents.utils.text import message_text

import sys
from AtomAgents.utils.logger import configure_agent_logger


import os as _os


class AgentConfig:
    """LLM settings for the hosted models. Env-overridable (ATOMAGENTS_*)."""

    model: str = _os.environ.get("ATOMAGENTS_MODEL", "gpt-5")
    temperature: float = float(_os.environ.get("ATOMAGENTS_TEMPERATURE", 1.0))
    max_tokens: int = int(_os.environ.get("ATOMAGENTS_MAX_TOKENS", 40960))
    # "minimal" or "low" or "medium" or "high"
    reasoning_effort: str = _os.environ.get("ATOMAGENTS_REASONING_EFFORT", "minimal")


def _startup_banner() -> str:
    """One line naming the configuration this process will actually run with.

    AgentConfig.model is ATOMAGENTS_MODEL, a placeholder that defaults to
    "gpt-5" and is NOT the model any agent uses — the real one comes from the
    resolved provider. Printing it alone made the banner claim "gpt-5" during a
    verified claude-opus-5 run, and WS26's per-arm configuration had to be
    reverse-engineered from log signatures afterwards because nothing recorded
    it. Everything a run needs to be identified later goes on this line.
    """
    try:
        from AtomAgents import config as _c

        provider = _c.default_provider()
        model = (_c.discover_providers().get(provider) or {}).get("model", "?")
        flags = (
            f"exec_tools={int(bool(_c.SUPERVISOR_EXEC_TOOLS))} "
            f"triage={int(bool(_c.ENABLE_TRIAGE))} "
            f"stage_guard={int(bool(_c.STAGE_GUARD))} "
            f"hitl={int(bool(_c.ENABLE_HITL))} "
            f"embed={_c.EMBED_PROVIDER}"
        )
    except Exception as exc:  # noqa: BLE001 - a banner must never block startup
        provider, model, flags = "?", "?", f"(config unavailable: {exc})"
    return (
        f"[OpenAI Backend] LLM config: provider={provider} model={model} "
        f"temp={AgentConfig.temperature} max_tokens={AgentConfig.max_tokens} "
        f"effort={AgentConfig.reasoning_effort} {flags}"
    )


print(_startup_banner())

configure_agent_logger(verbose=False)

# Ensure project root is on the path
sys.path.append(str(Path(__file__).parent.parent))

# Self-contained initialisation (optional MCP client)
try:
    from AtomAgents.config import make_mcp_client

    client = make_mcp_client()
except Exception:
    # MCP is optional; continue without external tools
    client = None


def _initialize_components():
    try:
        from AtomAgents.agents.simulation_manager import Supervisor
        from AtomAgents.agents.model_building_agent import Modeler
        from AtomAgents.agents.simulator_agent import SimulatorAgent
        from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory
        from AtomAgents.utils.logger import configure_agent_logger
        from AtomAgents.workflows.state_definitions import State as GraphState
        from langgraph.graph import StateGraph, END
    except Exception:
        print("[OpenAI Backend] Failed to import AtomAgents components:")
        import traceback as _tb

        _tb.print_exc()
        return None

    configure_agent_logger(verbose=False)

    from AtomAgents.config import LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL

    llm_config = dict(
        model=LOCAL_LLM_MODEL,
        temperature=1.0,
        max_tokens=16000,
        base_url=LOCAL_LLM_BASE_URL,
    )

    # NOTE: no workspace or memory DB is created here. Doing so per call left
    # an orphaned <jobs>/backend-<timestamp>/memory.db on every graph build; the
    # memory is created in _get_or_build_graph once the real session directory
    # is known.
    return {
        "Supervisor": Supervisor,
        "Modeler": Modeler,
        "SimulatorAgent": SimulatorAgent,
        "GraphState": GraphState,
        "StateGraph": StateGraph,
        "END": END,
        "llm_config": llm_config,
        "SQLiteRAGMemory": SQLiteRAGMemory,
    }


def _assemble_graph(cfg, tools, checkpointer=None):
    # Optional per-agent prompt loaders in cfg["prompt_loaders"]
    pls = cfg.get("prompt_loaders") if isinstance(cfg, dict) else None

    # In-process OKR tools bound to this session's memory: register_artefact,
    # record_metric, record_attempt, okr_update. SQL = detail; README = view.
    try:
        from AtomAgents.utils.okr_tools import build_okr_tools

        okr_tools = build_okr_tools(
            cfg["memory"], str(cfg["project_root"].resolve())
        )
    except Exception as e:
        print(f"[OpenAI Backend] OKR tools unavailable: {e}")
        okr_tools = []
    tools = list(tools) + okr_tools

    # Real HITL: agents may route to a Human node that interrupts the graph
    # until the user's next message resumes it (durable sessions only).
    hitl = bool(atom_config.ENABLE_HITL)
    human = ["Human"] if hitl else []

    sup_kwargs = dict(
        llm_config=cfg["llm_config"],
        memory=cfg["memory"],
        Collages=["Modeler", "Simulator", "END"] + human,
        tools=tools,
        project_root=str(cfg["project_root"].resolve()),
        enable_okr_table=True,
    )
    if pls and pls.get("Supervisor"):
        sup_kwargs["prompt_loader"] = pls["Supervisor"]
    sup = cfg["Supervisor"](**sup_kwargs)

    mod_kwargs = dict(
        llm_config=cfg["llm_config"],
        memory=cfg["memory"],
        Collages=["Supervisor", "Simulator", "END"] + human,
        tools=tools,
        project_root=str(cfg["project_root"].resolve()),
    )
    if pls and pls.get("Modeler"):
        mod_kwargs["prompt_loader"] = pls["Modeler"]
    mod = cfg["Modeler"](**mod_kwargs)

    sim_kwargs = dict(
        llm_config=cfg["llm_config"],
        memory=cfg["memory"],
        Collages=["Supervisor", "Modeler", "END"] + human,
        tools=tools,
        config={"recursion_limit": 50},
        project_root=str(cfg["project_root"].resolve()),
    )
    if pls and pls.get("Simulator"):
        sim_kwargs["prompt_loader"] = pls["Simulator"]
    sim = cfg["SimulatorAgent"](**sim_kwargs)

    graph = cfg["StateGraph"](state_schema=cfg["GraphState"])
    # Node defaults (must precede add_node): bounded retry for transient
    # proxy/network failures + graceful END-with-error instead of a 500 when
    # retries exhaust. Interrupts (HITL) bypass both — verified.
    from langgraph.types import RetryPolicy

    from AtomAgents.utils.routing import agent_error_handler

    graph.set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=atom_config.NODE_MAX_ATTEMPTS),
        error_handler=agent_error_handler,
    )
    # arun is the real implementation; the production graph is driven with
    # astream, so nodes are the async callables.
    graph.add_node("Supervisor", sup.arun)
    graph.add_node("Modeler", mod.arun)
    graph.add_node("Simulator", sim.arun)
    if hitl:
        graph.add_node("Human", human_node)

    graph.set_entry_point("Supervisor")
    END = cfg["END"]
    human_edge = {"Human": "Human"} if hitl else {}
    # handoff_router = send_to lookup + per-turn handoff budget backstop
    # (config.MAX_HANDOFFS agent runs per user turn, then forced END).
    graph.add_conditional_edges(
        "Supervisor",
        partial(handoff_router, agent="Supervisor"),
        {"Modeler": "Modeler", "Simulator": "Simulator", "END": END, **human_edge},
    )
    graph.add_conditional_edges(
        "Modeler",
        partial(handoff_router, agent="Modeler"),
        {"Supervisor": "Supervisor", "Simulator": "Simulator", "END": END, **human_edge},
    )
    graph.add_conditional_edges(
        "Simulator",
        partial(handoff_router, agent="Simulator"),
        {"Supervisor": "Supervisor", "Modeler": "Modeler", "END": END, **human_edge},
    )
    if hitl:
        graph.add_conditional_edges(
            "Human",
            partial(handoff_router, agent="Human"),
            {
                "Supervisor": "Supervisor",
                "Modeler": "Modeler",
                "Simulator": "Simulator",
                "END": END,
            },
        )

    # Compile WITH a checkpointer: without one every request started the graph
    # from an empty state, which is what caused the Supervisor to re-bootstrap
    # and replay identical turns 8x per session.
    if checkpointer is None:
        checkpointer = _make_checkpointer(cfg["project_root"])
    return graph.compile(checkpointer=checkpointer)


async def _apending_interrupt(graph, run_config) -> bool:
    """True when the thread in ``run_config`` is paused on a langgraph
    interrupt (HITL waiting for the user's answer). Async: AsyncSqliteSaver
    raises InvalidStateError when its sync get_tuple is called from its own
    loop — aget_state is mandatory here."""
    try:
        snapshot = await graph.aget_state(run_config)
    except Exception:
        return False
    tasks = getattr(snapshot, "tasks", ()) or ()
    return any(getattr(t, "interrupts", None) for t in tasks)


def _make_checkpointer(project_root: Path):
    """Durable per-session graph state at <session>/graph_state.db, with an
    in-memory fallback (still fixes continuity within a backend process, since
    compiled graphs are cached per session key).

    The serde allowlist registers ReportMessage for msgpack: without it,
    langgraph's strict-msgpack mode (CVE-2026-28277 machinery, currently
    warn-only) will silently deserialize the subclass to a plain dict — no
    exception — breaking every marker/isinstance check downstream. NOTE an
    explicit allowlist puts THIS serializer in strict mode for everything
    else; core message types are on langgraph's built-in safe list, so only
    our own classes need registering here.
    """
    try:
        import sqlite3 as _sqlite3

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite import SqliteSaver

        from AtomAgents.agents.messages import ReportMessage

        conn = _sqlite3.connect(
            str(Path(project_root) / "graph_state.db"), check_same_thread=False
        )
        return SqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[ReportMessage])
        )
    except Exception as e:
        print(f"[OpenAI Backend] SqliteSaver unavailable ({e}); using InMemorySaver")
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


async def _make_checkpointer_async(project_root: Path):
    """Async twin of _make_checkpointer for the astream-driven production
    graph (sync SqliteSaver.aput raises NotImplementedError). MUST be called
    on the event loop — AsyncSqliteSaver.__init__ captures the running loop.
    Returns (saver, aiosqlite_connection); the caller owns closing the
    connection at shutdown (aiosqlite's worker thread is non-daemon and hangs
    the process otherwise — verified)."""
    try:
        import aiosqlite
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from AtomAgents.agents.messages import ReportMessage

        conn = await aiosqlite.connect(str(Path(project_root) / "graph_state.db"))
        saver = AsyncSqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[ReportMessage])
        )
        return saver, conn
    except Exception as e:
        print(
            f"[OpenAI Backend] AsyncSqliteSaver unavailable ({e}); using InMemorySaver"
        )
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), None


from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# ---------------------------- OpenAI API Models --------------------------


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="atomagent", description="Model to use")
    # OpenAI-compatible wire format; converted to BaseMessage internally
    messages: List[Dict[str, Any]] = Field(
        ..., description="List of messages (OpenAI format)"
    )
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=16000, gt=0)
    stream: Optional[bool] = Field(default=False)
    session_dir: Optional[str] = Field(
        None,
        description=(
            "Path to a session workspace directory. If relative, it is created/used under the jobs dir (runs/jobs by default)."
        ),
    )
    image_dir: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a directory containing image files to include with this request."
        ),
    )


class ChatHistoryRequest(BaseModel):
    session_dir: str = Field(..., description="Path to the session workspace directory")


class ChatHistoryResponse(BaseModel):
    messages: List[BaseMessage] = Field(..., description="List of chat messages")


class ChatCompletionChoice(BaseModel):
    index: int
    # OpenAI-compatible message payload for the wire format
    message: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "atomagents"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ------------------------ Responses API wire models ------------------------
# The primary surface (POST /v1/responses). Every emitted response object is
# kept parseable by openai.types.responses.Response — enforced by tests.


class ResponsesRequest(BaseModel):
    """POST /v1/responses request. Unknown OpenAI fields (tools, text,
    include, truncation, ...) are ignored rather than rejected (pydantic
    default extra="ignore")."""

    model: str = "atomagent"
    input: Union[str, List[Dict[str, Any]]] = ""
    instructions: Optional[str] = None
    stream: Optional[bool] = False
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None  # accepted, ignored (backend owns LLM config)
    store: Optional[bool] = None  # accepted, ignored; echoed as false
    previous_response_id: Optional[str] = None  # rejected with 400 — use session_dir
    metadata: Optional[Dict[str, str]] = None  # echoed back
    # AtomAgents extensions (same semantics as the chat surface)
    session_dir: Optional[str] = None
    image_dir: Optional[str] = None


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # cache_write_tokens is REQUIRED by openai>=2.53 client-side validation —
    # a bare {"cached_tokens": 0} fails Response.model_validate.
    input_tokens_details: Dict[str, int] = Field(
        default_factory=lambda: {"cached_tokens": 0, "cache_write_tokens": 0}
    )
    output_tokens_details: Dict[str, int] = Field(
        default_factory=lambda: {"reasoning_tokens": 0}
    )


class ResponseOutputText(BaseModel):
    type: str = "output_text"
    text: str = ""
    annotations: List[Any] = Field(default_factory=list)


class ResponseMessageItem(BaseModel):
    """One agent message as a Responses output item.

    ``metadata`` is an AtomAgents extension: SDK clients read it via
    ``item.model_extra["metadata"]`` — the chat-chunk surface could never
    expose atom_* fields through typed SDK models.
    """

    type: str = "message"
    id: str = ""
    status: str = "completed"
    role: str = "assistant"
    content: List[ResponseOutputText] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class ResponseObject(BaseModel):
    id: str
    object: str = "response"
    created_at: int = 0
    # completed | incomplete (HITL pause / cancelled) | failed | in_progress.
    status: str = "completed"
    model: str = "atomagent"
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    store: bool = False
    output: List[ResponseMessageItem] = Field(default_factory=list)
    output_text: str = ""
    # MUST stay None for the HITL pause: the SDK's incomplete_details.reason
    # is a closed enum (max_output_tokens|content_filter) — custom reasons
    # fail client-side validation. Ours ride in metadata.atom_status instead.
    incomplete_details: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    parallel_tool_calls: bool = False
    tool_choice: str = "auto"
    tools: List[Any] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    usage: ResponseUsage = Field(default_factory=ResponseUsage)


# ---------------------------- Helper Functions ----------------------------


def _extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> tuple[str, List[str]]:
    """Extract text content and base64 images from OpenAI message format"""
    text_parts = []
    images = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            # Handle multimodal content
            for content_part in content:
                if content_part.get("type") == "text":
                    text_parts.append(f"{role}: {content_part.get('text', '')}")
                elif content_part.get("type") == "image_url":
                    image_url = content_part.get("image_url", {})
                    url = image_url.get("url", "")
                    if url.startswith("data:image"):
                        # Extract base64 data
                        try:
                            header, data = url.split(",", 1)
                            images.append(data)
                        except ValueError:
                            continue

    combined_text = "\n".join(text_parts)
    return combined_text, images


def _extract_from_responses_input(
    input_value: Union[str, List[Dict[str, Any]], None],
    instructions: Optional[str] = None,
) -> tuple[str, List[str]]:
    """Extract ``(text, base64_images)`` from a Responses-API ``input``.

    Mirrors ``_extract_latest_user_message``'s philosophy: ``instructions``
    and in-band system/developer items are hoisted ahead of the text, only
    the NEWEST user message is used (conversation continuity lives in the
    session's checkpointed graph state, not the payload), and non-message
    items (function_call, reasoning, output refs) are ignored.
    """

    def _text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") in ("input_text", "text")
            ]
            return "\n".join(t for t in parts if t)
        return ""

    system_bits: List[str] = []
    if instructions:
        system_bits.append(str(instructions))

    if input_value is None or isinstance(input_value, str):
        text = input_value or ""
        return ("\n\n".join([*system_bits, text]) if system_bits else text), []

    for item in input_value:
        if isinstance(item, dict) and item.get("role") in ("system", "developer"):
            t = _text_of(item.get("content"))
            if t:
                system_bits.append(t)

    text_parts: List[str] = []
    images: List[str] = []
    for item in reversed(input_value):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("input_text", "text"):
                    t = part.get("text", "")
                    if t:
                        text_parts.append(t)
                elif ptype in ("input_image", "image_url"):
                    # Responses shape: image_url is a BARE data-URL string;
                    # chat shape: {"image_url": {"url": ...}} — accept both.
                    url = part.get("image_url", "")
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    if (
                        isinstance(url, str)
                        and url.startswith("data:")
                        and "," in url
                    ):
                        images.append(url.split(",", 1)[1])
        break  # newest user message only

    text = "\n".join(p for p in text_parts if p)
    if system_bits:
        text = "\n\n".join([*system_bits, text]) if text else "\n\n".join(system_bits)
    return text, images


def _extract_latest_user_message(
    messages: List[Dict[str, Any]],
) -> tuple[str, List[str]]:
    """Extract text + images from ONLY the newest user message.

    Conversation continuity comes from the graph checkpointer; flattening the
    client's whole cumulative history into one blob re-injected the original
    prompt (and any uploaded image) on every single turn.
    """
    # Client-supplied system messages are steering, not turns — hoist them
    # instead of dropping them (ultrareview merged_bug_001).
    system_bits = [
        m.get("content")
        for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    for message in reversed(messages):
        if message.get("role", "user") != "user":
            continue
        content = message.get("content")
        text_parts: List[str] = []
        images: List[str] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:image"):
                        try:
                            _, data = url.split(",", 1)
                            images.append(data)
                        except ValueError:
                            continue
        # No role prefix: the old flattener emitted "user: ..." as a turn
        # separator; on a single message it was pure prompt pollution.
        text = "\n".join(t for t in text_parts if t)
        if system_bits:
            text = "\n\n".join([*system_bits, text] if text else system_bits)
        return text, images
    return "", []


def _save_base64_image(base64_data: str, project_root: Path) -> Optional[str]:
    """Save base64 image data to file and return path (deduped by content)."""
    try:
        # Create uploads directory
        uploads_dir = project_root / "uploads"
        uploads_dir.mkdir(exist_ok=True)

        # Decode base64 data
        image_data = base64.b64decode(base64_data)

        # Content-addressed name: the same uploaded image arriving on every
        # turn maps to one file instead of a new copy per request.
        import hashlib

        digest = hashlib.sha256(image_data).hexdigest()[:16]
        file_path = uploads_dir / f"image_{digest}.png"
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(image_data)

        return str(file_path)
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return None


def _load_images_from_dir(image_dir: str) -> List[str]:
    """Read image files from a directory and return base64-encoded strings.

    Supports common image extensions: png, jpg, jpeg, gif, bmp, webp.
    """
    try:
        p = Path(image_dir)
        if not p.exists() or not p.is_dir():
            return []
        out: List[str] = []
        exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        for fp in sorted(p.iterdir()):
            if fp.is_file() and fp.suffix.lower() in exts:
                try:
                    raw = fp.read_bytes()
                    out.append(base64.b64encode(raw).decode("utf-8"))
                except Exception:
                    continue
        return out
    except Exception as e:
        print(f"Error reading image_dir {image_dir}: {e}")
        return []


# ---------------------------- FastAPI App ----------------------------------

app = FastAPI(
    title="AtomAgents OpenAI Compatible API",
    description="OpenAI v1 ChatCompletion API compatible interface for AtomAgents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
# Maintain separate compiled graphs and workspaces per exposed model id
_APP_GRAPHS: Dict[str, Any] = {}
# Cooperative cancellation: /v1/chat/cancel sets a flag; the stream loop
# honors it at the next agent (node) boundary — a running LAMMPS/tool call
# cannot be aborted mid-flight.
CANCEL_FLAGS: Dict[str, bool] = {}
_PROJECT_ROOTS: Dict[str, Path] = {}
_BUILD_LOCKS: Dict[str, asyncio.Lock] = {}
# aiosqlite connections behind AsyncSqliteSavers, closed at shutdown (their
# worker threads are non-daemon — the process hangs otherwise).
_CHECKPOINTER_CONNS: Dict[str, Any] = {}
# Per-session run locks: agents are per-session singletons with mutable run
# state (msg_storage, loop guards), so concurrent runs of ONE session must
# serialize. Different sessions stay fully concurrent. For anonymous
# requests the key is the bare model id — their runs share agent objects
# too, so serializing them is correct, not merely conservative.
_RUN_LOCKS: Dict[str, asyncio.Lock] = {}
# LRU order of SESSIONFUL graph keys (anonymous keys are excluded — their
# workspace is timestamp-minted per build, so caching them is the point).
# Oldest first; touched on every _get_or_build_graph hit.
_SESSION_LRU: "OrderedDict[str, bool]" = OrderedDict()
# SIGTERM drain: set once, never cleared. Streams end at the next node
# boundary with a resume hint; durability="sync" means the superstep is
# already checkpointed, so the next request resumes the thread.
_DRAINING = False


def _begin_drain() -> None:
    global _DRAINING
    if _DRAINING:
        return
    _DRAINING = True
    print(
        "[OpenAI Backend] SIGTERM — draining: streams end at the next node "
        "boundary (state checkpointed); handing off to uvicorn shutdown."
    )
    # Re-raise as SIGINT so uvicorn's own graceful stop machinery proceeds
    # (our handler replaced its SIGTERM one).
    signal.raise_signal(signal.SIGINT)


def _touch_session(graph_key: str, sessionful: bool) -> None:
    """Move a sessionful key to the most-recently-used end of the LRU."""
    if not sessionful:
        return
    _SESSION_LRU.pop(graph_key, None)
    _SESSION_LRU[graph_key] = True


async def _close_session(graph_key: str) -> None:
    """Close a session's aiosqlite connection and drop every registry entry.

    Continuity is unaffected: the on-disk checkpoint rebuilds the thread on
    the session's next request. Also used per-key by shutdown_event.
    """
    conn = _CHECKPOINTER_CONNS.pop(graph_key, None)
    if conn is not None:
        try:
            await conn.close()
        except Exception as e:
            print(f"[OpenAI Backend] checkpointer close failed for {graph_key}: {e}")
    _APP_GRAPHS.pop(graph_key, None)
    _PROJECT_ROOTS.pop(graph_key, None)
    _SESSION_LRU.pop(graph_key, None)
    _RUN_LOCKS.pop(graph_key, None)
    _BUILD_LOCKS.pop(graph_key, None)
    CANCEL_FLAGS.pop(graph_key, None)


async def _evict_lru_sessions() -> None:
    """Evict least-recently-used sessionful graphs beyond MAX_SESSIONS.

    Called OUTSIDE any build lock (one victim build lock at a time — no
    lock-ordering cycles). Keys whose run lock is held are skipped: a run
    is in flight and closing its checkpointer would fail it mid-superstep.
    """
    limit = int(getattr(atom_config, "MAX_SESSIONS", 0) or 0)
    if limit <= 0:
        return
    while len(_SESSION_LRU) > limit:
        victim = next(
            (
                k
                for k in _SESSION_LRU  # oldest first
                if not (k in _RUN_LOCKS and _RUN_LOCKS[k].locked())
            ),
            None,
        )
        if victim is None:
            return  # every candidate is mid-run; retry on the next build
        lock = _BUILD_LOCKS.setdefault(victim, asyncio.Lock())
        async with lock:
            if victim not in _SESSION_LRU:
                continue  # concurrently evicted
            run_lock = _RUN_LOCKS.get(victim)
            if run_lock is not None and run_lock.locked():
                continue  # started running meanwhile — pick another victim
            await _close_session(victim)
            print(f"[OpenAI Backend] Evicted LRU session graph: {victim}")


async def _acquire_run_lock(graph_key: str) -> asyncio.Lock:
    """Acquire the per-session run lock; 409 on configured timeout."""
    lock = _RUN_LOCKS.setdefault(graph_key, asyncio.Lock())
    timeout = atom_config.SESSION_LOCK_TIMEOUT
    if timeout and timeout > 0:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=409,
                detail="session busy: another run is active for this session_dir",
            )
    else:
        await lock.acquire()
    return lock


@app.on_event("startup")
async def startup_event():
    """Lightweight startup; graphs are built lazily on first request."""
    # Explicit default executor: sync inner nodes + to_thread epilogues both
    # draw on it, and the stdlib default (min(32, cpu+4)) starves under
    # concurrent sessions.
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=atom_config.EXECUTOR_WORKERS)
    )
    # Graceful drain: SIGTERM sets the flag (streams end with a resume hint)
    # then re-raises as SIGINT for uvicorn's own graceful stop.
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _begin_drain)
    except (NotImplementedError, RuntimeError, ValueError):
        pass  # non-main thread / platform without loop signal support
    print(
        "[OpenAI Backend] Startup complete. Graphs will be built on demand; "
        f"executor workers={atom_config.EXECUTOR_WORKERS}."
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Close every AsyncSqliteSaver connection: aiosqlite's worker thread is
    NON-daemon — without this, uvicorn hangs at exit (verified)."""
    for key in list(_CHECKPOINTER_CONNS):
        await _close_session(key)
    print(f"[OpenAI Backend] Shutdown: checkpointer connections closed.")


async def _get_or_build_graph(
    graph_key: str, model_id: str, session_dir: Optional[str]
) -> tuple[Any, Path]:
    """Return (graph, project_root) for the key, building it if missing.

    graph_key uniquely identifies a compiled graph. We use
    - model_id only when session_dir is None (shared model graph), or
    - model_id + session_dir when the caller wants a per-session workspace.
    """
    sessionful = session_dir is not None
    # Fast path
    if graph_key in _APP_GRAPHS and graph_key in _PROJECT_ROOTS:
        _touch_session(graph_key, sessionful)
        return _APP_GRAPHS[graph_key], _PROJECT_ROOTS[graph_key]

    # Ensure only one build per model at a time
    lock = _BUILD_LOCKS.setdefault(graph_key, asyncio.Lock())
    async with lock:
        if graph_key in _APP_GRAPHS and graph_key in _PROJECT_ROOTS:
            _touch_session(graph_key, sessionful)
            return _APP_GRAPHS[graph_key], _PROJECT_ROOTS[graph_key]

        if _initialize_components is None:
            raise RuntimeError("AtomAgents components not available")

        # Blocking construction (component init, sqlite schema, embeddings
        # client, prompt loaders, agent/LLM construction) runs OFF the loop —
        # inside this lock it used to stall every concurrent request.
        cfg, project_root = await asyncio.to_thread(
            _prepare_build_cfg, model_id, session_dir
        )

        # Fetch tools on demand
        tools = []
        if client:
            try:
                tools = await client.get_tools()
                print(f"[OpenAI Backend] Loaded {len(tools)} MCP tools (on demand)")
            except Exception as e:
                print(f"[OpenAI Backend] Failed to get MCP tools: {e}")

        # Dry-run emulation (ATOMAGENTS_EMULATE_TOOLS): swap listed MCP tools
        # for LLM-backed stand-ins. Applied here — before the okr_tools merge
        # in _assemble_graph — so board tools always stay real.
        emu_names = [
            n.strip() for n in atom_config.EMULATE_TOOLS.split(",") if n.strip()
        ]
        if emu_names and tools:
            try:
                from AtomAgents.llm_factory import make_llm_from_config
                from AtomAgents.utils.tool_emulation import emulate_tools

                emu_llm = await asyncio.to_thread(
                    make_llm_from_config, cfg["llm_config"]
                )
                tools = emulate_tools(tools, emu_names, emu_llm)
                print(f"[OpenAI Backend] DRY RUN — emulating tools: {emu_names}")
            except Exception as e:
                print(f"[OpenAI Backend] Tool emulation failed (tools real): {e}")
        # NOTE: no tool.return_direct — only honored by prebuilt ReAct agents;
        # it was inert in this hand-rolled graph.

        # Async checkpointer must be built ON the loop (loop-bound __init__);
        # its aiosqlite connection is registered for shutdown close — the
        # worker thread is non-daemon and hangs the process otherwise.
        checkpointer, conn = await _make_checkpointer_async(project_root)
        if conn is not None:
            _CHECKPOINTER_CONNS[graph_key] = conn
        graph = await asyncio.to_thread(_assemble_graph, cfg, tools, checkpointer)
        _APP_GRAPHS[graph_key] = graph
        _PROJECT_ROOTS[graph_key] = project_root
        print(f"[OpenAI Backend] Built graph for key: {graph_key}")
    # Outside the build lock: LRU bookkeeping + eviction (one victim build
    # lock at a time — never nested under this key's own build lock).
    _touch_session(graph_key, sessionful)
    await _evict_lru_sessions()
    return graph, project_root


def _prepare_build_cfg(model_id: str, session_dir: Optional[str]):
    """BLOCKING build preparation — always call via asyncio.to_thread."""
    cfg = _initialize_components()
    if cfg is None:
        raise RuntimeError("Failed to initialize AtomAgents components")

    # Configure model-specific LLM/embedding.
    # Model ids are `pipeline[@provider]`: the pipeline picks prompts and
    # defaults (atomagent / atomagent_local / fenicsagent); the provider
    # picks the API backend (see AtomAgents.config.discover_providers).
    from AtomAgents.llm_factory import parse_model_id, resolve_provider

    pipeline, provider = parse_model_id(model_id)
    if pipeline == "atomagent_local" and provider is None:
        provider = "local"  # legacy alias for atomagent@local
    provider_name, provider_cfg = resolve_provider(provider)  # raises ValueError on unknown

    cfg["llm_config"] = dict(
        provider=provider_name,
        model=provider_cfg.get("model") or AgentConfig.model,
        temperature=AgentConfig.temperature,
        max_tokens=AgentConfig.max_tokens,
        reasoning_effort=AgentConfig.reasoning_effort,
    )
    print(
        f"[OpenAI Backend] pipeline={pipeline} provider={provider_name} "
        f"model={cfg['llm_config']['model']}"
    )

    # Workspace selection:
    # - If session_dir is provided, use it directly (absolute) or under the
    #   jobs dir (relative). JOBS_DIR defaults to runs/jobs (ATOMAGENTS_JOBS_DIR).
    # - Otherwise, create a shared model workspace under <jobs>/backend-<model>-<timestamp>.
    from AtomAgents.config import JOBS_DIR

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if session_dir:
        p = Path(session_dir)
        project_root = p if p.is_absolute() else (Path(JOBS_DIR) / p)
    else:
        project_root = Path(JOBS_DIR) / f"backend-{model_id}-{ts}"
    project_root.mkdir(parents=True, exist_ok=True)
    cfg["project_root"] = project_root

    # Create the session memory bound to this workspace. The full schema
    # (messages + FTS + OKR tables) comes from SQLiteRAGMemory itself; the
    # old reduced backend copy is gone.
    # Resolve + PROBE the embedding backend. The previous version constructed
    # OpenAIEmbeddings without ever calling it, so a base URL that serves no
    # /v1/embeddings (any chat-only gateway) surfaced only as a per-call
    # "storing without vector" warning and semantic recall was silently off for
    # the entire session. build_embeddings resolves this once and says so.
    from AtomAgents.utils.embeddings import build_embeddings

    embedding_model, embedding_desc = build_embeddings()
    print(f"[OpenAI Backend] Embeddings: {embedding_desc}")
    cfg["embedding_desc"] = embedding_desc
    cfg["memory"] = cfg["SQLiteRAGMemory"](
        sqlite_path=str((project_root / "memory.db").resolve()),
        project_root=str(project_root.resolve()),
        embedding_model=embedding_model,
    )

    # Configure per-agent prompt loaders for fenicsagent
    if pipeline == "fenicsagent":
        root = Path(__file__).parent.parent
        fenics_dir = str((root / "AtomAgents" / "templates" / "fenicsx").resolve())
        cfg["prompt_loaders"] = {
            "Supervisor": PromptLoader(templates_dir=fenics_dir),
            "Simulator": PromptLoader(templates_dir=fenics_dir),
            "Modeler": PromptLoader(templates_dir=fenics_dir),
        }
    return cfg, project_root


# ---------------------------- API Endpoints --------------------------------


@app.post("/v1/chat/history")
async def get_chat_history(request: ChatHistoryRequest) -> ChatHistoryResponse:
    """Get chat history for a specific session."""
    session_dir = request.session_dir
    memory_db_path = Path(session_dir) / "memory.db"
    if not memory_db_path.exists():
        raise HTTPException(
            status_code=404, detail=f"No chat history found for session: {session_dir}"
        )
    # Fetch chat history from the memory database
    memory = SQLiteRAGMemory(
        sqlite_path=str(memory_db_path), project_root=str(session_dir)
    )
    messages = memory.get_all_raw_messages()
    return ChatHistoryResponse(messages=messages)


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models (OpenAI compatible)

    We expose logical model identifiers:
    - atomagent        → default AtomAgents pipeline (text or multimodal)
    - atomagent_local  → local/runtime-focused pipeline (also supports multimodal)
    - fenicsagent      → FEniCSx-flavoured prompts (Supervisor/Simulator)
    Both accept pure text messages or OpenAI-style multimodal content arrays.
    """
    now = int(datetime.now().timestamp())
    ids = ["atomagent", "atomagent_local", "fenicsagent"]
    # Provider-qualified variants: pipeline@provider for every configured
    # API backend (see AtomAgents.config.discover_providers).
    try:
        from AtomAgents.config import discover_providers

        for provider in sorted(discover_providers()):
            for pipeline in ("atomagent", "fenicsagent"):
                ids.append(f"{pipeline}@{provider}")
    except Exception as e:
        print(f"[OpenAI Backend] Provider discovery failed: {e}")
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(id=i, object="model", created=now, owned_by="atomagents")
            for i in ids
        ],
    )


class CancelRequest(BaseModel):
    model: str = Field(default="atomagent")
    session_dir: Optional[str] = Field(default=None)


@app.post("/v1/chat/cancel")
async def cancel_chat(request: CancelRequest) -> Dict[str, Any]:
    """Request cancellation of the active run for a session.

    Takes effect at the next agent boundary (a running tool/LAMMPS call
    finishes first).
    """
    key = (
        f"{request.model}:{request.session_dir}"
        if request.session_dir
        else request.model
    )
    CANCEL_FLAGS[key] = True
    return {"cancelling": key}


@dataclass
class RunContext:
    """Everything a surface emitter needs to drive one graph run.

    Built by ``_prepare_run`` so the session/graph resolution, image handling,
    thread-identity, and HITL-resume rules live in exactly one place for both
    /v1/chat/completions and /v1/responses.
    """

    graph: Any
    project_root: Optional[Path]
    graph_key: str
    model_id: str
    run_config: Dict[str, Any]
    graph_input: Any
    prompt: str


async def _prepare_run(
    model_id: str,
    session_dir: Optional[str],
    image_dir: Optional[str],
    text_content: str,
    images: List[str],
    request_id: str,
) -> RunContext:
    """Resolve the session graph and build the graph input for one request."""
    graph_key = f"{model_id}:{session_dir}" if session_dir else model_id
    try:
        graph, project_root = await _get_or_build_graph(
            graph_key, model_id, session_dir
        )
    except ValueError as e:
        # e.g. unknown @provider suffix in the model id
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # If a directory of images is provided, load them and append
    images = list(images or [])
    if image_dir:
        try:
            imgs_from_dir = _load_images_from_dir(image_dir)
            if imgs_from_dir:
                images.extend(imgs_from_dir)
        except Exception as e:
            print(f"[OpenAI Backend] Failed to load images from dir {image_dir}: {e}")

    # Save images (if any) under the model's project workspace for traceability
    image_paths = []
    if images and project_root:
        for img in images:
            p = _save_base64_image(img, project_root)
            if p:
                image_paths.append(p)

    enhanced_prompt = text_content
    if image_paths:
        enhanced_prompt = f"[{len(image_paths)} image(s) uploaded]\n{text_content}"

    if graph is None:
        # Caller renders its surface-appropriate echo/fallback response.
        return RunContext(
            graph=None,
            project_root=project_root,
            graph_key=graph_key,
            model_id=model_id,
            run_config={},
            graph_input=None,
            prompt=enhanced_prompt,
        )

    # Prepare input for the agent graph as multimodal content
    input_content: List[Dict[str, Any]] = []
    if enhanced_prompt:
        input_content.append({"type": "text", "text": enhanced_prompt})
    for b64 in images:
        input_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    input_msg = HumanMessage(content=input_content if input_content else enhanced_prompt)
    print(
        f"[OpenAI Backend] Received request: {model_id} - {str(input_msg.content)[:300]}"
    )
    # Thread identity for the checkpointer + proper config placement (the old
    # code put recursion_limit INSIDE input, where langgraph silently drops it).
    # Thread isolation (ultrareview bug_003): without a session_dir the graph
    # cache key collapses to the bare model id — using that as thread_id would
    # merge ALL anonymous callers of a model into one checkpointed
    # conversation (cross-client prompt/image leakage). Anonymous requests get
    # a per-request thread instead; continuity requires session_dir.
    run_config = {
        "configurable": {
            "thread_id": graph_key if session_dir else f"anon-{request_id}"
        },
        "recursion_limit": atom_config.GRAPH_RECURSION_LIMIT,
    }
    # HITL resume: if this durable thread is paused on an interrupt, the new
    # user message is the ANSWER to the pending question — resume the graph
    # (which routes it back to the asking agent) instead of injecting a fresh
    # conversational turn at the Supervisor.
    graph_input: Any = {"messages": [input_msg]}
    if session_dir and await _apending_interrupt(graph, run_config):
        from langgraph.types import Command

        graph_input = Command(resume=enhanced_prompt)
        print(f"[OpenAI Backend] Resuming pending HITL interrupt on {graph_key}")
    return RunContext(
        graph=graph,
        project_root=project_root,
        graph_key=graph_key,
        model_id=model_id,
        run_config=run_config,
        graph_input=graph_input,
        prompt=enhanced_prompt,
    )


@dataclass
class AtomEvent:
    """Normalized graph event; both API surfaces are thin emitters over it."""

    # working | token | message | awaiting_user | cancelled.
    # token = a raw-LLM text fragment from a tagged prose call (responses
    # surface only; the chat surface ignores tokens — byte-pinned format).
    kind: str
    agent: str = ""
    stage: str = ""
    msg_type: str = ""  # "ai" | "tool" for message events
    text: str = ""  # flattened text, no surface decoration
    parts: Any = None  # ORIGINAL message content (str or parts list)


def _event_from_message(msg) -> Optional[AtomEvent]:
    """Map one graph message to its AtomEvent (or None for non-ai/tool).

    Shared by the batch (updates-mode) and incremental (custom-mode) paths so
    the agent/stage parsing cannot drift between them.
    """
    msg_type = getattr(msg, "type", "")
    if msg_type not in ("ai", "tool"):
        return None
    raw = getattr(msg, "content", "")
    # Canonical flattener: provider-aware (excludes reasoning/thinking
    # blocks), preserves the "\n" join contract.
    text = message_text(msg)
    agent = getattr(msg, "name", None) or ""
    stage = ""
    m = re.match(r"#\s+(\S+)\s+-\s+(\w+)", text)
    if m:
        agent = agent or m.group(1)
        stage = m.group(2)
    if not stage:
        # Tool-call act messages carry no "# Agent - stage" header (their
        # content is the provider block list with tool_use blocks — never
        # rewritten); the agents stamp the stage into additional_kwargs
        # instead. Without this fallback the token-segment adoption key
        # (agent, stage) missed and the same act preamble shipped as TWO
        # output items on /v1/responses (A2).
        stage = (getattr(msg, "additional_kwargs", None) or {}).get("atom_stage", "")
    return AtomEvent(
        "message", agent=agent, stage=stage, msg_type=msg_type, text=text, parts=raw
    )


async def aiter_graph_events(
    graph,
    graph_input,
    run_config,
    *,
    cancel_key: Optional[str] = None,
    include_progress: bool = True,
):
    """Drive the agent graph asynchronously, yielding normalized ``AtomEvent``s.

    ``cancelled`` is terminal. Exceptions are deliberately NOT caught here —
    each emitter maps them to its own wire format (chat: error chunk;
    responses: response.failed; non-stream: HTTP 500).

    ``include_progress`` subscribes to langgraph's CUSTOM stream, which
    carries the typed progress events agents publish via get_stream_writer()
    ("working" markers). The non-streaming paths keep it off, preserving
    their historical ``stream_mode="updates"`` invocation exactly.
    """
    # Seed the dedupe set with the thread's checkpointed history so a prior
    # turn's message can never be re-emitted, even if a node update carries
    # more than its delta (defense-in-depth behind the agents' delta-only
    # returns; also covers the HITL-resume replay, where the checkpoint holds
    # the pre-interrupt messages). Degrades to an empty seed for anon first
    # turns, graphs without a checkpointer, and test fakes. Known residual:
    # pending writes of an interrupted superstep are not in ``values`` —
    # unreachable with single-node supersteps.
    emitted_ids: set = set()
    try:
        vals = getattr(await graph.aget_state(run_config), "values", None)
        if isinstance(vals, dict):
            emitted_ids = {
                m.id for m in (vals.get("messages") or []) if getattr(m, "id", None)
            }
    except Exception:
        emitted_ids = set()
    seeded_count = len(emitted_ids)
    message_events = 0
    # durability="sync" (default) persists each superstep before the next
    # starts — true crash-resume for multi-hour simulation sessions.
    durability = atom_config.GRAPH_DURABILITY
    if include_progress:
        stream_iter = graph.astream(
            input=graph_input,
            config=run_config,
            stream_mode=["updates", "custom"],
            durability=durability,
        )
    else:

        async def _wrap_updates():
            async for step in graph.astream(
                input=graph_input,
                config=run_config,
                stream_mode="updates",
                durability=durability,
            ):
                yield ("updates", step)

        stream_iter = _wrap_updates()
    async for mode, payload in stream_iter:
        if cancel_key is not None and CANCEL_FLAGS.pop(cancel_key, None):
            yield AtomEvent("cancelled", text="\n[run cancelled by user]\n")
            return
        if _DRAINING:
            # SIGTERM drain: durability="sync" already persisted the last
            # superstep — the next request resumes this thread verbatim.
            yield AtomEvent(
                "cancelled",
                text="\n[server draining — run checkpointed; reply to resume]\n",
            )
            return
        if mode == "custom":
            # Typed progress events published by agent runs / the Human node.
            if isinstance(payload, dict) and payload.get("event") == "working":
                agent = payload.get("agent", "")
                if agent:
                    yield AtomEvent("working", agent=agent)
            elif isinstance(payload, dict) and payload.get("event") == "message":
                # Incremental publish (WS19.6): a completed inner message,
                # chronological. The trailing updates batch re-delivers the
                # same messages and hits this dedupe as a no-op.
                msg = payload.get("message")
                key = getattr(msg, "id", None) or id(msg)
                if key in emitted_ids:
                    continue
                emitted_ids.add(key)
                ev = _event_from_message(msg)
                if ev is not None:
                    message_events += 1
                    yield ev
            elif isinstance(payload, dict) and payload.get("event") == "token":
                text = payload.get("text", "")
                if text:
                    yield AtomEvent(
                        "token",
                        agent=payload.get("agent", ""),
                        stage=payload.get("stage", ""),
                        msg_type="ai",
                        text=text,
                    )
            continue
        # HITL pause: the graph is waiting on the user; the asking agent's
        # report (with the actual question) was already emitted.
        if isinstance(payload, dict) and "__interrupt__" in payload:
            for intr in payload.get("__interrupt__") or ():
                val = getattr(intr, "value", None)
                frm = val.get("from", "") if isinstance(val, dict) else ""
                text = (
                    "[awaiting your input"
                    + (f" — requested by {frm}" if frm else "")
                    + " — reply to continue]"
                )
                yield AtomEvent(
                    "awaiting_user",
                    agent=frm or "Human",
                    stage="await",
                    msg_type="ai",
                    text=text,
                    parts=text,
                )
            continue
        for v in payload.values():
            if not isinstance(v, dict):
                continue
            for msg in v.get("messages", []):
                # Yield each message once — the old code re-emitted the
                # entire history on every graph update.
                key = getattr(msg, "id", None) or id(msg)
                if key in emitted_ids:
                    continue
                emitted_ids.add(key)
                ev = _event_from_message(msg)
                if ev is not None:
                    message_events += 1
                    yield ev
    if seeded_count and not message_events:
        # A run that emitted nothing while the seed was non-empty is the
        # signature of an id collision with the checkpointed history — make
        # the empty response diagnosable instead of silent.
        print(
            f"[OpenAI Backend] aiter_graph_events: {seeded_count} seeded ids, "
            "zero message events emitted this run"
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (OpenAI compatible, legacy surface)."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(datetime.now().timestamp())

    # Only the newest user message is sent to the graph — prior turns live in
    # the session's checkpointed graph state, not in the request payload.
    try:
        text_content, images = _extract_latest_user_message(request.messages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid messages: {e}")

    model_id = request.model or "atomagent"
    ctx = await _prepare_run(
        model_id,
        request.session_dir or None,
        request.image_dir,
        text_content,
        images,
        completion_id,
    )

    # If the app graph is not ready, return a minimal echo response
    if ctx.graph is None:
        msg = {"role": "assistant", "content": f"[Echo] {ctx.prompt}"}
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[ChatCompletionChoice(index=0, message=msg, finish_reason="stop")],
            usage=ChatCompletionUsage(),
        )
    # Streaming mode ---------------------------------------------------------
    if request.stream:

        def _sse(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[
                    ChatCompletionChoice(index=0, delta=delta, finish_reason=finish)
                ],
            )
            return f"data: {chunk.model_dump_json()}\n\n"

        cancel_key = ctx.graph_key
        CANCEL_FLAGS.pop(cancel_key, None)

        run_lock = await _acquire_run_lock(ctx.graph_key)

        async def generate_stream():
            # Extra delta fields (atom_*) are ignored by plain OpenAI clients,
            # which still receive the concatenated text. The run lock was
            # acquired by the handler (so 409s precede streaming); this
            # generator owns releasing it.
            try:
                async for ev in aiter_graph_events(
                    ctx.graph,
                    ctx.graph_input,
                    ctx.run_config,
                    cancel_key=cancel_key,
                    include_progress=True,
                ):
                    if ev.kind == "cancelled":
                        yield _sse(
                            {"atom_event": "cancelled", "content": ev.text},
                            finish="stop",
                        )
                        yield "data: [DONE]\n\n"
                        return
                    if ev.kind == "working":
                        yield _sse({"atom_event": "working", "atom_agent": ev.agent})
                    elif ev.kind == "token":
                        # Chat surface is byte-pinned (golden frames): token
                        # deltas live on /v1/responses only.
                        continue
                    elif ev.kind == "awaiting_user":
                        yield _sse(
                            {
                                "content": "\n" + ev.text + "\n",
                                "atom_event": "awaiting_user",
                                "atom_agent": ev.agent,
                                "atom_stage": ev.stage,
                                "atom_type": ev.msg_type,
                            }
                        )
                    elif ev.kind == "message" and ev.text:
                        yield _sse(
                            {
                                "content": ev.text + "\n\n",
                                "atom_event": "message",
                                "atom_agent": ev.agent,
                                "atom_stage": ev.stage,
                                "atom_type": ev.msg_type,
                            }
                        )

                end = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChatCompletionChoice(index=0, delta={}, finish_reason="stop")
                    ],
                )
                yield f"data: {end.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            delta={"content": f"Error: {e}"},
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {err.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                run_lock.release()

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming mode -----------------------------------------------------
    run_lock = await _acquire_run_lock(ctx.graph_key)
    try:

        async def _run_graph():
            final_parts: List[Dict[str, Any]] = []
            final_text = ""
            async for ev in aiter_graph_events(
                ctx.graph,
                ctx.graph_input,
                ctx.run_config,
                cancel_key=None,
                include_progress=False,
            ):
                if ev.kind == "awaiting_user":
                    c: Any = ev.text
                elif ev.kind == "message" and ev.msg_type == "ai":
                    c = ev.parts
                else:
                    continue
                if isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") in {
                            "text",
                            "image_url",
                        }:
                            final_parts.append(p)
                else:
                    if final_parts:
                        final_parts.append({"type": "text", "text": str(c or "")})
                    else:
                        final_text += ("\n\n" if final_text else "") + str(c or "")
            return final_parts, final_text

        # The graph is async end-to-end now; blocking work inside agent runs
        # is offloaded by arun itself (to_thread epilogue).
        final_parts, final_text = await _run_graph()

        reply_content: Any = final_parts if final_parts else final_text
        reply = {"role": "assistant", "content": reply_content}
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChoice(index=0, message=reply, finish_reason="stop")
            ],
            usage=ChatCompletionUsage(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")
    finally:
        run_lock.release()


# --------------------- /v1/responses (primary surface) ---------------------

# response_id → graph_key, so /v1/responses/{id}/cancel can reach the run.
# Bounded: evict-oldest past the cap so it can never leak. /v1/chat/cancel
# (model+session_dir keyed) remains the authoritative cancel for both
# surfaces — it needs no response id, which is what the UI stop button has.
RESPONSE_KEYS: "OrderedDict[str, str]" = OrderedDict()
_RESPONSE_KEYS_MAX = 256


def _remember_response_key(response_id: str, graph_key: str) -> None:
    RESPONSE_KEYS[response_id] = graph_key
    while len(RESPONSE_KEYS) > _RESPONSE_KEYS_MAX:
        RESPONSE_KEYS.popitem(last=False)


def _item_from_event(ev: AtomEvent) -> ResponseMessageItem:
    md = {
        "atom_agent": ev.agent,
        "atom_stage": ev.stage,
        "atom_type": ev.msg_type or "ai",
    }
    if ev.kind == "awaiting_user":
        md["atom_event"] = "awaiting_user"
    return ResponseMessageItem(
        id=f"msg_{uuid.uuid4().hex[:16]}",
        content=[ResponseOutputText(text=ev.text)],
        metadata=md,
    )


# Terminal SSE event per final status. "cancelled" rides response.incomplete:
# the openai SDK's typed event union has no response.cancelled member, and
# incomplete is the semantically closest terminal generic clients handle.
_TERMINAL_EVENT = {
    "completed": "response.completed",
    "incomplete": "response.incomplete",
    "cancelled": "response.incomplete",
    "failed": "response.failed",
}


@app.post("/v1/responses")
async def create_response(request: ResponsesRequest):
    """Create a model response (OpenAI Responses API — the primary surface)."""
    if request.previous_response_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "server-side response storage is not implemented; use the "
                "session_dir extension for conversation continuity"
            ),
        )
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(datetime.now().timestamp())
    try:
        text_content, images = _extract_from_responses_input(
            request.input, request.instructions
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    model_id = request.model or "atomagent"
    ctx = await _prepare_run(
        model_id,
        request.session_dir or None,
        request.image_dir,
        text_content,
        images,
        response_id,
    )

    def snapshot(
        *,
        status: str,
        output: List[ResponseMessageItem],
        output_text: str = "",
        metadata: Optional[Dict[str, str]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> ResponseObject:
        return ResponseObject(
            id=response_id,
            created_at=created,
            status=status,
            model=model_id,
            instructions=request.instructions,
            output=output,
            output_text=output_text,
            metadata={**(request.metadata or {}), **(metadata or {})},
            error=error,
        )

    if ctx.graph is None:
        echo = AtomEvent("message", msg_type="ai", text=f"[Echo] {ctx.prompt}")
        item = _item_from_event(echo)
        return snapshot(status="completed", output=[item], output_text=echo.text)

    _remember_response_key(response_id, ctx.graph_key)
    CANCEL_FLAGS.pop(ctx.graph_key, None)

    async def _consume_events(on_item=None):
        """Shared drain loop: yields intermediate frames and finally
        ("end", None, (items, texts, status, metadata, error)).

        ``on_item`` lets the streaming emitter interleave per-item SSE frames
        while keeping the status bookkeeping identical to non-streaming.

        Token correlation (WS19.8): incremental publishing (WS19.6)
        guarantees at most ONE open token segment at a time. First token of a
        new (agent, stage) opens an item; the matching completed message
        ADOPTS it with the authoritative stage-bound text; a mismatching
        message / cancel / error / stream-end closes the orphan with its
        buffered raw preview so the terminal snapshot stays self-consistent.
        Non-streaming never sees tokens (include_progress off).
        """
        items: List[ResponseMessageItem] = []
        texts: List[str] = []
        status = "completed"
        metadata: Dict[str, str] = {}
        error: Optional[Dict[str, Any]] = None
        open_item = None  # (ResponseMessageItem, out_index)
        open_key = None  # (agent, stage)
        open_buf: List[str] = []

        def _close_open_segment():
            nonlocal open_item, open_key, open_buf
            if open_item is None:
                return None
            item, out_index = open_item
            preview = "".join(open_buf)
            item.content = [ResponseOutputText(text=preview)]
            texts[out_index] = preview
            closed = (item, out_index, preview)
            open_item = open_key = None
            open_buf = []
            return closed

        try:
            async for ev in aiter_graph_events(
                ctx.graph,
                ctx.graph_input,
                ctx.run_config,
                cancel_key=ctx.graph_key,
                include_progress=on_item is not None,
            ):
                if ev.kind == "cancelled":
                    status = "cancelled"
                    metadata["atom_status"] = "cancelled"
                    if on_item is not None:
                        closed = _close_open_segment()
                        if closed is not None:
                            yield ("segment_end", None, closed)
                        yield ("cancelled", None, None)
                    break
                if ev.kind == "working":
                    if on_item is not None:
                        yield ("working", ev, None)
                    continue
                if ev.kind == "token":
                    if on_item is None:
                        continue
                    key = (ev.agent, ev.stage)
                    if open_key != key:
                        closed = _close_open_segment()
                        if closed is not None:
                            yield ("segment_end", None, closed)
                        item = _item_from_event(
                            AtomEvent(
                                "message",
                                agent=ev.agent,
                                stage=ev.stage,
                                msg_type="ai",
                            )
                        )
                        out_index = len(items)
                        items.append(item)
                        texts.append("")
                        open_item, open_key = (item, out_index), key
                        yield ("segment_start", ev, (item, out_index))
                    open_buf.append(ev.text)
                    yield ("token", ev, open_item)
                    continue
                if ev.kind not in ("message", "awaiting_user") or not ev.text:
                    continue
                if open_item is not None and open_key == (ev.agent, ev.stage):
                    # ADOPT: the streamed segment becomes this authoritative
                    # item — done text carries the stage-bound header, deltas
                    # were the raw-LLM preview (documented wire contract).
                    item, out_index = open_item
                    item.content = [ResponseOutputText(text=ev.text)]
                    item.metadata.update(
                        {
                            "atom_agent": ev.agent,
                            "atom_stage": ev.stage,
                            "atom_type": ev.msg_type or "ai",
                        }
                    )
                    texts[out_index] = ev.text
                    open_item = open_key = None
                    open_buf = []
                    yield ("segment_done", ev, (item, out_index))
                    continue
                closed = _close_open_segment()
                if closed is not None and on_item is not None:
                    yield ("segment_end", None, closed)
                item = _item_from_event(ev)
                out_index = len(items)
                items.append(item)
                texts.append(ev.text)
                if ev.kind == "awaiting_user":
                    status = "incomplete"
                    metadata["atom_status"] = "awaiting_user"
                    if ev.agent:
                        metadata["atom_agent"] = ev.agent
                if on_item is not None:
                    yield ("item", ev, (item, out_index))
        except Exception as e:
            status = "failed"
            metadata["atom_status"] = "error"
            error = {"code": "server_error", "message": str(e)}
        # Never leave a segment dangling (error/stream-end paths).
        closed = _close_open_segment()
        if closed is not None and on_item is not None:
            yield ("segment_end", None, closed)
        yield ("end", None, (items, texts, status, metadata, error))

    # Streaming: typed named SSE events, no [DONE] sentinel ------------------
    if request.stream:
        run_lock = await _acquire_run_lock(ctx.graph_key)

        async def generate_responses_stream():
            seq = 0

            def frame(event_type: str, data: Dict[str, Any]) -> str:
                nonlocal seq
                payload = {"type": event_type, "sequence_number": seq, **data}
                seq += 1
                return (
                    f"event: {event_type}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )

            try:
                base = snapshot(status="in_progress", output=[])
                yield frame("response.created", {"response": base.model_dump()})
                yield frame("response.in_progress", {"response": base.model_dump()})

                result = None
                async for kind, ev, extra in _consume_events(on_item=True):
                    if kind == "working":
                        yield frame(
                            "response.atomagents.working", {"agent": ev.agent}
                        )
                    elif kind == "cancelled":
                        yield frame("response.atomagents.cancelled", {})
                    elif kind == "segment_start":
                        item, out_index = extra
                        pending = item.model_dump()
                        pending["status"] = "in_progress"
                        pending["content"] = []
                        yield frame(
                            "response.output_item.added",
                            {"output_index": out_index, "item": pending},
                        )
                    elif kind == "token":
                        item, out_index = extra
                        yield frame(
                            "response.output_text.delta",
                            {
                                "item_id": item.id,
                                "output_index": out_index,
                                "content_index": 0,
                                "delta": ev.text,
                                "logprobs": [],
                            },
                        )
                    elif kind in ("segment_done", "segment_end"):
                        if kind == "segment_done":
                            item, out_index = extra
                            done_text = item.content[0].text if item.content else ""
                        else:
                            item, out_index, done_text = extra
                        yield frame(
                            "response.output_text.done",
                            {
                                "item_id": item.id,
                                "output_index": out_index,
                                "content_index": 0,
                                "text": done_text,
                                "logprobs": [],
                            },
                        )
                        yield frame(
                            "response.output_item.done",
                            {"output_index": out_index, "item": item.model_dump()},
                        )
                    elif kind == "item":
                        item, out_index = extra
                        pending = item.model_dump()
                        pending["status"] = "in_progress"
                        pending["content"] = []
                        yield frame(
                            "response.output_item.added",
                            {"output_index": out_index, "item": pending},
                        )
                        yield frame(
                            "response.output_text.delta",
                            {
                                "item_id": item.id,
                                "output_index": out_index,
                                "content_index": 0,
                                "delta": ev.text,
                                "logprobs": [],
                            },
                        )
                        yield frame(
                            "response.output_text.done",
                            {
                                "item_id": item.id,
                                "output_index": out_index,
                                "content_index": 0,
                                "text": ev.text,
                                "logprobs": [],
                            },
                        )
                        yield frame(
                            "response.output_item.done",
                            {"output_index": out_index, "item": item.model_dump()},
                        )
                        if ev.kind == "awaiting_user":
                            yield frame(
                                "response.atomagents.awaiting_user",
                                {"agent": ev.agent},
                            )
                    elif kind == "end":
                        result = extra
                items, texts, status, metadata, error = result
                final = snapshot(
                    status=status,
                    output=items,
                    output_text="\n\n".join(texts),
                    metadata=metadata,
                    error=error,
                )
                yield frame(_TERMINAL_EVENT[status], {"response": final.model_dump()})
            finally:
                run_lock.release()

        return StreamingResponse(
            generate_responses_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming ----------------------------------------------------------
    async def _run_responses() -> ResponseObject:
        result = None
        async for kind, _ev, extra in _consume_events(on_item=None):
            if kind == "end":
                result = extra
        items, texts, status, metadata, error = result
        return snapshot(
            status=status,
            output=items,
            output_text="\n\n".join(texts),
            metadata=metadata,
            error=error,
        )

    run_lock = await _acquire_run_lock(ctx.graph_key)
    try:
        return await _run_responses()
    finally:
        run_lock.release()


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str):
    """Standard Responses-API cancel; /v1/chat/cancel remains authoritative."""
    key = RESPONSE_KEYS.get(response_id)
    if not key:
        raise HTTPException(status_code=404, detail=f"unknown response id {response_id}")
    CANCEL_FLAGS[key] = True
    return {"id": response_id, "object": "response", "status": "cancelled"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graphs": list(_APP_GRAPHS.keys()),
        "graph_ready": bool(_APP_GRAPHS),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AtomAgents OpenAI-compatible backend")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "backend.openai_backend:app", host=args.host, port=args.port, reload=args.reload
    )
