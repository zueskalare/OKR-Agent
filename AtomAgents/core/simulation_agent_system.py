"""
High-level simulation system orchestration for AtomAgents.

This module provides a thin job manager that composes the core agents
into a LangGraph workflow and runs them in the background. It exposes a
simple API similar to the demo under `demo_running/test.py`:

- Build a graph: Supervisor ⇄ Modeler ⇄ Simulator (+ optional Human)
- Submit a job with a natural-language description
- Poll status until completion

Two operation modes are supported:
- with_human: a Human node collects interactive input when routed
- no_human: a stub node bounces the flow back with a note (non-blocking)
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import logging

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy as _RetryPolicy
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from .. import config as _cfg
from ..workflows.state_definitions import State as GraphState
from ..memory.sqlite_rag_memory import SQLiteRAGMemory
from ..agents import Modeler, SimulatorAgent
from ..agents.simulation_manager import Supervisor
from ..agents.human_interput import human_node as default_human_node
from ..utils.routing import agent_error_handler, handoff_router


def _session_checkpointer(project_root: Path):
    """Durable graph state under the session dir; in-memory fallback.

    The serde allowlist registers ReportMessage for msgpack — without it,
    langgraph's strict-msgpack mode silently degrades the subclass to a
    plain dict on deserialization (see backend._make_checkpointer).
    """
    try:
        import sqlite3 as _sqlite3

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite import SqliteSaver

        from ..agents.messages import ReportMessage

        conn = _sqlite3.connect(
            str(Path(project_root) / "graph_state.db"), check_same_thread=False
        )
        return SqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[ReportMessage])
        )
    except Exception:
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver()
        except ModuleNotFoundError:
            return None


@dataclass
class _Job:
    job_id: str
    app: Any
    thread: threading.Thread
    status: str = "queued"
    send_to: Optional[str] = None
    last_messages: list[BaseMessage] = field(default_factory=list)
    error: Optional[str] = None
    project_root: Path | None = None


class SimulationAgentSystem:
    """Simple job manager that wires up the simulation agent framework.

    Parameters
    - general_llm: LLM used by coordination-heavy agents (e.g., Supervisor)
    - coder_llm: LLM used by execution-heavy agents (e.g., Simulator/Modeler)
    - prompt_loader: PromptLoader instance
    - workdir: base directory for job workspaces (jobs are created under it)
    - verbose: enable agent debug logging
    - with_human: whether to include an interactive Human node in the graph
    """

    def __init__(
        self,
        general_llm=None,
        coder_llm=None,
        prompt_loader=None,
        workdir: str | Path = ".",
        verbose: bool = True,
        with_human: bool = False,
    ) -> None:
        self.general_llm = general_llm
        self.coder_llm = coder_llm or general_llm
        self.prompt_loader = prompt_loader
        self.workdir = Path(workdir).resolve()
        self.verbose = verbose
        self.with_human = with_human

        self.jobs: dict[str, _Job] = {}
        self.logger = logging.getLogger("AtomAgents.SimulationAgentSystem")

        from ..config import JOBS_DIR

        # Sessions live under runs/jobs by default (ATOMAGENTS_JOBS_DIR).
        self.jobs_dir = (
            self.workdir / JOBS_DIR
            if not Path(JOBS_DIR).is_absolute()
            else Path(JOBS_DIR)
        )
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def submit_job(
        self,
        request: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        with_human: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a job workspace, build the agent graph, and run in background.

        request should contain either a plain `description` string or a
        pre-built list of `messages`.
        """

        job_id = str(uuid.uuid4())
        use_human = self.with_human if with_human is None else bool(with_human)

        project_root = (self.jobs_dir / job_id).resolve()
        project_root.mkdir(parents=True, exist_ok=True)
        sqlite_path = project_root / "memory.db"

        # Memory shared by all agents
        memory = SQLiteRAGMemory(
            sqlite_path=str(sqlite_path), project_root=str(project_root)
        )

        from ..utils.okr_tools import build_okr_tools

        okr_tools = build_okr_tools(memory, str(project_root))

        # Construct agents
        # Explicitly avoid self-routing to prevent stuck loops. Provide per-agent
        # collage lists that never include the current agent. "Human" is only a
        # valid target when a real human node exists — offering it alongside a
        # stub let reports route to a bounce-back loop.
        human = ["Human"] if use_human else []
        collages_supervisor = ["Modeler", "Simulator", "END"] + human
        collages_modeler = ["Supervisor", "Simulator", "END"] + human
        collages_simulator = ["Supervisor", "Modeler", "END"] + human

        supervisor = Supervisor(
            llm=self.general_llm,
            memory=memory,
            Collages=collages_supervisor,
            tools=list(okr_tools),
            project_root=str(project_root),
            prompt_loader=self.prompt_loader,
            verbose=self.verbose,
        )
        modeler = Modeler(
            llm=self.coder_llm,
            memory=memory,
            Collages=collages_modeler,
            tools=list(okr_tools),
            project_root=str(project_root),
            prompt_loader=self.prompt_loader,
            verbose=self.verbose,
        )
        simulator = SimulatorAgent(
            llm=self.coder_llm,
            memory=memory,
            Collages=collages_simulator,
            tools=list(okr_tools),
            project_root=str(project_root),
            prompt_loader=self.prompt_loader,
            verbose=self.verbose,
            config={"recursion_limit": 50},
        )

        # Build node callables (thin wrappers around .run) -----------------
        def run_supervisor(state: Dict[str, Any]):
            out = supervisor.run(state)
            try:
                self.logger.info("Supervisor → send_to=%s", out.get("send_to"))
            except Exception:
                pass
            return out

        def run_modeler(state: Dict[str, Any]):
            out = modeler.run(state)
            try:
                self.logger.info("Modeler → send_to=%s", out.get("send_to"))
            except Exception:
                pass
            return out

        def run_simulator(state: Dict[str, Any]):
            out = simulator.run(state)
            try:
                self.logger.info("Simulator → send_to=%s", out.get("send_to"))
            except Exception:
                pass
            return out

        # Human node: interactive or stub based on mode --------------------
        def _human_stub(state: Dict[str, Any]):
            # Terminal: the old stub bounced back to the sender, which
            # immediately re-asked — an un-capped Supervisor↔Human ping-pong
            # bounded only by the recursion limit.
            note = AIMessage(
                content=(
                    "# Human - reply\n[AUTOMATED NOTE] No human-in-the-loop "
                    "available; ending the turn instead of bouncing back."
                ),
                name="Human",
            )
            # Delta-only update: state already holds the inbound history
            # (add_messages); re-returning it replays prior turns on the wire.
            return {"messages": [note], "send_to": "END"}

        human_callable: Callable[[Dict[str, Any]], Dict[str, Any]]
        human_callable = default_human_node if use_human else _human_stub

        # Assemble the graph -----------------------------------------------
        graph = StateGraph(state_schema=GraphState)
        graph.set_node_defaults(
            retry_policy=_RetryPolicy(max_attempts=_cfg.NODE_MAX_ATTEMPTS),
            error_handler=agent_error_handler,
        )
        graph.add_node("Supervisor", run_supervisor)
        graph.add_node("Modeler", run_modeler)
        graph.add_node("Simulator", run_simulator)
        graph.add_node("Human", human_callable)

        graph.set_entry_point("Supervisor")

        graph.add_conditional_edges(
            "Supervisor",
            partial(handoff_router, agent="Supervisor"),
            {
                "Modeler": "Modeler",
                "Simulator": "Simulator",
                "Human": "Human",
                "END": END,
            },
        )
        graph.add_conditional_edges(
            "Modeler",
            partial(handoff_router, agent="Modeler"),
            {
                "Supervisor": "Supervisor",
                "Simulator": "Simulator",
                "Human": "Human",
                "END": END,
            },
        )
        graph.add_conditional_edges(
            "Simulator",
            partial(handoff_router, agent="Simulator"),
            {
                "Supervisor": "Supervisor",
                "Modeler": "Modeler",
                "Human": "Human",
                "END": END,
            },
        )
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

        app = graph.compile(checkpointer=_session_checkpointer(project_root))

        # Prepare input messages ------------------------------------------
        messages = request.get("messages")
        if messages is None:
            desc = request.get("description", "")
            messages = [HumanMessage(content=desc)]

        # Start background execution --------------------------------------
        job = _Job(job_id=job_id, app=app, thread=None, status="queued", project_root=project_root)
        self.jobs[job_id] = job

        def _runner():
            job.status = "running"
            try:
                for update in app.stream(
                    input={"messages": messages},
                    # recursion_limit belongs in config; inside `input` it was
                    # silently dropped.
                    config={
                        "configurable": {"thread_id": job_id},
                        "recursion_limit": _cfg.GRAPH_RECURSION_LIMIT,
                    },
                    stream_mode="updates",
                    durability=_cfg.GRAPH_DURABILITY,
                ):
                    for node, payload in update.items():
                        # payload is a state dict from a node
                        job.last_messages = payload.get("messages", [])
                        job.send_to = payload.get("send_to")
                        try:
                            self.logger.info(
                                "Graph update from %s → send_to=%s",
                                node,
                                job.send_to,
                            )
                        except Exception:
                            pass
                # When stream completes, consider job done
                job.status = "completed"
            except Exception as e:
                job.error = str(e)
                job.status = "error"

        t = threading.Thread(target=_runner, name=f"job-{job_id}", daemon=True)
        job.thread = t
        t.start()

        return {"job_id": job_id, "project_root": str(project_root)}

    def get_status(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found"}
        last = job.last_messages[-1] if job.last_messages else None
        return {
            "status": job.status,
            "send_to": job.send_to,
            "last_message": getattr(last, "content", None),
            "error": job.error,
            "project_root": str(job.project_root) if job.project_root else None,
        }


def build_simulation_graph(
    work_dir: str | Path,
    *,
    memory_file: str = "memory.db",
    tools: Optional[list[Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    with_human: bool = False,
) -> tuple[Any, SQLiteRAGMemory, Path]:
    """Build and return a compiled LangGraph app for the simulation workflow.

    Returns (app_graph, memory, project_root). Inspired by backend/Build_AtomAgent.py
    but kept minimal (no MCP wiring). Pass `tools` if you have MCP tools.
    """
    project_root = Path(work_dir).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = (
        project_root / memory_file if not str(memory_file).startswith("/") else Path(memory_file)
    )

    memory = SQLiteRAGMemory(
        sqlite_path=str(sqlite_path.resolve()), project_root=str(project_root)
    )

    from ..utils.okr_tools import build_okr_tools

    tools = list(tools or []) + build_okr_tools(memory, str(project_root))

    # Instantiate agents; allow llm_config passthrough so BaseAgent can construct LLM
    # Avoid self in collage lists to prevent self-routing selections
    collages_supervisor = ["Modeler", "Simulator", "END"] + (["Human"] if with_human else [])
    collages_modeler = ["Supervisor", "Simulator", "END"] + (["Human"] if with_human else [])
    collages_simulator = ["Supervisor", "Modeler", "END"] + (["Human"] if with_human else [])

    sup_kwargs: Dict[str, Any] = dict(
        memory=memory,
        Collages=collages_supervisor,
        tools=tools or [],
        project_root=str(project_root),
        enable_okr_table=True,
    )
    if llm_config:
        sup_kwargs["llm_config"] = llm_config

    mod_kwargs: Dict[str, Any] = dict(
        memory=memory,
        Collages=collages_modeler,
        tools=tools or [],
        project_root=str(project_root),
    )
    if llm_config:
        mod_kwargs["llm_config"] = llm_config

    sim_kwargs: Dict[str, Any] = dict(
        memory=memory,
        Collages=collages_simulator,
        tools=tools or [],
        project_root=str(project_root),
        config={"recursion_limit": 30},
    )
    if llm_config:
        sim_kwargs["llm_config"] = llm_config

    supervisor = Supervisor(**sup_kwargs)
    modeler = Modeler(**mod_kwargs)
    simulator = SimulatorAgent(**sim_kwargs)

    graph = StateGraph(state_schema=GraphState)
    graph.set_node_defaults(
        retry_policy=_RetryPolicy(max_attempts=_cfg.NODE_MAX_ATTEMPTS),
        error_handler=agent_error_handler,
    )
    graph.add_node("Supervisor", supervisor.run)
    graph.add_node("Modeler", modeler.run)
    graph.add_node("Simulator", simulator.run)

    if with_human:
        graph.add_node("Human", default_human_node)

    graph.set_entry_point("Supervisor")

    sup_edges = {
        "Modeler": "Modeler",
        "Simulator": "Simulator",
        "END": END,
    }
    if with_human:
        sup_edges["Human"] = "Human"
    graph.add_conditional_edges(
        "Supervisor", partial(handoff_router, agent="Supervisor"), sup_edges
    )

    mod_edges = {
        "Supervisor": "Supervisor",
        "Simulator": "Simulator",
        "END": END,
    }
    if with_human:
        mod_edges["Human"] = "Human"
    graph.add_conditional_edges(
        "Modeler", partial(handoff_router, agent="Modeler"), mod_edges
    )

    sim_edges = {
        "Supervisor": "Supervisor",
        "Modeler": "Modeler",
        "END": END,
    }
    if with_human:
        sim_edges["Human"] = "Human"
    graph.add_conditional_edges(
        "Simulator", partial(handoff_router, agent="Simulator"), sim_edges
    )

    if with_human:
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

    app = graph.compile(checkpointer=_session_checkpointer(project_root))
    return app, memory, project_root
