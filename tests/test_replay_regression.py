"""R2 regression: each run persists only ITS OWN messages to memory.db.

Before the fix, _run_epilogue handed _save_run_messages the whole session
history while the dedupe set reset per run, so every turn re-saved (and
re-embedded) all prior HumanMessages and own-name messages: rows grew
k·n(n+1)/2 (the pre-demo review measured 4→12→24 over three turns). Fixed
growth is k·n, with k = rows produced by one turn.
"""

import asyncio
import sqlite3
from typing import Annotated, Any, TypedDict

import pytest
from fakes import FakeLLM
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from test_memory_schema import StubEmb

from AtomAgents.agents.model_building_agent import Modeler
from AtomAgents.agents.simulation_manager import (
    Supervisor,
    SupervisorPlanOut,
    TriageOut,
    UpdateTask,
)
from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory


class S(TypedDict):
    messages: Annotated[list[Any], add_messages]
    send_to: str | None


@pytest.fixture()
def mem(tmp_path):
    return SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "memory.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )


def _row_count(mem):
    with sqlite3.connect(mem.sqlite_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]


def _supervisor_app(sup):
    g = StateGraph(S)
    g.add_node("Supervisor", sup.arun)
    g.set_entry_point("Supervisor")
    g.add_conditional_edges("Supervisor", lambda s: s.get("send_to"), {"END": END})
    return g.compile(checkpointer=InMemorySaver())


def test_memory_rows_grow_linearly_across_turns(tmp_path, mem):
    prompts = ["hi", "hello there", "hey friend"]
    sup = Supervisor(
        llm=FakeLLM(
            [TriageOut(verdict="converse", reply=f"answer {i}") for i in range(3)]
        ),
        memory=mem,
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    app = _supervisor_app(sup)
    cfg = {"configurable": {"thread_id": "mem-linear"}}

    counts = []

    async def go():
        for prompt in prompts:
            await app.ainvoke({"messages": [HumanMessage(prompt)]}, config=cfg)
            counts.append(_row_count(mem))

    asyncio.run(go())

    k = counts[0]
    assert k > 0
    assert counts == [k, 2 * k, 3 * k], (
        f"quadratic re-persistence is back: rows went {counts}"
    )
    with sqlite3.connect(mem.sqlite_path) as conn:
        for prompt in prompts:
            n = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE msg_text = ?", (prompt,)
            ).fetchone()[0]
            assert n == 1, f"prompt {prompt!r} persisted {n}× (expected once)"
        agents = conn.execute(
            "SELECT COUNT(DISTINCT agent_name) FROM messages"
        ).fetchone()[0]
    assert agents == 1  # Supervisor-only script — no cross-agent copies


class _AutoStructured:
    """Schema-introspecting structured fake for agents whose schemas are
    function-local (Modeler.plan's PlanOut) — same pattern as
    test_structure.py."""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, msgs, *args, **kwargs):
        fields = self.schema.model_fields
        kw = {}
        if "plan" in fields:
            kw["plan"] = "plan text"
            kw["send_to"] = "report"
        if "report" in fields:
            kw["report"] = "model built"
            args_ = getattr(fields["send_to"].annotation, "__args__", ("END",))
            kw["send_to"] = "END" if "END" in args_ else args_[0]
        return self.schema(**kw)

    async def ainvoke(self, msgs, *args, **kwargs):
        return self.invoke(msgs, *args, **kwargs)


class _AutoLLM:
    def invoke(self, msgs, *args, **kwargs):
        return AIMessage(content="working on it")

    async def ainvoke(self, msgs, *args, **kwargs):
        return self.invoke(msgs, *args, **kwargs)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _AutoStructured(schema)

    def with_config(self, *args, **kwargs):
        return self

    def bind(self, **kwargs):
        return self


def test_handoff_persists_human_message_once_under_entry_agent(tmp_path, mem):
    """A Supervisor→Modeler turn stores the user's HumanMessage exactly once,
    under the ENTRY agent — the recipient's run must not re-save it."""
    prompt = "build a bcc iron model"
    sup = Supervisor(
        llm=FakeLLM([]),
        memory=mem,
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    sup.llm.script = [
        TriageOut(verdict="task"),
        AIMessage(content="# Supervisor - think\nassign the build"),
        UpdateTask(update_task=False),
        SupervisorPlanOut(plan="hand off to Modeler", send_to="report"),
        sup._Report_Schema(report="handing off", send_to="Modeler"),
    ]
    mod = Modeler(
        llm=_AutoLLM(),
        memory=mem,
        project_root=str(tmp_path),
        tools=[],
        Collages=["Supervisor", "END"],
    )

    g = StateGraph(S)
    g.add_node("Supervisor", sup.arun)
    g.add_node("Modeler", mod.arun)
    g.set_entry_point("Supervisor")
    g.add_conditional_edges(
        "Supervisor", lambda s: s.get("send_to"), {"Modeler": "Modeler", "END": END}
    )
    g.add_conditional_edges("Modeler", lambda s: s.get("send_to"), {"END": END})
    app = g.compile(checkpointer=InMemorySaver())

    asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(prompt)]},
            config={"configurable": {"thread_id": "handoff"}},
        )
    )

    with sqlite3.connect(mem.sqlite_path) as conn:
        human_rows = conn.execute(
            "SELECT agent_name FROM messages WHERE msg_text = ?", (prompt,)
        ).fetchall()
        modeler_rows = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE agent_name = 'Modeler'"
        ).fetchone()[0]
    assert human_rows == [("Supervisor",)], (
        f"HumanMessage rows (agent_name): {human_rows}"
    )
    assert modeler_rows > 0  # the Modeler still persisted its own messages
