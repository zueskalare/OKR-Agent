"""Human-in-the-loop: interrupt pauses the run, the next message resumes it
back to the asking agent; anonymous threads degrade to END instead of
stranding an unresumable interrupt."""

from typing import Annotated, Any, List, Optional, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from AtomAgents.agents.human_interput import human_node
from AtomAgents.agents.messages import ReportMessage


class S(TypedDict):
    messages: Annotated[List[Any], add_messages]
    send_to: Optional[str]


@pytest.fixture()
def app():
    def asker(state):
        return {
            "messages": [
                ReportMessage("Need the target temperature. What T?", name="Modeler")
            ],
            "send_to": "Human",
        }

    def modeler(state):
        return {"messages": [], "send_to": "END"}

    g = StateGraph(S)
    g.add_node("Asker", asker)
    g.add_node("Human", human_node)
    g.add_node("Modeler", modeler)
    g.set_entry_point("Asker")
    g.add_conditional_edges("Asker", lambda s: s.get("send_to"), {"Human": "Human"})
    g.add_conditional_edges(
        "Human", lambda s: s.get("send_to"), {"Modeler": "Modeler", "END": END}
    )
    g.add_edge("Modeler", END)
    return g.compile(checkpointer=InMemorySaver())


def _interrupts(app, cfg):
    return [i for t in app.get_state(cfg).tasks for i in (t.interrupts or ())]


def test_interrupt_and_resume_to_asker(app):
    cfg = {"configurable": {"thread_id": "session-x"}}
    app.invoke({"messages": [HumanMessage("run it")]}, config=cfg)

    pending = _interrupts(app, cfg)
    assert pending, "expected a pending interrupt"
    assert pending[0].value["from"] == "Modeler"
    assert "temperature" in pending[0].value["question"]

    out = app.invoke(Command(resume="300 K please"), config=cfg)
    human_msgs = [m for m in out["messages"] if getattr(m, "name", None) == "human"]
    assert human_msgs and "300 K" in str(human_msgs[-1].content)
    assert not _interrupts(app, cfg)  # resumed to completion


def test_anon_thread_ends_instead_of_interrupting(app):
    cfg = {"configurable": {"thread_id": "anon-123"}}
    out = app.invoke({"messages": [HumanMessage("run it")]}, config=cfg)
    assert not _interrupts(app, cfg)
    notes = [m for m in out["messages"] if getattr(m, "name", None) == "Human"]
    assert notes and "HITL unavailable" in str(notes[-1].content)


def test_resume_update_is_delta_only(app):
    """R1 at the human node: the resume update carries ONLY the new
    HumanMessage — re-returning the full history replayed the prior turn
    through the backend's per-request dedupe (WS15 known issue)."""
    cfg = {"configurable": {"thread_id": "session-delta"}}
    app.invoke({"messages": [HumanMessage("run it")]}, config=cfg)
    assert _interrupts(app, cfg)

    updates = list(
        app.stream(Command(resume="42 K"), config=cfg, stream_mode="updates")
    )
    human_updates = [u["Human"] for u in updates if "Human" in u]
    assert human_updates, f"expected a Human node update, got {updates}"
    msgs = human_updates[0]["messages"]
    assert len(msgs) == 1 and "42 K" in str(msgs[0].content)


def test_anon_update_is_delta_only(app):
    cfg = {"configurable": {"thread_id": "anon-delta"}}
    updates = list(
        app.stream(
            {"messages": [HumanMessage("run it")]}, config=cfg, stream_mode="updates"
        )
    )
    human_updates = [u["Human"] for u in updates if "Human" in u]
    assert human_updates
    msgs = human_updates[0]["messages"]
    assert len(msgs) == 1 and "HITL unavailable" in str(msgs[0].content)


def test_pending_interrupt_helper(app):
    import asyncio

    from backend.openai_backend import _apending_interrupt

    cfg = {"configurable": {"thread_id": "session-y"}}
    assert asyncio.run(_apending_interrupt(app, cfg)) is False  # untouched thread
    app.invoke({"messages": [HumanMessage("run it")]}, config=cfg)
    assert asyncio.run(_apending_interrupt(app, cfg)) is True
    app.invoke(Command(resume="answer"), config=cfg)
    assert asyncio.run(_apending_interrupt(app, cfg)) is False
