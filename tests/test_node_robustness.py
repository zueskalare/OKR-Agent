"""Outer-graph node robustness (WS18.3): retry + graceful error handler,
with HITL interrupts verified exempt from both."""

from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, RetryPolicy, interrupt

from AtomAgents.utils.routing import agent_error_handler


class S(TypedDict):
    messages: Annotated[List[Any], add_messages]
    send_to: Optional[str]


def _graph_with_defaults(node_fn, checkpointer=None):
    g = StateGraph(S)
    g.set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.01),
        error_handler=agent_error_handler,
    )
    g.add_node("Modeler", node_fn)
    g.set_entry_point("Modeler")
    g.add_conditional_edges("Modeler", lambda s: s.get("send_to"), {"END": END})
    return g.compile(checkpointer=checkpointer)


def test_transient_failure_retried_then_succeeds():
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("proxy hiccup")
        return {"messages": [], "send_to": "END"}

    out = _graph_with_defaults(flaky).invoke({"messages": []})
    assert calls["n"] == 2
    assert out["send_to"] == "END"


def test_exhausted_retries_end_gracefully_with_error_note():
    def broken(state):
        raise RuntimeError("solver exploded")

    out = _graph_with_defaults(broken).invoke({"messages": []})
    assert out["send_to"] == "END"  # graceful END, not an exception
    note = out["messages"][-1]
    assert "[node-error]" in note.content
    assert "solver exploded" in note.content
    assert note.name == "Modeler"


def test_interrupt_exempt_from_retry_and_handler():
    calls = {"n": 0}

    def human(state):
        calls["n"] += 1
        ans = interrupt({"q": "?"})
        return {"messages": [HumanMessage(str(ans))], "send_to": "END"}

    app = _graph_with_defaults(human, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    app.invoke({"messages": []}, config=cfg)
    pending = [i for t in app.get_state(cfg).tasks for i in (t.interrupts or ())]
    assert pending and calls["n"] == 1  # no retry storm on interrupt
    out = app.invoke(Command(resume="answer"), config=cfg)
    assert any("answer" in str(m.content) for m in out["messages"])
    # no [node-error] note anywhere — the handler never fired
    assert not any("[node-error]" in str(m.content) for m in out["messages"])
