"""Checkpoint serialization of ReportMessage (Tier 0.1).

langgraph's strict-msgpack machinery (CVE-2026-28277) currently only WARNS
about unregistered types — but under strict mode (the announced future
default) an unregistered subclass silently deserializes to a plain dict, no
exception, breaking every marker/isinstance check downstream. Both
checkpointer factories now register ReportMessage via
JsonPlusSerializer(allowed_msgpack_modules=[...]); these tests pin that
behavior in both the permissive present and the strict future.
"""

import warnings
from typing import Annotated, Any, List, Optional, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from AtomAgents.agents.messages import ReportMessage
from AtomAgents.core.simulation_agent_system import _session_checkpointer
from backend.openai_backend import _make_checkpointer


class S(TypedDict):
    messages: Annotated[List[Any], add_messages]
    send_to: Optional[str]


def _report_graph(checkpointer):
    def node(state):
        return {
            "messages": [
                ReportMessage("Supervisor:\nall done", name="Supervisor"),
                ToolMessage("tool out", tool_call_id="t1"),
            ],
            "send_to": "END",
        }

    g = StateGraph(S)
    g.add_node("agent", node)
    g.set_entry_point("agent")
    g.add_edge("agent", END)
    return g.compile(checkpointer=checkpointer)


@pytest.mark.parametrize("factory", [_make_checkpointer, _session_checkpointer])
def test_reportmessage_survives_checkpoint_roundtrip(tmp_path, factory):
    cfg = {"configurable": {"thread_id": "t1"}}
    app = _report_graph(factory(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the old code warned on deserialize
        app.invoke({"messages": [HumanMessage("go")]}, config=cfg)
        # fresh saver on the same file = a real deserialize from disk
        app2 = _report_graph(factory(tmp_path))
        state = app2.get_state(cfg)
    reports = [m for m in state.values["messages"] if isinstance(m, ReportMessage)]
    assert reports, "ReportMessage degraded to a plain message/dict on round-trip"
    assert reports[0].content == "Supervisor:\nall done"
    assert reports[0].name == "Supervisor"
    assert reports[0].additional_kwargs["atom_message_type"] == "report"
    # core message types still round-trip under the (now-strict) serializer
    assert any(isinstance(m, ToolMessage) for m in state.values["messages"])
    assert any(isinstance(m, HumanMessage) for m in state.values["messages"])


def test_allowlisted_serializer_is_strict_and_keeps_subclass():
    serde = JsonPlusSerializer(allowed_msgpack_modules=[ReportMessage])
    msg = ReportMessage("body", name="Supervisor")
    kind, blob = serde.dumps_typed(msg)
    out = serde.loads_typed((kind, blob))
    assert isinstance(out, ReportMessage)
    assert out.content == "body"


def test_strict_mode_without_reportmessage_degrades_silently():
    """Documents the failure mode the fix prevents: a strict serializer whose
    allowlist lacks ReportMessage yields a NON-ReportMessage silently — no
    exception. (An explicit allowlist puts the serializer in strict mode;
    LANGGRAPH_STRICT_MSGPACK is only read at module import, so the env-var
    route can't be tested in-process.)"""
    permissive = JsonPlusSerializer()
    kind, blob = permissive.dumps_typed(ReportMessage("body", name="Supervisor"))
    strict_without = JsonPlusSerializer(
        allowed_msgpack_modules=[("some.other.module", "OtherType")]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = strict_without.loads_typed((kind, blob))
    assert not isinstance(out, ReportMessage)
