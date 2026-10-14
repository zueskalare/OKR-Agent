"""Run-scoped loop-guard notices (WS25, A5).

Regression evidence: the act-budget notice ("Tools are no longer available")
persisted into SHARED outer history and the memory DB; a downstream agent
with a fresh budget and tools bound read it as a runtime property and refused
to call tools — a 28-handoff Supervisor⇄Simulator ping-pong with zero tool
calls (runs/jobs/session-nb-md-20260806-131234). Internal notices now stay
inside the owning agent's run: dropped from the returned delta and never
persisted, while report() still sees them.
"""

import copy

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from AtomAgents import config
from AtomAgents.agents.model_building_agent import Modeler
from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory

from fakes import FakeLLM
from test_memory_schema import StubEmb


@tool
def noisy_tool(x: str) -> str:
    """A tool the fake model wants to call forever."""
    return "ok"


class FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, msgs, *args, **kwargs):
        fields = self.schema.model_fields
        kwargs = {}
        if "plan" in fields:
            kwargs["plan"] = "keep going"
        if "send_to" in fields:
            args_ = getattr(fields["send_to"].annotation, "__args__", ("act",))
            kwargs["send_to"] = "act" if "act" in args_ else args_[0]
        if "report" in fields:
            kwargs["report"] = "wrapped up after budget"
            sargs = getattr(fields["send_to"].annotation, "__args__", ("END",))
            kwargs["send_to"] = sargs[0]
        return self.schema(**kwargs)


class LoopingLLM:
    """Calls the tool with DIFFERENT args every act — evades the
    identical-retry guard; only the act budget can stop it."""

    def __init__(self):
        self.n = 0
        self.tools_bound = False

    def invoke(self, msgs, *args, **kwargs):
        if self.tools_bound:
            self.n += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "noisy_tool",
                        "args": {"x": f"call-{self.n}"},
                        "id": f"t{self.n}",
                    }
                ],
            )
        return AIMessage(content="final summary without tools")

    def bind_tools(self, tools):
        bound = copy.copy(self)
        bound.tools_bound = True
        return bound

    def with_structured_output(self, schema):
        return FakeStructured(schema)


def run_budget_exhausting_modeler(tmp_path, monkeypatch, memory=None):
    monkeypatch.setattr(config, "MAX_ACT_CYCLES", 3)
    m = Modeler(
        llm=LoopingLLM(),
        memory=memory,
        Collages=["Supervisor", "END"],
        tools=[noisy_tool],
        project_root=str(tmp_path),
        config={"recursion_limit": 60},
    )
    result = m.run({"messages": [HumanMessage(content="loop forever please")]})
    # The budget (not the recursion limit) ended the run, in a report.
    assert any(type(x).__name__ == "ReportMessage" for x in result["messages"])
    return result


def test_act_budget_notice_not_in_returned_delta(tmp_path, monkeypatch):
    result = run_budget_exhausting_modeler(tmp_path, monkeypatch)
    leaked = [
        m
        for m in result["messages"]
        if "act budget" in str(getattr(m, "content", "")).lower()
        or "loop-guard" in str(getattr(m, "content", ""))
    ]
    assert not leaked, [str(m.content)[:120] for m in leaked]


def test_act_budget_notice_not_persisted(tmp_path, monkeypatch):
    mem = SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "memory.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )
    run_budget_exhausting_modeler(tmp_path, monkeypatch, memory=mem)
    rows = mem.get_raw_messages_by_agent(agent_name="Modeler")
    texts = [str(r) for r in rows]
    assert rows, "run should persist real messages"
    assert not any("loop-guard" in t or "no longer available" in t for t in texts)


def test_downstream_agent_history_is_clean(tmp_path, monkeypatch):
    """Feed the exhausted run's returned delta to a fresh Simulator-shaped
    check: nothing internal arrives, so nothing can poison its stages."""
    result = run_budget_exhausting_modeler(tmp_path, monkeypatch)
    from AtomAgents.agents.base_agent import BaseAgent

    assert not any(BaseAgent._is_internal(m) for m in result["messages"])


def test_loop_guard_notice_is_agent_scoped_text(tmp_path):
    m = Modeler(
        llm=FakeLLM(),
        memory=None,
        Collages=["Supervisor", "END"],
        tools=[noisy_tool],
        project_root=str(tmp_path),
    )
    note = m._loop_guard_message(
        f"[loop-guard] {m.__name__}'s own act budget for THIS run is exhausted "
        "— tools are withheld for the remainder of this run only."
    )
    assert note.additional_kwargs["atom_message_type"] == "loop_guard"
    assert m._is_internal(note)
    assert "Modeler" in note.content
    assert "THIS run" in note.content


def test_report_message_is_not_internal(tmp_path):
    from AtomAgents.agents.base_agent import BaseAgent
    from AtomAgents.agents.messages import ReportMessage

    assert not BaseAgent._is_internal(ReportMessage(content="real report"))
    assert not BaseAgent._is_internal(AIMessage(content="plain message"))


def test_identical_retry_note_is_internal_and_unpersisted(tmp_path, monkeypatch):
    """Three identical tool calls trigger the retry note; it must be marked
    internal and skipped by _save_run_messages."""
    mem = SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "memory.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )

    class SameCallLLM(LoopingLLM):
        def invoke(self, msgs, *args, **kwargs):
            if self.tools_bound:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "noisy_tool", "args": {"x": "same"}, "id": "t1"}
                    ],
                )
            return AIMessage(content="final summary")

    m = Modeler(
        llm=SameCallLLM(),
        memory=mem,
        Collages=["Supervisor", "END"],
        tools=[noisy_tool],
        project_root=str(tmp_path),
        config={"recursion_limit": 60},
    )
    result = m.run({"messages": [HumanMessage(content="retry forever")]})
    assert any(type(x).__name__ == "ReportMessage" for x in result["messages"])
    # Not in the returned delta...
    assert not any(
        "identical" in str(getattr(x, "content", "")) for x in result["messages"]
    )
    # ...and not in the DB.
    rows = mem.get_raw_messages_by_agent(agent_name="Modeler")
    assert not any("identical" in str(r) for r in rows)


def test_loop_guard_notice_not_published_on_wire(tmp_path, monkeypatch):
    """Review fix: internal notices must not ride the incremental custom
    stream either — published as message events they could even ADOPT the
    open act token segment and replace the real act output item in the
    client transcript."""
    captured = []

    def fake_writer():
        return lambda payload: captured.append(payload)

    from AtomAgents.agents.base_agent import BaseAgent

    monkeypatch.setattr(BaseAgent, "_stream_writer", staticmethod(fake_writer))
    monkeypatch.setattr(config, "TOKEN_STREAM", True)
    run_budget_exhausting_modeler(tmp_path, monkeypatch)

    message_events = [p for p in captured if p.get("event") == "message"]
    assert message_events, "expected message events on the custom stream"
    leaked = [
        p
        for p in message_events
        if "loop-guard" in str(getattr(p.get("message"), "content", ""))
    ]
    assert not leaked, [str(p["message"].content)[:120] for p in leaked]
