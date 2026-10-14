"""Supervisor triage gate: trivial turns answer directly in ONE LLM call.

Regression evidence: a bare "hi" used to bootstrap OKRs and draft the
Modeler into 26-75 messages of invented intake work
(runs/jobs/session-atomagent-cdfb8780-*, PLAN_BOOK WS13 incident).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from AtomAgents import config
from AtomAgents.agents.messages import ReportMessage
from AtomAgents.agents.simulation_manager import (
    Supervisor,
    SupervisorPlanOut,
    TriageOut,
    UpdateTask,
)
from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory

from fakes import FakeLLM
from test_memory_schema import StubEmb


@pytest.fixture()
def mem(tmp_path):
    return SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "memory.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )


def make_supervisor(tmp_path, mem, script):
    llm = FakeLLM(script)
    sup = Supervisor(
        llm=llm,
        memory=mem,
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    return sup, llm


def test_hi_converses_in_one_llm_call(tmp_path, mem):
    sup, llm = make_supervisor(
        tmp_path, mem, [TriageOut(verdict="converse", reply="Hi! I coordinate an MD team.")]
    )
    result = sup.run({"messages": [HumanMessage(content="hi")]})

    assert len(llm.calls) == 1, f"expected 1 LLM call, saw {llm.calls}"
    assert result["send_to"] == "END"
    final = result["messages"][-1]
    assert isinstance(final, ReportMessage)
    assert final.content.startswith("# Supervisor - reply")
    assert "Hi! I coordinate" in final.content
    assert mem.get_all_tasks() == []  # no OKR bootstrap
    assert llm.bound_tools is None  # tools never bound
    assert sup._short_circuit is None  # cleared for the next turn


def test_converse_without_reply_uses_default_greeting(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [TriageOut(verdict="converse", reply="")])
    result = sup.run({"messages": [HumanMessage(content="hello?")]})
    assert result["send_to"] == "END"
    assert "Supervisor" in result["messages"][-1].content
    assert len(llm.calls) == 1


def test_clarify_asks_and_ends(tmp_path, mem):
    sup, llm = make_supervisor(
        tmp_path,
        mem,
        [TriageOut(verdict="clarify", reply="Which material, and what property?")],
    )
    result = sup.run({"messages": [HumanMessage(content="simulate something")]})

    assert len(llm.calls) == 1
    assert result["send_to"] == "END"
    assert "?" in result["messages"][-1].content
    assert mem.get_all_tasks() == []


def test_task_verdict_runs_pipeline(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [])
    llm.script = [
        TriageOut(verdict="task"),
        AIMessage(content="# Supervisor - think\nreal work"),
        UpdateTask(update_task=False),
        SupervisorPlanOut(plan="assign the build", send_to="act"),
        AIMessage(content="# Supervisor - act\nDecision: assign Modeler"),
        sup._Report_Schema(report="handing off", send_to="Modeler"),
    ]
    result = sup.run({"messages": [HumanMessage(content="tensile test on a CNT bundle")]})

    assert len(llm.calls) == 6
    assert result["send_to"] == "Modeler"
    assert isinstance(result["messages"][-1], ReportMessage)


def test_triage_skipped_on_colleague_handoff(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [])
    out = sup.triage(
        {"messages": [ReportMessage(content="Modeler:\ndone", name="Modeler")]}
    )
    assert out["send_to"] == "think"
    assert llm.calls == []  # deterministic skip — no LLM call


def test_triage_disabled_by_config(tmp_path, mem, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TRIAGE", False)
    sup, llm = make_supervisor(tmp_path, mem, [])
    out = sup.triage({"messages": [HumanMessage(content="hi")]})
    assert out["send_to"] == "think"
    assert llm.calls == []


def test_triage_llm_failure_falls_through(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [RuntimeError("provider down")])
    out = sup.triage({"messages": [HumanMessage(content="hi")]})
    assert out["send_to"] == "think"  # the gate must never break a run
    assert sup._short_circuit is None


def test_clarify_without_question_biases_to_task(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [TriageOut(verdict="clarify", reply="")])
    out = sup.triage({"messages": [HumanMessage(content="do the thing")]})
    assert out["send_to"] == "think"
    assert sup._triage_verdict == "task"


def test_arun_equivalence_with_run(tmp_path, mem):
    """arun is the real implementation; run() wraps it — same outcome on the
    same script (WS19.3)."""
    import asyncio

    script = [TriageOut(verdict="converse", reply="Hi there!")]
    sup_a, llm_a = make_supervisor(tmp_path, mem, list(script))
    out_async = asyncio.run(sup_a.arun({"messages": [HumanMessage(content="hi")]}))

    sup_s, llm_s = make_supervisor(tmp_path, mem, list(script))
    out_sync = sup_s.run({"messages": [HumanMessage(content="hi")]})

    assert out_async["send_to"] == out_sync["send_to"] == "END"
    assert (
        out_async["messages"][-1].content == out_sync["messages"][-1].content
    )
    assert llm_a.calls == llm_s.calls == [("structured", "TriageOut")]


def test_run_from_inside_a_running_loop(tmp_path, mem):
    """run() called on a loop thread must not deadlock — it hops to a worker
    thread with a copied context (T3 guard)."""
    import asyncio

    sup, llm = make_supervisor(
        tmp_path, mem, [TriageOut(verdict="converse", reply="looped!")]
    )

    async def call_run_on_loop():
        return sup.run({"messages": [HumanMessage(content="hi")]})

    out = asyncio.run(call_run_on_loop())
    assert out["send_to"] == "END"
    assert "looped!" in out["messages"][-1].content


def test_structured_invoke_retries_with_error_feedback(tmp_path, mem):
    """A failed structured call is retried with the error appended as
    feedback; the retry succeeding rescues the run (WS18.5)."""
    sup, llm = make_supervisor(
        tmp_path,
        mem,
        [ValueError("not valid json"), TriageOut(verdict="converse", reply="ok!")],
    )
    out = sup.triage({"messages": [HumanMessage(content="hi")]})
    assert out["send_to"] == "end"  # rescued by the retry
    assert len(llm.calls) == 2
    assert sup._short_circuit["reply"] == "ok!"


def test_structured_invoke_raises_after_retry_budget(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [ValueError("a"), ValueError("b")])
    import pytest as _pytest

    with _pytest.raises(Exception):
        sup._structured_invoke(TriageOut, [HumanMessage(content="x")], retries=1)


def test_agent_run_emits_working_via_custom_stream(tmp_path, mem):
    """The 'working' typing indicator now rides langgraph's custom stream
    (emitted from BaseAgent.run), replacing the debug task-event scrape."""
    from langgraph.graph import END, StateGraph

    sup, llm = make_supervisor(
        tmp_path, mem, [TriageOut(verdict="converse", reply="hi!")]
    )
    g = StateGraph(Supervisor.SubState)
    g.add_node("Supervisor", sup.run)
    g.set_entry_point("Supervisor")
    g.add_edge("Supervisor", END)
    app = g.compile()

    events = list(
        app.stream(
            {"messages": [HumanMessage(content="hi")]},
            stream_mode=["updates", "custom"],
        )
    )
    # Since WS19.6 the custom channel also carries incremental message
    # events; this test pins the WORKING emission specifically.
    working = [
        p
        for mode, p in events
        if mode == "custom" and isinstance(p, dict) and p.get("event") == "working"
    ]
    assert working == [{"event": "working", "agent": "Supervisor"}]


def test_force_report_skips_plan_probe(tmp_path, mem):
    sup, llm = make_supervisor(tmp_path, mem, [])
    sup._force_report = True
    out = sup.plan({"messages": [HumanMessage(content="x")]})
    assert out["send_to"] == "report"
    assert llm.calls == []  # zero LLM calls on a forced wrap-up
