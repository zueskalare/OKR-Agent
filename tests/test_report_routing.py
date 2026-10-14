"""Report routing: END-first-class coercion, normalization, handoff budget.

Regression evidence: the old fallback for an invalid report target was
``valid_next[0]`` — for the Supervisor that silently drafted the Modeler
(observed live on a bare "hi", session-atomagent-cdfb8780-*).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from AtomAgents import config
from AtomAgents.agents.base_agent import BaseAgent
from AtomAgents.agents.messages import ReportMessage
from AtomAgents.agents.model_building_agent import Modeler
from AtomAgents.agents.simulation_manager import Supervisor
from AtomAgents.utils.routing import handoff_router, reports_since_last_human

from fakes import FakeLLM


def make_agent(cls, tmp_path, collages, script):
    llm = FakeLLM(script)
    agent = cls(llm=llm, project_root=str(tmp_path), tools=[], Collages=collages)
    return agent, llm


# ---------------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**Next → Simulator**", "Simulator"),
        ("Next -> Modeler", "Modeler"),
        ("end", "END"),
        ("Done", "END"),
        ("Simulator Agent", "Simulator"),
        ("modeler agent", "Modeler"),
        ("user", "Human"),
        ("Nobody", "Nobody"),  # unmatched passes through so the guard fires
        ("END", "END"),
    ],
)
def test_normalize_next_target(raw, expected):
    valid = ["Modeler", "Simulator", "END", "Human"]
    assert BaseAgent._normalize_next_target(raw, valid) == expected


# ---------------------------------------------------------------------------
# Coercion in report()
# ---------------------------------------------------------------------------


def test_unknown_target_coerces_to_end_not_modeler(tmp_path):
    sup, llm = make_agent(
        Supervisor, tmp_path, ["Modeler", "Simulator", "END"], []
    )
    # model_construct bypasses the Literal validation, like a provider quirk
    llm.script = [sup._Report_Schema.model_construct(report="?", send_to="Nobody")]
    result = sup.report({"messages": [HumanMessage(content="hi")]})
    assert result["send_to"] == "END"  # NEVER valid_next[0] (= Modeler)


def test_self_target_hands_back_to_supervisor(tmp_path):
    mod, llm = make_agent(
        Modeler, tmp_path, ["Supervisor", "Simulator", "END"], []
    )
    llm.script = [mod._Report_Schema.model_construct(report="more", send_to="Modeler")]
    result = mod.report({"messages": [HumanMessage(content="build it")]})
    assert result["send_to"] == "Supervisor"


def test_report_prompt_sells_end(tmp_path):
    sup, _ = make_agent(Supervisor, tmp_path, ["Modeler", "Simulator", "END"], [])
    prompt = sup._build_report_user_prompt()
    assert "`END` is a first-class outcome" in prompt
    assert "still pick the most appropriate" not in prompt


def test_report_prompt_mentions_human_only_when_offered(tmp_path):
    sup, _ = make_agent(
        Supervisor, tmp_path, ["Modeler", "Simulator", "END", "Human"], []
    )
    assert "Choose `Human` only" in sup._build_report_user_prompt()
    sup2, _ = make_agent(Supervisor, tmp_path, ["Modeler", "Simulator", "END"], [])
    assert "Choose `Human`" not in sup2._build_report_user_prompt()


# ---------------------------------------------------------------------------
# Handoff budget
# ---------------------------------------------------------------------------


def test_reports_counted_since_newest_human_only():
    msgs = [
        HumanMessage("turn 1"),
        ReportMessage("r1"),
        ReportMessage("r2"),
        HumanMessage("turn 2"),
        AIMessage("think"),
        ReportMessage("r3"),
        # DB round-trip shape: plain AIMessage carrying the marker
        AIMessage("r4", additional_kwargs={"atom_message_type": "report"}),
    ]
    assert reports_since_last_human(msgs) == 2  # cross-turn-leak regression


def test_handoff_router_caps(monkeypatch):
    monkeypatch.setattr(config, "MAX_HANDOFFS", 2)
    msgs = [HumanMessage("go"), ReportMessage("r1"), ReportMessage("r2")]
    state = {"messages": msgs, "send_to": "Modeler"}
    assert handoff_router(state, agent="Supervisor") == "END"
    state_under = {"messages": msgs[:2], "send_to": "Modeler"}
    assert handoff_router(state_under, agent="Supervisor") == "Modeler"


def test_report_budget_forces_end_with_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_HANDOFFS", 3)
    sup, llm = make_agent(
        Supervisor, tmp_path, ["Modeler", "Simulator", "END"], []
    )
    llm.script = [sup._Report_Schema(report="keep going", send_to="Modeler")]
    msgs = [HumanMessage("go"), ReportMessage("r1"), ReportMessage("r2")]
    result = sup.report({"messages": msgs})
    assert result["send_to"] == "END"
    assert "[loop-guard]" in result["messages"][-1].content
