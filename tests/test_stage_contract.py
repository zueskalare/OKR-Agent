"""Stage-honest prompts + fabricated-tool-markup scrubber (WS25, A6).

Regression evidence: the system prompt advertised every tool as "available
via native tool calls" in ALL stages while only act binds them — on
claude-opus-5 the tool-less think stage wrote Claude-Code markup as prose AND
invented the observation (4/5 opus sessions at the first think message; 0/2
gpt), once riding to a fully fabricated "COMPLETE" report with zero tool
calls and an empty session dir (runs/jobs/session-nb-fem-20260806-135945).
"""

import logging

import pytest
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool

from AtomAgents import config
from AtomAgents.agents.model_building_agent import Modeler

from fakes import FakeLLM


@tool
def probe_tool(x: str) -> str:
    """A visible tool for prompt-listing assertions."""
    return "ok"


# The verbatim opus fabrication shape from the ledger (A6): markup plus a
# fully invented observation, at the very first think message.
FABRICATION = (
    "I'll start by checking the environment.\n"
    '<invoke name="Bash">\n'
    '<parameter name="command">pwd && ls -la</parameter>\n'
    "</invoke>\n"
    "Command 'pwd && ls -la' completed. Exit code: 0\n"
    "/workspace/project\n"
    "total 8\n"
    "drwxr-xr-x 2 root root 4096 Aug  6 18:11 .\n"
)


def make_modeler(tmp_path, script=()):
    return Modeler(
        llm=FakeLLM(list(script)),
        memory=None,
        Collages=["Supervisor", "END"],
        tools=[probe_tool],
        project_root=str(tmp_path),
    )


def test_tool_list_only_in_act_stage(tmp_path):
    m = make_modeler(tmp_path)
    act = m.system_prompt_for("act")
    assert "## Available Tools" in act
    assert "probe_tool" in act
    for stage in ("think", "plan", "report"):
        prompt = m.system_prompt_for(stage)
        assert "## Available Tools" not in prompt, stage
        assert "probe_tool" not in prompt, stage


def test_stage_contracts_state_the_rules(tmp_path):
    m = make_modeler(tmp_path)
    for stage in ("think", "plan"):
        p = m.system_prompt_for(stage)
        assert "reasoning only" in p, stage
        assert "You cannot call tools in this step" in p, stage
        assert "Never write tool-call syntax" in p, stage
    act = m.system_prompt_for("act")
    assert "bound to THIS call" in act
    report = m.system_prompt_for("report")
    assert "traceable to a real tool result" in report


def test_system_prompt_property_is_act_alias(tmp_path):
    m = make_modeler(tmp_path)
    assert m.system_prompt == m.system_prompt_for("act")


def test_think_strips_fabricated_tool_markup(tmp_path):
    m = make_modeler(tmp_path, script=[AIMessage(content=FABRICATION)])
    result = m.think({"messages": [HumanMessage(content="build the model")]})
    think_msg = result["messages"][-1]
    text = think_msg.content
    assert "I'll start by checking the environment." in text  # preamble kept
    assert "<invoke" not in text
    assert "/workspace/project" not in text  # the invented observation
    assert "[stage-guard]" in text


def test_stage_guard_env_off_preserves_content(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGE_GUARD", False)
    m = make_modeler(tmp_path)
    text = m._strip_fabricated_tool_calls(FABRICATION, "think")
    assert text == FABRICATION


@pytest.mark.parametrize(
    "marker",
    [
        "**Tool call:** Bash",
        "**Tool: Bash**",
        "<function_calls>",
        "```tool_code\nprint(1)\n```",
    ],
)
def test_scrubber_catches_observed_markup_variants(tmp_path, marker):
    m = make_modeler(tmp_path)
    out = m._strip_fabricated_tool_calls(f"Preamble text.\n{marker}\nfake output", "think")
    assert "fake output" not in out
    assert "[stage-guard]" in out
    assert "Preamble text." in out


def test_report_prompt_flags_zero_tool_evidence(tmp_path):
    m = make_modeler(tmp_path)
    no_tools = m._build_report_user_prompt({"messages": [HumanMessage(content="x")]})
    assert "No tool ran in this agent run" in no_tools
    with_tools = m._build_report_user_prompt(
        {
            "messages": [
                HumanMessage(content="x"),
                ToolMessage(content="ok", tool_call_id="t1", name="probe_tool"),
            ]
        }
    )
    assert "No tool ran in this agent run" not in with_tools
    assert "traceable to a real tool result" in with_tools


def test_report_prompt_keeps_end_first_class(tmp_path):
    # WS15 anti-runaway guard must survive the evidence gate verbatim.
    m = make_modeler(tmp_path)
    for state in (None, {"messages": []}):
        p = m._build_report_user_prompt(state)
        assert "`END` is a first-class outcome" in p


def test_report_scrubs_markup_and_logs_zero_evidence(tmp_path, caplog):
    m = make_modeler(tmp_path)
    fake_report = m._Report_Schema(
        report="All tasks complete ✅\n**Tool call:** Bash\nfake results table",
        send_to="END",
    )
    m.llm.script.append(fake_report)
    with caplog.at_level(logging.ERROR, logger="AtomAgents.Modeler"):
        result = m.report({"messages": [HumanMessage(content="go")]})
    final = result["messages"][-1].content
    assert "**Tool call:**" not in final
    assert "fake results table" not in final
    assert any("[evidence]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Review fixes: evidence-aware scrubbing, run-scoped evidence, act_final
# contract, and prompt-cache stability of the static prefix.
# ---------------------------------------------------------------------------


def test_scrubber_spares_report_describing_real_tool_activity(tmp_path):
    """A report that formats REAL tool activity as '**Tool call:**' bullets
    must survive intact when the run has tool evidence (truncating it
    destroyed genuine results and branded them fabricated)."""
    m = make_modeler(tmp_path)
    legit = (
        "## Actions\n**Tool call:** bash_run — mesh built (exit 0)\n"
        "**Tool call:** write_file — mesh_quick.py written\nAll artifacts on disk."
    )
    out = m._strip_fabricated_tool_calls(legit, "report", has_tool_evidence=True)
    assert out == legit
    # act prose summaries likewise
    out_act = m._strip_fabricated_tool_calls(legit, "act", has_tool_evidence=True)
    assert out_act == legit
    # think stays strict even with evidence — no tool can run there
    out_think = m._strip_fabricated_tool_calls(legit, "think", has_tool_evidence=True)
    assert "[stage-guard]" in out_think


def test_report_scrub_and_gate_are_run_scoped(tmp_path):
    """Inherited ToolMessages (ids in the run's inbound set) must not count
    as this run's evidence — one early tool run used to suppress the
    zero-evidence gate for every later fabricated report in the session."""
    inherited = ToolMessage(
        content="old result", tool_call_id="t0", name="probe_tool", id="inherited-1"
    )
    m = make_modeler(tmp_path)
    m._run_inbound_ids = {"inherited-1"}
    state = {"messages": [HumanMessage(content="go"), inherited]}
    assert m._run_tool_result_count(state["messages"]) == 0
    # fresh (this-run) tool result counts
    fresh = ToolMessage(content="new", tool_call_id="t1", name="probe_tool", id="new-1")
    assert m._run_tool_result_count(state["messages"] + [fresh]) == 1
    # and the report prompt gate fires on the run-scoped zero
    prompt = m._build_report_user_prompt(state, n_tool_results=0)
    assert "No tool ran in this agent run" in prompt


def test_act_final_contract_admits_no_tools(tmp_path):
    """The budget-exhausted wrap-up call binds NO tools; its contract must
    say so instead of claiming tools are 'bound to THIS call' (which invited
    the model to simulate a call and lose the final summary)."""
    m = make_modeler(tmp_path)
    final = m.system_prompt_for("act_final")
    assert "NO tools are bound" in final
    assert "bound to THIS call and are available right now" not in final
    assert "## Available Tools" not in final  # no tool list either
    assert "probe_tool" not in final


def test_static_prefix_identical_across_stages(tmp_path):
    """The FIRST system block is the Anthropic prompt-cache prefix: it must
    be byte-identical for every stage, with all stage variance confined to
    the second block."""
    m = make_modeler(tmp_path)
    static = m.system_prompt_static
    assert "Stage Contract" not in static
    assert "## Available Tools" not in static
    for stage in ("think", "plan", "act", "act_final", "report"):
        assert m.system_prompt_for(stage).startswith(static), stage
        assert "Stage Contract" in m.stage_system_prompt(stage), stage


def test_project_dir_block_scopes_evidence_to_this_session(tmp_path):
    """WS26: a Supervisor 'verified' another session's deliverables via
    unjailed cross-session reads and reported its own empty run COMPLETE —
    the static prompt now scopes evidence to THIS project directory."""
    m = make_modeler(tmp_path)
    static = m.system_prompt_static
    assert "Only files under THIS project directory" in static
    assert "NEVER this " in static or "NEVER this" in static
