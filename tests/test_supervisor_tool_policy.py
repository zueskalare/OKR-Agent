"""Coordinator-only Supervisor act toolset (WS25, A7).

Regression evidence: 5/5 claude-opus-5 runs were single-agent — the
Supervisor did every task itself (`{'Supervisor': 18}`) despite strong
delegation prompts, because its act stage bound the full execution toolset.
The constraint is now structural: execution/write tools are withheld from
the Supervisor's act binding (ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 restores
the legacy shape); Modeler/Simulator are unaffected.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from AtomAgents import config
from AtomAgents.agents.model_building_agent import Modeler
from AtomAgents.agents.simulation_manager import Supervisor
from AtomAgents.agents.simulator_agent import SimulatorAgent
from AtomAgents.utils.tool_policy import (
    EXECUTION_TOOLS,
    READ_ONLY_TOOLS,
    is_execution_tool,
    split_tools,
)

from fakes import FakeLLM


def _mk_tool(name: str):
    def _f(x: str = "") -> str:
        return "ok"

    return StructuredTool.from_function(_f, name=name, description=f"{name} tool")


MIXED = [
    "code_runner",
    "bash_run",
    "lammps_run",
    "write_file",
    "file_reader",
    "okr_update",
]


def make_supervisor(tmp_path, tool_names=MIXED, script=()):
    return Supervisor(
        llm=FakeLLM(list(script)),
        memory=None,
        project_root=str(tmp_path),
        tools=[_mk_tool(n) for n in tool_names],
        Collages=["Modeler", "Simulator", "END"],
    )


def _names(tools):
    return {t.name for t in tools}


def test_supervisor_withholds_execution_tools(tmp_path):
    sup = make_supervisor(tmp_path)
    assert _names(sup.act_tools) == {"file_reader", "okr_update"}


def test_supervisor_act_binds_only_coordination_tools(tmp_path):
    sup = make_supervisor(
        tmp_path, script=[AIMessage(content="Decision: assign J01 to Modeler")]
    )
    sup.act({"messages": [HumanMessage(content="assign the mesh work")]})
    assert sup.llm.bound_tools is not None, "act must still bind coordination tools"
    assert _names(sup.llm.bound_tools) == {"file_reader", "okr_update"}


def test_supervisor_exec_tools_env_flag_restores_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SUPERVISOR_EXEC_TOOLS", True)
    sup = make_supervisor(tmp_path)
    assert _names(sup.act_tools) == set(MIXED)


def test_modeler_and_simulator_keep_all_tools(tmp_path):
    for cls in (Modeler, SimulatorAgent):
        agent = cls(
            llm=FakeLLM(),
            memory=None,
            project_root=str(tmp_path),
            tools=[_mk_tool(n) for n in MIXED],
            Collages=["Supervisor", "END"],
        )
        assert _names(agent.act_tools) == set(MIXED), cls.__name__


def test_supervisor_act_prompt_lists_only_bound_tools(tmp_path):
    # A6/A7 coherence: never advertise what isn't bound.
    sup = make_supervisor(tmp_path)
    act_prompt = sup.system_prompt_for("act")
    assert "file_reader" in act_prompt
    assert "okr_update" in act_prompt
    for withheld in ("code_runner", "bash_run", "lammps_run", "write_file"):
        assert withheld not in act_prompt, withheld


def test_supervisor_tool_node_narrowed(tmp_path):
    sup = make_supervisor(tmp_path)
    node_tools = set(sup.ToolNode.tools_by_name)
    assert "file_reader" in node_tools
    assert "okr_update" in node_tools
    for withheld in ("code_runner", "bash_run", "lammps_run", "write_file"):
        assert withheld not in node_tools, withheld


# Full real inventory, pinned name by name: an MCP tool rename must fail here
# instead of silently changing the Supervisor's capabilities.
@pytest.mark.parametrize(
    "name",
    sorted(EXECUTION_TOOLS),
)
def test_real_execution_tools_classified(name):
    assert is_execution_tool(_mk_tool(name)), name


@pytest.mark.parametrize(
    "name",
    sorted(READ_ONLY_TOOLS),
)
def test_real_coordination_tools_classified(name):
    assert not is_execution_tool(_mk_tool(name)), name


def test_classification_matches_live_mcp_inventory():
    """The policy name sets must cover the ACTUAL @mcp.tool registrations —
    a tool added to the service but missing from the policy is only covered
    by the schema heuristic, which this test surfaces."""
    import re
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent
        / "AtomAgents"
        / "mcp_service"
        / "mcp_service.py"
    ).read_text()
    registered = re.findall(r"@mcp\.tool\(\)\s*\ndef\s+(\w+)\(", src)
    assert registered, "no @mcp.tool registrations found — did the service move?"
    known = EXECUTION_TOOLS | READ_ONLY_TOOLS
    missing = [n for n in registered if n not in known]
    assert not missing, f"MCP tools not covered by tool_policy name sets: {missing}"


def test_split_tools_preserves_order(tmp_path):
    tools = [_mk_tool(n) for n in MIXED]
    coordination, execution = split_tools(tools)
    assert [t.name for t in coordination] == ["file_reader", "okr_update"]
    assert [t.name for t in execution] == [
        "code_runner",
        "bash_run",
        "lammps_run",
        "write_file",
    ]


# ---------------------------------------------------------------------------
# Execution Boundary injection (review fix): the boundary statement lives in
# code and tracks the ACTUAL policy, so the prompt can never claim "NO
# execution tools" while bash_run/write_file are listed as bound (the legacy
# ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 mode shipped exactly that contradiction
# when the text was hard-coded in the yaml template).
# ---------------------------------------------------------------------------


def test_execution_boundary_injected_when_coordinator_only(tmp_path):
    sup = make_supervisor(tmp_path)
    act_prompt = sup.system_prompt_for("act")
    assert "## Execution Boundary" in act_prompt
    assert "NO execution or write tools" in act_prompt
    assert "delegating IS the action" in act_prompt


def test_execution_boundary_absent_in_legacy_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SUPERVISOR_EXEC_TOOLS", True)
    sup = make_supervisor(tmp_path)
    act_prompt = sup.system_prompt_for("act")
    # No contradiction: exec tools are listed AND no boundary text denies them.
    assert "## Execution Boundary" not in act_prompt
    assert "NO execution or write tools" not in act_prompt
    assert "bash_run" in act_prompt
    assert "write_file" in act_prompt


def test_act_tools_snapshot_ignores_post_construction_flag_flip(tmp_path, monkeypatch):
    """The policy is resolved ONCE at construction: the ToolNode registry and
    the bind set always agree; flipping the flag mid-life changes neither."""
    sup = make_supervisor(tmp_path)
    before = _names(sup.act_tools)
    monkeypatch.setattr(config, "SUPERVISOR_EXEC_TOOLS", True)
    assert _names(sup.act_tools) == before  # unchanged — snapshot semantics
    assert set(sup.ToolNode.tools_by_name) == before


def test_unknown_path_only_tool_stays_with_supervisor(tmp_path):
    """A read-only foreign tool that merely takes a `path` argument must not
    be withheld (generic path params are not an execution signal)."""

    def _reader(path: str) -> str:
        return "contents"

    reader = StructuredTool.from_function(
        _reader, name="page_reader", description="read a page"
    )
    assert not is_execution_tool(reader)
    coordination, execution = split_tools([reader])
    assert [t.name for t in coordination] == ["page_reader"]
    assert execution == []


def test_unknown_tool_with_exec_signals_is_withheld(tmp_path):
    def _writer(dest: str, content: str = "") -> str:
        return "ok"

    writer = StructuredTool.from_function(
        _writer, name="page_saver", description="save a page"
    )
    assert is_execution_tool(writer)  # `dest` param
    assert is_execution_tool(_mk_tool("report_uploader"))  # name hint


def test_policy_prompt_declares_send_to_dispatch(tmp_path):
    """WS26: opus/fable concluded delegation was impossible because no
    'handoff tool' exists — send_to IS the dispatch channel and must be
    stated as such wherever the boundary is injected."""
    sup = make_supervisor(tmp_path)
    p = sup.system_prompt_for("act")
    assert "## Dispatch mechanism" in p
    assert "`send_to`" in p
    assert "protocol violation" in p
    assert "BY DESIGN" in p
