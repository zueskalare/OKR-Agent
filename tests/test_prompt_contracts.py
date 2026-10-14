"""Prompt-file contracts the triage/loop-control work depends on — a cheap
guard against future prompt edits silently reopening the 'hi' hole."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
MD_SUP = REPO / "AtomAgents" / "templates" / "prompts" / "supervisor.yaml"
FEN_SUP = REPO / "AtomAgents" / "templates" / "fenicsx" / "supervisor.yaml"


@pytest.fixture(params=[MD_SUP, FEN_SUP], ids=["md", "fenicsx"])
def sup_yaml(request):
    return yaml.safe_load(request.param.read_text())["Supervisor"]


def test_supervisor_has_triage_key(sup_yaml):
    assert "triage" in sup_yaml
    triage = sup_yaml["triage"]
    for verdict in ("converse", "clarify", "task"):
        assert verdict in triage
    assert "When in doubt choose `task`" in triage


def test_supervisor_has_no_task_handling(sup_yaml):
    tpl = sup_yaml["template"]
    assert "No-Task Handling" in tpl
    assert "do NOT bootstrap OKRs" in tpl
    # both bootstrap triggers are gated on an actionable goal
    assert "actionable simulation goal" in tpl


def test_end_descriptions_sell_end():
    for p in (
        REPO / "AtomAgents" / "templates" / "prompts" / "end.yaml",
        REPO / "AtomAgents" / "templates" / "fenicsx" / "end.yaml",
    ):
        desc = yaml.safe_load(p.read_text())["END"]["description"]
        assert "no further agent action" in desc


def test_human_descriptions_gate_hitl():
    for p in (
        REPO / "AtomAgents" / "templates" / "prompts" / "human.yaml",
        REPO / "AtomAgents" / "templates" / "fenicsx" / "human.yaml",
    ):
        desc = yaml.safe_load(p.read_text())["Human"]["description"]
        assert "PAUSES" in desc and "ONLY when genuinely blocked" in desc


def test_report_bias_clause_gone():
    # the clause that steered reports away from END lives in code, not yaml —
    # assert against the built prompt via the class constant location
    from AtomAgents.agents import base_agent

    src = Path(base_agent.__file__).read_text()
    assert "still pick the most appropriate" not in src


# ---------------------------------------------------------------------------
# WS25 (A6/A7/A8): stage-honest tool protocol in the templates. The injected
# system-prompt block is the single source of truth for the tool inventory;
# templates must not hand-write tool lists, instruct textual call formats,
# reference tools that don't exist, or mandate README writes the framework
# performs in code.
# ---------------------------------------------------------------------------

ALL_TEMPLATES = sorted((REPO / "AtomAgents" / "templates").rglob("*.yaml"))


def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


def test_no_template_instructs_textual_tool_calls():
    # A8: fenicsx supervisor said "Use function-call JSON format" while the
    # injected block mandated native tool calls — two rival protocols in ONE
    # prompt, and the FEM flavor degraded worst on claude-opus-5.
    for path in ALL_TEMPLATES:
        for text in _walk_strings(yaml.safe_load(path.read_text())):
            assert "function-call JSON" not in text, path
            assert "JSON format; the framework will execute" not in text, path


def test_templates_do_not_hardcode_tool_inventories():
    for path in ALL_TEMPLATES:
        for text in _walk_strings(yaml.safe_load(path.read_text())):
            assert "## Available Tools" not in text, path


def test_templates_reference_only_real_tool_names():
    # A8 sibling: fenicsx modeler advertised read_file / execute_python /
    # list_directory — none of which exist (real: file_reader, code_runner).
    fictional = ("`read_file`", "`execute_python`", "`list_directory`")
    for path in ALL_TEMPLATES:
        content = path.read_text()
        for name in fictional:
            assert name not in content, f"{path} references fictional tool {name}"


def test_supervisor_templates_do_not_mandate_readme_tool_write(sup_yaml):
    tpl = sup_yaml["template"] + sup_yaml.get("act", "")
    assert "README (REQUIRED)" not in tpl
    assert "automatically" in tpl  # framework-mirrored, not agent-written


def test_supervisor_templates_do_not_hardcode_execution_boundary(sup_yaml):
    # A7 prompt half lives in code (Supervisor._policy_prompt), injected ONLY
    # when the coordinator-only policy is actually active — hard-coding it in
    # the template contradicted ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 (the A8
    # dual-protocol lie, reintroduced). See test_supervisor_tool_policy for
    # the injection behavior.
    tpl = sup_yaml["template"]
    assert "## Execution Boundary" not in tpl


def test_triage_never_converses_on_work_instructions(sup_yaml):
    # WS26: opus/fable triaged dispatch/resume orders as `converse` and
    # answered from a blocked-belief narrative — the pipeline (the only path
    # that can delegate) never ran again. Work instructions are always task.
    triage = " ".join(sup_yaml["triage"].split())
    assert "ALWAYS `task`" in triage
    assert "blocked, impossible, or complete" in triage
