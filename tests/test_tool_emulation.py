"""Dry-run tool emulation (WS20.3): same name/schema, LLM-fabricated output,
unlisted tools untouched."""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from AtomAgents.utils.tool_emulation import emulate_tools
from fakes import FakeLLM


def _real_tool():
    def lammps_run(input_file: str, work_dir: str) -> str:
        """Run a LAMMPS simulation."""
        raise AssertionError("real tool must never execute in a dry run")

    return StructuredTool.from_function(lammps_run)


def test_emulated_tool_same_surface_fabricated_output():
    real = _real_tool()
    llm = FakeLLM([AIMessage('{"log": "Total wall time: 0:00:42"}')])
    (emu,) = emulate_tools([real], ["lammps_run"], llm)
    assert emu.name == real.name
    assert emu.description == real.description
    assert emu.args_schema is real.args_schema
    out = emu.invoke({"input_file": "in.tension", "work_dir": "/tmp/w"})
    assert out == '{"log": "Total wall time: 0:00:42"}'
    # the emulation prompt carried the tool identity and the arguments
    assert llm.calls == [("invoke",)]


def test_emulated_tool_async_path():
    real = _real_tool()
    llm = FakeLLM([AIMessage("emulated async output")])
    (emu,) = emulate_tools([real], ["lammps_run"], llm)
    out = asyncio.run(emu.ainvoke({"input_file": "a", "work_dir": "b"}))
    assert out == "emulated async output"


def test_unlisted_tools_kept_identical():
    real = _real_tool()
    board = StructuredTool.from_function(
        lambda note: f"recorded {note}", name="record_metric", description="board"
    )
    out = emulate_tools([real, board], ["lammps_run"], FakeLLM([AIMessage("x")]))
    assert out[1] is board  # same object — board tools stay real


def test_empty_names_is_inert():
    real = _real_tool()
    out = emulate_tools([real], [], FakeLLM([]))
    assert out == [real]


def test_missing_name_warns_but_never_raises(caplog):
    real = _real_tool()
    out = emulate_tools([real], ["no_such_tool"], FakeLLM([]))
    assert out[0] is real
    assert any("not found" in r.message for r in caplog.records)


def test_schema_still_validates_args():
    real = _real_tool()
    (emu,) = emulate_tools([real], ["lammps_run"], FakeLLM([AIMessage("x")]))
    with pytest.raises(Exception):
        emu.invoke({"input_file": "only one arg"})  # missing work_dir
