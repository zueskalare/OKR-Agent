"""Responses-API content-block tolerance.

With ``output_version="responses/v1"`` (langchain-openai on the Responses
path) ``AIMessage.content`` is ALWAYS a block list, typically led by a
``{"type": "reasoning"}`` item with an ``encrypted_content`` blob. Every
text choke point must flatten that shape instead of crashing or persisting
multi-KB opaque payloads.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from AtomAgents.agents.messages import ReportMessage
from AtomAgents.agents.simulation_manager import Supervisor, TriageOut
from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory

from fakes import FakeLLM
from test_memory_schema import StubEmb

# A realistic responses/v1 content list.
BLOCKS = [
    {
        "type": "reasoning",
        "id": "rs_123",
        "summary": [],
        "encrypted_content": "gAAAA" + "x" * 500,
    },
    {"type": "text", "text": "# Supervisor - think\nreal analysis", "id": "msg_1"},
]


@pytest.fixture()
def mem(tmp_path):
    return SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "memory.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )


def test_flatten_query_handles_block_lists():
    f = SQLiteRAGMemory._flatten_query
    assert f("plain") == "plain"
    assert f(BLOCKS) == "# Supervisor - think\nreal analysis"
    # reasoning-only content → empty, not ValueError
    assert f([BLOCKS[0]]) == ""
    assert f([{"type": "input_text", "text": "a"}, {"type": "output_text", "text": "b"}]) == "a\nb"
    assert f({"type": "function_call", "name": "x"}) == ""
    assert f(None) == ""


def test_load_memory_variables_never_raises_on_blocks(mem):
    mem.save_context(
        inputs=[HumanMessage(content="seed")],
        outputs=[AIMessage(content="prior fact", name="Modeler")],
        agent_name="Modeler",
    )
    # Old code: BLOCKS[0] is a reasoning dict → ValueError("Unsupported query format")
    out = mem.load_memory_variables({"input": BLOCKS, "agent_name": "Supervisor"})
    assert isinstance(out["history"], list)
    # reasoning-only query degrades to empty history, still no raise
    out2 = mem.load_memory_variables({"input": [BLOCKS[0]]})
    assert out2 == {"history": []}


def test_encrypted_content_stripped_on_persist(mem):
    mem.save_context(
        inputs=[],
        outputs=[AIMessage(content=BLOCKS, name="Supervisor")],
        agent_name="Supervisor",
    )
    rows = mem.get_all_raw_messages()
    dumped = json.dumps(rows, default=str)
    assert "encrypted_content" not in dumped
    assert "gAAAA" not in dumped
    # the text block survived
    assert "real analysis" in dumped


def test_content_text_and_stage_content_flatten(tmp_path):
    llm = FakeLLM([])
    sup = Supervisor(llm=llm, project_root=str(tmp_path), tools=[], Collages=["END"])
    assert sup._content_text(BLOCKS) == "# Supervisor - think\nreal analysis"
    staged = sup._stage_content("think", BLOCKS)
    assert staged.startswith("# Supervisor - think")
    assert "encrypted_content" not in staged


def test_agent_run_completes_with_block_content(tmp_path, mem):
    """A full triage-converse run whose scripted LLM output uses block-list
    content must complete and emit a flattened text reply."""
    llm = FakeLLM([TriageOut(verdict="converse", reply="Hi from blocks!")])
    sup = Supervisor(
        llm=llm,
        memory=mem,
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    # the incoming human message itself arrives as content parts (multimodal)
    result = sup.run(
        {"messages": [HumanMessage(content=[{"type": "text", "text": "hi"}])]}
    )
    assert result["send_to"] == "END"
    final = result["messages"][-1]
    assert isinstance(final, ReportMessage)
    assert "Hi from blocks!" in final.content
