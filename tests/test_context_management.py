"""Context management (WS20.2): tool-result clearing, RAG snippet caps,
profile plumbing, client-side tool caps — with the disabled paths inert."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from AtomAgents import config
from AtomAgents.agents.simulation_manager import Supervisor
from fakes import FakeLLM


@pytest.fixture()
def sup(tmp_path):
    return Supervisor(
        llm=FakeLLM([]), project_root=str(tmp_path), tools=[], Collages=["END"]
    )


def _messages_with_tools(n_tools=5, tool_size=4000):
    msgs = [HumanMessage("go")]
    for i in range(n_tools):
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "t", "args": {}, "id": f"c{i}"}],
            )
        )
        msgs.append(ToolMessage("X" * tool_size, tool_call_id=f"c{i}"))
    msgs.append(AIMessage("final"))
    return msgs


def test_edit_context_clears_old_tool_results(sup, monkeypatch):
    monkeypatch.setattr(config, "TOOL_CLEAR_TRIGGER", 1000)  # force trigger
    monkeypatch.setattr(config, "TOOL_CLEAR_KEEP", 2)
    msgs = _messages_with_tools(n_tools=5)
    out = sup._edit_context(msgs)
    tool_texts = [str(m.content) for m in out if isinstance(m, ToolMessage)]
    cleared = [t for t in tool_texts if "X" * 100 not in t]
    intact = [t for t in tool_texts if "X" * 100 in t]
    assert len(intact) == 2  # keep the most recent 2
    assert len(cleared) == 3  # older ones replaced with placeholders
    # non-tool content untouched
    assert any("final" in str(m.content) for m in out)


def test_edit_context_inert_below_trigger(sup, monkeypatch):
    monkeypatch.setattr(config, "TOOL_CLEAR_TRIGGER", 10_000_000)
    monkeypatch.setattr(config, "SUMMARIZE", False)
    msgs = _messages_with_tools(n_tools=2)
    out = sup._edit_context(msgs)
    assert [str(m.content) for m in out] == [str(m.content) for m in msgs]


def test_edit_context_never_raises(sup, monkeypatch):
    monkeypatch.setattr(config, "TOOL_CLEAR_TRIGGER", 1)
    # a message shape the editor might choke on → falls back to the original
    weird = [HumanMessage("x"), object()]
    out = sup._edit_context(weird)  # must not raise
    assert len(out) == 2


def test_rag_snippets_capped(tmp_path, monkeypatch):
    from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory
    from test_memory_schema import StubEmb

    monkeypatch.setattr(config, "RAG_SNIPPET_CHARS", 50)
    mem = SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "m.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )
    mem.save_context(
        inputs=[HumanMessage("q")],
        outputs=[AIMessage("Y" * 500, name="Modeler")],
        agent_name="Modeler",
    )
    out = mem.load_memory_variables({"input": "Y query", "agent_name": "Supervisor"})
    for snippet in out["history"]:
        assert len(snippet) <= 50 + len(" …[snippet truncated]")


def test_profile_plumbed_into_models(monkeypatch):
    from AtomAgents.llm_factory import make_llm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "MAX_INPUT_TOKENS", 55_000)
    llm = make_llm("openai", model="gpt-5")
    assert llm.profile == {"max_input_tokens": 55_000}


def test_client_side_tool_cap(monkeypatch):
    from AtomAgents.agents.tool_node_agent import MCPToolNode

    monkeypatch.setattr(config, "TOOL_OUTPUT_MAX_BYTES", 100)
    msg = ToolMessage("Z" * 1000, tool_call_id="t")
    capped = MCPToolNode._cap_output(msg)
    assert len(str(capped.content)) < 300
    assert "truncated client-side" in str(capped.content)
    # under the cap → untouched
    small = ToolMessage("ok", tool_call_id="t")
    assert MCPToolNode._cap_output(small).content == "ok"


def test_cap_truncates_text_blocks_but_not_media(monkeypatch):
    """_cap_output runs AFTER _unwrap_multimodal (A1): on block lists it must
    bound oversized text blocks (foreign servers) while leaving base64
    image/file payloads whole — vision needs the full body."""
    from AtomAgents.agents.tool_node_agent import MCPToolNode

    monkeypatch.setattr(config, "TOOL_OUTPUT_MAX_BYTES", 100)
    url = "data:image/png;base64," + "A" * 1000
    msg = ToolMessage(
        [
            {"type": "text", "text": "T" * 1000},
            {"type": "image_url", "image_url": {"url": url}},
        ],
        tool_call_id="t",
    )
    capped = MCPToolNode._cap_output(msg)
    text_block, image_block = capped.content
    assert "truncated client-side" in text_block["text"]
    assert len(text_block["text"]) < 300
    assert image_block["image_url"]["url"] == url
