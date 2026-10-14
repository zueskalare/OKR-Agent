"""MCPToolNode async execution path (WS19.2): result passthrough, timeout
containment, error containment, multimodal unwrap — on both drivers."""

import asyncio
import json
from typing import Annotated, Any, List, Optional, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from AtomAgents.agents.tool_node_agent import MCPToolNode


class S(TypedDict):
    messages: Annotated[List[Any], add_messages]
    send_to: Optional[str]


@tool
async def quick_tool(x: str) -> str:
    """Returns a marker for x."""
    return f"quick:{x}"


@tool
async def slow_tool(x: str) -> str:
    """Sleeps past the node timeout."""
    await asyncio.sleep(5)
    return "never"


@tool
async def image_tool(x: str) -> str:
    """Returns a JSON-encoded multimodal payload."""
    return json.dumps(
        [
            {"type": "text", "text": "rendered"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )


# Well over TOOL_OUTPUT_MAX_BYTES (64 KiB): the size class every real
# md_visualize / file_reader PNG lands in.
BIG_B64 = "A" * 200_000


@tool
async def big_image_tool(x: str) -> str:
    """Returns a large JSON-encoded multimodal payload (real file_reader shape)."""
    return json.dumps(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + BIG_B64}}]
    )


@tool
async def dict_payload_tool(x: str) -> str:
    """Returns dict-shaped JSON that mentions a data URL."""
    return json.dumps({"error": "could not render data:image/png;base64,AAAA"})


def _graph(node):
    def caller(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": state["send_to"], "args": {"x": "v"}, "id": "t1"}
                    ],
                )
            ]
        }

    g = StateGraph(S)
    g.add_node("caller", caller)
    g.add_node("tools", node)
    g.set_entry_point("caller")
    g.add_edge("caller", "tools")
    g.add_edge("tools", END)
    return g.compile()


def _tool_msg(out):
    return next(m for m in out["messages"] if isinstance(m, ToolMessage))


@pytest.fixture()
def node(monkeypatch):
    n = MCPToolNode(
        tools=[quick_tool, slow_tool, image_tool, big_image_tool, dict_payload_tool]
    )
    monkeypatch.setattr(MCPToolNode, "TIMEOUT_SECONDS", 0.3)
    # Pin the client-side cap so the big-payload tests are deterministic
    # regardless of ATOMAGENTS_TOOL_OUTPUT_MAX in the environment.
    from AtomAgents import config

    monkeypatch.setattr(config, "TOOL_OUTPUT_MAX_BYTES", 64 * 1024)
    return n


def test_async_driver_passthrough(node):
    out = asyncio.run(
        _graph(node).ainvoke({"messages": [HumanMessage("go")], "send_to": "quick_tool"})
    )
    assert _tool_msg(out).content == "quick:v"


def test_async_driver_timeout(node):
    out = asyncio.run(
        _graph(node).ainvoke({"messages": [HumanMessage("go")], "send_to": "slow_tool"})
    )
    msg = _tool_msg(out)
    assert msg.status == "error"
    assert "timed out after" in str(msg.content)


def test_async_driver_multimodal_unwrap(node):
    out = asyncio.run(
        _graph(node).ainvoke({"messages": [HumanMessage("go")], "send_to": "image_tool"})
    )
    content = _tool_msg(out).content
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"


def test_sync_driver_still_works(node):
    out = _graph(node).invoke(
        {"messages": [HumanMessage("go")], "send_to": "quick_tool"}
    )
    assert _tool_msg(out).content == "quick:v"


def test_sync_driver_timeout(node):
    out = _graph(node).invoke(
        {"messages": [HumanMessage("go")], "send_to": "slow_tool"}
    )
    msg = _tool_msg(out)
    assert msg.status == "error"
    assert "timed out after" in str(msg.content)


def test_sync_driver_multimodal_unwrap(node):
    out = _graph(node).invoke(
        {"messages": [HumanMessage("go")], "send_to": "image_tool"}
    )
    assert isinstance(_tool_msg(out).content, list)


def test_async_driver_unwraps_image_larger_than_cap(node):
    """A1 regression: payloads over TOOL_OUTPUT_MAX_BYTES must unwrap BEFORE
    the client-side cap — the old cap-first order spliced a truncation marker
    into the base64 body, json.loads failed, and vision silently died for
    every real render (the shipped test fixture was ~110 bytes)."""
    out = asyncio.run(
        _graph(node).ainvoke(
            {"messages": [HumanMessage("go")], "send_to": "big_image_tool"}
        )
    )
    content = _tool_msg(out).content
    assert isinstance(content, list), "large multimodal payload was not unwrapped"
    assert content[0]["image_url"]["url"] == "data:image/png;base64," + BIG_B64
    assert "truncated client-side" not in json.dumps(content)


def test_sync_driver_unwraps_image_larger_than_cap(node):
    out = _graph(node).invoke(
        {"messages": [HumanMessage("go")], "send_to": "big_image_tool"}
    )
    content = _tool_msg(out).content
    assert isinstance(content, list)
    assert content[0]["image_url"]["url"] == "data:image/png;base64," + BIG_B64


def test_non_list_multimodal_json_stays_capped_text(node):
    """dict/scalar JSON that merely mentions a data URL must NOT be assigned
    to ToolMessage.content — it stays a (cappable) string."""
    out = asyncio.run(
        _graph(node).ainvoke(
            {"messages": [HumanMessage("go")], "send_to": "dict_payload_tool"}
        )
    )
    assert isinstance(_tool_msg(out).content, str)
