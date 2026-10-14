"""Canonical flattener (utils.text.message_text) — WS18.1.

Five hand-rolled flatteners consolidated onto one implementation. The two
load-bearing contracts: text blocks join with "\\n" (message.text would
concatenate WITHOUT a separator — verified difference, and stage parsing +
golden-frame wire tests depend on the join), and reasoning/thinking blocks
never leak into text.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from AtomAgents.utils.text import message_text


def test_plain_and_none():
    assert message_text("hello") == "hello"
    assert message_text(None) == ""
    assert message_text(42) == "42"


def test_newline_join_contract():
    content = [{"type": "text", "text": "part A"}, {"type": "text", "text": "part B"}]
    assert message_text(content) == "part A\npart B"
    # via a message too (message.text would give 'part Apart B')
    assert message_text(AIMessage(content=content)) == "part A\npart B"


def test_reasoning_and_thinking_excluded():
    responses_shape = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "th"}]},
            {"type": "text", "text": "answer"},
        ]
    )
    anthropic_shape = AIMessage(
        content=[
            {"type": "thinking", "thinking": "secret", "signature": "s"},
            {"type": "text", "text": "answer"},
        ]
    )
    assert message_text(responses_shape) == "answer"
    assert message_text(anthropic_shape) == "answer"
    assert "secret" not in message_text(anthropic_shape)


def test_raw_content_block_variants():
    assert message_text([{"type": "input_text", "text": "a"}, {"type": "output_text", "text": "b"}]) == "a\nb"
    assert message_text({"type": "function_call", "name": "x"}) == ""
    assert message_text(["bare string", {"type": "text", "text": "block"}]) == "bare string\nblock"


def test_message_shapes():
    assert message_text(ToolMessage(content="tool out", tool_call_id="t")) == "tool out"
    assert (
        message_text(
            HumanMessage(
                content=[
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ]
            )
        )
        == "hi"
    )


def test_delegates_share_the_implementation():
    from AtomAgents.agents.base_agent import BaseAgent
    from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory

    blocks = [{"type": "reasoning", "id": "r"}, {"type": "text", "text": "x"}]
    assert BaseAgent._content_text(blocks) == "x"
    assert SQLiteRAGMemory._flatten_query(blocks) == "x"
