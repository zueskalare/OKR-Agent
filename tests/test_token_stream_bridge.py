"""Langgraph contract pins for the WS19 token bridge (probes P4/P5/P6).

These pin the langgraph 1.2.10 behaviors the async core depends on; if a
future upgrade changes any of them, these fail loudly BEFORE the framework
misbehaves in production.
"""

import asyncio
import threading
from typing import Annotated, Any, List, Optional, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class S(TypedDict):
    messages: Annotated[List[Any], add_messages]
    send_to: Optional[str]


def _fake(text):
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def _inner_graph(tagged: bool):
    def think(state):
        llm = _fake("alpha beta gamma")
        cfg = {"tags": ["atom_stream"]} if tagged else None
        r = llm.invoke([HumanMessage("go")], config=cfg)
        return {"messages": [r], "send_to": "END"}

    g = StateGraph(S)
    g.add_node("think", think)
    g.set_entry_point("think")
    g.add_edge("think", END)
    return g.compile()


def _outer_with_bridge(inner, collect):
    """Async outer node consuming the inner stream — the arun pattern."""

    async def outer(state):
        final = {}
        async for mode, payload in inner.astream(
            {"messages": state["messages"]}, stream_mode=["values", "messages"]
        ):
            if mode == "messages":
                chunk, meta = payload
                collect.append((chunk, meta))
            else:
                final = payload
        return {"messages": final.get("messages", []), "send_to": "END"}

    g = StateGraph(S)
    g.add_node("Sup", outer)
    g.set_entry_point("Sup")
    g.add_conditional_edges("Sup", lambda s: s.get("send_to"), {"END": END})
    return g.compile()


def test_inner_stream_preserves_tags_and_tokens():
    """P4: inside the INNER stream, sync llm.invoke auto-upgrades to
    streaming; tags + inner-node attribution survive; concatenation equals
    the final content."""
    collected: List[Any] = []
    inner = _inner_graph(tagged=True)
    app = _outer_with_bridge(inner, collected)
    out = asyncio.run(app.ainvoke({"messages": [HumanMessage("hi")]}))

    chunks = [c for c, m in collected if isinstance(c, AIMessageChunk)]
    assert len(chunks) >= 3, "expected per-token chunks, got whole-message"
    metas = [m for _, m in collected]
    assert all("atom_stream" in (m.get("tags") or ()) for _, m in collected)
    assert all(m.get("langgraph_node") == "think" for m in metas)
    joined = "".join(str(c.content) for c in chunks)
    final_texts = [str(m.content) for m in out["messages"] if getattr(m, "type", "") == "ai"]
    assert joined in final_texts


def test_untagged_call_filtered_by_tag_gate():
    """The structured-JSON leak guard: untagged calls produce chunks WITHOUT
    the tag, so a tag filter drops all of them."""
    collected: List[Any] = []
    inner = _inner_graph(tagged=False)
    app = _outer_with_bridge(inner, collected)
    asyncio.run(app.ainvoke({"messages": [HumanMessage("hi")]}))
    tagged = [c for c, m in collected if "atom_stream" in (m.get("tags") or ())]
    assert tagged == []
    assert collected, "chunks still flow — only the tag filter excludes them"


def test_values_diff_assigns_ids_exactly_once():
    """P4/T1: values-mode payloads are post-reducer; add_messages assigns ids,
    so an id-seeded diff publishes each message exactly once."""

    def a(state):
        return {"messages": [AIMessage(content="one", name="A")]}

    def b(state):
        return {"messages": [AIMessage(content="two", name="A")], "send_to": "END"}

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    inner = g.compile()

    async def run():
        seen: set = set()
        published: List[str] = []
        async for mode, payload in inner.astream(
            {"messages": [HumanMessage("hi", id="h1")]},
            stream_mode=["values"],
        ):
            _, vals = mode, payload
            for msg in vals.get("messages", ()):
                mid = getattr(msg, "id", None)
                assert mid is not None, "post-reducer messages must carry ids"
                if mid in seen:
                    continue
                seen.add(mid)
                published.append(str(msg.content))
        return published

    published = asyncio.run(run())
    assert published == ["hi", "one", "two"]


def test_hitl_under_async_saver(tmp_path):
    """P5: sync human node + interrupt under astream + AsyncSqliteSaver;
    resume works; ReportMessage survives via the serde allowlist."""
    import aiosqlite
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from AtomAgents.agents.messages import ReportMessage

    def asker(state):
        return {
            "messages": [ReportMessage("Approve?", name="Supervisor")],
            "send_to": "Human",
        }

    def human(state):
        ans = interrupt({"q": "?"})
        return {"messages": [HumanMessage(str(ans), name="human")], "send_to": "END"}

    async def run():
        conn = await aiosqlite.connect(str(tmp_path / "graph_state.db"))
        saver = AsyncSqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[ReportMessage])
        )
        g = StateGraph(S)
        g.add_node("Asker", asker)
        g.add_node("Human", human)
        g.set_entry_point("Asker")
        g.add_conditional_edges("Asker", lambda s: s.get("send_to"), {"Human": "Human"})
        g.add_conditional_edges("Human", lambda s: s.get("send_to"), {"END": END})
        app = g.compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "t"}}
        await app.ainvoke({"messages": [HumanMessage("go")]}, config=cfg)
        snap = await app.aget_state(cfg)
        assert any(t.interrupts for t in snap.tasks)
        out = await app.ainvoke(Command(resume="approved"), config=cfg)
        assert any("approved" in str(m.content) for m in out["messages"])
        assert any(isinstance(m, ReportMessage) for m in out["messages"])
        await conn.close()

    asyncio.run(run())


def _events_through_backend(sup, monkeypatch, token_stream: bool):
    """Drive a REAL Supervisor (FakeLLM) through aiter_graph_events on a real
    outer graph; return the AtomEvent (kind, agent, stage, text-head) list."""
    from AtomAgents import config
    from backend.openai_backend import aiter_graph_events

    monkeypatch.setattr(config, "TOKEN_STREAM", token_stream)

    from langgraph.checkpoint.memory import InMemorySaver

    g = StateGraph(S)
    g.add_node("Supervisor", sup.arun)
    g.set_entry_point("Supervisor")
    g.add_conditional_edges("Supervisor", lambda s: s.get("send_to"), {"END": END})
    # durability (passed by aiter_graph_events) requires a checkpointer on
    # async graphs — production graphs always have one.
    app = g.compile(checkpointer=InMemorySaver())

    async def go():
        out = []
        async for ev in aiter_graph_events(
            app,
            {"messages": [HumanMessage("hi")]},
            {"configurable": {"thread_id": "t"}},
            cancel_key=None,
            include_progress=True,
        ):
            out.append((ev.kind, ev.agent, ev.stage, ev.text[:30]))
        return out

    return asyncio.run(go())


def test_incremental_messages_emitted_once(tmp_path, monkeypatch):
    """WS19.6: with TOKEN_STREAM on, each message arrives exactly once (via
    the custom channel, chronologically); the updates batch is a no-op."""
    from fakes import FakeLLM

    from AtomAgents.agents.simulation_manager import Supervisor, TriageOut

    sup = Supervisor(
        llm=FakeLLM([TriageOut(verdict="converse", reply="hello!")]),
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    events = _events_through_backend(sup, monkeypatch, token_stream=True)
    kinds = [k for k, *_ in events]
    assert kinds == ["working", "message"]  # exactly one message event
    assert events[1][2] == "reply"


def test_token_stream_off_falls_back_to_batch(tmp_path, monkeypatch):
    """TOKEN_STREAM=0 → identical event list, sourced from the updates batch."""
    from fakes import FakeLLM

    from AtomAgents.agents.simulation_manager import Supervisor, TriageOut

    sup = Supervisor(
        llm=FakeLLM([TriageOut(verdict="converse", reply="hello!")]),
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    events = _events_through_backend(sup, monkeypatch, token_stream=False)
    kinds = [k for k, *_ in events]
    assert kinds == ["working", "message"]
    assert events[1][2] == "reply"


def _multi_turn_events(sup, monkeypatch, token_stream: bool, prompts):
    """Like _events_through_backend, but drives several user turns on ONE
    checkpointed thread — the multi-turn session shape the R1 replay bug
    corrupted."""
    from AtomAgents import config
    from backend.openai_backend import aiter_graph_events

    monkeypatch.setattr(config, "TOKEN_STREAM", token_stream)

    from langgraph.checkpoint.memory import InMemorySaver

    g = StateGraph(S)
    g.add_node("Supervisor", sup.arun)
    g.set_entry_point("Supervisor")
    g.add_conditional_edges("Supervisor", lambda s: s.get("send_to"), {"END": END})
    app = g.compile(checkpointer=InMemorySaver())

    async def go():
        turns = []
        for prompt in prompts:
            out = []
            async for ev in aiter_graph_events(
                app,
                {"messages": [HumanMessage(prompt)]},
                {"configurable": {"thread_id": "multi"}},
                cancel_key=None,
                include_progress=True,
            ):
                out.append((ev.kind, ev.agent, ev.stage, ev.text))
            turns.append(out)
        return turns

    return asyncio.run(go())


@pytest.mark.parametrize("token_stream", [True, False])
def test_second_turn_emits_only_its_own_messages(tmp_path, monkeypatch, token_stream):
    """R1 regression: turn N must not replay turns 1..N-1. The full-history
    node returns × per-request dedupe used to re-emit every prior turn's
    messages in the trailing updates batch (with TOKEN_STREAM off, that batch
    is the ONLY wire source — the exact live-reproduced failure)."""
    from fakes import FakeLLM

    from AtomAgents.agents.simulation_manager import Supervisor, TriageOut

    sup = Supervisor(
        llm=FakeLLM(
            [
                TriageOut(verdict="converse", reply="first answer"),
                TriageOut(verdict="converse", reply="second answer"),
            ]
        ),
        project_root=str(tmp_path),
        tools=[],
        Collages=["Modeler", "Simulator", "END"],
    )
    turn1, turn2 = _multi_turn_events(
        sup, monkeypatch, token_stream, ["hi", "hi again"]
    )
    assert [k for k, *_ in turn1] == ["working", "message"]
    kinds2 = [k for k, *_ in turn2]
    assert kinds2 == ["working", "message"], f"stale replay in turn 2: {turn2}"
    assert "second answer" in turn2[1][3]
    assert all("first answer" not in text for *_, text in turn2)


def test_updates_batch_seeded_against_checkpointed_history(monkeypatch):
    """R1 layer 2: even when a node update carries prior-turn messages, the
    backend seeds its dedupe set from the thread's checkpointed state and
    refuses to re-emit them."""
    from types import SimpleNamespace

    from test_chat_golden_frames import FakeGraph

    from backend.openai_backend import aiter_graph_events

    prior = AIMessage(
        content="# Supervisor - reply\nold answer", name="Supervisor", id="old1"
    )
    fresh = AIMessage(
        content="# Supervisor - reply\nnew answer", name="Supervisor", id="new1"
    )

    class SeededFakeGraph(FakeGraph):
        async def aget_state(self, config):
            return SimpleNamespace(tasks=(), values={"messages": [prior]})

    graph = SeededFakeGraph(
        [("updates", {"Supervisor": {"messages": [prior, fresh]}})]
    )

    async def go():
        return [
            (ev.kind, ev.text)
            async for ev in aiter_graph_events(
                graph,
                {"messages": []},
                {"configurable": {"thread_id": "t"}},
                cancel_key=None,
                include_progress=True,
            )
        ]

    events = asyncio.run(go())
    assert [k for k, _ in events] == ["message"]
    assert "new answer" in events[0][1]
    assert all("old answer" not in text for _, text in events)


class _StubSubGraph:
    """Scripted inner stream for hermetic bridge-filter tests."""

    def __init__(self, script):
        self.script = script

    async def astream(self, inputs, config=None, stream_mode=None):
        assert "messages" in stream_mode, "bridge must subscribe to messages mode"
        for item in self.script:
            yield item


def _bridge_tokens(tmp_path, monkeypatch, script):
    from fakes import FakeLLM

    from AtomAgents import config
    from AtomAgents.agents.simulation_manager import Supervisor

    monkeypatch.setattr(config, "TOKEN_STREAM", True)
    sup = Supervisor(
        llm=FakeLLM([]), project_root=str(tmp_path), tools=[], Collages=["END"]
    )
    sup._SubGraph = _StubSubGraph(script)
    events = []
    monkeypatch.setattr(sup, "_stream_writer", lambda: events.append)
    asyncio.run(sup._adrive_subgraph({"messages": []}))
    return events


def test_bridge_republishes_tagged_tokens_only(tmp_path, monkeypatch):
    tagged_meta = {"tags": ["atom_stream"], "langgraph_node": "think"}
    untagged_meta = {"tags": None, "langgraph_node": "plan"}
    script = [
        ("messages", (AIMessageChunk(content="tok1"), tagged_meta)),
        ("messages", (AIMessageChunk(content='{"send_to":'), untagged_meta)),  # JSON
        ("messages", (AIMessageChunk(content="tok2"), tagged_meta)),
        ("values", {"messages": [], "send_to": "end"}),
    ]
    events = _bridge_tokens(tmp_path, monkeypatch, script)
    tokens = [e for e in events if e.get("event") == "token"]
    assert [t["text"] for t in tokens] == ["tok1", "tok2"]
    assert all(t["stage"] == "think" and t["agent"] == "Supervisor" for t in tokens)
    assert not any('{"send_to"' in str(e) for e in events)  # JSON never leaks


def test_bridge_drops_reasoning_chunks(tmp_path, monkeypatch):
    tagged_meta = {"tags": ["atom_stream"], "langgraph_node": "think"}
    reasoning_chunk = AIMessageChunk(
        content=[{"type": "reasoning", "id": "rs", "summary": []}]
    )
    script = [
        ("messages", (reasoning_chunk, tagged_meta)),
        ("messages", (AIMessageChunk(content="visible"), tagged_meta)),
        ("values", {"messages": [], "send_to": "end"}),
    ]
    events = _bridge_tokens(tmp_path, monkeypatch, script)
    tokens = [e for e in events if e.get("event") == "token"]
    assert [t["text"] for t in tokens] == ["visible"]


def test_aiosqlite_thread_gone_after_close(tmp_path):
    """P6: aiosqlite spawns a NON-daemon worker thread; without close the
    process hangs at exit. Pin that close() actually retires it."""
    import aiosqlite

    async def run():
        conn = await aiosqlite.connect(str(tmp_path / "x.db"))
        names_open = {t.name for t in threading.enumerate()}
        await conn.close()
        return names_open

    before = {t.name for t in threading.enumerate()}
    names_open = asyncio.run(run())
    after = {t.name for t in threading.enumerate()}
    assert any("_connection_worker" in n or "Thread-" in n for n in names_open - before)
    assert after <= before | set()  # no lingering aiosqlite worker threads
