"""Golden-frame regression for /v1/chat/completions.

WS16.3 rebuilt the endpoint as a thin emitter over ``iter_graph_events``;
these tests pin the exact wire frames a scripted fake graph produces so the
compat surface cannot drift while /v1/responses evolves next to it.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import backend.openai_backend as ob


class FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeGraph:
    """Mimics a compiled langgraph app for the backend's stream contract.

    ``script`` is a list of (mode, payload) tuples. In list stream-mode the
    tuples are yielded as-is; in plain "updates" mode only the payloads of
    updates entries are yielded (matching langgraph's API).
    """

    def __init__(self, script, cancel_after=None):
        self.script = script
        self.cancel_after = cancel_after  # (index, cancel_key) → set flag mid-run

    def stream(self, input=None, config=None, stream_mode=None, durability=None):
        list_mode = isinstance(stream_mode, list)
        for i, (mode, payload) in enumerate(self.script):
            if self.cancel_after and i == self.cancel_after[0]:
                ob.CANCEL_FLAGS[self.cancel_after[1]] = True
            if not list_mode and mode != "updates":
                continue
            yield (mode, payload) if list_mode else payload

    async def astream(self, input=None, config=None, stream_mode=None, durability=None):
        for item in self.stream(
            input=input, config=config, stream_mode=stream_mode, durability=durability
        ):
            yield item

    def get_state(self, config):
        return SimpleNamespace(tasks=())

    async def aget_state(self, config):
        return self.get_state(config)


SCRIPT = [
    ("custom", {"event": "working", "agent": "Supervisor"}),
    (
        "updates",
        {
            "Supervisor": {
                "messages": [
                    AIMessage(
                        content="# Supervisor - think\nplan it",
                        name="Supervisor",
                        id="m1",
                    ),
                    ToolMessage(content="tool says hi", tool_call_id="t1", id="m2"),
                    AIMessage(content="", name="Supervisor", id="m3"),  # empty → skipped in stream
                ]
            }
        },
    ),
]

INTERRUPT_SCRIPT = SCRIPT + [
    ("updates", {"__interrupt__": (FakeInterrupt({"question": "T?", "from": "Modeler"}),)}),
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    holder = {"graph": None}

    async def fake_get_or_build(graph_key, model_id, session_dir):
        return holder["graph"], tmp_path

    monkeypatch.setattr(ob, "_get_or_build_graph", fake_get_or_build)
    ob.CANCEL_FLAGS.clear()
    return TestClient(ob.app), holder


def _stream_deltas(resp_text):
    frames = []
    done = 0
    for line in resp_text.splitlines():
        if line == "data: [DONE]":
            done += 1
            continue
        if not line.startswith("data: "):
            continue
        chunk = json.loads(line[6:])
        choice = chunk["choices"][0]
        frames.append((choice.get("delta"), choice.get("finish_reason")))
    return frames, done


def _post(client, payload):
    return client.post("/v1/chat/completions", json=payload)


def test_stream_golden_frames(client):
    tc, holder = client
    holder["graph"] = FakeGraph(SCRIPT)
    r = _post(tc, {"model": "atomagent", "stream": True,
                   "messages": [{"role": "user", "content": "go"}]})
    assert r.status_code == 200
    frames, done = _stream_deltas(r.text)
    assert frames == [
        ({"atom_event": "working", "atom_agent": "Supervisor"}, None),
        (
            {
                "content": "# Supervisor - think\nplan it\n\n",
                "atom_event": "message",
                "atom_agent": "Supervisor",
                "atom_stage": "think",
                "atom_type": "ai",
            },
            None,
        ),
        (
            {
                "content": "tool says hi\n\n",
                "atom_event": "message",
                "atom_agent": "",
                "atom_stage": "",
                "atom_type": "tool",
            },
            None,
        ),
        ({}, "stop"),
    ]
    assert done == 1


def test_stream_awaiting_user_frames(client):
    tc, holder = client
    holder["graph"] = FakeGraph(INTERRUPT_SCRIPT)
    r = _post(tc, {"model": "atomagent", "stream": True,
                   "messages": [{"role": "user", "content": "go"}]})
    frames, done = _stream_deltas(r.text)
    awaiting = [d for d, _ in frames if d and d.get("atom_event") == "awaiting_user"]
    assert awaiting == [
        {
            "content": "\n[awaiting your input — requested by Modeler — reply to continue]\n",
            "atom_event": "awaiting_user",
            "atom_agent": "Modeler",
            "atom_stage": "await",
            "atom_type": "ai",
        }
    ]
    assert frames[-1] == ({}, "stop") and done == 1


def test_stream_cancel_terminal(client):
    tc, holder = client
    holder["graph"] = FakeGraph(SCRIPT, cancel_after=(1, "atomagent:sess-c"))
    r = _post(tc, {"model": "atomagent", "session_dir": "sess-c", "stream": True,
                   "messages": [{"role": "user", "content": "go"}]})
    frames, done = _stream_deltas(r.text)
    assert frames[-1] == (
        {"atom_event": "cancelled", "content": "\n[run cancelled by user]\n"},
        "stop",
    )
    assert done == 1  # cancel ends the stream — no extra end frame


def test_non_stream_concatenates_ai_text(client):
    tc, holder = client
    holder["graph"] = FakeGraph(INTERRUPT_SCRIPT)
    r = _post(tc, {"model": "atomagent",
                   "messages": [{"role": "user", "content": "go"}]})
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    # ai text + empty ai message separator + awaiting marker; tool excluded
    assert content == (
        "# Supervisor - think\nplan it\n\n\n\n"
        "[awaiting your input — requested by Modeler — reply to continue]"
    )
