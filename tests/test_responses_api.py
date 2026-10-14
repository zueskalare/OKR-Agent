"""/v1/responses surface: wire models, input transformer, endpoint behavior.

Every response object the backend emits must stay parseable by the official
SDK (``openai.types.responses.Response``) — that validation catches the two
traps found during design: the closed ``incomplete_details.reason`` enum and
the required ``usage.input_tokens_details.cache_write_tokens`` field.
"""

import json

import pytest
from fastapi.testclient import TestClient
from openai.types.responses import Response as SDKResponse

import backend.openai_backend as ob
from backend.openai_backend import (
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputText,
    _extract_from_responses_input,
)
from test_chat_golden_frames import SCRIPT, INTERRUPT_SCRIPT, FakeGraph


# ---------------------------------------------------------------------------
# Input transformer
# ---------------------------------------------------------------------------


def test_string_input():
    assert _extract_from_responses_input("hi") == ("hi", [])


def test_string_input_with_instructions():
    text, imgs = _extract_from_responses_input("hi", instructions="be terse")
    assert text == "be terse\n\nhi"
    assert imgs == []


def test_newest_user_message_only():
    items = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": [{"type": "input_text", "text": "new question"}]},
    ]
    assert _extract_from_responses_input(items) == ("new question", [])


def test_system_and_developer_hoisted():
    items = [
        {"role": "system", "content": "sys rule"},
        {"role": "developer", "content": [{"type": "input_text", "text": "dev rule"}]},
        {"role": "user", "content": "q"},
    ]
    text, _ = _extract_from_responses_input(items, instructions="top rule")
    assert text == "top rule\n\nsys rule\n\ndev rule\n\nq"


def test_input_image_bare_string_and_chat_object_forms():
    items = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "see image"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ],
        }
    ]
    text, imgs = _extract_from_responses_input(items)
    assert text == "see image"
    assert imgs == ["AAAA", "BBBB"]


def test_non_message_items_ignored():
    items = [
        {"type": "function_call", "name": "t", "arguments": "{}"},
        {"type": "reasoning", "id": "rs_1"},
        {"role": "user", "content": "real"},
    ]
    assert _extract_from_responses_input(items) == ("real", [])


def test_empty_input():
    assert _extract_from_responses_input(None) == ("", [])
    assert _extract_from_responses_input([]) == ("", [])


# ---------------------------------------------------------------------------
# Wire-model ↔ SDK round trip
# ---------------------------------------------------------------------------


def _sample_response(**overrides):
    obj = ResponseObject(
        id="resp_" + "0" * 32,
        created_at=1754000000,
        model="atomagent",
        output=[
            ResponseMessageItem(
                id="msg_" + "0" * 16,
                content=[ResponseOutputText(text="# Supervisor - reply\nhello")],
                metadata={
                    "atom_agent": "Supervisor",
                    "atom_stage": "reply",
                    "atom_type": "ai",
                },
            )
        ],
        output_text="# Supervisor - reply\nhello",
    )
    return obj.model_copy(update=overrides)


def test_response_object_parses_with_sdk():
    parsed = SDKResponse.model_validate(_sample_response().model_dump())
    assert parsed.id.startswith("resp_")
    assert parsed.output[0].model_extra["metadata"]["atom_agent"] == "Supervisor"
    assert parsed.output_text == "# Supervisor - reply\nhello"


@pytest.mark.parametrize(
    "status,metadata",
    [
        ("completed", {}),
        ("incomplete", {"atom_status": "awaiting_user", "atom_agent": "Supervisor"}),
        ("cancelled", {"atom_status": "cancelled"}),
        ("failed", {"atom_status": "error"}),
    ],
)
def test_all_terminal_statuses_sdk_valid(status, metadata):
    obj = _sample_response(status=status, metadata=metadata)
    if status == "failed":
        obj = obj.model_copy(
            update={"error": {"code": "server_error", "message": "boom"}}
        )
    parsed = SDKResponse.model_validate(obj.model_dump())
    assert parsed.status == status
    # HITL/cancel reasons ride in metadata — incomplete_details must be None
    assert parsed.incomplete_details is None


# ---------------------------------------------------------------------------
# Endpoint behavior (fake graph, hermetic)
# ---------------------------------------------------------------------------


class RaisingGraph(FakeGraph):
    def stream(self, input=None, config=None, stream_mode=None, durability=None):
        yield from super().stream(input=input, config=config, stream_mode=stream_mode)
        raise RuntimeError("solver exploded")

    async def astream(self, input=None, config=None, stream_mode=None, durability=None):
        async for item in super().astream(
            input=input, config=config, stream_mode=stream_mode
        ):
            yield item
        raise RuntimeError("solver exploded")


@pytest.fixture()
def rclient(tmp_path, monkeypatch):
    holder = {"graph": None}

    async def fake_get_or_build(graph_key, model_id, session_dir):
        return holder["graph"], tmp_path

    monkeypatch.setattr(ob, "_get_or_build_graph", fake_get_or_build)
    ob.CANCEL_FLAGS.clear()
    ob.RESPONSE_KEYS.clear()
    return TestClient(ob.app), holder


def _events(sse_text):
    """Parse named SSE frames → list of (event_name, payload_dict)."""
    out = []
    name = None
    for line in sse_text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[6:])))
    return out


def test_stream_event_sequence_and_metadata(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(SCRIPT)
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "stream": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in r.text
    events = _events(r.text)
    names = [n for n, _ in events]
    assert names == [
        "response.created",
        "response.in_progress",
        "response.atomagents.working",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    seqs = [p["sequence_number"] for _, p in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    done_items = [p["item"] for n, p in events if n == "response.output_item.done"]
    assert done_items[0]["metadata"] == {
        "atom_agent": "Supervisor",
        "atom_stage": "think",
        "atom_type": "ai",
    }
    assert done_items[1]["metadata"]["atom_type"] == "tool"
    # every full response snapshot must parse with the official SDK
    for n, p in events:
        if "response" in p:
            SDKResponse.model_validate(p["response"])


def test_stream_hitl_incomplete(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(INTERRUPT_SCRIPT)
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "stream": True},
    )
    events = _events(r.text)
    assert events[-1][0] == "response.incomplete"
    final = events[-1][1]["response"]
    assert final["status"] == "incomplete"
    assert final["metadata"]["atom_status"] == "awaiting_user"
    assert final["metadata"]["atom_agent"] == "Modeler"
    assert final["incomplete_details"] is None
    assert "response.atomagents.awaiting_user" in [n for n, _ in events]
    SDKResponse.model_validate(final)


def test_stream_cancel(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(SCRIPT, cancel_after=(1, "atomagent:sess-rc"))
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "stream": True,
              "session_dir": "sess-rc"},
    )
    events = _events(r.text)
    names = [n for n, _ in events]
    assert "response.atomagents.cancelled" in names
    assert events[-1][0] == "response.incomplete"
    assert events[-1][1]["response"]["status"] == "cancelled"


def test_stream_failure(rclient):
    tc, holder = rclient
    holder["graph"] = RaisingGraph(SCRIPT)
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "stream": True},
    )
    events = _events(r.text)
    assert events[-1][0] == "response.failed"
    final = events[-1][1]["response"]
    assert final["status"] == "failed"
    assert "solver exploded" in final["error"]["message"]
    SDKResponse.model_validate(final)


def test_non_stream_matches_streamed_text(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(SCRIPT)
    stream_r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "stream": True},
    )
    deltas = [
        p["delta"]
        for n, p in _events(stream_r.text)
        if n == "response.output_text.delta"
    ]
    holder["graph"] = FakeGraph(SCRIPT)
    plain_r = tc.post("/v1/responses", json={"model": "atomagent", "input": "go"})
    assert plain_r.status_code == 200
    body = plain_r.json()
    assert body["output_text"] == "\n\n".join(deltas)
    assert body["status"] == "completed"
    SDKResponse.model_validate(body)


def test_previous_response_id_rejected(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(SCRIPT)
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "previous_response_id": "resp_x"},
    )
    assert r.status_code == 400
    assert "session_dir" in r.json()["detail"]


def test_cancel_endpoint(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(SCRIPT)
    assert tc.post("/v1/responses/resp_nope/cancel").status_code == 404
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "session_dir": "sess-k"},
    )
    rid = r.json()["id"]
    c = tc.post(f"/v1/responses/{rid}/cancel")
    assert c.status_code == 200
    assert c.json()["status"] == "cancelled"
    assert ob.CANCEL_FLAGS.get("atomagent:sess-k") is True
