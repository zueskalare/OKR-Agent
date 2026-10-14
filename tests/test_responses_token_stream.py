"""/v1/responses per-token streaming (WS19.8): the token→item correlation
state machine — real deltas, adoption by the authoritative message, orphan
and cancel closure, SDK validity throughout."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from openai.types.responses import Response as SDKResponse

import backend.openai_backend as ob
from test_chat_golden_frames import FakeGraph


def _tok(stage, text):
    return (
        "custom",
        {"event": "token", "agent": "Supervisor", "stage": stage, "text": text},
    )


def _msg(stage, body, mid):
    return (
        "custom",
        {
            "event": "message",
            "agent": "Supervisor",
            "message": AIMessage(
                content=f"# Supervisor - {stage}\n{body}", name="Supervisor", id=mid
            ),
        },
    )


THINK_TEXT = "# Supervisor - think\nfull think text"

TOKENS_SCRIPT = [
    ("custom", {"event": "working", "agent": "Supervisor"}),
    _tok("think", "raw1 "),
    _tok("think", "raw2"),
    _msg("think", "full think text", "m1"),
    # trailing updates batch re-delivers the same message → dedupe no-op
    (
        "updates",
        {
            "Supervisor": {
                "messages": [
                    AIMessage(content=THINK_TEXT, name="Supervisor", id="m1")
                ]
            }
        },
    ),
]

ORPHAN_SCRIPT = [
    _tok("think", "orphan tokens"),
    _msg("plan", "the plan", "m2"),  # different stage → think segment orphaned
]


@pytest.fixture()
def rclient(tmp_path, monkeypatch):
    holder = {"graph": None}

    async def fake_get_or_build(graph_key, model_id, session_dir):
        return holder["graph"], tmp_path

    monkeypatch.setattr(ob, "_get_or_build_graph", fake_get_or_build)
    ob.CANCEL_FLAGS.clear()
    ob._RUN_LOCKS.clear()
    return TestClient(ob.app), holder


def _events(sse_text):
    out = []
    name = None
    for line in sse_text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[6:])))
    return out


def _post_stream(tc, session="s-tok"):
    return tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "stream": True,
              "session_dir": session},
    )


def test_token_segment_adopted_by_message(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(TOKENS_SCRIPT)
    events = _events(_post_stream(tc).text)
    names = [n for n, _ in events]
    assert names == [
        "response.created",
        "response.in_progress",
        "response.atomagents.working",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    deltas = [p["delta"] for n, p in events if n == "response.output_text.delta"]
    assert deltas == ["raw1 ", "raw2"]  # raw-LLM preview
    done = next(p for n, p in events if n == "response.output_text.done")
    assert done["text"] == THINK_TEXT  # authoritative stage-bound text
    final = events[-1][1]["response"]
    assert len(final["output"]) == 1  # adopted, not duplicated
    assert final["output"][0]["content"][0]["text"] == THINK_TEXT
    assert final["output"][0]["metadata"]["atom_stage"] == "think"
    SDKResponse.model_validate(final)


def test_tokenless_message_keeps_quartet(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph([_msg("plan", "no tokens preceded", "m9")])
    events = _events(_post_stream(tc).text)
    names = [n for n, _ in events]
    assert names[2:] == [
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]


def test_orphan_segment_closed_with_preview(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(ORPHAN_SCRIPT)
    events = _events(_post_stream(tc).text)
    dones = [p for n, p in events if n == "response.output_text.done"]
    assert dones[0]["text"] == "orphan tokens"  # buffered preview
    assert "the plan" in dones[1]["text"]
    final = events[-1][1]["response"]
    assert len(final["output"]) == 2
    assert final["output"][0]["content"][0]["text"] == "orphan tokens"
    SDKResponse.model_validate(final)


def test_cancel_mid_segment_closes_item(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(
        [_tok("think", "partial"), _tok("think", " more")],
        cancel_after=(1, "atomagent:s-tok"),
    )
    events = _events(_post_stream(tc).text)
    names = [n for n, _ in events]
    assert "response.output_item.done" in names  # segment closed
    assert "response.atomagents.cancelled" in names
    final = events[-1][1]["response"]
    assert final["status"] == "cancelled"
    assert final["output"][0]["content"][0]["text"] == "partial"
    SDKResponse.model_validate(final)


def test_non_stream_never_sees_tokens(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(TOKENS_SCRIPT)
    r = tc.post(
        "/v1/responses",
        json={"model": "atomagent", "input": "go", "session_dir": "s-tok"},
    )
    body = r.json()
    # include_progress off → only the updates batch message; one item,
    # authoritative text, no preview residue
    assert len(body["output"]) == 1
    assert body["output"][0]["content"][0]["text"] == THINK_TEXT
    assert body["output_text"] == THINK_TEXT
    SDKResponse.model_validate(body)


# ---------------------------------------------------------------------------
# A2 (WS25): act messages carrying preamble text + tool calls have no
# "# Agent - stage" header (their content is never rewritten), so the
# adoption key (agent, stage) missed and the SAME text shipped as TWO output
# items. The agents now stamp additional_kwargs["atom_stage"]; the emitter
# falls back to it.
# ---------------------------------------------------------------------------


def _act_tool_call_msg(mid="m5", stamped=True):
    kwargs = {"atom_stage": "act"} if stamped else {}
    return (
        "custom",
        {
            "event": "message",
            "agent": "Supervisor",
            "message": AIMessage(
                content="Let me check.",
                name="Supervisor",
                id=mid,
                tool_calls=[{"name": "bash_run", "args": {"cmd": "ls"}, "id": "t1"}],
                additional_kwargs=kwargs,
            ),
        },
    )


def test_act_preamble_with_tool_calls_emits_one_item(rclient):
    tc, holder = rclient
    holder["graph"] = FakeGraph(
        [_tok("act", "Let me "), _tok("act", "check."), _act_tool_call_msg()]
    )
    events = _events(_post_stream(tc).text)
    names = [n for n, _ in events]
    assert names.count("response.output_item.added") == 1
    assert names.count("response.output_item.done") == 1
    final = events[-1][1]["response"]
    assert len(final["output"]) == 1  # adopted — was 2 (duplicated text) pre-fix
    assert final["output"][0]["content"][0]["text"] == "Let me check."
    assert final["output"][0]["metadata"]["atom_stage"] == "act"
    assert final["output_text"] == "Let me check."
    SDKResponse.model_validate(final)


def test_event_from_message_falls_back_to_stage_metadata():
    ev = ob._event_from_message(
        AIMessage(
            content="Let me check.",
            name="Supervisor",
            additional_kwargs={"atom_stage": "act"},
        )
    )
    assert ev.stage == "act"
    # An explicit header still wins over the metadata.
    ev2 = ob._event_from_message(
        AIMessage(
            content="# Supervisor - plan\nbody",
            name="Supervisor",
            additional_kwargs={"atom_stage": "act"},
        )
    )
    assert ev2.stage == "plan"


def test_act_tool_call_message_carries_stage_metadata(tmp_path):
    """Agent side of A2: the act tool-call branch stamps atom_stage."""
    from langchain_core.tools import tool as lc_tool

    from AtomAgents.agents.model_building_agent import Modeler
    from fakes import FakeLLM

    @lc_tool
    def probe_tool(x: str) -> str:
        """probe"""
        return "ok"

    m = Modeler(
        llm=FakeLLM(
            [
                AIMessage(
                    content="Let me check.",
                    tool_calls=[
                        {"name": "probe_tool", "args": {"x": "1"}, "id": "t1"}
                    ],
                )
            ]
        ),
        memory=None,
        Collages=["Supervisor", "END"],
        tools=[probe_tool],
        project_root=str(tmp_path),
    )
    from langchain_core.messages import HumanMessage

    result = m.act({"messages": [HumanMessage(content="go")]})
    action = result["messages"][-1]
    assert action.tool_calls
    assert action.additional_kwargs.get("atom_stage") == "act"
