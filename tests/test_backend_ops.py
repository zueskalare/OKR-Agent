"""Backend ops hardening (WS19.5): per-session run lock, off-loop build,
shutdown closes checkpointer connections."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import backend.openai_backend as ob
from test_chat_golden_frames import SCRIPT, FakeGraph


def test_same_session_runs_serialize():
    """Two concurrent 'runs' on one graph_key must not interleave; different
    keys stay concurrent. Tests _acquire_run_lock + release contract at the
    asyncio level (hermetic — TestClient portal threading stays out of it)."""
    ob._RUN_LOCKS.clear()
    trace: list = []

    async def fake_run(key, label):
        lock = await ob._acquire_run_lock(key)
        try:
            trace.append(f"enter-{label}")
            await asyncio.sleep(0.02)
            trace.append(f"exit-{label}")
        finally:
            lock.release()

    async def go():
        await asyncio.gather(
            fake_run("atomagent:s1", "a"),
            fake_run("atomagent:s1", "b"),
        )
        same = list(trace)
        trace.clear()
        await asyncio.gather(
            fake_run("atomagent:s1", "a"),
            fake_run("atomagent:OTHER", "c"),
        )
        return same, list(trace)

    same, mixed = asyncio.run(go())
    assert same in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    )
    # different sessions interleave freely (both enter before either exits)
    assert mixed[0].startswith("enter") and mixed[1].startswith("enter")


def test_lock_timeout_yields_409(monkeypatch):
    from AtomAgents import config
    from fastapi import HTTPException

    monkeypatch.setattr(config, "SESSION_LOCK_TIMEOUT", 0.05)
    ob._RUN_LOCKS.clear()

    async def go():
        first = await ob._acquire_run_lock("atomagent:s-busy")
        try:
            with pytest.raises(HTTPException) as exc:
                await ob._acquire_run_lock("atomagent:s-busy")
            return exc.value
        finally:
            first.release()

    err = asyncio.run(go())
    assert err.status_code == 409
    assert "session busy" in err.detail


def test_build_cfg_runs_off_loop(monkeypatch, tmp_path):
    """_prepare_build_cfg is the blocking half — _get_or_build_graph must
    run it via to_thread (i.e. NOT on the event loop)."""
    seen = {}

    def fake_prepare(model_id, session_dir):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        raise RuntimeError("stop-here")  # abort the build after the probe

    monkeypatch.setattr(ob, "_prepare_build_cfg", fake_prepare)
    monkeypatch.setattr(ob, "_initialize_components", lambda: {"ok": True})
    ob._APP_GRAPHS.pop("probe-key", None)
    ob._PROJECT_ROOTS.pop("probe-key", None)

    async def go():
        with pytest.raises(RuntimeError, match="stop-here"):
            await ob._get_or_build_graph("probe-key", "atomagent", "probe-sess")

    asyncio.run(go())
    assert seen["on_loop"] is False


def test_shutdown_closes_checkpointer_conns():
    closed = []

    class FakeConn:
        async def close(self):
            closed.append(True)

    ob._CHECKPOINTER_CONNS["k1"] = FakeConn()
    ob._CHECKPOINTER_CONNS["k2"] = FakeConn()
    asyncio.run(ob.shutdown_event())
    assert closed == [True, True]
    assert not ob._CHECKPOINTER_CONNS
