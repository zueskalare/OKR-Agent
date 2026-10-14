"""Session lifecycle (WS20.4): LRU eviction of sessionful graphs + graceful
SIGTERM drain. Continuity is the on-disk checkpoint — eviction only drops
in-memory registries and closes the aiosqlite conn."""

import asyncio

import pytest

import backend.openai_backend as ob
from test_chat_golden_frames import SCRIPT, FakeGraph


class FakeConn:
    def __init__(self, closed):
        self._closed = closed

    async def close(self):
        self._closed.append(self)


@pytest.fixture(autouse=True)
def clean_registries():
    regs = (
        ob._APP_GRAPHS,
        ob._PROJECT_ROOTS,
        ob._CHECKPOINTER_CONNS,
        ob._BUILD_LOCKS,
        ob._RUN_LOCKS,
        ob._SESSION_LRU,
        ob.CANCEL_FLAGS,
    )
    saved = [dict(r) for r in regs]
    for r in regs:
        r.clear()
    yield
    for r, s in zip(regs, saved):
        r.clear()
        r.update(s)


def _seed(key, closed):
    ob._APP_GRAPHS[key] = object()
    ob._PROJECT_ROOTS[key] = f"/tmp/{key}"
    ob._CHECKPOINTER_CONNS[key] = FakeConn(closed)
    ob._touch_session(key, sessionful=True)


def test_evicts_oldest_closes_conn_drops_registries(monkeypatch):
    monkeypatch.setattr(ob.atom_config, "MAX_SESSIONS", 2)
    closed = []
    for k in ("m:s1", "m:s2", "m:s3"):
        _seed(k, closed)
    asyncio.run(ob._evict_lru_sessions())
    assert "m:s1" not in ob._APP_GRAPHS
    assert "m:s1" not in ob._SESSION_LRU
    assert "m:s1" not in ob._CHECKPOINTER_CONNS
    assert len(closed) == 1  # exactly the evicted conn
    assert set(ob._SESSION_LRU) == {"m:s2", "m:s3"}


def test_touch_refreshes_recency(monkeypatch):
    monkeypatch.setattr(ob.atom_config, "MAX_SESSIONS", 2)
    closed = []
    for k in ("m:s1", "m:s2", "m:s3"):
        _seed(k, closed)
    ob._touch_session("m:s1", sessionful=True)  # s1 becomes MRU
    asyncio.run(ob._evict_lru_sessions())
    assert set(ob._SESSION_LRU) == {"m:s3", "m:s1"}  # s2 was oldest


def test_eviction_skips_held_run_locks(monkeypatch):
    monkeypatch.setattr(ob.atom_config, "MAX_SESSIONS", 1)
    closed = []
    for k in ("m:s1", "m:s2"):
        _seed(k, closed)

    async def go():
        lock = ob._RUN_LOCKS.setdefault("m:s1", asyncio.Lock())
        await lock.acquire()  # s1 is mid-run
        try:
            await ob._evict_lru_sessions()
        finally:
            lock.release()

    asyncio.run(go())
    assert "m:s1" in ob._SESSION_LRU  # oldest but running — kept
    assert "m:s2" not in ob._SESSION_LRU  # next-oldest evicted instead


def test_anonymous_keys_never_tracked():
    ob._touch_session("bare-model-id", sessionful=False)
    assert not ob._SESSION_LRU


def test_zero_limit_disables_eviction(monkeypatch):
    monkeypatch.setattr(ob.atom_config, "MAX_SESSIONS", 0)
    closed = []
    for k in ("m:s1", "m:s2", "m:s3"):
        _seed(k, closed)
    asyncio.run(ob._evict_lru_sessions())
    assert len(ob._SESSION_LRU) == 3 and not closed


def test_drain_ends_stream_with_resume_hint(monkeypatch):
    monkeypatch.setattr(ob, "_DRAINING", True)

    async def collect():
        return [
            ev
            async for ev in ob.aiter_graph_events(
                FakeGraph(SCRIPT),
                {"messages": []},
                {"configurable": {"thread_id": "t"}},
            )
        ]

    events = asyncio.run(collect())
    assert len(events) == 1
    assert events[0].kind == "cancelled"
    assert "draining" in events[0].text and "resume" in events[0].text


def test_begin_drain_idempotent_sets_flag(monkeypatch):
    raised = []
    monkeypatch.setattr(ob.signal, "raise_signal", lambda s: raised.append(s))
    monkeypatch.setattr(ob, "_DRAINING", False)
    ob._begin_drain()
    ob._begin_drain()  # second call is a no-op
    assert ob._DRAINING is True
    assert raised == [ob.signal.SIGINT]
