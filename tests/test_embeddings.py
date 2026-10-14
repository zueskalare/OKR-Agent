"""Contracts for the embedding-backend resolver.

The bug these pin: OpenAIEmbeddings was constructed but never called, so a base
URL serving no /v1/embeddings degraded to "storing without vector" with only a
per-call WARNING. A WS27 pilot logged 22 of them in one run and semantic recall
had been off the whole time. The resolver must PROBE, must say what it
resolved to, and must never return a backend it has not exercised.
"""

import logging

import pytest

from AtomAgents.utils import embeddings as emb


class _FakeEmb:
    """Minimal LangChain-embedding-shaped double."""

    def __init__(self, vector=(0.1, 0.2, 0.3), boom=None):
        self._vector = list(vector)
        self._boom = boom
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        if self._boom is not None:
            raise self._boom
        return list(self._vector)

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def cfg(monkeypatch):
    from AtomAgents import config as _cfg

    monkeypatch.setattr(_cfg, "EMBED_PROVIDER", "auto", raising=False)
    monkeypatch.setattr(_cfg, "EMBED_MODEL", "text-embedding-3-large", raising=False)
    monkeypatch.setattr(_cfg, "EMBED_BASE_URL", "", raising=False)
    monkeypatch.setattr(_cfg, "EMBED_LOCAL_MODEL", "fake/local", raising=False)
    return _cfg


def test_none_disables_without_pretending(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "EMBED_PROVIDER", "none")
    model, desc = emb.build_embeddings()
    assert model is None
    assert desc.startswith("none")


def test_remote_is_probed_not_merely_constructed(cfg, monkeypatch):
    """The whole point: a backend is only returned after a real call."""
    fake = _FakeEmb()
    monkeypatch.setattr(emb, "_build_remote", lambda: fake)
    model, desc = emb.build_embeddings()
    assert model is fake
    assert fake.calls == 1, "resolver returned a backend it never exercised"
    assert desc.startswith("remote:")


def test_remote_returning_empty_vector_is_a_failure(cfg, monkeypatch):
    """A 200 that yields nothing is exactly the gateway's behaviour."""
    monkeypatch.setattr(emb, "_build_remote", lambda: _FakeEmb(vector=()))
    local = _FakeEmb()
    monkeypatch.setattr(emb, "LocalEmbeddings", lambda name, device=None: local)
    model, desc = emb.build_embeddings()
    assert model is local and desc.startswith("local:")


def test_auto_falls_back_to_local_when_remote_404s(cfg, monkeypatch):
    def _boom():
        raise RuntimeError("Error code: 404")

    monkeypatch.setattr(emb, "_build_remote", _boom)
    local = _FakeEmb()
    monkeypatch.setattr(emb, "LocalEmbeddings", lambda name, device=None: local)
    model, desc = emb.build_embeddings()
    assert model is local
    assert local.calls == 1
    assert desc == "local:fake/local"


def test_remote_mode_never_silently_falls_back(cfg, monkeypatch):
    """provider=remote means a dead endpoint is loud, not papered over."""
    monkeypatch.setattr(cfg, "EMBED_PROVIDER", "remote")

    def _boom():
        raise RuntimeError("Error code: 404")

    monkeypatch.setattr(emb, "_build_remote", _boom)
    monkeypatch.setattr(
        emb, "LocalEmbeddings", lambda name, device=None: pytest.fail("must not fall back")
    )
    model, desc = emb.build_embeddings()
    assert model is None
    assert "remote probe failed" in desc


def test_total_failure_is_logged_at_error(cfg, monkeypatch, caplog):
    def _boom():
        raise RuntimeError("Error code: 404")

    monkeypatch.setattr(emb, "_build_remote", _boom)
    monkeypatch.setattr(
        emb, "LocalEmbeddings", lambda name, device=None: _FakeEmb(boom=RuntimeError("no torch"))
    )
    with caplog.at_level(logging.ERROR):
        model, desc = emb.build_embeddings()
    assert model is None
    assert any(r.levelno >= logging.ERROR for r in caplog.records), (
        "an unusable embedding backend must be an ERROR, not a silent downgrade"
    )
    # both causes survive into the description that lands in the run manifest
    assert "no torch" in desc


def test_local_embeddings_is_lazy():
    """Constructing must not load the model — build_embeddings constructs
    candidates while deciding."""
    local = emb.LocalEmbeddings("definitely/not-a-real-model")
    assert local._model is None


def test_remote_uses_its_own_base_url_and_key(cfg, monkeypatch):
    """The whole point of a separate embedding config: the remote embedder must
    NOT silently inherit OPENAI_BASE_URL / OPENAI_API_KEY, because on a gateway
    deployment that URL serves no /v1/embeddings."""
    import sys
    import types

    seen = {}

    class _Cap:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def embed_query(self, text):
            return [0.1, 0.2]

    fake_mod = types.ModuleType("langchain_openai")
    fake_mod.OpenAIEmbeddings = _Cap
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)
    monkeypatch.setattr(cfg, "EMBED_PROVIDER", "remote")
    monkeypatch.setattr(cfg, "EMBED_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(cfg, "EMBED_API_KEY", "sk-test-not-a-real-key")

    model, desc = emb.build_embeddings()
    assert model is not None
    assert seen["base_url"] == "https://api.openai.com/v1"
    assert seen["api_key"] == "sk-test-not-a-real-key"
    assert seen["max_retries"] == 1, "a dead endpoint must fail fast, not retry"
    assert "api.openai.com" in desc


def test_local_device_is_passed_through(cfg, monkeypatch):
    seen = {}

    class _Cap:
        def __init__(self, name, device=None):
            seen["name"] = name
            seen["device"] = device

        def embed_query(self, text):
            return [0.1]

    monkeypatch.setattr(cfg, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(cfg, "EMBED_DEVICE", "cpu")
    monkeypatch.setattr(emb, "LocalEmbeddings", _Cap)
    model, _ = emb.build_embeddings()
    assert model is not None
    assert seen == {"name": "fake/local", "device": "cpu"}


def test_blank_device_means_let_the_library_choose(cfg, monkeypatch):
    seen = {}

    class _Cap:
        def __init__(self, name, device=None):
            seen["device"] = device

        def embed_query(self, text):
            return [0.1]

    monkeypatch.setattr(cfg, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(cfg, "EMBED_DEVICE", "")
    monkeypatch.setattr(emb, "LocalEmbeddings", _Cap)
    emb.build_embeddings()
    assert seen["device"] is None


def test_memory_default_resolves_through_build_embeddings(monkeypatch, tmp_path):
    """SQLiteRAGMemory's own default was the second silent-404 site: a bare
    OpenAIEmbeddings() inheriting the gateway URL. Any caller that does not
    pass an embedder must now go through the probing resolver."""
    from AtomAgents.memory import sqlite_rag_memory as srm

    fake = _FakeEmb(vector=(0.5, 0.5))
    # SQLiteRAGMemory imports build_embeddings lazily inside __init__, so
    # patching it on the module it is imported FROM is what takes effect.
    monkeypatch.setattr(emb, "build_embeddings", lambda logger=None: (fake, "local:fake"))
    mem = srm.SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "m.db"), project_root=str(tmp_path)
    )
    assert mem.embedding_model is fake
    assert mem.embedding_desc == "local:fake"
    assert mem.embedding_fn("hello") == [0.5, 0.5]
