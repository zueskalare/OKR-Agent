"""Embedding-backend selection for :class:`SQLiteRAGMemory`.

Why this module exists
----------------------
``OpenAIEmbeddings`` inherits ``OPENAI_BASE_URL`` from the environment. On a
chat-only gateway deployment that URL serves chat completions but **no**
``/v1/embeddings`` (verified: 404 with an empty body), so every embed call
fails and ``SQLiteRAGMemory`` quietly downgrades to "storing without vector"
with a per-call WARNING. A WS27 pilot logged 22 of those in a single 6.5-minute
run, and semantic recall had been off for the whole session without anyone
noticing.

The failure mode was not "no embeddings" — it was "no embeddings, silently".
:func:`build_embeddings` probes the configured backend **once**, states what it
resolved to, and falls back to a local model instead of degrading in place.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

_logger = logging.getLogger("AtomAgents.utils.embeddings")

# A single short probe string. Kept tiny so the startup cost is one cheap call
# on the remote path and one forward pass on the local path.
_PROBE_TEXT = "embedding probe"


class LocalEmbeddings:
    """sentence-transformers embeddings behind the LangChain embedding API.

    Loads lazily so that merely constructing the object (as ``build_embeddings``
    does while deciding) never pays the model-load cost, and so an import-time
    failure surfaces at the probe rather than at process start.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self._device} if self._device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure()
        vectors = model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _probe(model: Any) -> None:
    """Raise unless ``model`` returns a non-empty vector for a real call."""
    vector = model.embed_query(_PROBE_TEXT)
    if not vector:
        raise RuntimeError("embedding backend returned an empty vector")


def _build_remote() -> Any:
    from langchain_openai import OpenAIEmbeddings

    from AtomAgents import config as _cfg

    kwargs: dict[str, Any] = {"model": _cfg.EMBED_MODEL}
    # Both of these default to empty, i.e. "inherit OPENAI_BASE_URL /
    # OPENAI_API_KEY". Inheriting the base URL is exactly what broke recall on
    # a gateway deployment, so setting them explicitly is the supported path.
    if _cfg.EMBED_BASE_URL:
        kwargs["base_url"] = _cfg.EMBED_BASE_URL
    if getattr(_cfg, "EMBED_API_KEY", ""):
        kwargs["api_key"] = _cfg.EMBED_API_KEY
    # Keep the probe fast: one attempt, short timeout. A slow-failing embedding
    # endpoint must not stall session construction.
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("max_retries", 1)
    return OpenAIEmbeddings(**kwargs)


def build_embeddings(logger: logging.Logger | None = None) -> tuple[Any, str]:
    """Resolve the embedding backend, probing it before handing it back.

    Returns ``(embedding_model, description)``. ``embedding_model`` is ``None``
    only when semantic recall is genuinely unavailable — and in that case the
    reason is logged at ERROR, not swallowed. ``description`` is short enough to
    put in a run manifest, which is the point: a campaign must be able to prove
    afterwards whether recall was on.
    """
    log = logger or _logger

    from AtomAgents import config as _cfg

    provider = (_cfg.EMBED_PROVIDER or "auto").strip().lower()

    if provider == "none":
        log.warning(
            "Semantic recall DISABLED by ATOMAGENTS_EMBED_PROVIDER=none; "
            "keyword/FTS recall still works."
        )
        return None, "none (disabled by config)"

    remote_error: Exception | None = None
    if provider in ("auto", "remote"):
        try:
            model = _build_remote()
            _probe(model)
            target = _cfg.EMBED_BASE_URL or "provider default"
            desc = f"remote:{_cfg.EMBED_MODEL} @ {target}"
            log.info("Embeddings: %s (probe OK)", desc)
            return model, desc
        except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
            remote_error = exc
            if provider == "remote":
                # Explicitly demanded remote: fail loudly rather than pretend.
                log.error(
                    "Embeddings UNAVAILABLE: remote endpoint %s rejected the "
                    "probe (%s). Semantic recall is OFF. Set "
                    "ATOMAGENTS_EMBED_PROVIDER=local or point "
                    "ATOMAGENTS_EMBED_BASE_URL at a working endpoint.",
                    _cfg.EMBED_BASE_URL or "provider default",
                    exc,
                )
                return None, f"none (remote probe failed: {exc})"
            log.warning(
                "Embeddings: remote probe failed (%s); falling back to the "
                "local model. Set ATOMAGENTS_EMBED_PROVIDER=local to skip this "
                "probe, or =remote to make it fatal.",
                exc,
            )

    try:
        model = LocalEmbeddings(
            _cfg.EMBED_LOCAL_MODEL, device=getattr(_cfg, "EMBED_DEVICE", "") or None
        )
        _probe(model)
        desc = f"local:{_cfg.EMBED_LOCAL_MODEL}"
        log.info("Embeddings: %s (probe OK)", desc)
        return model, desc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Embeddings UNAVAILABLE: local model %s failed (%s)%s. Semantic "
            "recall is OFF for this session; keyword/FTS recall still works.",
            _cfg.EMBED_LOCAL_MODEL,
            exc,
            f"; remote also failed ({remote_error})" if remote_error else "",
        )
        return None, f"none (local failed: {exc})"
