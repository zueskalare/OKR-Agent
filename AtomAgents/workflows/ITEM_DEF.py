"""Shared workflow items.

Historically this module constructed a ChatOpenAI client, an OpenAI client,
an InMemorySaver, and an InMemoryStore at import time — none of which were
used by the live code paths (the checkpointer that WOULD have fixed session
continuity sat here unused). It now exposes lazy factories with no import-time
side effects; the real checkpointer wiring lives in the graph builders.
"""

from __future__ import annotations


def make_checkpointer():
    """In-memory checkpointer factory (durable SqliteSaver wiring lives in the
    graph builders, keyed to the session directory)."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ModuleNotFoundError:
        return None


def make_store(dims: int = 1536):
    try:
        from langgraph.store.memory import InMemoryStore
    except ModuleNotFoundError:
        return None

    def embed_texts(texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return [e.embedding for e in response.data]
        except Exception:
            return [[] for _ in texts]

    return InMemoryStore(index={"dims": dims, "embed": embed_texts})


# Backwards-compatible names (previously eager instances).
LLM = None
Checkpointer = None
Store = None

__all__ = ["LLM", "Checkpointer", "Store", "make_checkpointer", "make_store"]
