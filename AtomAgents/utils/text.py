"""Canonical message-text flattening for the framework.

Historically FIVE independent flatteners existed (BaseAgent._content_text,
SQLiteRAGMemory._flatten_query, SQLiteRAGMemory._get_text_content,
prefill_safe_messages' private _text, and an inline copy in the backend's
graph-event loop). This module is the single implementation they all
delegate to.

Design notes (verified against langchain-core 1.5.3):
- Given a BaseMessage, we use ``message.content_blocks`` — langchain's
  provider-aware normalization, which classifies anthropic ``thinking`` and
  Responses ``reasoning`` items as non-text so they never leak into prompts,
  persistence, or the wire.
- We deliberately do NOT use ``message.text``: it concatenates multiple text
  blocks WITHOUT a separator ("part Apart B"), while the framework contract
  since inception joins with "\\n" — stage-header parsing and golden-frame
  wire tests depend on that.
- Never raises; unknown shapes degrade to "" or str(value).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - keep importable in minimal envs

    class BaseMessage:  # type: ignore
        pass


_TEXTUAL_BLOCK_TYPES = ("text", "input_text", "output_text")


def message_text(value: Any) -> str:
    """Flatten a message OR raw content value to plain text.

    Accepts a BaseMessage, a str, a content-block list, a single block dict,
    or None. Text-ish blocks are joined with "\\n"; reasoning/thinking/
    tool-call/image blocks are ignored.
    """
    if isinstance(value, BaseMessage):
        try:
            blocks = value.content_blocks
        except Exception:
            return message_text(getattr(value, "content", ""))
        parts = [
            str(b.get("text") or "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") in _TEXTUAL_BLOCK_TYPES:
            return str(value.get("text") or "")
        return ""
    if isinstance(value, list):
        parts = [message_text(item) for item in value]
        return "\n".join(p for p in parts if p)
    return "" if value is None else str(value)


__all__ = ["message_text"]
