"""
LLM factory: build a chat model for any configured provider.

Providers are named API backends discovered from the environment (see
``config.discover_providers``). Requests select one with a ``model@provider``
id; agents receive whatever the factory builds, so the rest of the framework
is provider-agnostic.

The factory owns the parameter-mapping layer: one ``llm_config`` dict works
across providers — kwargs a backend does not support (e.g. ``reasoning_effort``
on Anthropic) are dropped with a debug log instead of crashing the request.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from . import config as _config
from .config import default_provider, discover_providers

logger = logging.getLogger("AtomAgents.llm_factory")

# Process-wide limiter shared by every model the factory builds, so the cap
# applies across the whole multi-agent fan-out, not per agent. Lazy: only
# built when ATOMAGENTS_LLM_RPS > 0.
_shared_rate_limiter = None


def _rate_limiter():
    global _shared_rate_limiter
    rps = float(getattr(_config, "LLM_RPS", 0.0) or 0.0)
    if rps <= 0:
        return None
    if _shared_rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _shared_rate_limiter = InMemoryRateLimiter(
            requests_per_second=rps, check_every_n_seconds=0.1
        )
        logger.info("LLM rate limiter enabled: %.2f req/s (process-wide)", rps)
    return _shared_rate_limiter


def cache_system_blocks(messages):
    """Stamp anthropic ``cache_control`` on the FIRST system message.

    Marks a prompt-cache breakpoint after the large static agent prompt, so
    every subsequent call in the session reads the prefix from the provider
    cache. Only the first system message is marked: later stage prompts vary
    per node and would fragment the cache. No-op for non-system-leading
    lists; never raises.
    """
    try:
        from langchain_core.messages import SystemMessage

        out = list(messages)
        for i, msg in enumerate(out):
            if not isinstance(msg, SystemMessage):
                continue
            content = msg.content
            if isinstance(content, str):
                blocks = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
                if isinstance(blocks[-1], dict) and blocks[-1].get("type") == "text":
                    blocks[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    break
            else:
                break
            out[i] = SystemMessage(content=blocks)
            break  # first system message only
        return out
    except Exception:  # pragma: no cover - cache marking must never break a call
        return messages


def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """Split a ``pipeline@provider`` id into (pipeline, provider|None).

    ``atomagent`` → ("atomagent", None);
    ``atomagent@anthropic`` → ("atomagent", "anthropic").
    """
    if "@" in (model_id or ""):
        pipeline, _, provider = model_id.partition("@")
        return pipeline.strip(), provider.strip() or None
    return (model_id or "").strip(), None


def resolve_provider(name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Return (name, provider_config); validates that the provider exists."""
    providers = discover_providers()
    name = name or default_provider()
    if name not in providers:
        raise ValueError(
            f"Unknown LLM provider {name!r}; available: {sorted(providers)}"
        )
    return name, providers[name]


def _allow_claude_via_openai() -> bool:
    # Env var FIRST, read live: the ValueError's remediation text advertises
    # ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI=1, and it must work from a running
    # kernel/notebook — the config attribute is frozen at import time, so
    # checking it first made the env path dead code.
    raw = os.environ.get("ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI")
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(getattr(_config, "ALLOW_CLAUDE_VIA_OPENAI", False))


def _assert_model_provider_compatible(
    model: Optional[str], api_type: str, provider: str, strict: bool = True
) -> None:
    """Fail fast (or warn) on the known-broken model/route pairing.

    A claude-* model served through an OpenAI-compatible route answers
    structured-output calls with fenced/prose JSON — TriageOut, UpdateTask,
    TaskOperations and SupervisorPlanOut all fail validation and the turn
    dies in agent_error_handler having produced nothing (observed live with
    ATOMAGENTS_OPENAI_MODEL=claude-opus-5: 129 s to a dead turn; PLAN_BOOK
    WS24). The only supported Claude route is the anthropic provider.

    ``strict`` controls the failure mode: True raises (the stock ``openai``
    registry provider — the misconfiguration observed live); False logs an
    error and continues (custom providers and legacy explicit-endpoint
    configs, where the operator may deliberately point Claude at a gateway
    that genuinely translates the surface, e.g. LiteLLM-style).
    ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI=1 downgrades strict mode to the log
    line too. Model detection is substring-based so gateway spellings
    (us.anthropic.claude-*, bedrock/anthropic.claude-*) are covered.
    """
    lowered = (model or "").strip().lower()
    if not lowered:
        return
    if api_type == "openai" and ("claude" in lowered or "anthropic" in lowered):
        msg = (
            f"Model {model!r} is a Claude model but provider {provider!r} "
            "speaks the OpenAI-compatible surface — every structured call "
            "will fail (fenced/prose JSON instead of schema output). Route "
            "it through the anthropic provider instead: use a "
            "'<pipeline>@anthropic' model id, or set "
            "ATOMAGENTS_LLM_PROVIDER=anthropic (and "
            "ATOMAGENTS_ANTHROPIC_MODEL). Set "
            "ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI=1 to bypass this check."
        )
        if not strict or _allow_claude_via_openai():
            logger.error("%s", msg)
        else:
            raise ValueError(msg)
    elif api_type == "anthropic" and (
        "gpt" in lowered or re.search(r"\bo\d", lowered)
    ):
        logger.warning(
            "Model %r looks like an OpenAI model on the anthropic provider "
            "%r — check ATOMAGENTS_ANTHROPIC_MODEL / the model id.",
            model,
            provider,
        )


def prefill_safe_messages(messages):
    """Return *messages* adjusted to satisfy Anthropic's conversation rules.

    Anthropic validates strictly where OpenAI-compatible routes are lenient:
    - the conversation must end with a user turn (no "assistant prefill");
      agents legitimately end context with prior-stage AIMessages, so a
      continuation user message is appended;
    - every tool_result must sit immediately after the assistant message
      carrying its tool_use, and every tool_use needs results before the
      conversation moves on — violations are repaired (orphan results become
      plain user text, unanswered tool_use gets placeholder results) with a
      warning, so an upstream assembly bug degrades instead of 400ing the run;
    - the first non-system message must not be assistant.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    def _text(content) -> str:
        from .utils.text import message_text

        return message_text(content)

    msgs = []
    open_ids: list = []  # tool_use ids of the last assistant turn still awaiting results
    for m in messages:
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid in open_ids:
                open_ids.remove(tcid)
                msgs.append(m)
            else:
                logger.warning(
                    "prefill_safe: orphan tool_result %s (no matching tool_use "
                    "in the preceding assistant message) — converting to text",
                    tcid,
                )
                msgs.append(
                    HumanMessage(content=f"[tool result {tcid}]\n{_text(m.content)}")
                )
            continue
        if open_ids:
            logger.warning(
                "prefill_safe: %d unanswered tool_use id(s) before a "
                "non-tool message — inserting placeholder results",
                len(open_ids),
            )
            msgs.extend(
                ToolMessage(content="(no result recorded)", tool_call_id=i)
                for i in open_ids
            )
            open_ids = []
        if isinstance(m, AIMessage) and (getattr(m, "tool_calls", None) or []):
            open_ids = [tc["id"] for tc in m.tool_calls]
        msgs.append(m)

    for i, m in enumerate(msgs):
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, AIMessage):
            msgs.insert(i, HumanMessage(content="(context follows)"))
        break
    if open_ids:
        msgs.extend(
            ToolMessage(content="(no result recorded)", tool_call_id=i)
            for i in open_ids
        )
    elif msgs and isinstance(msgs[-1], AIMessage):
        msgs.append(HumanMessage(content="Continue."))
    return msgs


_prefill_safe_cls = None


def _prefill_safe_anthropic():
    """ChatAnthropic subclass that normalizes messages on every call path.

    bind_tools/with_structured_output return RunnableBindings that delegate
    back to these methods, so overriding the four generate/stream entry
    points covers plain, tool-bound, structured, and streaming calls alike.
    """
    global _prefill_safe_cls
    if _prefill_safe_cls is None:
        from langchain_anthropic import ChatAnthropic

        def _prep(messages):
            msgs = prefill_safe_messages(messages)
            # Prompt-cache breakpoint on the static agent prompt (env-gated;
            # read at call time so tests/ops can flip it live).
            if getattr(_config, "ANTHROPIC_PROMPT_CACHE", True):
                msgs = cache_system_blocks(msgs)
            return msgs

        class _ModelDumpDict(dict):
            """A dict that satisfies langchain-anthropic's `.model_dump()` calls.

            The real Anthropic SDK parses `context_management`/`container`
            into pydantic models; through an OpenAI-compatible proxy
            (a translating gateway) they arrive as SDK-unknown extra fields and stay
            plain dicts, so upstream's `.model_dump()` on the final
            `message_delta` event crashed EVERY streamed call
            (`AttributeError: 'dict' object has no attribute 'model_dump'` —
            killed all three opus backend demo runs at the first prose
            stream; non-streamed calls were unaffected, which is why the
            token-stream-off notebooks never saw it).
            """

            def model_dump(self, *args, **kwargs):
                return dict(self)

        def _normalize_stream_event(event):
            """Make proxy-shaped stream events safe for the upstream parser."""
            try:
                cm = getattr(event, "context_management", None)
                if type(cm) is dict:
                    object.__setattr__(event, "context_management", _ModelDumpDict(cm))
                delta = getattr(event, "delta", None)
                container = getattr(delta, "container", None) if delta is not None else None
                if type(container) is dict:
                    object.__setattr__(delta, "container", _ModelDumpDict(container))
            except Exception:
                pass  # never break a stream over defensive normalization
            return event

        class PrefillSafeChatAnthropic(ChatAnthropic):
            def _generate(self, messages, *args, **kwargs):
                return super()._generate(_prep(messages), *args, **kwargs)

            async def _agenerate(self, messages, *args, **kwargs):
                return await super()._agenerate(_prep(messages), *args, **kwargs)

            def _stream(self, messages, *args, **kwargs):
                return super()._stream(_prep(messages), *args, **kwargs)

            async def _astream(self, messages, *args, **kwargs):
                async for chunk in super()._astream(_prep(messages), *args, **kwargs):
                    yield chunk

            def _make_message_chunk_from_anthropic_event(self, event, *args, **kwargs):
                return super()._make_message_chunk_from_anthropic_event(
                    _normalize_stream_event(event), *args, **kwargs
                )

        PrefillSafeChatAnthropic._normalize_stream_event = staticmethod(
            _normalize_stream_event
        )

        _prefill_safe_cls = PrefillSafeChatAnthropic
    return _prefill_safe_cls


def make_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    **extra: Any,
):
    """Build a LangChain chat model for *provider*.

    Unknown/unsupported kwargs are filtered per api_type rather than raised —
    callers pass one config for all backends.
    """
    name, cfg = resolve_provider(provider)
    api_type = cfg.get("api_type", "openai")
    model = model or cfg.get("model")
    # Hard-fail only for the stock openai provider (the live-observed
    # misconfiguration); custom providers may be deliberate translating
    # gateways — loud error, never fatal.
    _assert_model_provider_compatible(model, api_type, name, strict=(name == "openai"))

    if api_type == "anthropic":
        kwargs: Dict[str, Any] = {
            "model": model,
            # Anthropic requires max_tokens; keep generous headroom for
            # structured outputs (reports/SendTo).
            "max_tokens": max_tokens or 16000,
            # Proxy-named models are absent from langchain's profile tables;
            # declaring the window enables fraction-based context triggers.
            "profile": {"max_input_tokens": _config.MAX_INPUT_TOKENS},
        }
        if reasoning_effort is not None:
            # Native since langchain-anthropic 1.1.3: Literal["max","xhigh",
            # "high","medium","low"] (auto-enables adaptive thinking). Map the
            # OpenAI-vocabulary value we get from shared config; unknown
            # values are dropped rather than 422ing the constructor.
            effort = {"minimal": "low"}.get(reasoning_effort, reasoning_effort)
            if effort in ("max", "xhigh", "high", "medium", "low"):
                kwargs["reasoning_effort"] = effort
            else:
                logger.debug(
                    "Dropping unmapped reasoning_effort=%r for anthropic provider %s",
                    reasoning_effort,
                    name,
                )
        if temperature is not None:
            if "reasoning_effort" in kwargs:
                # Anthropic rejects temperature != 1 when thinking is enabled,
                # and reasoning_effort auto-enables adaptive thinking — the
                # value cannot be honored on this path. Loud, not silent: the
                # notebooks pass temperature=1.0 believing it applies.
                if abs(float(temperature) - 1.0) > 1e-9:
                    logger.warning(
                        "Anthropic provider %s: dropping temperature=%s — "
                        "reasoning_effort=%r enables thinking, and the "
                        "Messages API rejects temperature != 1 with thinking "
                        "on. Unset reasoning_effort to control temperature.",
                        name,
                        temperature,
                        kwargs["reasoning_effort"],
                    )
                else:
                    logger.debug(
                        "Anthropic provider %s: temperature=1.0 is the "
                        "default under thinking; omitting.",
                        name,
                    )
            else:
                kwargs["temperature"] = min(float(temperature), 1.0)
        if cfg.get("base_url"):
            kwargs["anthropic_api_url"] = cfg["base_url"]
        if cfg.get("api_key"):
            kwargs["anthropic_api_key"] = cfg["api_key"]
        limiter = _rate_limiter()
        if limiter is not None:
            kwargs["rate_limiter"] = limiter
        logger.info("LLM: provider=%s api=anthropic model=%s", name, model)
        return _prefill_safe_anthropic()(**kwargs)

    # default: OpenAI-compatible
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": model,
        "profile": {"max_input_tokens": _config.MAX_INPUT_TOKENS},
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    if cfg.get("api_key"):
        kwargs["api_key"] = cfg["api_key"]

    api = "openai"
    if cfg.get("use_responses_api"):
        # Route through POST /v1/responses. langchain auto-converts
        # reasoning_effort → reasoning={"effort": ...} and max_tokens →
        # max_output_tokens on this path, so those kwargs stay as-is.
        api = "responses"
        kwargs["use_responses_api"] = True
        kwargs["output_version"] = "responses/v1"
        # Reasoning summaries: verified live that the proxy accepts
        # summary="auto" but never returns one — so this stays opt-in.
        summary = os.environ.get("ATOMAGENTS_REASONING_SUMMARY")
        if summary and reasoning_effort is not None:
            kwargs.pop("reasoning_effort", None)
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": summary}
        # Escape hatch: "0" forces stateless replay (no server-side item
        # references) if the upstream ever 400s with "Item 'rs_…' not found".
        store = os.environ.get("ATOMAGENTS_OPENAI_RESPONSES_STORE")
        if store is not None:
            kwargs["store"] = store.strip().lower() not in ("0", "false", "no", "off", "")
    limiter = _rate_limiter()
    if limiter is not None:
        kwargs["rate_limiter"] = limiter
    logger.info("LLM: provider=%s api=%s model=%s", name, api, model)
    return ChatOpenAI(**kwargs)


def make_llm_from_config(llm_config: Dict[str, Any]):
    """Build a chat model from an agent-style ``llm_config`` dict.

    Recognized keys: provider, model, temperature, max_tokens,
    reasoning_effort, base_url, api_key. A base_url/api_key present in the
    dict (legacy local configs) defines an ad-hoc OpenAI-compatible endpoint
    that bypasses the registry, preserving old behavior.
    """
    cfg = dict(llm_config or {})
    provider = cfg.pop("provider", None)

    if provider is None and cfg.get("base_url"):
        # Legacy path: explicit endpoint in the config itself — a deliberate
        # custom setup, so the claude-route guard warns instead of raising.
        _assert_model_provider_compatible(
            cfg.get("model"), "openai", "legacy-endpoint", strict=False
        )
        from langchain_openai import ChatOpenAI

        known = {
            "model",
            "temperature",
            "max_tokens",
            "reasoning_effort",
            "base_url",
            "api_key",
            # Responses-API keys — must pass through or they are silently lost.
            "use_responses_api",
            "output_version",
            "reasoning",
            "store",
        }
        dropped = set(cfg) - known
        for key in dropped:
            logger.debug("Dropping unsupported llm_config key %r", key)
        return ChatOpenAI(**{k: v for k, v in cfg.items() if k in known})

    return make_llm(
        provider=provider,
        model=cfg.get("model"),
        temperature=cfg.get("temperature"),
        max_tokens=cfg.get("max_tokens"),
        reasoning_effort=cfg.get("reasoning_effort"),
    )
