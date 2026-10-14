"""Multi-provider LLM support: registry discovery, model-id routing, factory."""

import asyncio

import pytest

from AtomAgents import config as cfg
from AtomAgents.llm_factory import (
    make_llm,
    make_llm_from_config,
    parse_model_id,
    prefill_safe_messages,
    resolve_provider,
)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ATOMAGENTS_LLM_PROVIDER",
        "ATOMAGENTS_OPENAI_MODEL",
        "ATOMAGENTS_ANTHROPIC_MODEL",
        "ATOMAGENTS_OPENAI_USE_RESPONSES",
        "ATOMAGENTS_LOCAL_USE_RESPONSES",
        "ATOMAGENTS_REASONING_SUMMARY",
        "ATOMAGENTS_OPENAI_RESPONSES_STORE",
        "ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_discovery_local_only(clean_env):
    providers = cfg.discover_providers()
    assert set(providers) >= {"local"}
    assert "openai" not in providers
    assert "anthropic" not in providers
    assert cfg.default_provider() == "local"


def test_discovery_from_standard_env(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("OPENAI_BASE_URL", "http://proxy:8000/v1")
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    clean_env.setenv("ANTHROPIC_BASE_URL", "http://proxy:8000")
    providers = cfg.discover_providers()
    assert providers["openai"]["base_url"] == "http://proxy:8000/v1"
    assert providers["openai"]["api_type"] == "openai"
    assert providers["anthropic"]["api_key"] == "tok-test"
    assert providers["anthropic"]["api_type"] == "anthropic"
    # openai preferred as default when nothing is pinned
    assert cfg.default_provider() == "openai"


def test_custom_provider_via_env(clean_env):
    clean_env.setenv("ATOMAGENTS_PROVIDER_MYPROX_BASE_URL", "http://myprox:9000/v1")
    clean_env.setenv("ATOMAGENTS_PROVIDER_MYPROX_API_KEY_ENV", "MYPROX_KEY")
    clean_env.setenv("MYPROX_KEY", "zzz")
    clean_env.setenv("ATOMAGENTS_PROVIDER_MYPROX_MODEL", "my-model")
    providers = cfg.discover_providers()
    assert providers["myprox"]["api_key"] == "zzz"
    assert providers["myprox"]["model"] == "my-model"


def test_explicit_default_provider_validated(clean_env):
    clean_env.setenv("ATOMAGENTS_LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        cfg.default_provider()


def test_parse_model_id():
    assert parse_model_id("atomagent") == ("atomagent", None)
    assert parse_model_id("atomagent@anthropic") == ("atomagent", "anthropic")
    assert parse_model_id("fenicsagent@local") == ("fenicsagent", "local")


def test_resolve_unknown_provider_raises(clean_env):
    with pytest.raises(ValueError):
        resolve_provider("does-not-exist")


def test_factory_openai(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("OPENAI_BASE_URL", "http://proxy:8000/v1")
    llm = make_llm("openai", model="gpt-5", reasoning_effort="minimal", max_tokens=100)
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-5"
    assert "proxy:8000" in str(llm.openai_api_base)
    # Responses API is the default for the openai (proxy) provider.
    assert llm.use_responses_api is True
    assert llm.output_version == "responses/v1"
    # reasoning_effort stays a plain kwarg — langchain converts it on the
    # Responses path; the explicit reasoning dict is opt-in via env.
    assert llm.reasoning_effort == "minimal"
    assert llm.reasoning is None


def test_factory_openai_responses_env_off(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("ATOMAGENTS_OPENAI_USE_RESPONSES", "0")
    llm = make_llm("openai", model="gpt-5")
    assert not llm.use_responses_api


def test_factory_local_stays_chat_completions(clean_env):
    llm = make_llm("local")
    assert type(llm).__name__ == "ChatOpenAI"
    assert not llm.use_responses_api
    assert llm.output_version is None


def test_factory_reasoning_summary_opt_in(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("ATOMAGENTS_REASONING_SUMMARY", "auto")
    llm = make_llm("openai", model="gpt-5", reasoning_effort="high")
    assert llm.reasoning == {"effort": "high", "summary": "auto"}
    assert llm.reasoning_effort is None


def test_factory_responses_store_escape_hatch(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("ATOMAGENTS_OPENAI_RESPONSES_STORE", "0")
    llm = make_llm("openai", model="gpt-5")
    assert llm.store is False


def test_custom_provider_use_responses(clean_env):
    clean_env.setenv("ATOMAGENTS_PROVIDER_MYPROX_BASE_URL", "http://myprox:9000/v1")
    clean_env.setenv("ATOMAGENTS_PROVIDER_MYPROX_API_KEY_ENV", "MYPROX_KEY")
    clean_env.setenv("MYPROX_KEY", "k-test")
    clean_env.setenv("ATOMAGENTS_PROVIDER_MYPROX_USE_RESPONSES", "1")
    llm = make_llm("myprox", model="gpt-5")
    assert llm.use_responses_api is True
    # and default-off without the env flag
    clean_env.delenv("ATOMAGENTS_PROVIDER_MYPROX_USE_RESPONSES")
    llm2 = make_llm("myprox", model="gpt-5")
    assert not llm2.use_responses_api


def test_factory_anthropic_filters_params(clean_env):
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    clean_env.setenv("ANTHROPIC_BASE_URL", "http://proxy:8000")
    # reasoning_effort passes through natively (langchain-anthropic >= 1.1.3);
    # max_tokens defaulted; temperature OMITTED because thinking is enabled
    # (Anthropic rejects temperature != 1 with thinking on).
    llm = make_llm("anthropic", reasoning_effort="high", temperature=1.5)
    from langchain_anthropic import ChatAnthropic

    assert isinstance(llm, ChatAnthropic)
    assert llm.max_tokens == 16000
    assert llm.reasoning_effort == "high"
    assert llm.temperature is None
    assert "proxy:8000" in str(llm.anthropic_api_url)


def test_cache_system_blocks_marks_first_system_only():
    from langchain_core.messages import HumanMessage, SystemMessage

    from AtomAgents.llm_factory import cache_system_blocks

    msgs = cache_system_blocks(
        [
            SystemMessage(content="big static agent prompt"),
            SystemMessage(content="stage prompt"),
            HumanMessage(content="hi"),
        ]
    )
    first = msgs[0].content
    assert isinstance(first, list)
    assert first[0]["cache_control"] == {"type": "ephemeral"}
    assert first[0]["text"] == "big static agent prompt"
    # second system message untouched (stage prompts vary per node —
    # marking them would fragment the cache)
    assert msgs[1].content == "stage prompt"
    # non-system-leading lists are a no-op
    out = cache_system_blocks([HumanMessage(content="x")])
    assert out[0].content == "x"


def test_rate_limiter_shared_and_opt_in(clean_env, monkeypatch):
    from AtomAgents import config
    from AtomAgents import llm_factory

    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    # disabled by default
    monkeypatch.setattr(llm_factory, "_shared_rate_limiter", None)
    monkeypatch.setattr(config, "LLM_RPS", 0.0)
    assert make_llm("openai", model="gpt-5").rate_limiter is None
    # enabled: one shared instance across models
    monkeypatch.setattr(config, "LLM_RPS", 2.0)
    a = make_llm("openai", model="gpt-5")
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    b = make_llm("anthropic")
    assert a.rate_limiter is not None
    assert a.rate_limiter is b.rate_limiter  # process-wide, not per-agent


def test_factory_anthropic_effort_mapping_and_temperature(clean_env):
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    # openai vocabulary maps: minimal -> low
    llm = make_llm("anthropic", reasoning_effort="minimal")
    assert llm.reasoning_effort == "low"
    # unknown value dropped, not passed (would fail the Literal validator)
    llm2 = make_llm("anthropic", reasoning_effort="turbo")
    assert llm2.reasoning_effort is None
    # without effort, temperature still passes through clamped
    llm3 = make_llm("anthropic", temperature=1.5)
    assert llm3.temperature == 1.0
    assert llm3.reasoning_effort is None


def test_prefill_safe_messages():
    """Anthropic routes reject conversations ending with an assistant turn;
    the normalizer must append a user turn (or tool results) without touching
    already-valid conversations."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    # Trailing AIMessage (prior-stage think/plan output) → append user turn.
    fixed = prefill_safe_messages(
        [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="# Supervisor - think\n..."),
        ]
    )
    assert isinstance(fixed[-1], HumanMessage)

    # Trailing AIMessage with unresolved tool_calls → placeholder tool results.
    fixed = prefill_safe_messages(
        [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "note", "args": {}, "id": "toolu_1"}],
            ),
        ]
    )
    assert isinstance(fixed[-1], ToolMessage)
    assert fixed[-1].tool_call_id == "toolu_1"

    # First non-system message assistant → user turn inserted before it.
    fixed = prefill_safe_messages(
        [SystemMessage(content="sys"), AIMessage(content="prior"), HumanMessage(content="go")]
    )
    assert isinstance(fixed[1], HumanMessage)
    assert isinstance(fixed[-1], HumanMessage)

    # Already valid → untouched.
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    assert prefill_safe_messages(msgs) == msgs


def test_prefill_safe_tool_pairing_repair():
    """A tool_result must directly follow its tool_use, and a tool_use must be
    answered before the conversation moves on — Anthropic 400s otherwise
    (seen live: replace-semantics state wiped history down to the orphan
    tool result). The normalizer repairs both instead of crashing the run."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    # Orphan tool_result (no tool_use anywhere before it) → user text.
    fixed = prefill_safe_messages(
        [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            ToolMessage(content='{"ok": true}', tool_call_id="toolu_orphan"),
        ]
    )
    assert not any(isinstance(m, ToolMessage) for m in fixed)
    assert any(
        isinstance(m, HumanMessage) and "toolu_orphan" in str(m.content) for m in fixed
    )

    # Properly paired tool call → untouched (plus trailing-AI guard only).
    good = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "toolu_a"}]),
        ToolMessage(content="ok", tool_call_id="toolu_a"),
    ]
    fixed = prefill_safe_messages(good)
    assert fixed == good

    # tool_use answered late (a Human slipped in between) → placeholder result
    # inserted right after the assistant turn, late real result becomes text.
    fixed = prefill_safe_messages(
        [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "toolu_b"}]),
            HumanMessage(content="interruption"),
            ToolMessage(content="late", tool_call_id="toolu_b"),
        ]
    )
    assert isinstance(fixed[2], ToolMessage) and fixed[2].tool_call_id == "toolu_b"
    assert isinstance(fixed[-1], HumanMessage) and "late" in str(fixed[-1].content)

    # Duplicate tool_result for one id → second copy becomes text (duplicate
    # tool_result ids are rejected too).
    fixed = prefill_safe_messages(
        [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "toolu_c"}]),
            ToolMessage(content="one", tool_call_id="toolu_c"),
            ToolMessage(content="two", tool_call_id="toolu_c"),
        ]
    )
    assert sum(isinstance(m, ToolMessage) for m in fixed) == 1


def test_substate_messages_accumulate_across_tool_node():
    """SubState.messages must use the add_messages reducer: the prebuilt
    ToolNode returns ONLY the new ToolMessage, and replace semantics wiped
    the conversation at every tools boundary (the live anthropic 400)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.graph import END, StateGraph

    from AtomAgents.agents.base_agent import BaseAgent

    def act(state):
        return {
            "messages": list(state["messages"])
            + [
                AIMessage(
                    content="calling",
                    tool_calls=[{"name": "t", "args": {}, "id": "toolu_x"}],
                )
            ],
            "send_to": "act",
        }

    def tools(state):
        # Mimic langgraph's prebuilt ToolNode: returns ONLY the new message.
        return {"messages": [ToolMessage(content="ok", tool_call_id="toolu_x")]}

    g = StateGraph(BaseAgent.SubState)
    g.add_node("act", act)
    g.add_node("tools", tools)
    g.set_entry_point("act")
    g.add_edge("act", "tools")
    g.add_edge("tools", END)
    result = g.compile().invoke({"messages": [HumanMessage(content="hi")], "send_to": "act"})

    types = [type(m).__name__ for m in result["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage"], types


def test_factory_anthropic_is_prefill_safe(clean_env):
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    clean_env.setenv("ANTHROPIC_BASE_URL", "http://proxy:8000")
    llm = make_llm("anthropic")
    assert type(llm).__name__ == "PrefillSafeChatAnthropic"


def test_factory_legacy_explicit_endpoint(clean_env):
    # Old-style llm_config with its own base_url bypasses the registry.
    llm = make_llm_from_config(
        {
            "model": "qwen2.5-vl-72b-instruct",
            "temperature": 1.0,
            "max_tokens": 16000,
            "base_url": "http://localhost:11434/v1/",
            "api_key": "local",
        }
    )
    assert type(llm).__name__ == "ChatOpenAI"
    assert "11434" in str(llm.openai_api_base)


def test_factory_legacy_whitelist_passes_responses_keys(clean_env):
    # The legacy explicit-endpoint path must not silently drop the new keys.
    llm = make_llm_from_config(
        {
            "model": "gpt-5",
            "base_url": "http://localhost:9000/v1",
            "api_key": "k",
            "use_responses_api": True,
            "output_version": "responses/v1",
        }
    )
    assert llm.use_responses_api is True
    assert llm.output_version == "responses/v1"


def test_backend_lists_provider_variants(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    import backend.openai_backend as ob

    resp = asyncio.run(ob.list_models())
    ids = {m.id for m in resp.data}
    assert {"atomagent", "atomagent_local", "fenicsagent"} <= ids
    assert "atomagent@anthropic" in ids
    assert "atomagent@openai" in ids
    assert "fenicsagent@local" in ids


def test_claude_via_openai_provider_fails_fast(clean_env):
    """claude-* through the OpenAI-compatible surface answers structured
    calls with fenced/prose JSON — every TriageOut/PlanOut/TaskOperations
    call fails and the turn dies producing nothing (observed live, 129 s).
    The factory must refuse the pairing up front, naming the fix."""
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("OPENAI_BASE_URL", "http://proxy:8000/v1")
    with pytest.raises(ValueError, match="@anthropic"):
        make_llm("openai", model="claude-opus-5")


def test_claude_via_openai_escape_hatch(clean_env, monkeypatch, caplog):
    import logging

    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(cfg, "ALLOW_CLAUDE_VIA_OPENAI", True, raising=False)
    with caplog.at_level(logging.ERROR, logger="AtomAgents.llm_factory"):
        llm = make_llm("openai", model="claude-opus-5")
    assert type(llm).__name__ == "ChatOpenAI"
    assert any("anthropic" in r.getMessage() for r in caplog.records)


def test_legacy_endpoint_config_warns_not_raises(clean_env, caplog):
    """An explicit base_url in the config is a deliberate custom endpoint —
    possibly a gateway that genuinely translates the surface — so the claude
    guard is loud (error log) but never fatal on this path."""
    import logging

    with caplog.at_level(logging.ERROR, logger="AtomAgents.llm_factory"):
        llm = make_llm_from_config(
            {
                "model": "claude-opus-5",
                "base_url": "http://localhost:1234/v1",
                "api_key": "k",
            }
        )
    assert type(llm).__name__ == "ChatOpenAI"
    assert any("anthropic" in r.getMessage() for r in caplog.records)


def test_custom_openai_provider_warns_not_raises(clean_env, caplog):
    """Custom ATOMAGENTS_PROVIDER_* backends may be translating gateways —
    only the stock 'openai' provider hard-fails on a claude model."""
    import logging

    clean_env.setenv("ATOMAGENTS_PROVIDER_GW_BASE_URL", "http://gw:4000/v1")
    clean_env.setenv("ATOMAGENTS_PROVIDER_GW_API_TYPE", "openai")
    clean_env.setenv("ATOMAGENTS_PROVIDER_GW_MODEL", "claude-opus-5")
    clean_env.setenv("ATOMAGENTS_PROVIDER_GW_API_KEY_ENV", "GW_KEY")
    clean_env.setenv("GW_KEY", "k")
    with caplog.at_level(logging.ERROR, logger="AtomAgents.llm_factory"):
        llm = make_llm("gw")
    assert type(llm).__name__ == "ChatOpenAI"
    assert any("anthropic" in r.getMessage() for r in caplog.records)


def test_guard_catches_gateway_claude_spellings(clean_env):
    """Bedrock/LiteLLM-style spellings of the same broken pairing must hit
    the guard too — prefix matching let them straight through to the 129-s
    runtime failure the check exists to prevent."""
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    for spelling in ("us.anthropic.claude-opus-5", "bedrock/anthropic.claude-opus-5"):
        with pytest.raises(ValueError, match="@anthropic"):
            make_llm("openai", model=spelling)


def test_escape_hatch_env_var_works_at_runtime(clean_env, caplog):
    """The ValueError's remediation text advertises the env var — it must
    work when exported in a running process (env is read live, before the
    import-frozen config attribute)."""
    import logging

    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("ATOMAGENTS_ALLOW_CLAUDE_VIA_OPENAI", "1")
    with caplog.at_level(logging.ERROR, logger="AtomAgents.llm_factory"):
        llm = make_llm("openai", model="claude-opus-5")
    assert type(llm).__name__ == "ChatOpenAI"
    assert any("anthropic" in r.getMessage() for r in caplog.records)


def test_gpt_via_anthropic_warns_only(clean_env, caplog):
    import logging

    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    clean_env.setenv("ANTHROPIC_BASE_URL", "http://proxy:8000")
    with caplog.at_level(logging.WARNING, logger="AtomAgents.llm_factory"):
        llm = make_llm("anthropic", model="gpt-5.6-sol")
    # never fatal — warn and build
    assert type(llm).__name__ == "PrefillSafeChatAnthropic"
    assert any("OpenAI model" in r.getMessage() for r in caplog.records)


def test_anthropic_temperature_drop_is_logged(clean_env, caplog):
    """The drop itself is required (thinking forbids temperature != 1), but
    it must be loud: the notebooks pass temperature=1.0 believing it
    applies. Non-1.0 → warning; exactly 1.0 → silent (debug only)."""
    import logging

    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    with caplog.at_level(logging.WARNING, logger="AtomAgents.llm_factory"):
        make_llm("anthropic", temperature=1.5, reasoning_effort="high")
    assert any(
        "dropping temperature" in r.getMessage().lower() for r in caplog.records
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="AtomAgents.llm_factory"):
        make_llm("anthropic", temperature=1.0, reasoning_effort="high")
    assert not any("temperature" in r.getMessage().lower() for r in caplog.records)


def test_anthropic_stream_event_normalizer_shims_proxy_dicts(monkeypatch):
    """Gateway compat: `context_management`/`container` arrive as
    plain dicts (SDK-unknown extra fields) and upstream's `.model_dump()`
    calls crashed EVERY streamed anthropic call. The PrefillSafe subclass
    normalizes events before the upstream parser sees them."""
    from types import SimpleNamespace

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy:8000")
    llm = make_llm("anthropic")
    norm = type(llm)._normalize_stream_event

    ev = SimpleNamespace(
        type="message_delta",
        context_management={"applied_edits": []},
        delta=SimpleNamespace(container={"id": "c1"}),
    )
    out = norm(ev)
    assert out.context_management.model_dump() == {"applied_edits": []}
    assert out.delta.container.model_dump(mode="json") == {"id": "c1"}
    # dict semantics preserved for any other consumer
    assert dict(out.context_management) == {"applied_edits": []}

    # already-parsed objects (real SDK) pass through untouched
    cm_obj = SimpleNamespace(model_dump=lambda: {"x": 1})
    ev2 = SimpleNamespace(type="message_delta", context_management=cm_obj, delta=None)
    assert norm(ev2).context_management is cm_obj
    # events without the fields are untouched
    ev3 = SimpleNamespace(type="content_block_delta")
    assert norm(ev3) is ev3
