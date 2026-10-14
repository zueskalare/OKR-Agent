"""ARM R — a single ReAct agent on claude-opus-5.

Control 1: one agent, one simple prompt, the same 17 tools AtomAgents' workers
get. No supervisor, no staged prompts, no delegation, no stage-guard.

Inherited substrate, to be reported as such rather than as a ReAct property:
the model comes from `AtomAgents.llm_factory.make_llm`, so this arm inherits
the prefill repair, the streaming shim, the prompt-cache breakpoint and the
temperature suppression that the AtomAgents work built. Those are fixes to a
shared gateway problem; attributing them to either architecture would be wrong.

Run standalone:

    venv/bin/python -m benchmarks.arms.react_arm \\
        --task fem_demo0 --session-dir runs/jobs/session-x
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.mcp_client import MCPClient  # noqa: E402
from benchmarks.okr_tools import OKRStore  # noqa: E402
from benchmarks import toolbind as tb  # noqa: E402

PROMPTS = REPO / "benchmarks" / "prompts"
BRIEFS = REPO / "benchmarks" / "briefs"

# Shared budget: whichever binds first is recorded. Deliberately generous --
# the point is that ONE cap binds for every arm, not that the cap is tight.
MAX_LLM_CALLS = 120
MAX_TOOL_CALLS = 240
WALL_CLOCK_SEC = 10800  # 3 h, matching the longest observed AtomAgents cell

# LangGraph's default recursion_limit=25 silently truncates a real run mid-way
# and would read as the agent giving up. Set above the tool-call cap so OUR
# budget is what binds, never the framework's. Module-level so the contract
# test pins the value actually passed, not a formula re-derived in the test.
RECURSION_LIMIT = 2 * MAX_TOOL_CALLS + 50

# Context editing. Measured, not assumed: without it the demo1 cell died at
# 74 tool calls with `prompt is too long: 1007846 tokens > 1000000 maximum`,
# 17 lammps_run log tails being most of it.
#
# This is a parity fix, not a favour to the baseline. AtomAgents declares
# MAX_INPUT_TOKENS=120_000 and summarizes long histories (base_agent.py:692),
# so an unmanaged ReAct arm would be measuring *context management* rather than
# architecture. ClearToolUsesEdit is LangChain's own implementation of
# Anthropic's context-editing pattern: it drops stale TOOL RESULTS only, and
# leaves the agent's reasoning, its plan and its recent observations intact —
# the minimal intervention, adding no planning scaffolding.
#
# The trigger is deliberately >3x AtomAgents' declared per-agent window, so the
# baseline is never the starved side of the comparison, and so one of the
# shared budget caps above is what actually binds.
# Attempt 1 (trigger 400k, approximate counting) STILL overflowed, at 1 068 114
# tokens. Two independent causes, both measured:
#
#   1. `count_tokens_approximately` assumes ~4 chars/token. Base64 tokenizes far
#      worse: on a realistic 1.5 MB PNG payload it reported 500 017 tokens
#      against the model's exact 1 364 866 -- a 2.73x UNDERCOUNT. At a real
#      791 329 tokens the middleware saw ~290 k and never crossed its trigger.
#      Fixed by counting with the model, not the heuristic.
#   2. Individual results were enormous (single calls adding ~250 k tokens).
#      Fixed at source by the 64 KiB cap in toolbind (ARM A parity).
#
# With results capped, `keep` is now bounded: 3 x 64 KiB is small.
CONTEXT_TRIGGER_TOKENS = 350_000
KEEP_RECENT_TOOL_RESULTS = 3

AGENT_NAME = "ReActAgent"


def _is_error_result(text: str) -> bool:
    """The live advisory flag on a tool result, matching the scorer's markers.

    Load-bearing in exactly one case: when the 20 000-char transcript
    truncation destroys the payload's JSON, the scorer falls back to this flag
    because it was computed on the FULL text. It can false-positive on payload
    CONTENT quoting these markers; the scorer therefore consults it only for
    truncated results.
    """
    if '"status": "error"' in text or '"isError": true' in text:
        return True
    m = re.search(r'"returncode":\s*(-?\d+)', text)
    return bool(m and int(m.group(1)) != 0)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    out.append("[thinking]")
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(p for p in out if p)
    return str(content)


def build_agent(session_dir: Path, flavor: str, store: OKRStore, client: MCPClient):
    from dotenv import load_dotenv
    from langchain.agents import create_agent

    load_dotenv(str(REPO / ".env"))
    # Never inherit an ambient default: the provider is set EXPLICITLY from
    # BENCH_PROVIDER (anthropic unless the runner says otherwise), so the run
    # can never quietly land on the wrong route.
    provider = os.environ.get("BENCH_PROVIDER", "anthropic")
    os.environ["ATOMAGENTS_LLM_PROVIDER"] = provider
    if provider == "openai":
        # The Responses surface returns reasoning blocks carrying
        # encrypted_content, and ChatOpenAI.get_num_tokens_from_messages — the
        # "model" token counter the context middleware relies on — raises
        # ValueError('Unrecognized content block type') on them (killed
        # ws29b-demo2-react_gpt-t1 in 16 s). The chat surface returns plain
        # text; this arm uses no Responses-only feature.
        os.environ.setdefault("ATOMAGENTS_OPENAI_USE_RESPONSES", "0")

    from AtomAgents.llm_factory import make_llm

    default_model = "claude-opus-5" if provider == "anthropic" else "gpt-5.6-sol"
    llm = make_llm(
        provider=provider,
        model=os.environ.get("BENCH_MODEL", default_model),
        reasoning_effort="high",
    )

    specs = tb.tool_specs(client, store)
    tb.assert_lock(specs)
    tools = tb.build_langchain_tools(specs)

    prompt_file = PROMPTS / ("baseline_fem.md" if flavor == "fem" else "baseline_md.md")
    system_prompt = prompt_file.read_text(encoding="utf-8").replace(
        "{PROJECT_ROOT}", str(session_dir)
    )
    if "{PROJECT_ROOT}" in system_prompt or "PROJECT_ROOT" in system_prompt:
        raise RuntimeError("PROJECT_ROOT survived substitution in the composed prompt")

    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

    context_mw = ContextEditingMiddleware(
        # "model", not the default "approximate": the heuristic undercounts
        # base64 by 2.73x, which is exactly why attempt 1 sailed past its
        # trigger and died at 1 068 114 tokens.
        token_count_method="model",
        edits=[
            ClearToolUsesEdit(
                trigger=CONTEXT_TRIGGER_TOKENS,
                keep=KEEP_RECENT_TOOL_RESULTS,
                clear_tool_inputs=False,   # keep WHAT it ran; drop only the output
                # Never clear the bookkeeping read-back: it is how the agent
                # checks its own deliverables, and clearing it would induce
                # exactly the double-registration noise we are measuring.
                exclude_tools=("list_artefacts",),
                placeholder="[earlier tool output cleared to free context]",
            )
        ]
    )
    agent = create_agent(
        model=llm, tools=tools, system_prompt=system_prompt, middleware=[context_mw]
    )
    return agent, llm, tools, system_prompt, prompt_file


def run(task: str, session_dir: Path, transcript_path: Path, max_turns: int = 4,
        continue_text: str = "Continue.") -> dict:
    lock = json.loads((BRIEFS / "briefs.lock.json").read_text())
    spec = lock["tasks"][task]
    brief = (BRIEFS / spec["brief_file"]).read_text(encoding="utf-8")
    flavor = "fem" if spec["model"] == "fenicsagent" else "md"

    session_dir.mkdir(parents=True, exist_ok=True)
    store = OKRStore(session_dir / "memory.db", session_dir)

    result: dict[str, Any] = {
        "arm": "react",
        "task": task,
        "flavor": flavor,
        "max_turns": max_turns,
        "llm_calls": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "tools_bound": 0,
        "cap_hit": None,
    }

    try:
        return _run_agent(task, spec, brief, flavor, session_dir, transcript_path,
                          max_turns, continue_text, store, result)
    except Exception as exc:  # noqa: BLE001
        # A failure BEFORE the turn loop (MCP unreachable, lock drift, model
        # construction) must still leave a structured record: without this the
        # comparison sees only a bare exit code and a run that built nothing
        # reads the same as one that never started.
        result.setdefault("error", f"{type(exc).__name__}: {exc}")
        result.setdefault("wall_clock_s", 0.0)
        return result


def _run_agent(task: str, spec: dict, brief: str, flavor: str, session_dir: Path,
               transcript_path: Path, max_turns: int, continue_text: str,
               store: OKRStore, result: dict) -> dict:
    with MCPClient() as client:
        agent, llm, tools, system_prompt, prompt_file = build_agent(
            session_dir, flavor, store, client
        )
        result["tools_bound"] = len(tools)
        result["model"] = llm.model
        result["prompt_file"] = prompt_file.name
        result["prompt_sha256"] = __import__("hashlib").sha256(
            system_prompt.encode()
        ).hexdigest()

        t0 = time.time()
        fh = transcript_path.open("w", encoding="utf-8")
        seen = 0

        def emit(kind: str, payload: dict) -> None:
            fh.write(json.dumps({"t": round(time.time() - t0, 2), "kind": kind, **payload}) + "\n")
            fh.flush()

        emit("system", {"text": system_prompt, "prompt_file": prompt_file.name})
        emit("brief", {"text": brief, "sha256": spec["sha256"]})
        store.log_message(AGENT_NAME, json.dumps({"role": "system"}), system_prompt)
        store.log_message(AGENT_NAME, json.dumps({"role": "user"}), brief)

        convo: list[Any] = [{"role": "user", "content": brief}]
        result["turns"] = []
        try:
            # Same continuation budget ARM A gets: one kickoff plus up to three
            # byte-identical nudges. Without it the arms would differ in how
            # many chances they had, not just in architecture.
            for turn in range(1, max_turns + 1):
                sent = "brief" if turn == 1 else continue_text
                if turn > 1:
                    convo = convo + [{"role": "user", "content": continue_text}]
                    emit("continue", {"text": continue_text, "turn": turn})
                    store.log_message(AGENT_NAME, json.dumps({"role": "user"}), continue_text)
                t_turn = time.time()
                before_tools = result["tool_calls"]
                chunk = None
                for chunk in agent.stream(
                    {"messages": convo},
                    config={"recursion_limit": RECURSION_LIMIT},
                    stream_mode="values",
                ):
                    msgs = chunk.get("messages", [])
                    for m in msgs[seen:]:
                        seen = len(msgs)
                        mtype = getattr(m, "type", "?")
                        text = _text_of(getattr(m, "content", ""))
                        calls = list(getattr(m, "tool_calls", []) or [])
                        if mtype == "ai":
                            result["llm_calls"] += 1
                            result["tool_calls"] += len(calls)
                            emit(
                                "assistant",
                                {
                                    "text": text,
                                    "tool_calls": [
                                        {"name": c.get("name"), "args": c.get("args")}
                                        for c in calls
                                    ],
                                    "usage": getattr(m, "usage_metadata", None),
                                },
                            )
                            store.log_message(AGENT_NAME, json.dumps({"role": "assistant"}), text)
                        elif mtype == "tool":
                            name = getattr(m, "name", "?")
                            is_err = _is_error_result(text)
                            result["tool_errors"] += int(is_err)
                            emit("tool_result", {"name": name, "text": text[:20000], "error": is_err})
                            store.log_message(
                                AGENT_NAME, json.dumps({"role": "tool", "name": name}), text[:20000]
                            )

                    elapsed = time.time() - t0
                    if result["llm_calls"] >= MAX_LLM_CALLS:
                        result["cap_hit"] = f"MAX_LLM_CALLS={MAX_LLM_CALLS}"
                    elif result["tool_calls"] >= MAX_TOOL_CALLS:
                        result["cap_hit"] = f"MAX_TOOL_CALLS={MAX_TOOL_CALLS}"
                    elif elapsed >= WALL_CLOCK_SEC:
                        result["cap_hit"] = f"WALL_CLOCK_SEC={WALL_CLOCK_SEC}"
                    if result["cap_hit"]:
                        emit("cap", {"cap": result["cap_hit"]})
                        break

                if chunk is not None:
                    convo = list(chunk.get("messages", convo))
                gained = result["tool_calls"] - before_tools
                result["turns"].append(
                    {
                        "turn": turn,
                        "sent": sent,
                        "seconds": round(time.time() - t_turn, 1),
                        "tool_calls_gained": gained,
                    }
                )
                if result["cap_hit"]:
                    break
                # Mirrors ARM A's early stop: a nudge that produces no further
                # tool use will not produce one next time either.
                if turn > 1 and gained == 0:
                    result["stopped_early"] = f"turn {turn} produced no further tool calls"
                    emit("stopped_early", {"reason": result["stopped_early"]})
                    break
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            emit("error", {"error": result["error"]})
        finally:
            fh.close()

        result["wall_clock_s"] = round(time.time() - t0, 1)
        result["okr_counts"] = store.counts()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["demo1", "demo2", "fem_demo0"])
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--transcript", default="")
    ap.add_argument("--result", default="", help="write the result dict here as JSON")
    args = ap.parse_args()

    session_dir = Path(args.session_dir).resolve()
    transcript = Path(args.transcript) if args.transcript else session_dir / "transcript.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)

    out = run(args.task, session_dir, transcript)
    if args.result:
        Path(args.result).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("RESULT_JSON " + json.dumps(out))
    return 0 if not out.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
