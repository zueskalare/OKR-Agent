"""ARM C — a flat sequential CrewAI crew on claude-opus-5.

Control 2: three agents, three tasks, `Process.sequential`,
`allow_delegation=False`. That flat topology *is* the control -- it is what a
competent CrewAI user writes without AtomAgents' staged-prompt machinery.

Runs under its own interpreter (crewai 0.201.1 / Python 3.12), launched as a
subprocess by the runner, and writes the same `memory.db` tables as every other
arm so one `session_stats()` covers all three.

This module deliberately does NOT reuse
`Comparison/AgentsBuild/CrewAI/atomagent/`. That scaffold carries five defects
that each silently produce a wrong campaign, all confirmed in its source:

1. `mcp_tools.py` swallows any MCP failure into `return ()` and `@lru_cache`es
   the empty tuple for the life of the process. CrewAI's own
   `if not tools: return []` never complains, the agent is handed the
   `no_tools` prompt slice, and it narrates a LAMMPS run it never performed.
2. `system_template=` without `prompt_template=` is silently discarded
   (`crewai/utilities/prompts.py:96-102`), so the prompt it loads never
   reaches the model -- and it also downgrades the agent out of the
   system/user split.
3. `llm: gpt-5` is hardcoded six times in `agents.yaml`; nothing would have
   pointed this arm at opus at all.
4. `gpt-5` matches no key in litellm's context-window table, so the window
   silently collapses to the 8192 default.
5. `crewai.LLM` construction failures are swallowed by `create_llm` into
   `llm=None` at debug level.

Each is asserted against below rather than merely avoided.
"""

from __future__ import annotations

import argparse
import hashlib
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

# The gateway's real context window. Left unset, litellm's table lookup misses
# a proxy-named model and CrewAI silently plans against 8192 tokens.
CONTEXT_WINDOW = 850_000
MAX_TOKENS = 16_000

# Mirrors ARM R's WALL_CLOCK_SEC. CrewAI has no wall-clock cap of its own, so
# max_execution_time per agent is the closest equivalent.
WALL_CLOCK_SEC = 10800

# CrewAI's per-agent iteration ceiling. It CAN bind before the shared
# MAX_TOOL_CALLS budget (3 agents x 40 < 240) and CrewAI hits it silently —
# "Maximum iterations reached. Requesting final answer." went unrecorded in
# two of the three WS28 cells (demo2's Model Builder, fem_demo0's Simulation
# Analyst). Detected via step counting below and recorded as cap_hit, so a
# framework-truncated stage can never again read as a voluntary stop.
MAX_ITER = 40


def _neutralize_react_markers(thought: str) -> str:
    """Defuse ReAct markers the model echoed inside its visible thought.

    CrewAI's parser regex-splits the WHOLE shim output at the FIRST
    `Action:`/`Action Input:` and treats any `Final Answer:` as terminal, so a
    thought quoting an earlier step ("as before: Action Input: {...}") makes
    the parser swallow garbage as the tool arguments — measured at 16/64,
    6/63 and 22/96 of the recorded cells' tool calls. Only the appended real
    Action pair may remain parseable.
    """
    return re.sub(
        r"(?i)\b(Action\s*Input|Action|Observation|Final\s*Answer)\s*:",
        r"\1 -",
        thought,
    )


def _prefill_safe(messages: list[dict]) -> list[dict]:
    """The dict-message twin of `llm_factory.prefill_safe_messages`.

    CrewAI's ReAct loop appends the model's own partial output as an assistant
    turn and calls again, i.e. it prefills. Anthropic rejects that outright:
    "This model does not support assistant message prefill. The conversation
    must end with a user message." ARM A and ARM R inherit this repair from
    `llm_factory`; ARM C has to carry its own copy, and it is the same
    shared-substrate fix rather than anything CrewAI-specific.
    """
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            continue
        if m.get("role") == "assistant":
            msgs.insert(i, {"role": "user", "content": "(context follows)"})
        break
    if msgs and msgs[-1].get("role") == "assistant":
        # Hand the partial back as context and ask for the next step, rather
        # than dropping it -- the partial carries the agent's Thought.
        msgs.append({"role": "user", "content": "Continue."})
    return msgs


def _native_tool_schemas(tools: list) -> list[dict]:
    """CrewAI BaseTools -> OpenAI-style function schemas for the wire."""
    out = []
    for t in tools:
        try:
            params = t.args_schema.model_json_schema()
            params.pop("title", None)
            for p in (params.get("properties") or {}).values():
                p.pop("title", None)
        except Exception:  # noqa: BLE001
            params = {"type": "object", "properties": {}}
        # `description` on a CrewAI tool is the whole rendered block; the first
        # paragraph is the human description and is what a function schema wants.
        desc = (t.description or "").strip()
        out.append(
            {
                "type": "function",
                "function": {"name": t.name, "description": desc[:1024], "parameters": params},
            }
        )
    return out


def make_native_tool_llm(base_kwargs: dict, tool_schemas: list[dict]):
    """The ARM C counterpart of ARM A/R's stream shim.

    The gateway prepends a cached system block -- "You are a Claude agent, built
    on Anthropic's Claude Agent SDK." -- to every request. It puts the model in
    native tool-use mode, and it is not disableable from the client. Measured
    consequence, reproduced on claude-opus-5, claude-sonnet-4-6 and
    claude-opus-4-8 alike: given CrewAI's ReAct *text* protocol the model
    ignores the `Action:` format entirely and emits native tool-call syntax
    (`<invoke name="Write">`, `**Tool Use: bash**`) for tools it does not have.
    CrewAI 0.201.1 drives agent tools ONLY through that text protocol -- its
    executor calls `llm.call()` with no `tools=` argument at all -- so on this
    gateway an unmodified CrewAI agent executes exactly zero tools and narrates
    instead. That measures the proxy, not the framework.

    This shim changes only the WIRE ENCODING of a tool request: the tools go out
    in the native `tools=` array, and the model's tool_call comes back rendered
    as the `Thought:/Action:/Action Input:` text CrewAI's own parser expects.
    CrewAI still chooses the tool, executes it, feeds back the Observation and
    decides when to finish. No orchestration is bypassed.

    Report it the way the plan reports ARM R's inherited fixes: shared-substrate
    repair for a gateway defect, not a property of either architecture.
    """
    import json as _json

    import litellm
    from crewai import LLM

    class NativeToolCallingLLM(LLM):
        shim = "native-tool-calling (gateway ReAct-text incompatibility)"

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            from_task=None,
            from_agent=None,
        ):
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            messages = _prefill_safe(messages)

            params = {
                "model": self.model,
                "messages": messages,
                "api_base": self.api_base,
                "api_key": self.api_key,
                "max_tokens": self.max_tokens,
                "stream": False,
                "tools": tool_schemas,
                # CrewAI's loop executes exactly ONE action per iteration, so a
                # parallel tool-use response would have all but the first call
                # silently dropped -- burning iterations re-requesting work it
                # already asked for. Observed doing exactly that before this
                # line was added.
                "parallel_tool_calls": False,
            }
            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort

            resp = litellm.completion(**params)
            msg = resp.choices[0].message
            text = (msg.content or "").strip()
            calls = list(getattr(msg, "tool_calls", None) or [])

            if calls:
                call = calls[0]
                try:
                    args = _json.loads(call.function.arguments or "{}")
                except Exception:  # noqa: BLE001
                    args = {}
                thought = _neutralize_react_markers(
                    text or f"I need to use {call.function.name}."
                )
                return (
                    f"Thought: {thought}\n"
                    f"Action: {call.function.name}\n"
                    f"Action Input: {_json.dumps(args)}"
                )

            if "Final Answer:" in text or "Action:" in text:
                return text
            return f"Thought: I now know the final answer\nFinal Answer: {text}"

    llm = NativeToolCallingLLM(**base_kwargs)
    return llm


def build_llm(tool_schemas: list[dict] | None = None):
    from dotenv import load_dotenv

    load_dotenv(str(REPO / ".env"))
    import litellm
    from crewai import LLM

    provider = os.environ.get("BENCH_PROVIDER", "anthropic")
    default_model = "claude-opus-5" if provider == "anthropic" else "gpt-5.6-sol"
    model = os.environ.get("BENCH_MODEL", default_model)

    # Landmine 4: a missing registry entry degrades the context window
    # silently. litellm strips the provider prefix for this lookup, so the
    # bare id is what must be present. Gateway aliases (gpt-5.6-sol) exist in
    # no public registry, so register them explicitly with the gateway's real
    # window rather than letting the lookup fall back to 8192.
    if model not in litellm.model_cost:
        litellm.register_model({model: {
            "max_tokens": MAX_TOKENS,
            "max_input_tokens": CONTEXT_WINDOW,
            "max_output_tokens": MAX_TOKENS,
            "litellm_provider": provider,
            "mode": "chat",
        }})
    if model not in litellm.model_cost:
        raise RuntimeError(
            f"{model!r} absent from litellm.model_cost — the context window would "
            "silently fall back to 8192 and the crew would plan against the wrong budget"
        )

    if provider == "anthropic":
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        if not token:
            raise RuntimeError("ANTHROPIC_AUTH_TOKEN unset — the crew would hit api.anthropic.com")
        api_base = os.environ.get("ANTHROPIC_BASE_URL")
        if not api_base:
            raise RuntimeError("ANTHROPIC_BASE_URL unset — refusing to leave the local gateway")
    else:
        token = os.environ.get("OPENAI_API_KEY")
        if not token:
            raise RuntimeError("OPENAI_API_KEY unset — refusing to run the openai route")
        api_base = os.environ.get("OPENAI_BASE_URL")
        if not api_base:
            raise RuntimeError("OPENAI_BASE_URL unset — refusing to leave the local gateway")

    base_kwargs = dict(
        model=f"{provider}/{model}",
        api_base=api_base,
        api_key=token,
        reasoning_effort="high",   # litellm maps this to a FIXED 4096 thinking budget
        max_tokens=MAX_TOKENS,
        # Landmine: the crewai/litellm streaming path against this gateway is
        # unproven, and ARMs A/R each needed a dedicated shim for exactly that
        # class of proxy stream bug.
        stream=False,
    )
    if tool_schemas:
        llm = make_native_tool_llm(base_kwargs, tool_schemas)
    else:
        llm = LLM(**base_kwargs)
    llm.context_window_size = CONTEXT_WINDOW

    # Landmine 5: create_llm swallows construction failures into None.
    if llm is None or not getattr(llm, "model", None):
        raise RuntimeError("crewai.LLM construction failed silently")
    if str(llm.api_base) != api_base:
        raise RuntimeError(f"api_base not honoured: {llm.api_base!r}")
    return llm


def build_crew(task: str, session_dir: Path, store: OKRStore, client: MCPClient):
    from crewai import Agent, Crew, Process, Task

    lock = json.loads((BRIEFS / "briefs.lock.json").read_text())
    spec = lock["tasks"][task]
    brief = (BRIEFS / spec["brief_file"]).read_text(encoding="utf-8")
    flavor = "fem" if spec["model"] == "fenicsagent" else "md"

    prompt_file = PROMPTS / ("baseline_fem.md" if flavor == "fem" else "baseline_md.md")
    shared = prompt_file.read_text(encoding="utf-8").replace("{PROJECT_ROOT}", str(session_dir))
    if "PROJECT_ROOT" in shared:
        raise RuntimeError("PROJECT_ROOT survived substitution in the composed prompt")

    specs = tb.tool_specs(client, store)
    tb.assert_lock(specs)
    tools = tb.build_crewai_tools(specs)
    # Landmine 1: an empty toolset is indistinguishable from no toolset in
    # CrewAI and turns the agents into narrators.
    if len(tools) != tb.EXPECTED_TOTAL:
        raise RuntimeError(f"expected {tb.EXPECTED_TOTAL} tools, got {len(tools)}")

    llm = build_llm(_native_tool_schemas(tools))

    # The prompt goes through role/goal/backstory -- the path CrewAI actually
    # renders (landmine 2). `backstory` carries the shared baseline prompt
    # verbatim; only the identity and objective lines differ per role.
    def mk(role: str, goal: str) -> Agent:
        # NO function_calling_llm: setting it routes CrewAI's tool-argument
        # repair through instructor -> litellm.completion with neither api_base
        # nor api_key, which bypasses the shim, leaves the gateway, and fails
        # with an auth error — 64/24/88 such failures in the recorded WS28
        # cells. Unset, a bad Action Input gets the stock actionable
        # "tool arguments error" observation instead.
        return Agent(
            role=role,
            goal=goal,
            backstory=shared,
            tools=tools,
            llm=llm,
            allow_delegation=False,   # flat by design: this IS the control
            verbose=True,
            max_iter=MAX_ITER,
            max_execution_time=WALL_CLOCK_SEC,
        )

    domain = "finite-element" if flavor == "fem" else "molecular-dynamics"
    planner = mk(
        f"{domain.capitalize()} Planner",
        "Read the brief, probe the environment with the tools to establish what is "
        "actually installed and available, and write a concrete plan naming every "
        "file that must exist when the work is done.",
    )
    builder = mk(
        "Model Builder",
        "Build the model — geometry, mesh or atomistic structure — by writing and "
        "running real scripts with the tools, and verify on disk that the files exist.",
    )
    analyst = mk(
        "Simulation Analyst",
        "Run the simulation, post-process the results, check the brief's acceptance "
        "criteria against numbers your own run produced, and register every deliverable.",
    )

    # Task 1 carries the brief byte-identical to what ARMs A and R receive.
    t1 = Task(
        description=(
            f"{brief}\n\n---\nThis is stage 1 of 3. Establish the plan and the "
            "environment. Use the tools to check what is installed before "
            "committing to an approach."
        ),
        expected_output="A concrete plan naming every required output file and the approach for each.",
        agent=planner,
    )
    t2 = Task(
        description=(
            "Stage 2 of 3: build the model described in the brief. Write real "
            "scripts with write_file and run them with code_runner or bash_run. "
            "Confirm each file exists with file_reader before moving on."
        ),
        expected_output="The model files on disk, with their absolute paths and confirmed sizes.",
        agent=builder,
        context=[t1],
    )
    t3 = Task(
        description=(
            "Stage 3 of 3: run the simulation, post-process, validate against the "
            "brief's acceptance criteria, and write the remaining deliverables. "
            "Register every file with register_artefact and every key number with "
            "record_metric. Report only numbers that came from real tool results."
        ),
        expected_output="A final report listing every deliverable, its path, and the validation numbers actually obtained.",
        agent=analyst,
        context=[t1, t2],
    )

    # Framework-cap detection. CrewAI truncates an agent at max_iter with only
    # a console line ("Maximum iterations reached. Requesting final answer."),
    # so a force-truncated stage looks identical to a voluntary finish in every
    # recorded output. One step_callback fires per executor step and
    # task_callback advances the stage pointer; a stage whose tally reaches
    # MAX_ITER was cut off, and run() records that as cap_hit.
    step_tally = {"stage": 0, "counts": [0, 0, 0]}

    def _on_step(_step: Any) -> None:
        step_tally["counts"][min(step_tally["stage"], 2)] += 1

    def _on_task(_out: Any) -> None:
        step_tally["stage"] += 1

    crew = Crew(
        agents=[planner, builder, analyst],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
        step_callback=_on_step,
        task_callback=_on_task,
    )
    return crew, llm, tools, shared, prompt_file, brief, spec, flavor, step_tally


def run(task: str, session_dir: Path, transcript_path: Path) -> dict:
    session_dir.mkdir(parents=True, exist_ok=True)
    store = OKRStore(session_dir / "memory.db", session_dir)

    result: dict[str, Any] = {"arm": "crewai", "task": task, "cap_hit": None}

    with MCPClient() as client:
        crew, llm, tools, shared, prompt_file, brief, spec, flavor, step_tally = build_crew(
            task, session_dir, store, client
        )
        result.update(
            {
                "flavor": flavor,
                "tools_bound": len(tools),
                "model": llm.model,
                "context_window": llm.context_window_size,
                "prompt_file": prompt_file.name,
                "prompt_sha256": hashlib.sha256(shared.encode()).hexdigest(),
                "brief_sha256": spec["sha256"],
                "agents": [a.role for a in crew.agents],
                "max_iter": MAX_ITER,
            }
        )

        t0 = time.time()
        fh = transcript_path.open("w", encoding="utf-8")

        def emit(kind: str, payload: dict) -> None:
            fh.write(json.dumps({"t": round(time.time() - t0, 2), "kind": kind, **payload}) + "\n")
            fh.flush()

        emit("system", {"text": shared, "prompt_file": prompt_file.name})
        emit("brief", {"text": brief, "sha256": spec["sha256"]})
        store.log_message("Crew", json.dumps({"role": "system"}), shared)
        store.log_message("Crew", json.dumps({"role": "user"}), brief)

        try:
            out = crew.kickoff()
            final = str(out)
            result["final"] = final[:8000]
            emit("final", {"text": final})
            for i, t in enumerate(crew.tasks, 1):
                raw = str(getattr(t.output, "raw", "") or "")
                agent_role = getattr(t.agent, "role", f"task{i}")
                emit("task_output", {"stage": i, "agent": agent_role, "text": raw[:20000]})
                store.log_message(agent_role, json.dumps({"role": "assistant", "stage": i}), raw)
            usage = getattr(crew, "usage_metrics", None)
            if usage is not None:
                result["usage"] = json.loads(usage.model_dump_json()) if hasattr(usage, "model_dump_json") else str(usage)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            emit("error", {"error": result["error"]})
        finally:
            result["steps_per_stage"] = step_tally["counts"]
            capped = [
                f"stage {i + 1} ({crew.agents[i].role})"
                for i, n in enumerate(step_tally["counts"])
                if n >= MAX_ITER
            ]
            if capped:
                result["cap_hit"] = f"max_iter={MAX_ITER}: " + ", ".join(capped)
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
