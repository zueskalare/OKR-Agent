"""
BaseAgent defines a four-step workflow for agents: think, plan, act, and report.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Literal, TypedDict

from langchain_core.tools import BaseTool
from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    PrivateAttr,
    create_model,
)  #! create model have some problem
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pathlib import Path

from .tool_node_agent import MCPToolNode
import logging
import json

# Module import (not value imports): loop budgets must be read as
# `_cfg.MAX_ACT_CYCLES` at call time so env overrides and test
# monkeypatching keep working.
from .. import config as _cfg

# Tag marking PROSE LLM calls whose tokens may stream to the user. Structured
# calls stay untagged BY DESIGN — that is the guard keeping raw JSON out of
# the token stream. Tags survive the inner stream (verified); they do NOT
# survive the invoked-subgraph boundary, which is why arun re-publishes.
TOKEN_TAG = "atom_stream"

# Simulated tool-call markup in PROSE stage output (A6): tool-trained models
# fill the think/plan tool gap by writing call syntax as text and *inventing
# the observation* that follows it. Matched forms observed live on
# claude-opus-5: XML-ish invoke/parameter/function_results blocks,
# "**Tool call:**" / "**Tool: Bash**" / "**Tool result:**" headers, and
# tool_code/tool_use/tool_result fences. The result-side forms matter as much
# as the invoke-side ones — a fabricated transcript can open with (or consist
# only of) the invented observation.
_FABRICATED_TOOL_MARKUP = re.compile(
    r"(<\s*/?\s*(?:antml:)?(?:invoke|function_calls|function_results|parameter)\b"
    r"|\*\*\s*Tool(?:\s+call|\s+result|\s+output)?\s*:"
    r"|```(?:tool_code|tool_use|tool_result)\b)",
    re.I,
)
from ..utils.prompt_loader import PromptLoader
from ..memory.sqlite_rag_memory import SQLiteRAGMemory
from ..utils.utils import _message_from_dict
from ..utils.readme_sync import get_section, ensure_readme
from .messages import ReportMessage


class BaseAgent(BaseModel, ABC):
    # --- Pydantic model configuration ---
    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True,
    #     validate_assignment=False,
    #     copy_on_model_validation="none",  # <- stops deepcopies that trigger RLock error
    # )
    __name__: str
    agent_type: str = Field(default="", description="Type of the agent")
    # --- Agent configuration ---
    # Store only config, not live LLM instance
    # NOTE: max_tokens is shared with reasoning tokens on reasoning models —
    # too small a budget silently truncates structured outputs (reports/SendTo).
    llm_config: dict = Field(
        default_factory=lambda: dict(
            model="gpt-5",
            temperature=1.0,
            max_tokens=16000,
            reasoning_effort="medium",  # "minimal" or "low" or "medium" or "high"
        )
    )
    tools: List[BaseTool] = Field(default_factory=list)
    prompt_loader: Optional[PromptLoader] = Field(
        default_factory=lambda: PromptLoader()
    )  # Placeholder for prompt loader
    memory: Optional[SQLiteRAGMemory] = None
    verbose: bool = True  # Enable debug logging by default
    handle_parsing_errors: bool = True
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("AtomAgents.BaseAgent")
    )
    project_root: str = Field(default_factory=str)
    Collages: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=lambda: dict(recursion_limit=100))
    msg_storage: List[BaseMessage] = Field(default_factory=list)
    # Runtime objects (not validated / not deepcopied)
    _llm: ChatOpenAI = PrivateAttr(default=None)
    _ToolNode: MCPToolNode = PrivateAttr(default=None)
    _SubGraph: StateGraph = PrivateAttr(default=None)
    _Report_Schema: BaseModel = PrivateAttr(default=None)
    # ids of messages already persisted incrementally this run (so the final
    # end-of-run save doesn't duplicate them)
    _persisted_ids: set = PrivateAttr(default_factory=set)
    # loop-guard state (reset each run via _reset_loop_guards)
    _no_progress_cycles: int = PrivateAttr(default=0)
    _last_tool_sig: Optional[str] = PrivateAttr(default=None)
    _same_tool_repeats: int = PrivateAttr(default=0)
    _force_report: bool = PrivateAttr(default=False)
    _act_cycles: int = PrivateAttr(default=0)
    # When a node decides the run needs no report LLM call (e.g. triage
    # answered directly), it stashes {"reply", "send_to", "stage", "reason"}
    # here and run()/stream() finalize from it instead of calling report().
    _short_circuit: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    # Streaming-disabled copy of the LLM for structured calls (lazy).
    _structured_llm_cache: Any = PrivateAttr(default=None)
    # Act-stage tool policy, resolved ONCE at construction: the ToolNode and
    # every act bind_tools call read this same snapshot, so the serving
    # registry and the bound set cannot diverge (flag changes require agent
    # reconstruction — deliberate).
    _act_tools: List[Any] = PrivateAttr(default_factory=list)
    # Ids of messages already in outer state when this run started; used to
    # scope tool-result evidence to THIS run (inherited ToolMessages from
    # other agents' earlier runs must not count). Set by arun, cleared in the
    # epilogue/failure resets; empty for direct node calls (tests).
    _run_inbound_ids: set = PrivateAttr(default_factory=set)

    def __init__(self, **data):
        super().__init__(**data)
        # build runtime objects AFTER pydantic init. The factory routes
        # llm_config["provider"] to the right backend (openai/anthropic/local)
        # and filters per-provider kwargs; passing a ready `llm` still wins.
        from ..llm_factory import make_llm_from_config

        self._llm = data.get("llm") or make_llm_from_config(self.llm_config)
        # The ToolNode serves exactly what act binds: a stray call to a tool
        # outside the agent's policy set returns "unknown tool" instead of
        # executing. The policy is resolved ONCE here (self.tools is already
        # populated by pydantic; subclass _select_act_tools overrides resolve
        # normally during base __init__) — act() reuses the same snapshot.
        self._act_tools = list(self._select_act_tools())
        self._ToolNode = MCPToolNode(tools=self._act_tools)
        # Expose a stable name attribute so that prompts & logging can reference
        # the agent's class name without risking an AttributeError.
        # NOTE: Instances do not normally have a ``__name__`` attribute – only
        # classes do.  A handful of places in the codebase (e.g. when loading
        # prompt templates or preparing report messages) were attempting to
        # access ``self.__name__`` which would raise an AttributeError at
        # runtime.  We therefore attach the attribute here so that existing
        # references keep working while also providing the correct value.

        self.agent_type = self.agent_type or self.__name__

        pipeline = StateGraph(state_schema=self.SubState)
        pipeline.add_node("think", self.think)
        pipeline.add_node("plan", self.plan)
        pipeline.add_node("act", self.act)
        pipeline.add_node("tool", self._ToolNode)

        pipeline.add_edge("think", "plan")
        pipeline.add_conditional_edges(
            "act",
            lambda state: state.get("send_to"),
            {"tools": "tool", "plan": "plan"},
        )
        pipeline.add_conditional_edges(
            "plan",
            lambda state: state.get("send_to"),
            {"act": "act", "report": END},
        )

        pipeline.add_edge("tool", "act")
        pipeline.set_entry_point("think")
        self._SubGraph = pipeline.compile()
        # initialize logger for this agent
        self.logger = logging.getLogger(f"AtomAgents.{self.agent_type}")
        if self.verbose:
            # enable debug output for this agent
            self.logger.setLevel(logging.DEBUG)

        if not self.project_root or not str(self.project_root).strip():
            # An empty project_root historically resolved to the process cwd
            # (and, through shell expansion, to filesystem root). Refuse.
            raise ValueError(
                f"Agent {self.__name__} requires an explicit project_root; "
                "got an empty value."
            )
        project_root = Path(self.project_root).resolve()
        project_root.mkdir(parents=True, exist_ok=True)
        self._Report_Schema = self.get_report_schema()
        self.logger.info(
            f"Agent {self.__name__} initialized with project root: {project_root}"
        )

    # expose properties
    @property
    def llm(self) -> ChatOpenAI:
        return self._llm

    @property
    def ToolNode(self) -> MCPToolNode:
        return self._ToolNode

    @property
    def SubGraph(self) -> StateGraph:
        return self._SubGraph

    class SubState(TypedDict):
        """
        SubState model for agent workflow.
        Contains fields for messages and an optional send_to field.

        ``messages`` MUST carry the add_messages reducer: agent nodes return
        the full list (id-merged, no dupes), but the prebuilt ToolNode returns
        ONLY the new ToolMessage — under plain replace semantics every tool
        call wiped the conversation down to the orphaned tool result
        (Anthropic rejects that outright; OpenAI silently accepted the
        amnesia).
        """

        messages: Annotated[List[Any], add_messages]
        send_to: Literal[
            "triage", "think", "plan", "act", "tools", "update_memory", "report", "end"
        ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _fill_project_root(self, text: str) -> str:
        """Substitute the PROJECT_ROOT placeholder conventions used across the
        prompt templates with the actual workspace path. Historically the
        literal `PROJECT_ROOT` reached the LLM unsubstituted."""
        root = str(getattr(self, "project_root", "") or "")
        if not root or not text:
            return text
        return (
            text.replace("PROJECT_ROOT", root)
            .replace("<project_root>", root)
            .replace("{project_root}", root)
        )

    def _select_act_tools(self) -> List[BaseTool]:
        """Resolve the act-stage tool policy — called ONCE from __init__.

        Subclasses override to narrow the set (the Supervisor binds
        coordination tools only — A7). Resolving at construction keeps the
        ToolNode registry and every bind_tools call on the same snapshot;
        policy-flag changes take effect on agent reconstruction, never
        mid-life (a live property here let the bound set and the serving
        registry diverge)."""
        return list(self.tools)

    @property
    def act_tools(self) -> List[BaseTool]:
        """The construction-time act-tool snapshot (bound in act, listed in
        the act prompt, served by the ToolNode — always the same set)."""
        return list(self._act_tools)

    def _policy_prompt(self, stage: str) -> str:
        """Runtime tool-policy statement injected into the stage-varying
        prompt block. Empty by default; the Supervisor emits its Execution
        Boundary here ONLY when the coordinator-only policy is actually
        active — putting it in the static yaml template contradicted the
        ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 legacy mode (the A8 dual-protocol
        lie, reintroduced)."""
        return ""

    def _stage_contract(self, stage: str) -> str:
        """The per-stage tool contract (A6).

        Advertising the full tool list as "available via native tool calls"
        in EVERY stage while only act binds them taught tool-trained models
        to *simulate* the call and invent its output in think/plan prose —
        observed live in 4/5 claude-opus-5 sessions, once escalating to a
        fully fabricated completion report with zero tool calls.
        ``act_final`` is the budget-exhausted wrap-up call: it binds NO
        tools, and telling the model otherwise (the plain act contract)
        invites exactly the simulation the budget guard exists to stop.
        """
        stage_l = (stage or "").lower()
        n_tools = len(self.act_tools)
        if stage_l in ("think", "plan"):
            return (
                f"\n## Stage Contract — {stage_l.upper()} (reasoning only)\n"
                "You cannot call tools in this step; tools are bound only in "
                f"the ACT step ({n_tools} tool(s) will be available there). "
                "Never write tool-call syntax or a transcript of tool output "
                "here. Never state a command's output, a directory listing, a "
                "version number, or a numeric result unless it already appears "
                "in a real tool result earlier in this conversation. If you "
                "need information from a tool, say in one plain sentence what "
                "you intend to call, and let ACT call it.\n"
            )
        if stage_l == "act":
            return (
                "\n## Stage Contract — ACT\n"
                "The tools listed below are bound to THIS call and are "
                "available right now. Any earlier `[loop-guard]` note saying "
                "tools were no longer available referred to a different "
                "agent's run and does not apply to you. Invoke tools as "
                "native tool calls only. Never simulate a call or its output; "
                "if you do not call a tool, say so plainly.\n"
            )
        if stage_l == "act_final":
            return (
                "\n## Stage Contract — ACT (final wrap-up, no tools)\n"
                "Your tool budget for this run is spent: NO tools are bound "
                "to this call and tool use is over for this run. Do not "
                "write tool-call syntax and do not invent tool output. "
                "Summarize in plain prose what was actually done (citing "
                "only real tool results earlier in this conversation), what "
                "remains, and any blockers.\n"
            )
        if stage_l == "report":
            return (
                "\n## Stage Contract — REPORT\n"
                "No tools are available in this step. Every path, value, and "
                "status you report must be traceable to a real tool result "
                "earlier in this conversation. If nothing was executed this "
                "run, say so plainly.\n"
            )
        return ""

    @property
    def system_prompt_static(self) -> str:
        """Stage-INVARIANT system prefix: yaml template + project-directory
        block. This is the first system message of every stage call and MUST
        stay byte-identical across think/plan/act/report — cache_system_blocks
        marks the first system block for the Anthropic prompt cache, and any
        per-stage variance here turns every stage transition into a full
        cache-prefix miss (write-billed). Stage-varying content belongs in
        stage_system_prompt, which rides the second (uncached) block.
        """
        base_prompt = self.prompt_loader.get_template(self.__name__).get("template", "")

        # Many agents (e.g. Supervisor, Modeler, Simulator) need to read and
        # write artefacts to a dedicated *project directory* created by the
        # Supervisor at the beginning of a job run; state it once here at the
        # common BaseAgent level.
        project_root = getattr(self, "project_root", None)
        if project_root:
            base_prompt += (
                "\n## Project Directory\n"
                f"All files **must** be read from or saved to the single "
                f"project directory located at: {project_root}. "
                "When referencing files use either an absolute path that starts "
                "with this directory or a path *relative* to it.  Do not create "
                "or access data outside of this folder.\n"
                "Only files under THIS project directory are this project's "
                "artifacts and evidence. Files found in other sessions' "
                "directories belong to other runs — they are NEVER this "
                "project's deliverables and must not be cited, verified, or "
                "reported as such.\n"
            )

        return self._fill_project_root(base_prompt)

    def stage_system_prompt(self, stage: str) -> str:
        """Stage-VARYING system block: policy statement + stage contract +
        (act only) the bound tool list. Call sites prepend this to the yaml
        stage prompt in the SECOND system message so the first stays
        cache-stable (see system_prompt_static)."""
        parts = [self._policy_prompt(stage), self._stage_contract(stage)]

        # Tools are invoked through native tool calling (bind_tools) — do NOT
        # instruct a textual call format here; that competes with the real
        # protocol. The list is exactly what act binds (act_tools), never the
        # full inventory — advertising unbound tools is the A6/A7 lie.
        act_tools = self.act_tools
        if (stage or "").lower() == "act" and act_tools:
            tool_list_prompt = [
                "\n## Available Tools",
                "The following tools are available via native tool calls:",
                "",
            ]

            bound_names = set()
            for tool in act_tools:
                try:
                    tool_name = getattr(tool, "name", str(tool))
                    tool_desc = getattr(tool, "description", "")
                except Exception:
                    tool_name, tool_desc = str(tool), ""
                bound_names.add(tool_name)
                tool_list_prompt.append(f"- **{tool_name}** — {tool_desc}")

            # Shell/file guidance only when those tools are actually bound —
            # naming unbound tools here is the same lie the stage gating
            # removes (the Supervisor binds neither after A7).
            if bound_names & {"bash_run", "write_file"}:
                tool_list_prompt.append(
                    "\nShell state does NOT persist between bash_run calls: environment "
                    "variables and `cd` are lost after each call. Always use absolute "
                    "paths. Author files with write_file, never with shell heredocs."
                )
            parts.append("\n".join(tool_list_prompt))

        return self._fill_project_root("".join(p for p in parts if p))

    def system_prompt_for(self, stage: str = "act") -> str:
        """Full assembled system prompt for ONE stage (static prefix + the
        stage-varying block). Prompt-building call sites use the two halves
        as separate system messages; this combined view serves tests and
        external callers."""
        return self.system_prompt_static + self.stage_system_prompt(stage)

    @property
    def system_prompt(self) -> str:
        """Back-compat alias: the act-stage prompt (the only stage that binds
        tools). Stage nodes must call system_prompt_for(<stage>) instead."""
        return self.system_prompt_for("act")

    # properties for prompts
    @property
    def think_prompt(self) -> str:
        """
        Prompt for the THINK stage, used to record incoming messages to memory.
        """
        tpl = self.prompt_loader.get_template(self.__name__)
        return self._fill_project_root(tpl.get("think", tpl.get("template", "")))

    @property
    def plan_prompt(self) -> str:
        """
        Prompt for the PLAN stage, used to generate a structured job plan.
        """
        tpl = self.prompt_loader.get_template(self.__name__)
        return self._fill_project_root(tpl.get("plan", ""))

    @property
    def act_prompt(self) -> str:
        """
        Prompt for the ACT stage, used to execute actions based on the job plan.
        """
        tpl = self.prompt_loader.get_template(self.__name__)
        return self._fill_project_root(tpl.get("act", ""))

    def get_report_schema(self):
        """
        Returns a Pydantic model for the report schema.
        """
        send_to_type = self.get_send_to()
        DynamicReport = create_model(
            "DynamicReport",
            report=(str, ...),
            send_to=(send_to_type, ...),
        )
        return DynamicReport

    @property
    def collage_agents(self) -> str:
        """
        Returns a string of other agents that this agent can collaborate with.
        Used to generate the next agent to send the report to.
        """
        # Name + short description only. Interpolating every colleague's FULL
        # system prompt here roughly tripled the cost of every report call.
        role_template = "Agent Name: {name}\nDescription: {description}\n"
        roles = []
        for _agent in self.Collages:
            try:
                description = self.prompt_loader.get_template(_agent).get(
                    "description", ""
                )
            except Exception:
                description = ""
            roles.append(role_template.format(name=_agent, description=description))
        return (
            "Colleague Agents:\n" + "\n".join(roles)
            if roles
            else "No other colleges available."
        )

    @abstractmethod
    def think(self, inputs: Dict) -> Any:
        """Generate internal reasoning based on inputs."""
        ...

    @abstractmethod
    def plan(self, thought: Dict) -> Any:
        """Create a plan of actions based on thoughts."""
        ...

    @abstractmethod
    def act(self, plan: Dict) -> Any:
        """Perform the planned actions and return the results."""
        ...

    def get_send_to(self) -> Any:
        """
        Dynamically create a Literal type for the 'send_to' field.
        """
        if not self.Collages:
            return str  # Fallback to string if no collages are defined
        return Literal[tuple(self.Collages)]

    def get_current_status(self) -> List[BaseMessage]:
        try:
            current_status = json.dumps(self.get_okr_context(), default=str)
        except Exception as e:
            self.logger.warning("Could not serialize OKR context: %s", e)
            current_status = "{}"
        if len(current_status) > 16000:
            current_status = current_status[:16000] + "... [truncated]"
        msgs: List[BaseMessage] = [
            AIMessage(content="Current Task status:\n" + current_status)
        ]
        # Attach snapshot of the shared README sections (Tasks + this agent)
        try:
            if getattr(self, "project_root", None):
                ensure_readme(self.project_root)
                tasks_md = get_section(self.project_root, "Tasks")
                my_md = get_section(
                    self.project_root, getattr(self, "__name__", self.agent_type)
                )
                bundle = []
                if tasks_md:
                    bundle.append("[README: Tasks]\n" + tasks_md)
                if my_md:
                    bundle.append(
                        f"[README: {getattr(self, '__name__', self.agent_type)}]\n"
                        + my_md
                    )
                if bundle:
                    msgs.append(AIMessage(content="\n\n".join(bundle)))
        except Exception:
            # Non-fatal if README is missing or unreadable
            pass
        return msgs

    def _board_summary_line(self) -> str:
        """One-line task-board summary for cheap prompts (triage). Never the
        up-to-16 KB get_current_status() blob."""
        if not self.memory:
            return "Task board: unavailable (no memory)."
        try:
            tasks = self.memory.get_all_tasks() or []
        except Exception as e:
            self.logger.debug("Board summary failed: %s", e)
            return "Task board: unavailable."
        if not tasks:
            return "Task board: 0 tasks (empty session)."

        def _status(t: Any) -> str:
            return str(t.get("status", "")).lower() if isinstance(t, dict) else ""

        closed = ("done", "achieved", "pass", "completed", "finished")
        open_n = sum(1 for t in tasks if _status(t) not in closed)
        return f"Task board: {len(tasks)} task(s), {open_n} open."

    def _recent_own_messages(
        self, limit: int = 4, max_chars: int = 400
    ) -> List[BaseMessage]:
        """This agent's last few DB messages, hard-truncated — a compact
        recall window for cheap prompts. Deliberately NOT get_workflow_steps(),
        which globs every project *.md at up to 12 kB each."""
        if self.memory is None:
            return []
        try:
            raw = self.memory.get_raw_messages_by_agent(agent_name=self.__name__)
        except Exception as e:
            self.logger.debug("Recent-own-messages fetch failed: %s", e)
            return []
        msgs: List[BaseMessage] = []
        for item in raw[-(limit * 3):]:
            try:
                msgs.append(_message_from_dict(item))
            except Exception:
                continue
        out: List[BaseMessage] = []
        for m in self.filter_tool_messages(msgs)[-limit:]:
            text = self._content_text(m.content)
            if len(text) > max_chars:
                text = text[:max_chars] + " …[truncated]"
            out.append(
                AIMessage(content=text, name=getattr(m, "name", None) or self.__name__)
            )
        return out

    def _build_report_user_prompt(
        self, state: Optional[Dict] = None, n_tool_results: Optional[int] = None
    ) -> str:
        """Construct the user message for the reporting step.

        The evidence gate: a run with ZERO tool results gets an explicit "do
        not claim artifacts" instruction, because a fabricated think
        transcript once rode all the way to a 'COMPLETE ✅' report with an
        empty session dir (A6 escalation, session-nb-fem-20260806-135945).
        ``n_tool_results`` is the RUN-scoped count (report() passes it via
        _run_tool_result_count); when omitted, it falls back to counting
        ``state`` whole — acceptable only for direct callers/tests that build
        a single-run state.
        """

        def _format_agent(agent: Any) -> str:
            if isinstance(agent, str):
                return agent
            return getattr(agent, "__name__", getattr(agent, "name", str(agent)))

        if self.Collages:
            colleague_names = ", ".join(_format_agent(agent) for agent in self.Collages)
            handoff_instructions = (
                "Select the best colleague from this list and return their exact agent "
                "name in the structured `send_to` field: "
                f"{colleague_names}."
            )
        else:
            handoff_instructions = "No colleagues are listed; set the structured `send_to` field to `Supervisor`."

        colleague_context = self.collage_agents
        human_hint = (
            "Choose `Human` only when you are genuinely blocked on the user's input — a missing "
            "physical parameter, approval for a long/expensive or destructive step, or an ambiguity "
            "only the user can resolve.\n"
            if "Human" in (self.Collages or [])
            else ""
        )
        evidence_hint = (
            "Report only what this conversation shows actually happened: every file path, "
            "metric, and status you state must be traceable to a real tool result above. "
            "Do not present estimated or expected values as computed results.\n"
        )
        if n_tool_results is None and state is not None:
            n_tool_results = sum(
                isinstance(m, ToolMessage) for m in state.get("messages", []) or []
            )
        if n_tool_results == 0:
            evidence_hint += (
                "**No tool ran in this agent run.** Do not claim any artifact was "
                "produced, any file was written, or any value was computed.\n"
            )
        return (
            "Use the workflow steps to produce a concise markdown hand-off report for your teammates. "
            "Summarise key actions, results, blockers, and clear next steps so the work can continue seamlessly.\n"
            f"{handoff_instructions}\n"
            "Never choose yourself as the recipient.\n"
            "`END` is a first-class outcome: choose it when the work is complete, the user's question "
            "has been answered, or no further agent action is needed (including when you are waiting "
            "on the user to reply). Do NOT invent follow-up work for a colleague just to keep the "
            "conversation going.\n"
            f"{evidence_hint}"
            f"{human_hint}"
            f"{colleague_context}"
        )

    @staticmethod
    def _normalize_next_target(raw: str, valid: List[str]) -> str:
        """Map a sloppy LLM-emitted target ("**Next → Simulator**", "end",
        "Simulator Agent") onto the exact collage name. Returns ``raw``
        unchanged when nothing matches so the caller's guard still fires."""
        import re as _re

        def _key(s: str) -> str:
            return _re.sub(r"[^a-z0-9]", "", s.lower())

        text = _re.sub(r"\*+", "", raw or "")
        text = _re.sub(r"(?i)next\s*(?:→|->)?\s*", "", text).strip()
        k = _key(text)
        if not k:
            return raw
        by_key = {_key(v): v for v in valid}
        if k in by_key:
            return by_key[k]
        aliases = {
            "end": "END",
            "finish": "END",
            "finished": "END",
            "done": "END",
            "stop": "END",
            "terminate": "END",
            "user": "Human",
            "humanuser": "Human",
            "simulatoragent": "Simulator",
            "modeleragent": "Modeler",
            "modelbuildingagent": "Modeler",
            "supervisoragent": "Supervisor",
            "atomagentmanager": "Supervisor",
        }
        alias = aliases.get(k)
        if alias and alias in valid:
            return alias
        return raw

    def _edit_context(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Context management pass applied before every LLM call (WS20.2).

        - ClearToolUsesEdit (default ON): once the approximate token count
          exceeds config.TOOL_CLEAR_TRIGGER, older tool results are replaced
          with placeholders, keeping the TOOL_CLEAR_KEEP most recent intact —
          bulky LAMMPS/tool outputs otherwise replay into every subsequent
          act/report prompt for the rest of the run.
        - SummarizationMiddleware (default OFF — prompt-content-changing):
          summarizes long histories using the non-streaming LLM copy
          (summarization must never stream tokens into the bridge).
        Never raises; on any failure the original messages are returned.
        """
        try:
            msgs = list(messages)
            if _cfg.TOOL_CLEAR_TRIGGER > 0:
                from langchain.agents.middleware.context_editing import (
                    ClearToolUsesEdit,
                )
                from langchain_core.messages.utils import (
                    count_tokens_approximately,
                )

                ClearToolUsesEdit(
                    trigger=_cfg.TOOL_CLEAR_TRIGGER, keep=_cfg.TOOL_CLEAR_KEEP
                ).apply(msgs, count_tokens=count_tokens_approximately)
            if _cfg.SUMMARIZE:
                from langchain.agents.middleware import SummarizationMiddleware

                mw = SummarizationMiddleware(
                    model=self._structured_llm(),
                    trigger=("fraction", 0.8),
                    keep=("messages", 20),
                )
                update = mw.before_model({"messages": msgs}, None)
                if update and update.get("messages"):
                    msgs = list(update["messages"])
            return msgs
        except Exception as e:
            self.logger.warning("[context-edit] skipped: %s", e)
            return list(messages)

    def _invoke(self, message: List[BaseMessage], llm=None):
        """Prose LLM call: context-edited + token-tagged (the bridge streams
        ONLY these calls). All six think/act call sites route through here."""
        return (llm or self.llm).invoke(
            self._edit_context(message), config={"tags": [TOKEN_TAG]}
        )

    def _structured_llm(self):
        """The LLM with streaming force-disabled, for structured calls.

        With the token bridge active, langgraph's messages-mode callback
        makes langchain auto-upgrade EVERY invoke to streaming — and the
        streaming json_schema path enforces strict mode
        (additionalProperties: false), which open-dict schemas like
        TaskOperations cannot satisfy (live 400 'invalid_json_schema').
        Structured calls must never stream; this restores their exact
        non-streaming request shape.
        """
        if self._structured_llm_cache is not None:
            return self._structured_llm_cache
        llm = self.llm
        try:
            llm = llm.model_copy(update={"disable_streaming": True})
        except Exception:
            pass  # test fakes have no model_copy — they never stream anyway
        self._structured_llm_cache = llm
        return llm

    def _structured_invoke(self, schema, prompt, *, retries: int = 2):
        """Invoke ``with_structured_output(schema)`` with bounded
        error-feedback retries (the ToolStrategy.handle_errors pattern ported
        to our hand-rolled call sites). On failure the error is appended as a
        message so the model can self-correct; the final failure re-raises
        for the caller's own handling. Success path is one call — identical
        cost to the previous bare ``.invoke``.
        """
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = list(prompt)
        messages = self._edit_context(messages)
        llm = self._structured_llm().with_structured_output(schema)
        last_err: Optional[BaseException] = None
        for attempt in range(1, retries + 2):
            try:
                out = llm.invoke(messages)
                if out is None:
                    raise ValueError("structured output returned None")
                return out
            except Exception as e:
                last_err = e
                if attempt >= retries + 1:
                    break
                self.logger.warning(
                    "[structured-retry] %s attempt %d failed: %s",
                    getattr(schema, "__name__", schema),
                    attempt,
                    e,
                )
                messages = messages + [self._structured_feedback(e)]
        raise last_err

    @staticmethod
    def _structured_feedback(e: BaseException) -> HumanMessage:
        """The error-feedback message of the structured retry ladder — shared
        so a future async twin (_astructured_invoke) reuses the exact loop
        shape: invoke → on failure append this → bounded retries → re-raise."""
        return HumanMessage(
            content=(
                f"Your previous response failed validation: {e}. "
                "Respond again, matching the required schema exactly."
            )
        )

    def _finalize_short_circuit(self, state: Dict) -> Dict[str, Any]:
        """Close the run WITHOUT the report() LLM call.

        A node sets ``self._short_circuit = {"reply": str, "send_to": str,
        "stage": str, "reason": str}``; run()/stream() turn it into the single
        ReportMessage this run emits. This keeps the one-report-per-run
        invariant (handoff counting, memory persistence, one frontend bubble)
        while saving the whole report round-trip on turns that need no agent
        work."""
        sc = self._short_circuit or {}
        valid = list(self.Collages or [])
        send_to = sc.get("send_to") or ("END" if "END" in valid else "Supervisor")
        msg = ReportMessage(
            content=self._stage_content(sc.get("stage", "reply"), sc.get("reply", "")),
            name=self.__name__,
        )
        msgs = list(state.get("messages", []))
        msgs.append(msg)
        self.logger.warning(
            "[short-circuit] %s ending turn without report LLM call: reason=%s send_to=%s",
            self.__name__,
            sc.get("reason", "unspecified"),
            send_to,
        )
        return {"messages": msgs, "send_to": send_to}

    def report(self, state: Dict) -> Dict[str, Any]:
        """
        Produce a standard markdown workflow report showing workflow step, last action, OKR status, and next/last agent.
        """

        self.logger.warning(
            f"Agent {self.__name__} reporting step: len_storage: {len(self.msg_storage)}\t len_messages: {len(state.get('messages', []))}"
        )

        # DynamicReport = create_model(
        #     "DynamicReport",
        #     report=(str, ...),
        #     send_to=(send_to_type, ...),
        # )

        # Run-scoped tool evidence: inherited ToolMessages (other agents',
        # earlier turns) must not vouch for THIS run's claims.
        n_tool_results = self._run_tool_result_count(state.get("messages", []))
        report_prompt = self._build_report_user_prompt(
            state, n_tool_results=n_tool_results
        )
        steps = state.get("messages", [])
        # Skip a leading human input query if present; never drop real steps.
        if steps and isinstance(steps[0], HumanMessage):
            steps = steps[1:]
        prompt_temp = ChatPromptTemplate.from_messages(
            [
                # Two system blocks: the static prefix stays byte-identical
                # across stages so the Anthropic prompt cache keeps hitting.
                SystemMessage(content=self.system_prompt_static),
                SystemMessage(content=self.stage_system_prompt("report")),
                HumanMessage(content=report_prompt),
                MessagesPlaceholder(variable_name="steps"),
            ]
        )
        prompt = prompt_temp.format_messages(steps=steps)
        self.logger.debug("Calling llm with message for the report step")
        report = self._structured_invoke(self._Report_Schema, prompt)
        content = self._strip_fabricated_tool_calls(
            report.report, "report", has_tool_evidence=n_tool_results > 0
        )
        send_to = report.send_to

        # Log-only fabricated-completion detector: a completion claim with
        # zero tool results this run is the A6 escalation signature. No
        # coercion — conversational turns legitimately say "done".
        if n_tool_results == 0 and re.search(
            r"\b(complete|completed|done)\b|✅", content, re.I
        ):
            self.logger.error(
                "[evidence] %s report claims completion with 0 tool results this run",
                self.__name__,
            )

        # Validate and normalise next-hop to avoid self-loops or invalid targets
        valid_next = list(self.Collages or [])
        if isinstance(send_to, str):
            send_to = self._normalize_next_target(send_to, valid_next)
            me = getattr(self, "__name__", self.agent_type)
            if send_to == me or (valid_next and send_to not in valid_next):
                if send_to == me:
                    # The agent wants to keep working: hand back to the
                    # coordinator rather than killing the run.
                    fallback = (
                        "Supervisor"
                        if "Supervisor" in valid_next
                        else ("END" if "END" in valid_next else send_to)
                    )
                else:
                    # Unknown/hallucinated target: END is the safe stop.
                    # Never valid_next[0] — for the Supervisor that silently
                    # drafted the Modeler (observed live on a bare "hi").
                    fallback = (
                        "END"
                        if "END" in valid_next
                        else ("Supervisor" if "Supervisor" in valid_next else send_to)
                    )
                self.logger.warning(
                    "[route-guard] Report selected invalid/self target '%s'; coercing to '%s' (valid=%s)",
                    send_to,
                    fallback,
                    valid_next,
                )
                send_to = fallback

        # Per-turn handoff budget: this report is run N+1 of the turn. Force
        # END (with a user-visible notice) instead of another hop; the outer
        # handoff_router is the silent backstop behind this check.
        from ..utils.routing import reports_since_last_human

        used = reports_since_last_human(state.get("messages", []))
        if send_to != "END" and used + 1 >= _cfg.MAX_HANDOFFS and "END" in valid_next:
            self.logger.error(
                "[loop-guard] handoff budget reached (%d/%d agent runs this turn) — forcing END",
                used + 1,
                _cfg.MAX_HANDOFFS,
            )
            content += (
                f"\n\n[loop-guard] Handoff budget ({_cfg.MAX_HANDOFFS} agent runs per "
                "user turn) reached — ending this turn here. Reply to continue."
            )
            send_to = "END"

        res_message = ReportMessage(
            content=f"{self.__name__}:\n{content}", name=self.__name__
        )
        # Copy: appending to the caller's list mutated shared state in place.
        res_messages = list(state.get("messages", []))
        res_messages.append(res_message)
        self.logger.info(
            "Prepared report message: send_to=%s; candidates=%s",
            send_to,
            valid_next,
        )
        return {
            "messages": res_messages,
            "send_to": send_to,
        }

    def filter_tool_messages(self, msg: List[BaseMessage]) -> bool:
        """Filter out tool messages from a list of messages."""

        # Filter out tool messages from the result
        def filter_tool_messages(msg: BaseMessage) -> bool:
            """Filter out tool messages from the result."""
            if isinstance(msg, ToolMessage):
                return False
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    return False
            return True

        return [msg for msg in msg if filter_tool_messages(msg)]

    # ------------------------------------------------------------------
    # Run machinery: arun is the ONE real implementation; run() wraps it.
    # ------------------------------------------------------------------

    def _run_prologue(self, inputs: Dict[str, Any]) -> None:
        self.logger.info(
            "Agent '%s' run starting with inputs: %s", self.agent_type, inputs
        )
        self._emit_working()
        # if last message is a human message, save it to msg_storage
        if inputs.get("messages"):
            last_message = inputs["messages"][-1]
            if isinstance(last_message, HumanMessage):
                self.msg_storage.append(last_message)

    def _run_epilogue(
        self, result: Dict[str, Any], inbound_ids: Optional[set] = None
    ) -> Dict[str, Any]:
        """BLOCKING (report LLM round-trip + sqlite/embedding writes).

        arun must call this via asyncio.to_thread — inlining it on the event
        loop stalls every concurrent request for the duration of the report
        call. This is the tier's number-one regression trap; keep it off-loop.
        """
        self.logger.info("Agent '%s' subgraph execution completed", self.agent_type)
        self.logger.debug(
            f"Agent {self.__name__} reporting step: len_storage: {len(self.msg_storage)}\t len_messages: {len(result.get('messages', []))}"
        )
        result["messages"] = self.msg_storage + result.get("messages", [])
        result = (
            self._finalize_short_circuit(result)
            if self._short_circuit
            else self.report(result)
        )
        # Delta-only node update: the inbound history is already in outer state
        # (add_messages), and the backend's updates-batch dedupe is per-request
        # — returning the full list replays prior turns on the wire and
        # re-embeds them into memory. id-None messages are always new; every
        # state-held message was id-stamped by the reducer. An empty/absent
        # inbound_ids (direct callers, legacy drivers) keeps the full list —
        # today's behavior.
        known_ids = inbound_ids or set()
        new_msgs = [
            m
            for m in result.get("messages", [])
            if getattr(m, "id", None) is None or m.id not in known_ids
        ]
        # Run-scoped internal notices (loop guards) stay out of the shared
        # outer history — report() above already saw them in result["messages"]
        # before this filter, which is exactly their intended scope (A5).
        new_msgs = [m for m in new_msgs if not self._is_internal(m)]
        result["messages"] = new_msgs
        self.logger.info(
            "Agent '%s' final report data: %s",
            self.__name__,
            len(result.get("messages", [])),
        )

        #! Save messages to memory
        # msg_storage holds this turn's HumanMessage; its id is inbound so the
        # delta filter drops it from the return — it must still be persisted,
        # once, by the entry agent.
        self._save_run_messages(self.msg_storage + new_msgs)
        self.logger.info(
            f"Returning response from run send to {result.get('send_to')}, where all the collages are {self.Collages}"
        )
        self.msg_storage = []  # Clear the message storage after run
        self._persisted_ids = set()
        self._short_circuit = None
        self._run_inbound_ids = set()
        return result

    def _run_failure(
        self,
        result: Optional[Dict[str, Any]],
        exc: BaseException,
        inbound_ids: Optional[set] = None,
    ) -> None:
        """BLOCKING partial-history save for the failure path."""
        self._short_circuit = None
        self.logger.error(
            "Error during agent '%s' run: %s — saving partial history",
            self.__name__,
            str(exc),
        )
        try:
            if result:
                known_ids = inbound_ids or set()
                new_msgs = [
                    m
                    for m in result.get("messages", [])
                    if (getattr(m, "id", None) is None or m.id not in known_ids)
                    and not self._is_internal(m)
                ]
                self._save_run_messages(self.msg_storage + new_msgs)
            elif self.msg_storage:
                self._save_run_messages(self.msg_storage)
        except Exception as save_err:
            self.logger.error("Partial-history save failed: %s", save_err)
        # Mirror the success-path resets: a stale msg_storage would be
        # prepended into the next run's update AND its memory save, and node
        # RetryPolicy re-runs a failed arun in-process, compounding the leak
        # per attempt. Stale _persisted_ids hold CPython id() values that a
        # new object can reuse, silently dropping a real message from
        # persistence.
        self.msg_storage = []
        self._persisted_ids = set()
        self._run_inbound_ids = set()

    @staticmethod
    def _stream_writer():
        """The OUTER node's custom-stream writer, or None when unavailable.
        Captured in the node body (never stored on self — no new per-agent
        mutable state under concurrency)."""
        try:
            from langgraph.config import get_stream_writer

            return get_stream_writer()
        except Exception:
            return None

    async def _adrive_subgraph(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Drive the inner subgraph asynchronously.

        values-mode payloads are post-reducer state, so the LAST payload is
        exactly what ``SubGraph.invoke`` would have returned — no manual
        re-implementation of add_messages needed. Each NEW message (id-diffed
        against the inbound history) is published on the custom stream as it
        completes, so the wire sees think/plan/act chronologically instead of
        one whole-run burst; the trailing updates batch dedupes by id.
        """
        writer = self._stream_writer()
        publish = writer is not None and _cfg.TOKEN_STREAM
        published: set = {
            m.id for m in (inputs.get("messages") or []) if getattr(m, "id", None)
        }
        final: Dict[str, Any] = {}
        # messages-mode taps the inner LLM streams: per-token chunks with
        # tags preserved (probe-pinned) — the raw material of the token
        # bridge. Only enabled when publishing.
        stream_mode = ["values", "messages"] if publish else ["values"]
        async for mode, payload in self.SubGraph.astream(
            inputs, config=self.config, stream_mode=stream_mode
        ):
            if mode == "messages":
                chunk, meta = payload
                # Tag gate: ONLY explicitly tagged prose calls stream —
                # untagged (structured) calls can never leak JSON tokens.
                if TOKEN_TAG not in (meta.get("tags") or ()):
                    continue
                # content_blocks-based flattener: reasoning/thinking chunk
                # blocks are classified non-text and cannot leak either.
                text = self._content_text(chunk)
                if not text:
                    continue
                writer(
                    {
                        "event": "token",
                        "agent": self.__name__,
                        "stage": meta.get("langgraph_node") or "",
                        "text": text,
                    }
                )
                continue
            final = payload
            if not publish:
                continue
            for msg in payload.get("messages", ()) or ():
                mid = getattr(msg, "id", None)
                if mid is None or mid in published:
                    continue
                published.add(mid)
                # Internal run-scoped notices (loop guards) stay off the wire
                # too: published as messages they can even ADOPT the open act
                # token segment (the retry note carries the act header) and
                # replace the real act output item in the client transcript.
                if getattr(msg, "type", "") in ("ai", "tool") and not self._is_internal(
                    msg
                ):
                    writer(
                        {"event": "message", "agent": self.__name__, "message": msg}
                    )
        return final

    def _publish_final_message(self, result: Dict[str, Any]) -> None:
        """Publish the post-subgraph message (report or short-circuit reply)
        on the custom stream so it lands in chronological order too. The id
        is ensured first — the outer updates batch re-delivers the same
        object and the backend dedupes on it."""
        if not _cfg.TOKEN_STREAM:
            return
        writer = self._stream_writer()
        if writer is None:
            return
        msgs = result.get("messages") or []
        if not msgs:
            return
        msg = msgs[-1]
        if getattr(msg, "id", None) is None:
            import uuid as _uuid

            msg.id = f"atom-{_uuid.uuid4().hex}"
        writer({"event": "message", "agent": self.__name__, "message": msg})

    async def arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async workflow — the real implementation, registered as the outer
        graph node by the production backend."""
        self._run_prologue(inputs)
        # Ids already in outer state before this run: the epilogue filters the
        # node update (and the memory save) down to messages NOT in this set,
        # so each run returns only what it produced.
        inbound_ids = {
            m.id for m in (inputs.get("messages") or []) if getattr(m, "id", None)
        }
        # Run-scope marker for tool-result evidence (report's zero-evidence
        # gate must not count inherited ToolMessages as this run's work).
        self._run_inbound_ids = set(inbound_ids)
        result = None
        try:
            result = await self._adrive_subgraph(inputs)
            result = await asyncio.to_thread(self._run_epilogue, result, inbound_ids)
            self._publish_final_message(result)
        except Exception as e:
            await asyncio.to_thread(self._run_failure, result, e, inbound_ids)
            raise
        return result

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sync entry point — valid OFF-loop only (tests, legacy drivers, the
        submit_job thread runner). On-loop callers must use arun."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(inputs))
        # Called from a loop thread (legacy code inside an async context):
        # run on a private loop in a worker thread. copy_context() is
        # REQUIRED — ThreadPoolExecutor does not propagate contextvars, and
        # without it _emit_working's stream writer silently dies.
        import concurrent.futures
        import contextvars

        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(ctx.run, asyncio.run, self.arun(inputs)).result()

    def _save_run_messages(self, messages: List[BaseMessage]) -> None:
        """Persist this run's messages: the triggering human input plus this
        agent's own outputs. Injected context (README/RAG/status blobs) and
        other agents' messages are excluded — historically everything was
        re-saved under this agent's name, polluting retrieval, and the save
        filter had been commented out entirely.
        """
        seen_ids: set = set(self._persisted_ids)
        to_save: List[BaseMessage] = []
        for msg in messages:
            if id(msg) in seen_ids:
                continue
            mid = getattr(msg, "id", None)
            if mid is not None and mid in seen_ids:
                continue
            # Run-scoped internal notices never reach the DB: cross-agent RAG
            # retrieval (exclude_agent=<caller>) deliberately serves OTHER
            # agents' rows, which is exactly the A5 poisoning path.
            if self._is_internal(msg):
                continue
            if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == self.__name__:
                to_save.append(msg)
                seen_ids.add(id(msg))
                if mid is not None:
                    seen_ids.add(mid)
        if to_save:
            self.__update_memory(outputs=to_save)
            self.logger.info(
                "Agent %s persisted %d/%d run messages",
                self.__name__,
                len(to_save),
                len(messages),
            )

    def persist_message(self, msg: BaseMessage) -> None:
        """Persist a single message immediately (crash-safe incremental save).

        Node methods call this as soon as they produce a message, so a crashed
        run no longer loses the whole session tail. The end-of-run save skips
        anything recorded here.
        """
        if self.memory is None or msg is None:
            return
        try:
            self.__update_memory(outputs=[msg])
            self._persisted_ids.add(id(msg))
            mid = getattr(msg, "id", None)
            if mid is not None:
                self._persisted_ids.add(mid)
        except Exception as e:
            self.logger.warning("Incremental persist failed: %s", e)

    def get_okr_context(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves OKR context from memory: current task, finished tasks, subtasks, key results.
        """
        if not self.memory:
            return {}
        # ------------------------------------------------------------------
        # We provide a comprehensive overview of the OKR progress so that the
        # reasoning components of the agent have the full picture available.
        # ------------------------------------------------------------------

        current_task = self.memory.get_current_task(job_id=job_id)

        # Historic / finished items kept for backward-compatibility – callers
        # might rely on those names.  They are now superseded by the *all_* keys
        # that contain the complete list regardless of status.
        finished_tasks = self.memory.get_finished_tasks(job_id=job_id)
        finished_subtasks = self.memory.get_finished_subtasks()
        finished_key_results = self.memory.get_finished_key_results()

        # Newly added full listings.
        all_tasks = self.memory.get_all_tasks(job_id=job_id)
        all_subtasks = self.memory.get_all_subtasks()
        all_key_results = self.memory.get_all_key_results()
        memory_schema = self.memory.get_schema()

        # Failure ledger + artefacts: retries must see what was already tried.
        recent_attempts: List[Dict[str, Any]] = []
        artefacts: List[Dict[str, Any]] = []
        try:
            if hasattr(self.memory, "get_recent_attempts"):
                recent_attempts = self.memory.get_recent_attempts(limit=15)
            if hasattr(self.memory, "get_all_artefacts"):
                artefacts = self.memory.get_all_artefacts()
        except Exception as e:
            self.logger.debug("Ledger/artefact fetch failed: %s", e)

        return {
            "memory_schema": memory_schema,
            "all_tasks": all_tasks,
            "all_subtasks": all_subtasks,
            "all_key_results": all_key_results,
            "current_task": current_task,
            "recent_attempts": recent_attempts,
            "artefacts": artefacts,
            # "finished_tasks": finished_tasks,
            # "finished_subtasks": finished_subtasks,
            # "finished_key_results": finished_key_results,
        }

    def get_relative_information(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves relative information from memory based on the input query.
        """
        if not self.memory:
            return {}
        relative_info = self.memory.load_memory_variables(query)
        return relative_info

    def get_workflow_steps(self, limit: int = 20) -> List[BaseMessage]:
        """Own history = rendered project board (README) + this agent's recent
        prior reasoning from the session DB.

        The DB part was commented out for months, which made every agent
        amnesiac: "OwnHistory" was just the README, whose Tasks section was
        empty — so the Supervisor re-bootstrapped and replayed identical turns.
        """
        self_his: List[BaseMessage] = []
        # Rendered project board for orientation
        try:
            path = Path(self.project_root)
            for readme in sorted(path.glob("*.md")):
                content = readme.read_text(encoding="utf-8", errors="replace")
                if len(content) > 12000:
                    content = content[:12000] + "\n... [truncated]"
                self_his.append(AIMessage(content=content, name=readme.stem))
        except OSError as e:
            self.logger.warning("Could not read project README(s): %s", e)

        # This agent's own recent messages from the session memory
        if self.memory is not None:
            try:
                workflow = self.memory.get_raw_messages_by_agent(
                    agent_name=self.__name__
                )
                msgs: List[BaseMessage] = []
                for raw in workflow:
                    try:
                        msgs.append(_message_from_dict(raw))
                    except Exception:
                        continue
                msgs = self.filter_tool_messages(msgs)
                # Rebuild text-only: Responses-API messages persist with
                # block-list content (incl. provider-specific reasoning
                # items); replaying those verbatim into another provider's
                # prompt (openai reasoning → anthropic) is a 400 waiting to
                # happen. Same normalization _recent_own_messages applies.
                for m in msgs[-limit:]:
                    cls = HumanMessage if isinstance(m, HumanMessage) else AIMessage
                    self_his.append(
                        cls(
                            content=self._content_text(m.content),
                            name=getattr(m, "name", None) or self.__name__,
                        )
                    )
            except Exception as e:
                self.logger.warning(
                    "Could not load own history for %s: %s", self.__name__, e
                )
        return self_his

    # ------------------------------------------------------------------
    # Loop guards
    # ------------------------------------------------------------------

    def _emit_working(self) -> None:
        """Publish a typed 'working' progress event via langgraph's custom
        stream (stream_mode="custom"). Replaces the old hack of scraping
        debug-mode task events for the UI typing indicator. No-ops when the
        graph isn't being streamed with custom mode; must never break a run.

        NOTE: must be called from the OUTER-graph node body (run/stream) —
        custom events emitted inside SubGraph.invoke do not propagate
        (verified against langgraph 1.2.10).
        """
        try:
            from langgraph.config import get_stream_writer

            get_stream_writer()({"event": "working", "agent": self.__name__})
        except Exception:
            pass

    def _loop_guard_message(self, text: str) -> AIMessage:
        """Run-scoped guidance for THIS agent's remaining stages (A5).

        Marked internal (`atom_message_type: loop_guard` — the same in-band
        marker convention ReportMessage uses) so it never enters the node's
        returned delta (shared outer history) nor the memory DB: a downstream
        agent with a fresh budget and tools bound read a persisted "Tools are
        no longer available" note as a runtime property and stopped calling
        tools entirely (28-handoff ping-pong, session-nb-md-20260806-131234).
        The owning agent still sees it for the remainder of its own run.
        """
        return AIMessage(
            content=text,
            name=self.__name__,
            additional_kwargs={"atom_message_type": "loop_guard"},
        )

    @staticmethod
    def _is_internal(msg: Any) -> bool:
        """True for run-scoped internal notices that must not leave the run."""
        marker = (getattr(msg, "additional_kwargs", None) or {}).get(
            "atom_message_type"
        )
        return marker == "loop_guard"

    def _mark_stage(self, msg: BaseMessage, stage: str) -> None:
        """Stamp the node's stage into additional_kwargs (A2).

        Tool-call act messages keep their raw content (the block list carries
        the provider's tool_use blocks — rewriting it would change what is
        re-sent and persisted), so they never get a `# Agent - stage` header;
        the backend's emitter falls back to this metadata to adopt the open
        token segment instead of shipping the same text twice.
        """
        try:
            msg.additional_kwargs["atom_stage"] = stage
        except Exception:
            pass

    def _reset_loop_guards(self) -> None:
        """Call at the start of each run (think node)."""
        self._no_progress_cycles = 0
        self._last_tool_sig = None
        self._same_tool_repeats = 0
        self._force_report = False
        self._act_cycles = 0
        self._short_circuit = None

    def _act_budget_exceeded(self) -> bool:
        """Count an act invocation; True once the per-run budget is spent.

        The no-progress and identical-retry guards don't bound a loop whose
        tool calls keep *varying* — observed live: one Supervisor run issued
        44 okr_update calls until the graph recursion limit killed it. Past
        the budget the caller must withhold tools and wrap up.
        """
        cap = _cfg.MAX_ACT_CYCLES
        self._act_cycles += 1
        if self._act_cycles > cap:
            if not self._force_report:
                self.logger.error(
                    "Agent %s: act-cycle budget exhausted (%d > %d) — forcing final answer",
                    self.__name__,
                    self._act_cycles,
                    cap,
                )
            self._force_report = True
            return True
        return self._force_report

    def _note_progress(self) -> None:
        self._no_progress_cycles = 0

    def _note_no_progress(self, where: str, cap: int = 3) -> bool:
        """Count a plan/act cycle that produced no tool call. Returns True when
        the cap is reached — the caller should route to report instead of
        looping (observed livelocks: 16-86 plan/act cycles with zero tool
        calls)."""
        self._no_progress_cycles += 1
        if self._no_progress_cycles >= cap:
            self.logger.error(
                "Agent %s: %d consecutive %s cycles without progress — forcing report",
                self.__name__,
                self._no_progress_cycles,
                where,
            )
            self._force_report = True
            return True
        return False

    def _tool_call_repeated(self, action: Any) -> bool:
        """True when the agent is about to re-issue the exact same tool call it
        already issued twice — the 'blind identical retry' pattern (same failing
        command re-run 3x with no diagnosis in real sessions)."""
        try:
            sig = json.dumps(
                [
                    (tc.get("name"), tc.get("args"))
                    for tc in (getattr(action, "tool_calls", None) or [])
                ],
                sort_keys=True,
                default=str,
            )
        except Exception:
            return False
        if sig == self._last_tool_sig:
            self._same_tool_repeats += 1
        else:
            self._same_tool_repeats = 0
            self._last_tool_sig = sig
        return self._same_tool_repeats >= 2

    @staticmethod
    def _content_text(content: Any) -> str:
        """Flatten a message or raw content to plain text. Delegates to the
        framework-wide canonical flattener (utils.text.message_text), which
        uses langchain's content_blocks normalization for messages."""
        from ..utils.text import message_text

        return message_text(content)

    def _run_tool_result_count(self, messages: List[BaseMessage]) -> int:
        """ToolMessages produced by THIS run — inherited ones (their ids were
        in outer state when the run started) don't count. Whole-conversation
        counting let one early tool run suppress the zero-evidence gate for
        every later fabricated report in the session."""
        inbound = self._run_inbound_ids or set()
        return sum(
            1
            for m in messages or []
            if isinstance(m, ToolMessage)
            and (getattr(m, "id", None) is None or m.id not in inbound)
        )

    def _strip_fabricated_tool_calls(
        self, text: str, stage: str, has_tool_evidence: bool = False
    ) -> str:
        """Truncate prose-stage output at the first simulated tool call (A6).

        Truncating at the marker — not merely deleting the markup — is the
        point: the invented *observation* always follows the markup, and it
        is the invented observation that poisons downstream stages (a plan
        stage read a fabricated `ls` transcript, concluded "all tasks
        complete", and the act stage never ran). Gated by config.STAGE_GUARD.

        Scope: think/plan are ALWAYS scrubbed — no tool can run there, so
        tool markup is fabrication by definition. act/report/reply are
        scrubbed only when the run has NO real tool evidence: when tools
        genuinely ran, describing them (even in "**Tool call:**" formatting)
        is the message's job, and truncating a legitimate hand-off destroys
        real results.
        """
        if not _cfg.STAGE_GUARD or not text:
            return text
        if has_tool_evidence and (stage or "").lower() not in ("think", "plan"):
            return text
        m = _FABRICATED_TOOL_MARKUP.search(text)
        if m is None:
            return text
        kept = text[: m.start()].rstrip()
        self.logger.error(
            "[stage-guard] %s/%s: stripped %d chars of simulated tool-call markup",
            self.__name__,
            stage,
            len(text) - len(kept),
        )
        return kept + (
            "\n\n[stage-guard] Simulated tool-call markup was removed here: no "
            "tool ran in this step. Any text that followed it (including "
            "apparent results) was discarded as fabricated."
        )

    def _stage_content(
        self, stage: str, content: Any, has_tool_evidence: bool = False
    ) -> str:
        """Bind the emitted content to the node's actual stage. Strips a
        leading LLM-emitted stage header that disagrees with the node (in real
        sessions up to 100% of a session's labelled messages carried the WRONG
        stage body, so the router advanced while the model re-emitted the
        previous stage). ``has_tool_evidence`` gates the fabrication scrubber
        for stages that may legitimately describe real tool activity."""
        import re as _re

        text = self._content_text(content)
        m = _re.match(r"\s*#{1,4}\s*(THINK|PLAN|ACT|REPORT)\b\s*:?\s*\n?", text, _re.I)
        if m:
            emitted = m.group(1).upper()
            if emitted != stage.upper():
                self.logger.warning(
                    "Agent %s: stage mismatch (node=%s, model emitted %s) — normalizing",
                    self.__name__,
                    stage.upper(),
                    emitted,
                )
            text = text[m.end():]
        text = self._strip_fabricated_tool_calls(text, stage, has_tool_evidence)
        return f"# {self.__name__} - {stage}\n{text}"

    def __update_memory(
        self,
        inputs: list = [],  # human message
        outputs: list = [],  # ai message
        job_id: Optional[str] = None,
        task_id: Optional[int] = None,
        key_result_id: Optional[int] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        """
        Save inputs and outputs to memory, tagging by job/task/key_result and agent_name.
        """
        if not self.memory:
            return
        # Use agent_type if no explicit name provided
        name = agent_name if agent_name else self.__name__
        # Delegate to memory
        all_types = [msg.__class__.__name__ for msg in inputs + outputs]
        self.logger.info(
            f"Updating memory with {len(inputs)} inputs and {len(outputs)} outputs of types: {all_types}"
        )

        self.memory.save_context(
            inputs=inputs,
            outputs=outputs,
            job_id=job_id,
            task_id=task_id,
            key_result_id=key_result_id,
            agent_name=name,
        )

    async def __acall__(self, inputs: Dict[str, Any]) -> Any:
        """Alias for arun: make the agent callable asynchronously."""
        return await self.arun(inputs)

    def __call__(self, inputs: Dict[str, Any]) -> Any:
        """Alias for run: make the agent callable synchronously."""
        return self.run(inputs)
