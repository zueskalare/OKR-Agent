"""
AtomAgent Manager implementation.

This agent is responsible for processing high-level descriptions and
managing the coordination between other specialized agents.
"""

import json
from typing import Dict, List, Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.graph import StateGraph, END

from .base_agent import BaseAgent, TOKEN_TAG
from .. import config as _cfg

from pydantic import ConfigDict, BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from ..utils.readme_sync import ensure_readme, update_section
from ..utils.okr_tools import sync_board, apply_task_operation
from ..utils.tool_policy import split_tools
from ..models.okr_models import TaskOperation, TaskOperations


# -----------------------------------------------------------------------------
# Shared pydantic models
# -----------------------------------------------------------------------------

# Whether an additional memory update cycle is required.  This is a *very* small
# helper model so we expose it at module level such that both the `plan` and the
# new `update_memory` graph nodes can re-use it.


class UpdateTask(BaseModel):
    """Flag returned by the LLM indicating a follow-up memory update is needed."""

    update_task: bool = False


class SupervisorPlanOut(BaseModel):
    """Structured PLAN output: the plan text plus where to go next.

    ``send_to="report"`` lets the Supervisor wrap up a run from PLAN — the
    escape Modeler/Simulator always had (PlanOut) and the Supervisor lacked,
    which is why its only exit was act's "end"."""

    plan: str = Field(description="The PLAN-stage content in markdown.")
    send_to: Literal["act", "report"] = "act"


class TriageOut(BaseModel):
    """First-hop classification of a user turn."""

    verdict: Literal["converse", "clarify", "task"] = "task"
    reply: str = Field(
        default="",
        description="Complete user-facing answer for converse/clarify; empty for task.",
    )
    rationale: str = Field(
        default="", description="One short line explaining the verdict, for logs only."
    )


# Used when the yaml template has no `triage` key — the gate must keep
# working even against a stale prompt file.
_TRIAGE_FALLBACK = """\
You are the Supervisor's intake gate for a multi-agent simulation lab.
Classify the LATEST user message into exactly one verdict:
- converse: greeting, thanks, small talk, meta questions about the team's
  capabilities, or a self-contained knowledge/explanation question that needs
  NO files, code, simulations, or agent work. Put the complete, helpful answer
  in `reply`. For greetings keep it to 1-3 sentences and briefly say what the
  team can do.
- clarify: a plausible simulation request that cannot be started because a
  decisive detail is missing (system/material, property of interest, or
  method). Put 1-3 targeted questions in `reply`. Do NOT propose a plan and do
  NOT create objectives.
- task: anything actionable — new work, a follow-up continuing work already on
  the board, or an approval/continuation of a question you asked earlier
  ("yes", "go ahead", "option B"). Leave `reply` empty.
When in doubt choose `task`. Never choose converse/clarify for a message that
names a concrete system to build, simulate, analyse, fix, or plot.
A message containing work instructions, dispatch orders, resume directives,
or an execution plan is ALWAYS `task` — never converse — even if earlier
turns reported the work blocked, impossible, or complete."""

_DEFAULT_GREETING = (
    "Hello! I'm the Supervisor of a simulation agent team — we build, run, and "
    "analyse molecular-dynamics (LAMMPS) and finite-element (FEniCSx) "
    "simulations from natural-language descriptions. Tell me what system you'd "
    "like to study and what you want to measure, and we'll take it from there."
)


class Supervisor(BaseAgent):
    """
    Supervisor agent handles high-level coordination and processes natural language
    descriptions into structured data for model building.
    """

    __name__: str = "Supervisor"
    enable_okr_table: bool = True  # Enable OKR table updates by default
    # Consecutive plan→update_memory rounds this run; plan forces "act" at the
    # cap so the plan↔update_memory ping-pong cannot recur.
    _okr_rounds: int = PrivateAttr(default=0)
    # Verdict of this run's triage gate (None when triage was skipped);
    # plan's OKR probe only fires for None/"task".
    _triage_verdict: Optional[str] = PrivateAttr(default=None)
    # True when the coordinator-only policy actually withheld tools at
    # construction — drives the Execution Boundary prompt injection so the
    # prompt can never contradict the real binding (legacy-flag mode included).
    _coordinator_only: bool = PrivateAttr(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _select_act_tools(self) -> List[BaseTool]:
        """Coordinator-only tool set: read/search/board tools, no exec/write.

        The Supervisor's charter already says it delegates domain work; on
        claude-opus-5 the prompt alone did not hold (5/5 runs single-agent,
        zero handoffs — finding A7), so the constraint is structural.
        Resolved ONCE at construction (see BaseAgent._select_act_tools);
        ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 at construction time restores the
        legacy all-tools shape.
        """
        if _cfg.SUPERVISOR_EXEC_TOOLS:
            self._coordinator_only = False
            return list(self.tools)
        coordination, withheld = split_tools(self.tools)
        self._coordinator_only = bool(withheld)
        if withheld:
            self.logger.info(
                "Supervisor act stage withholds execution tools: %s "
                "(ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 restores them)",
                sorted(getattr(t, "name", str(t)) for t in withheld),
            )
        return coordination

    def _policy_prompt(self, stage: str) -> str:
        """Execution Boundary statement — injected ONLY when the
        coordinator-only policy actually withheld tools, so the prompt and
        the real binding can never contradict each other (with
        ATOMAGENTS_SUPERVISOR_EXEC_TOOLS=1 the boundary text disappears
        along with the restriction)."""
        if not self._coordinator_only:
            return ""
        return (
            "\n## Execution Boundary (structural)\n"
            "You have NO execution or write tools in this runtime BY DESIGN — "
            "this is the coordinator role, not a malfunction. You cannot "
            "write files, run shells or Python, build models, mesh, or run "
            "simulations; your bound tools are read/search/board tools only. "
            "Your colleagues Modeler and Simulator hold the full execution "
            "toolset. Every step that creates or runs something MUST be "
            "delegated — emit a Decision/Assignment with acceptance criteria "
            "and target artifact paths and hand off (`**Next → Modeler**` for "
            "authoring/building, `**Next → Simulator**` for running/analysis). "
            "If your PLAN says work must be done and you have not delegated it "
            "this turn, delegating IS the action.\n"
            "\n## Dispatch mechanism\n"
            "The `send_to` field of your hand-off report IS the dispatch "
            "channel: naming a colleague there transfers control to them — no "
            "'handoff tool' exists or is needed, so never conclude that "
            "delegation is impossible because you see no dispatch tool in your "
            "tool list. Declaring a task blocked or terminal because YOU lack "
            "execution tools is a protocol violation: while Modeler or "
            "Simulator are listed as colleagues, handing off to them is always "
            "available and is the required action for pending build/run work. "
            "Reporting work as complete that you did not delegate and that no "
            "tool result shows is a failure.\n"
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        pipeline = StateGraph(self.SubState)

        pipeline.add_node("triage", self.triage)
        pipeline.add_node("think", self.think)
        pipeline.add_node("plan", self.plan)
        pipeline.add_node("act", self.act)
        # New node responsible for updating the OKR/task tables in memory
        if self.enable_okr_table:
            pipeline.add_node("update_memory", self.update_memory)
        pipeline.add_node("tools", self._ToolNode)

        # Triage is the first hop of every user turn: trivial/under-specified
        # messages short-circuit to END with a direct reply; real tasks fall
        # through to the think→plan→act pipeline.
        pipeline.set_entry_point("triage")
        pipeline.add_conditional_edges(
            "triage",
            lambda state: state.get("send_to"),
            {"think": "think", "end": END},
        )
        pipeline.add_edge("think", "plan")
        # The control flow now branches after *plan*: if a task update is
        # required we jump to the dedicated *update_memory* node, otherwise we
        # proceed directly to *act*.
        # "report" maps to END of the subgraph: run()/stream() then produce
        # the hand-off report. This escape existed for Modeler/Simulator but
        # not the Supervisor, whose only exit used to be act's "end".
        if self.enable_okr_table:
            pipeline.add_conditional_edges(
                "plan",
                lambda state: state.get("send_to"),
                {"update_memory": "update_memory", "act": "act", "report": END},
            )

            # After the memory has been synchronised we return to *plan* so that a
            # fresh plan can be produced using the latest OKR state.
            pipeline.add_edge("update_memory", "plan")
        else:
            pipeline.add_conditional_edges(
                "plan",
                lambda state: state.get("send_to"),
                {"act": "act", "report": END},
            )

        # Tools node remains unchanged
        pipeline.add_edge("tools", "act")

        # Decide next hop after *act*
        pipeline.add_conditional_edges(
            "act",
            lambda state: state.get("send_to"),
            {"plan": "plan", "end": END, "tools": "tools"},
        )

        self._SubGraph = pipeline.compile()
        # per-agent logger (inherited from BaseAgent); no module-level logger used here
        # BaseAgent.__init__ has set self.logger
        # Ensure shared README exists for this job workspace
        try:
            if getattr(self, "project_root", None):
                ensure_readme(self.project_root)
        except Exception as e:
            self.logger.error("Failed to ensure README in project_root: %s", e)

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the Supervisor agent."""
        # ``BaseAgent.system_prompt`` already appends the project-directory
        # information (if ``self.project_root`` is set), so we only need to
        # return the inherited prompt here.
        return super().system_prompt

    @property
    def triage_prompt(self) -> str:
        """System prompt for the triage gate (yaml key `triage`, with a
        hard fallback so a stale template cannot disable the gate)."""
        try:
            tpl = self.prompt_loader.get_template(self.__name__)
            text = self._fill_project_root(tpl.get("triage", ""))
        except Exception:
            text = ""
        return text or _TRIAGE_FALLBACK

    def triage(self, state: Dict) -> Dict[str, Any]:
        """First hop of a user turn: answer trivial or under-specified
        messages directly (one LLM call, then END) instead of engaging the
        full pipeline. A bare "hi" used to bootstrap OKRs and draft the
        Modeler into 26+ messages of invented intake work."""
        self._reset_loop_guards()
        self._triage_verdict = None
        messages = state.get("messages", [])
        last = messages[-1] if messages else None

        # Deterministic skips — no LLM call: gate disabled, or re-entry from
        # a colleague handoff (triage must never fire mid-pipeline).
        if not _cfg.ENABLE_TRIAGE or not isinstance(last, HumanMessage):
            return {"messages": messages, "send_to": "think"}

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=self.triage_prompt),
                    MessagesPlaceholder(variable_name="board"),
                    MessagesPlaceholder(variable_name="recent"),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            ).format_messages(
                messages=[last],
                board=[AIMessage(content=self._board_summary_line())],
                recent=self._recent_own_messages(limit=4, max_chars=400),
            )
            out = self._structured_invoke(TriageOut, prompt)
        except Exception as e:  # the gate must never break a run
            self.logger.warning(
                "[triage] failed (%s) — proceeding with full pipeline", e
            )
            return {"messages": messages, "send_to": "think"}

        verdict, reply = out.verdict, (out.reply or "").strip()
        self.logger.warning(
            "[triage] verdict=%s rationale=%s", verdict, (out.rationale or "")[:200]
        )

        if verdict == "converse" and not reply:
            reply = _DEFAULT_GREETING
        if verdict == "clarify" and not reply:
            # A clarify with no question is useless — bias to task.
            verdict = "task"

        self._triage_verdict = verdict
        if verdict in ("converse", "clarify"):
            self._short_circuit = {
                "reply": reply,
                "send_to": "END",
                "stage": "reply",
                "reason": f"triage:{verdict}",
            }
            return {"messages": messages, "send_to": "end"}
        return {"messages": messages, "send_to": "think"}

    def think(self, state: Dict) -> Dict[str, Any]:
        """
        Process the state and return an action or finish message.
        """
        # get all current task and subtask

        self._reset_loop_guards()
        self._okr_rounds = 0
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        self_his = self.get_workflow_steps()

        # Anti-re-bootstrap: if this session already has tasks or own history,
        # say so explicitly. Cold-booting Supervisors used to declare a NEW
        # project right after receiving a success handoff, regressing finished
        # work to FAIL.
        has_prior_work = False
        try:
            if self.memory is not None and self.memory.get_all_tasks():
                has_prior_work = True
        except Exception as e:
            self.logger.debug("Prior-work probe failed: %s", e)
        if not has_prior_work:
            has_prior_work = any(
                getattr(m, "name", None) == self.__name__ for m in self_his
            )
        if has_prior_work:
            self_his = self_his + [
                AIMessage(
                    content=(
                        "[RESUMING EXISTING SESSION] The history and task board "
                        "above belong to an ongoing project. CONTINUE that work; "
                        "do NOT re-bootstrap, re-create objectives, or restart "
                        "completed steps unless the user explicitly asks for a "
                        "new project."
                    ),
                    name=self.__name__,
                )
            ]

        query = {
            "input": self._content_text(last_message.content),
            "agent_name": self.__name__,
        }
        if not self.memory:
            relative_info = {}
        else:
            relative_info = self.memory.load_memory_variables(query)
        relative_messages = [AIMessage(content=json.dumps(relative_info))]

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                # Static prefix alone in the FIRST system block (prompt-cache
                # stable); the stage-varying contract rides the second block.
                SystemMessage(content=self.system_prompt_static),
                SystemMessage(
                    content=self.stage_system_prompt("think") + self.think_prompt
                ),
                MessagesPlaceholder(variable_name="OwnHistory"),
                MessagesPlaceholder(variable_name="TaskStatus"),
                MessagesPlaceholder(variable_name="RelativeInformation"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        message = chat_prompt.format_messages(
            messages=[last_message],
            TaskStatus=self.get_current_status(),
            RelativeInformation=relative_messages,
            OwnHistory=self_his,
        )

        response = self._invoke(message)
        response.content = self._stage_content("think", response.content)
        response.name = self.__name__
        self.persist_message(response)
        # Injected context (status/RAG/history) stays in the prompt only.
        messages = [last_message, response]
        return {
            "messages": messages,
            "send_to": "plan",
        }

    def _plan_message(self, chat_prompt, messages) -> str:
        """One structured PLAN call: appends the stage-bound plan message and
        returns where to go next ("act" or "report")."""
        prompt = chat_prompt.format_messages(
            messages=messages,
            saved=self.msg_storage,
            TaskStatus=self.get_current_status(),
        )
        plan_out = self._structured_invoke(SupervisorPlanOut, prompt)
        action_msg = AIMessage(
            content=self._stage_content("plan", plan_out.plan), name=self.__name__
        )
        self.persist_message(action_msg)
        messages.append(action_msg)
        send_to = plan_out.send_to
        if self._force_report and send_to != "report":
            self.logger.error("[loop-guard] %s: forcing plan→report", self.__name__)
            send_to = "report"
        return send_to

    def plan(self, state: Dict):
        """
        Generate a plan based on the current state.
        """

        messages = state.get("messages", [])

        # Loop guards already fired (act budget / no-progress): skip the OKR
        # probe and a fresh plan entirely — go straight to the hand-off
        # report. Drops a forced wrap-up from plan(2)+act(1)+report(1) LLM
        # calls to report(1).
        if self._force_report:
            self.logger.warning(
                "[loop-guard] %s: _force_report set — plan routes to report",
                self.__name__,
            )
            return {"messages": messages, "send_to": "report"}

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt_static),
                # was act_prompt — plan messages literally started "### ACT"
                SystemMessage(
                    content=self.stage_system_prompt("plan")
                    + (self.plan_prompt or self.act_prompt)
                ),
                MessagesPlaceholder(variable_name="TaskStatus"),
                MessagesPlaceholder(variable_name="saved"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # The OKR probe only fires on real task turns (triage verdict None =
        # gate skipped/disabled, "task" = confirmed actionable).
        if (
            self.enable_okr_table
            and self._okr_rounds < 2
            and self._triage_verdict in (None, "task")
        ):
            # whether we need to update the task
            updatetask_need_hint = (
                "Compare the TaskStatus table above with the conversation. "
                "Return update_task=true ONLY if the table is out of date "
                "(a task/subtask finished, changed status, or a new one is "
                "needed). If the table already reflects reality, return false."
            )
            message = chat_prompt.format_messages(
                messages=messages + [HumanMessage(content=updatetask_need_hint)],
                TaskStatus=self.get_current_status(),
                saved=self.msg_storage,
            )
            self.logger.debug(
                "Calling llm with message to decide whether task update is needed"
            )
            action = self._structured_invoke(UpdateTask, message)
            self.logger.info("Plan-stage UpdateTask flag from LLM: %s", action)

            update_task = action.update_task

            if update_task:
                next_step = "update_memory"
            else:
                # Only create a detailed plan once the OKR/task view is up-to-date.
                next_step = self._plan_message(chat_prompt, messages)

            out = {
                "messages": messages,
                "send_to": next_step,
            }
            self.logger.info("Supervisor.plan returning send_to=%s", out["send_to"])
            return out
        else:
            if self.enable_okr_table and self._okr_rounds >= 2:
                self.logger.warning(
                    "Supervisor: OKR round cap reached (%d) — proceeding "
                    "without another update_memory pass",
                    self._okr_rounds,
                )
            next_step = self._plan_message(chat_prompt, messages)
            out = {
                "messages": messages,
                "send_to": next_step,
            }
            self.logger.info("Supervisor.plan returning send_to=%s", out["send_to"])
            return out

    # ------------------------------------------------------------------
    # Memory update node (new)
    # ------------------------------------------------------------------

    def update_memory(self, state: Dict) -> Dict[str, Any]:
        """Synchronise OKR/task tables in memory.
        This is different from the __update_memory method in BaseAgent, which is designed to save messages
        to the memory.  This graph node runs only when the preceding *plan* step has detected and change the database OKR/task tables.

        This graph node runs only when the preceding *plan* step has detected
        that the conversation requires an update of the task/sub-task
        structure (completion status changes, new tasks etc.).  The heavy
        lifting is delegated to :meth:`update_task`, but the required
        operations are first generated via the LLM so that the human-readable
        conversation remains the single source of truth.
        """

        messages = state.get("messages", [])
        self._okr_rounds += 1

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt_static),
                SystemMessage(
                    content=self.stage_system_prompt("plan")
                    + (self.plan_prompt or self.act_prompt)
                ),
                MessagesPlaceholder(variable_name="TaskStatus"),
                MessagesPlaceholder(variable_name="saved"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Bounded retry: the old `while update_task:` retried forever on any
        # update_tasks error (a schema mismatch once re-issued the identical
        # command 120x).
        MAX_ATTEMPTS = 3
        for attempt in range(1, MAX_ATTEMPTS + 1):
            updatetask_hint = (
                "Compare the TaskStatus table with the conversation and return "
                "the TaskOperations needed to bring the table up to date "
                "(finished tasks/subtasks, status changes, new tasks). Return "
                "an EMPTY operations list if no change is needed."
            )

            self.logger.debug("Requesting TaskOperations from LLM (attempt %d)", attempt)

            message = chat_prompt.format_messages(
                messages=messages + [HumanMessage(content=updatetask_hint)],
                TaskStatus=self.get_current_status(),
                saved=self.msg_storage,
            )

            update_commands = self._structured_invoke(TaskOperations, message)

            self.logger.debug("TaskOperations returned: %s", update_commands)

            if not update_commands.operations:
                # A representable no-op: nothing to change, we're done.
                update_res = {"command": "[]", "status": "no_change"}
                messages.append(
                    AIMessage(content=json.dumps(update_res), name=self.__name__)
                )
                break

            # Apply the operations -------------------------------------------------
            try:
                self.update_tasks(update_commands)
                # Reflect the latest board (tasks + artefacts) into README
                try:
                    if getattr(self, "memory", None) and getattr(self, "project_root", None):
                        sync_board(self.memory, self.project_root)
                except Exception as e:
                    self.logger.error("Failed to write Tasks section to README: %s", e)

                update_res = {
                    "command": json.dumps(update_commands.model_dump()),
                    "status": "success",
                }
                messages.append(
                    AIMessage(content=json.dumps(update_res), name=self.__name__)
                )
                break
            except Exception as e:
                self.logger.error(
                    "Error updating tasks (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, e
                )
                update_res = {
                    "command": json.dumps(update_commands.model_dump()),
                    "status": "error",
                    "error": str(e),
                }
                messages.append(
                    AIMessage(content=json.dumps(update_res), name=self.__name__)
                )
                if attempt == MAX_ATTEMPTS:
                    self.logger.error(
                        "Giving up on task update after %d attempts", MAX_ATTEMPTS
                    )

        # After synchronisation, jump back to *plan* in order to generate the
        # next high-level plan with the freshly updated OKR state. (_okr_rounds
        # caps how often plan may route back here.)
        return {
            "messages": messages,
            "send_to": "plan",
        }

    def act(self, state: Dict) -> Dict[str, Any]:
        """
        Execute the plan based on the current state.
        """
        self.logger.debug("Executing act step in Supervisor agent.")
        messages = state.get("messages", [])
        last_message = messages[-1]
        force_final = self._act_budget_exceeded()
        if force_final:
            messages = messages + [
                self._loop_guard_message(
                    f"[loop-guard] {self.__name__}'s own act budget for THIS "
                    "run is exhausted — tools are withheld for the remainder "
                    "of this run only. Other agents and later runs are "
                    "unaffected. Produce your final decision/assignment "
                    "summary now."
                )
            ]
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt_static),
                # act_final: the wrap-up call binds NO tools — the plain act
                # contract ("bound to THIS call") would be a lie that invites
                # exactly the simulation the budget guard exists to stop.
                SystemMessage(
                    content=self.stage_system_prompt(
                        "act_final" if force_final else "act"
                    )
                    + self.act_prompt
                ),
                # Board view: act edits the task board, so it must SEE it —
                # without this the model re-added the same objective 44x.
                MessagesPlaceholder(variable_name="TaskStatus"),
                MessagesPlaceholder(variable_name="saved"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        act_tools = self.act_tools
        llm_with_tools = (
            self.llm
            if (force_final or not act_tools)
            else self.llm.bind_tools(act_tools)
        )
        message = chat_prompt.format_messages(
            messages=messages,
            saved=self.msg_storage,
            TaskStatus=self.get_current_status(),
        )
        action = self._invoke(message, llm=llm_with_tools)
        action.name = self.__name__  # Set the name of the action to the agent's name
        if action.tool_calls:
            if self._tool_call_repeated(action):
                # Count this as a no-progress cycle too: three blind-retry
                # bounces set _force_report so the next plan() wraps up
                # (Modeler/Simulator already did this; the Supervisor's guard
                # was dead code).
                self._note_no_progress("act")
                note = self._loop_guard_message(
                    f"# {self.__name__} - act\n[loop-guard] The identical "
                    "tool call was already attempted twice. Do not retry it "
                    "verbatim; analyse the previous error output and change "
                    "the approach."
                )
                messages.append(note)
                return {"messages": messages, "send_to": "plan"}
            self._note_progress()
            self._mark_stage(action, "act")
            messages.append(action)
            # NOTE: no msg_storage.extend(messages) — that grew O(N^2).
            self.logger.info("Tool calls is activated, sending to tools")
            out = {
                "messages": messages,
                "send_to": "tools",  # Tool calls will be handled by the tool node agent
            }
            self.logger.info("Supervisor.act returning send_to=%s", out["send_to"])
            return out
        elif isinstance(last_message, ToolMessage):
            # Tool result just came back: summarize and loop to plan. Real
            # tool evidence exists, so describing it (even in tool-call
            # formatting) is legitimate — the scrubber must not truncate.
            action.content = self._stage_content(
                "act",
                action.content,
                has_tool_evidence=self._run_tool_result_count(messages) > 0,
            )
            action.name = self.__name__
            messages[-1].name = self.__name__  # Update the last message's name
            self.persist_message(action)
            messages.append(action)
            out = {
                "messages": messages,
                "send_to": "plan",
            }
            self.logger.info("Supervisor.act returning send_to=%s", out["send_to"])
            return out
        else:
            # If no tool calls, treat as a regular action
            action.content = self._stage_content(
                "act",
                action.content,
                has_tool_evidence=self._run_tool_result_count(messages) > 0,
            )
            action.name = self.__name__
            self.persist_message(action)
            messages.append(action)
            # Mirror Supervisor decision to README under Supervisor section
            try:
                if getattr(self, "project_root", None):
                    update_section(self.project_root, self.__name__, action.content)
            except Exception as e:
                self.logger.error("Failed updating README section for %s: %s", self.__name__, e)
            out = {
                "messages": messages,
                "send_to": "end",
            }
            self.logger.info("Supervisor.act returning send_to=%s", out["send_to"])
            return out

    # ------------------------------------------------------------------
    # LLM-driven task update helper
    # ------------------------------------------------------------------

    def update_task_from_llm(self, instruction: str) -> List[TaskOperation]:
        """Generate and apply task operations based on an *instruction*.

        The underlying LLM is prompted with *instruction* and expected to return
        a JSON list adhering to :class:`TaskOperation` (thanks to
        ``with_structured_output``).  Parsed operations are then executed via
        :meth:`update_tasks` to synchronise the OKR tables in memory.
        """

        # Structured output with bounded error-feedback retries.
        cmd: TaskOperations = self._structured_invoke(TaskOperations, instruction)

        # Execute the operations.
        self.update_tasks(cmd)

        return cmd.operations

    # ------------------------------------------------------------------
    # Task CRUD helpers (add / update status / remove)
    # ------------------------------------------------------------------

    def update_tasks(self, operations: TaskOperations) -> None:
        """Execute a batch of task/subtask operations against ``self.memory``.

        The method is a convenience wrapper so the Supervisor can directly
        manipulate OKR tables stored in :pyclass:`SQLiteRAGMemory` (or any
        compatible memory implementation exposing the same CRUD API).
        """
        self.logger.info("updating the tasks with operations:\n%s", operations)
        if not self.memory:
            raise RuntimeError("Supervisor has no memory configured to update.")

        ops = (
            operations.operations
            if isinstance(operations, TaskOperations)
            else list(operations)
        )
        for op in ops:
            self.update_task(op)

    # Convenience wrapper for single operation ---------------------------------
    def update_task(self, op: TaskOperation) -> None:
        """Simpler one-shot variant accepting the same fields as TaskOperation.

        Example
        -------
        supervisor.update_task(op="update_task_status", job_id="JOB42", status="done")
        """

        self.logger.info("updating the tasks with operations:\n%s", op)
        if not self.memory:
            raise RuntimeError("Supervisor has no memory configured to update.")
        apply_task_operation(self.memory, op)
        self.logger.info("Task updated successfully.")

    def run(self, state: Dict) -> Dict[str, Any]:
        """
        Run the agent's main loop.
        """
        response = super().run(state)

        return response


# Alias for compatibility with SimulationAgentSystem
AtomAgentManager = Supervisor
