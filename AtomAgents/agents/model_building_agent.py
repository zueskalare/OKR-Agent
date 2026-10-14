"""
Model Building Agent implementation.

This agent is responsible for constructing molecular models for MD simulations,
including system setup, solvent addition, and use MCP tools for validation.
"""

import json
from typing import Dict, Any, Literal
from pydantic import ConfigDict, Field, BaseModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from .base_agent import BaseAgent, TOKEN_TAG
from ..utils.readme_sync import update_section


class Modeler(BaseAgent):
    """
    ModelBuildingAgent specializes in constructing molecular models for MD simulations.
    The task of this agent:
        - Understand the simulation model
        - Build the model structure and save the data file
        - The need the potential between the atoms and molecules, if need save to the file
    """

    __name__: str = "Modeler"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def think(self, state: Dict) -> Dict[str, Any]:
        """
        Process the state and return an action or finish message.
        """
        # get all current task and subtask

        self._reset_loop_guards()
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        query = {
            "input": self._content_text(last_message.content),
            "agent_name": self.__name__,
        }
        if not self.memory:
            self.logger.warning("No memory available for relative information.")
            relative_info = {}
        else:
            relative_info = self.memory.load_memory_variables(query)
        relative_messages = [AIMessage(content=json.dumps(relative_info))]
        current_status_message = self.get_current_status()

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                # Static prefix alone in the FIRST system block (prompt-cache
                # stable); the stage-varying contract rides the second block.
                SystemMessage(content=self.system_prompt_static),
                SystemMessage(
                    content=self.stage_system_prompt("think") + self.think_prompt
                ),
                MessagesPlaceholder(variable_name="TaskStatus"),
                MessagesPlaceholder(variable_name="RelativeInformation"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        message = chat_prompt.format_messages(
            messages=[last_message],
            TaskStatus=current_status_message,
            RelativeInformation=relative_messages,
        )
        response = self._invoke(message)
        response.content = self._stage_content("think", response.content)
        response.name = self.__name__
        self.persist_message(response)
        # Injected context (status/RAG) stays in the prompt only — re-adding it
        # to state bloated every later turn and was re-persisted to memory.
        messages = [last_message, response]
        self.logger.info("Agent %s think stage: \n%s", self.__name__, response.content)
        return {
            "messages": messages,
            "send_to": "plan",
        }

    def plan(self, state: Dict):
        """
        Generate a plan based on the current state.
        """
        self.logger.debug("Modeler plan stage")
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        query = {
            "input": self._content_text(last_message.content),
            "agent_name": self.__name__,
        }
        if not self.memory:
            relative_info = {}
        else:
            relative_info = self.memory.load_memory_variables(query)
        relative_messages = [AIMessage(content=json.dumps(relative_info))]
        chat_prompt = ChatPromptTemplate.from_messages(  # format the chat prompt
            [
                SystemMessage(content=self.system_prompt_static),
                SystemMessage(
                    content=self.stage_system_prompt("plan") + self.plan_prompt
                ),
                MessagesPlaceholder(variable_name="RelativeInformation"),
                MessagesPlaceholder(variable_name="msg_storage"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # One structured call produces both the plan text and the routing
        # decision (previously two full-context LLM calls per plan step).
        class PlanOut(BaseModel):
            plan: str = Field(description="The plan for the next step(s), in markdown.")
            send_to: Literal["act", "report"] = Field(
                default="act",
                description=(
                    "'report' if all planned work is finished, otherwise 'act'."
                ),
            )

        try:
            message = chat_prompt.format_messages(  # format the message
                messages=messages,
                msg_storage=self.msg_storage if self.msg_storage else [],
                RelativeInformation=relative_messages,
            )
        except KeyError as e:
            self.logger.error("Error formatting messages: %s", e)
            raise KeyError(f"Failed to format messages: {e}") from e
        plan_out = self._structured_invoke(PlanOut, message)
        action = AIMessage(
            content=self._stage_content("plan", plan_out.plan), name=self.__name__
        )
        self.persist_message(action)
        messages.append(action)
        send_to = plan_out.send_to
        if self._force_report and send_to != "report":
            self.logger.error(
                "Agent %s: no-progress cap reached — overriding plan routing to report",
                self.__name__,
            )
            send_to = "report"
        self.logger.info("Agent %s plan stage: next_step=%s", self.__name__, send_to)
        self.logger.info("Agent %s plan stage: \n%s", self.__name__, action.content)
        return {
            "messages": messages,
            "send_to": send_to,
        }

    def act(self, state: Dict) -> Dict[str, Any]:
        """
        Execute the plan based on the current state.
        """
        self.logger.debug("Modeler act stage")
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if isinstance(last_message, ToolMessage):
            messages[-1].name = self.__name__  # Update the last message's name
            self.logger.debug("Tool message returned from tool node")
        elif isinstance(last_message, AIMessage) and getattr(
            last_message, "tool_calls", None
        ):
            # Orphaned tool_calls (no matching ToolMessages) make the next API
            # request invalid. Close them out so the run can continue. (The old
            # guard's `raise` was wrapped in `except: pass` and never fired.)
            self.logger.error(
                "Agent %s: orphaned tool_calls in history; closing with error ToolMessages",
                self.__name__,
            )
            for tc in last_message.tool_calls:
                messages.append(
                    ToolMessage(
                        content='{"error": "tool call was never executed"}',
                        name=tc.get("name", "tool"),
                        tool_call_id=tc.get("id", ""),
                        status="error",
                    )
                )

        force_final = self._act_budget_exceeded()
        if force_final:
            messages = messages + [
                self._loop_guard_message(
                    f"[loop-guard] {self.__name__}'s own act budget for THIS "
                    "run is exhausted — tools are withheld for the remainder "
                    "of this run only. Other agents and later runs are "
                    "unaffected. Summarize the current state, outcomes, and "
                    "any blockers as your final answer."
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
                MessagesPlaceholder(variable_name="msg_storage"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        message = chat_prompt.format_messages(
            messages=messages, msg_storage=self.msg_storage if self.msg_storage else []
        )

        act_tools = self.act_tools
        llm_with_tools = (
            self.llm
            if (force_final or not act_tools)
            else self.llm.bind_tools(act_tools)
        )
        action = self._invoke(message, llm=llm_with_tools)
        self.logger.info("finished response from llm: %s", action.content)
        if action.tool_calls:
            action.name = (
                self.__name__
            )  # Set the name of the action to the agent's name
            if self._tool_call_repeated(action):
                # Third identical tool call in a row: refuse and force a
                # diagnosis pass instead of the blind-retry loop.
                note = self._loop_guard_message(
                    f"# {self.__name__} - act\n[loop-guard] The identical "
                    "tool call was already attempted twice. Do not retry it "
                    "verbatim; analyse the previous error output and change "
                    "the approach (or report the blocker)."
                )
                messages.append(note)
                self._note_no_progress("act")
                return {"messages": messages, "send_to": "plan"}
            self._note_progress()
            self._mark_stage(action, "act")
            messages.append(action)
            # NOTE: no msg_storage.extend(messages) here — that made storage
            # grow O(N^2) across the tool loop and replayed it into every
            # subsequent prompt. State already carries the conversation.
            self.logger.info("Tool calls is activated, sending to tools")
            self.logger.info(
                "Agent %s tool calls: %s", self.__name__, action.tool_calls
            )
            out = {
                "messages": messages,
                "send_to": "tools",  # Tool calls will be handled by the tool node agent
            }
            self.logger.info("Modeler.act returning send_to=%s", out["send_to"])
            return out
        else:
            # If no tool calls, treat as a regular action. Real tool evidence
            # from this run makes describing it legitimate — the scrubber
            # must not truncate a genuine activity summary.
            self._note_no_progress("plan-act")
            action.content = self._stage_content(
                "act",
                action.content,
                has_tool_evidence=self._run_tool_result_count(messages) > 0,
            )
            action.name = self.__name__
            self.persist_message(action)
            messages.append(action)
            # Persist latest ACT block to README under this agent's section
            try:
                if getattr(self, "project_root", None):
                    update_section(self.project_root, self.__name__, action.content)
            except Exception as e:
                self.logger.error(
                    "Failed updating README section for %s: %s", self.__name__, e
                )
            self.logger.info("Agent %s act stage: \n%s", self.__name__, action.content)
            out = {
                "messages": messages,
                "send_to": "plan",
            }
            self.logger.info("Modeler.act returning send_to=%s", out["send_to"])
            return out


ModelBuildingAgent = Modeler
