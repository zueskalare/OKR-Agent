"""
tool_node_agent.py

ToolNode implementation that invokes tools from the AtomAgents MCP service.
"""

from typing import Union, Sequence, Callable, Optional, Literal
from abc import ABC

from langchain_core.messages.tool import ToolCall, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
import asyncio
import concurrent.futures

from ..config import make_mcp_client, MCP_SSE_READ_TIMEOUT_SEC

# Shared MCP client, configured (URL + timeouts) in AtomAgents.config.
CLIENT = make_mcp_client()

from json import loads
import logging


class MCPToolNode(ToolNode, ABC):
    """
    ToolNode implementation that invokes tools from the AtomAgents MCP service.

    Async-first (WS19): the production graph drives agents with ``astream``,
    so tool calls take langgraph's native async path through
    ``awrap_tool_call`` (timeout + error containment + multimodal unwrap).
    The sync ``_run_one`` override remains for legacy synchronous drivers
    (examples, submit_job) and is simply off the production path.
    """

    __name__: str = "mcp_tool_node"

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "mcp_tool_node",
        tags: Optional[list[str]] = None,
    ):
        """
        Initialize the MCPToolNode with a list of tools.
        Args:
            tools (Sequence[Union[BaseTool, str]]): List of tool instances or tool names.
        """
        super().__init__(tools=tools, name=name, awrap_tool_call=self._awrap)
        # ToolNode is not a pydantic model; attach the logger as a plain attribute.
        self.logger = logging.getLogger("AtomAgents.MCPToolNode")

    # Budget for one tool call; slightly above the longest server-side tool
    # budget so the server, not the client, is what times out first.
    TIMEOUT_SECONDS = MCP_SSE_READ_TIMEOUT_SEC

    # ------------------------------------------------------------------
    # Shared outcome builders (both execution paths)
    # ------------------------------------------------------------------

    def _timeout_message(self, call: ToolCall) -> ToolMessage:
        return ToolMessage(
            content=(
                f'{{"error": "Tool call timed out after '
                f'{self.TIMEOUT_SECONDS:.0f}s."}}'
            ),
            name=call.get("name", "unknown"),
            tool_call_id=call.get("id", ""),
            status="error",
        )

    def _error_message(self, call: ToolCall, exc: BaseException) -> ToolMessage:
        self.logger.error("Tool call %s failed: %s", call.get("name"), exc)
        return ToolMessage(
            content=f'{{"error": "Tool call failed: {exc}"}}',
            name=call.get("name", "unknown"),
            tool_call_id=call.get("id", ""),
            status="error",
        )

    @staticmethod
    def _truncate_text(text: str, cap: int) -> str:
        head = text[: int(cap * 0.75)]
        tail = text[-int(cap * 0.25):]
        return (
            f"{head}\n... [{len(text) - len(head) - len(tail)} chars "
            f"truncated client-side] ...\n{tail}"
        )

    @staticmethod
    def _cap_output(msg: ToolMessage) -> ToolMessage:
        """Client-side tool-output cap (config.TOOL_OUTPUT_MAX_BYTES).

        Our own MCP service truncates server-side, but FOREIGN MCP servers'
        outputs were unbounded — and every oversized tool result replays into
        each subsequent act prompt for the rest of the run.

        Runs AFTER _unwrap_multimodal: str content is capped whole; unwrapped
        block lists get only their text blocks capped — base64 image/file
        payloads must stay intact for vision (the server already bounds
        binaries at ATOMAGENTS_TOOL_BINARY_MAX).
        """
        from .. import config as _config

        cap = int(getattr(_config, "TOOL_OUTPUT_MAX_BYTES", 0) or 0)
        if not cap:
            return msg
        content = msg.content
        if isinstance(content, str) and len(content) > cap:
            msg.content = MCPToolNode._truncate_text(content, cap)
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and len(block["text"]) > cap
                ):
                    block["text"] = MCPToolNode._truncate_text(block["text"], cap)
        return msg

    @staticmethod
    def _unwrap_multimodal(msg: ToolMessage) -> ToolMessage:
        """Unwrap JSON-encoded multimodal parts (images/PDFs) into real
        content blocks so vision models receive them as images, not text.

        Must run BEFORE _cap_output — truncation splices a marker into the
        middle of the base64 body, after which the payload can never parse.
        Only a JSON list is accepted: content blocks are lists, and assigning
        a dict/scalar to ToolMessage.content bypasses validation and breaks
        every downstream flattener.
        """
        content = msg.content
        if isinstance(content, str) and (
            "data:image/png;base64" in content
            or "data:image/jpeg;base64" in content
            or "data:application/pdf;base64" in content
        ):
            log = logging.getLogger("AtomAgents.MCPToolNode")
            try:
                loaded = loads(content)
            except ValueError:
                log.warning(
                    "multimodal-looking tool output (%d chars) is not valid "
                    "JSON; leaving as text — vision will not see it",
                    len(content),
                )
            else:
                if isinstance(loaded, list):
                    content = loaded
                else:
                    log.warning(
                        "multimodal-looking tool output parsed to %s, not a "
                        "block list; leaving as text",
                        type(loaded).__name__,
                    )
        msg.content = content
        return msg

    # ------------------------------------------------------------------
    # Async path (production): langgraph invokes this around every tool call
    # ------------------------------------------------------------------

    async def _awrap(self, request, execute):
        call = request.tool_call
        try:
            msg = await asyncio.wait_for(
                execute(request), timeout=self.TIMEOUT_SECONDS
            )
        except (asyncio.TimeoutError, TimeoutError):
            return self._timeout_message(call)
        except Exception as e:  # keep the graph alive; report the failure
            return self._error_message(call, e)
        if isinstance(msg, ToolMessage):
            # Unwrap BEFORE capping — the reverse order truncated the JSON
            # envelope mid-base64 and silently killed vision (A1).
            msg = self._cap_output(self._unwrap_multimodal(msg))
        return msg

    # ------------------------------------------------------------------
    # Sync path (legacy drivers only)
    # ------------------------------------------------------------------

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        tool_runtime,
    ) -> ToolMessage:
        # NOTE: the 3rd parameter is langgraph's ToolRuntime (historically
        # mis-named `config` here); it is passed through positionally.
        coro = asyncio.wait_for(
            self._arun_one(call, input_type, tool_runtime),
            timeout=self.TIMEOUT_SECONDS,
        )
        try:
            try:
                asyncio.get_running_loop()
                in_loop = True
            except RuntimeError:
                in_loop = False
            if in_loop:
                # We are on an event-loop thread; blocking here would deadlock.
                # Run the coroutine on its own loop in a worker thread.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    msg = ex.submit(asyncio.run, coro).result()
            else:
                msg = asyncio.run(coro)
        except (asyncio.TimeoutError, TimeoutError):
            return self._timeout_message(call)
        except Exception as e:
            return self._error_message(call, e)
        if isinstance(msg, ToolMessage):
            # Unwrap BEFORE capping — same ordering contract as _awrap.
            msg = self._cap_output(self._unwrap_multimodal(msg))
        return msg


__all__ = ["MCPToolNode", "CLIENT"]
