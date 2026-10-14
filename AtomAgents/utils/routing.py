"""Outer-graph routing helpers.

The Supervisor/Modeler/Simulator graph routes on ``state["send_to"]``, a
string chosen by each agent's report step. These helpers wrap that lookup
with a deterministic per-turn handoff budget so a runaway delegation loop
dies at ``config.MAX_HANDOFFS`` agent runs instead of grinding to the graph
recursion limit (each agent run costs 5+ LLM calls).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langgraph.errors import NodeError

from .. import config as _cfg

logger = logging.getLogger("AtomAgents.routing")


def reports_since_last_human(messages: List[Any]) -> int:
    """Count agent hand-off reports emitted since the newest HumanMessage.

    The outer graph state is checkpointed and ``add_messages`` never trims,
    so counting the WHOLE list would leak across turns — turn 4 of a session
    would start already over budget. Reports are detected via the
    ``atom_message_type: report`` marker that ReportMessage.__init__ always
    stamps (agents/messages.py) — the marker, not the class, because DB
    round-trips downgrade the subclass to a plain AIMessage. Not importing
    the class also keeps this module free of the agents package (base_agent
    imports us; importing agents back would be circular).
    """
    count = 0
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            break
        marker = (getattr(m, "additional_kwargs", None) or {}).get("atom_message_type")
        if marker == "report":
            count += 1
    return count


def handoff_router(state: Dict[str, Any], *, agent: str = "") -> Any:
    """Path function for the outer graph's conditional edges: honor
    ``send_to`` unless the per-turn handoff budget is spent, then force END.

    The report-step budget check fires one hop earlier and produces a
    user-visible notice; this router is the silent structural backstop for
    anything that slips past it (e.g. budgets lowered mid-session).
    """
    used = reports_since_last_human(state.get("messages", []))
    if used >= _cfg.MAX_HANDOFFS:
        logger.error(
            "[loop-guard] %s: handoff cap reached (%d/%d agent runs this turn) — routing END",
            agent or "router",
            used,
            _cfg.MAX_HANDOFFS,
        )
        return "END"
    return state.get("send_to")


def agent_error_handler(state: Dict[str, Any], error: NodeError) -> Any:
    """Default outer-graph error handler: after a node's retries exhaust,
    end the turn gracefully with a visible error message instead of a 500.

    Registered via ``StateGraph.set_node_defaults(error_handler=...)``. The
    ``error`` parameter MUST be annotated ``NodeError`` — langgraph injects
    it by type. Interrupts never reach this handler (verified against
    langgraph 1.2.10).
    """
    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    node = getattr(error, "node", "?")
    exc = getattr(error, "error", error)
    logger.error("[node-error] %s failed after retries: %s", node, exc)
    note = AIMessage(
        content=(
            f"# {node} - error\n[node-error] The {node} run failed after "
            f"retries: {type(exc).__name__}: {exc}. Ending this turn — reply "
            "to retry or adjust the request."
        ),
        name=str(node),
    )
    return Command(update={"messages": [note], "send_to": "END"})


__all__ = ["agent_error_handler", "handoff_router", "reports_since_last_human"]
