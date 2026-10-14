from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt


# NOTE: the `config` annotation must be RunnableConfig — langgraph inspects
# the type hint and only injects the runtime config when it matches.
def human_node(state: dict, config: Optional[RunnableConfig] = None, store=None):
    """Human-in-the-loop node.

    Uses langgraph's ``interrupt`` so the graph pauses through the
    checkpointer and the user's NEXT chat message resumes it (the backend
    detects the pending interrupt and streams ``Command(resume=<text>)``).
    The answer routes back to the agent that asked. The previous ``input()``
    implementation hung the backend event loop when no terminal was attached.

    Anonymous threads (``thread_id`` = ``anon-*``) get a fresh thread per
    request, so a pending interrupt could never be resumed — end the turn
    with a note instead of stranding the run.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"event": "working", "agent": "Human"})
    except Exception:
        pass
    messages = list(state.get("messages", []))
    last = messages[-1] if messages else None
    asker = getattr(last, "name", None) or "Supervisor"
    if asker == "Human":
        asker = "Supervisor"

    thread_id = ""
    if isinstance(config, dict):
        thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
    if thread_id.startswith("anon-"):
        note = AIMessage(
            content=(
                "# Human - reply\n[HITL unavailable] This request has no durable "
                "session (no session_dir), so the run cannot pause for user "
                "input. Ending the turn — reply with the missing details to "
                "continue."
            ),
            name="Human",
        )
        # Delta-only update: the inbound history is already in state
        # (add_messages) — returning it again replays prior turns through the
        # backend's per-request updates-batch dedupe.
        return {"messages": [note], "send_to": "END"}

    # The asking agent's report (which contains the actual question) is the
    # last message; surface its tail in the interrupt payload for clients
    # that want to render the ask without scrolling back.
    question = str(getattr(last, "content", "") or "")[-2000:] if last else ""
    value = interrupt({"question": question, "from": asker})

    human_message = HumanMessage(content=str(value), name="human")
    # Delta-only update (same rationale as the anon path above).
    return {"messages": [human_message], "send_to": asker}
