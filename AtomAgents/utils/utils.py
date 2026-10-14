from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk


def _message_from_dict(message: dict) -> BaseMessage:
    """Rebuild a message object from its serialized dict form.

    AI messages carrying the ``atom_message_type: report`` marker are restored
    as :class:`ReportMessage` (the subclass otherwise downgrades silently).
    """
    data = message.get("data", {})
    if data:
        message = data
    _type = message.get("type")
    if _type == "human":
        return HumanMessage(**message)
    if _type in ("ai", "ReportMessage"):
        additional_kwargs = message.get("additional_kwargs") or {}
        if (
            _type == "ReportMessage"
            or additional_kwargs.get("atom_message_type") == "report"
        ):
            # Local import: utils is imported by base_agent during the agents
            # package init, so a module-level import here would be circular.
            from ..agents.messages import ReportMessage

            payload = {k: v for k, v in message.items() if k != "type"}
            return ReportMessage(**payload)
        return AIMessage(**message)
    if _type == "system":
        return SystemMessage(**message)
    if _type == "chat":
        return ChatMessage(**message)
    if _type == "function":
        return FunctionMessage(**message)
    if _type == "tool":
        return ToolMessage(**message)
    if _type == "remove":
        return RemoveMessage(**message)
    if _type == "AIMessageChunk":
        return AIMessageChunk(**message)
    if _type == "HumanMessageChunk":
        return HumanMessageChunk(**message)
    if _type == "FunctionMessageChunk":
        return FunctionMessageChunk(**message)
    if _type == "ToolMessageChunk":
        return ToolMessageChunk(**message)
    if _type == "SystemMessageChunk":
        return SystemMessageChunk(**message)
    if _type == "ChatMessageChunk":
        return ChatMessageChunk(**message)
    msg = f"Got unexpected message type: {_type}"
    raise ValueError(msg)
