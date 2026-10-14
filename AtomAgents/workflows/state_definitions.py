"""
State definitions for the AtomAgent workflow.

This module defines the states and transitions for the molecular dynamics
simulation workflow graph.
"""
from typing import Optional, TypedDict,Annotated,Sequence
from langchain_core.messages import BaseMessage
try:
    from langgraph.managed import IsLastStep, RemainingSteps
    from langgraph.graph.message import add_messages
except ModuleNotFoundError:
    IsLastStep = None  # type: ignore
    RemainingSteps = None  # type: ignore
    def add_messages(x):  # type: ignore
        return x


    
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    send_to: Optional[str] = None
    













__all__ = ["State"]
