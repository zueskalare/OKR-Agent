"""
okr_models.py

Pydantic models defining the structure for OKR (Objectives and Key Results) inputs.
These models can be used as a BaseModel schema for LLMs to generate or validate
structured OKR data.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class KeyResultInput(BaseModel):
    """Represents a single Key Result under a Task."""
    id: Optional[int] = Field(
        None, description="Optional database ID of the key result (for updates)."
    )
    description: str = Field(..., description="Description of the key result.")
    achieved: bool = Field(
        False, description="Flag indicating whether this key result has been achieved."
    )
    messages: List[str] = Field(
        default_factory=list,
        description="Messages or notes associated with this key result.",
    )



class SubtaskInput(BaseModel):
    """Represents a Subtask under a Task."""
    id: Optional[int] = Field(
        None, description="Optional database ID of the subtask (for updates)."
    )
    description: str = Field(..., description="Description of the subtask.")
    status: Optional[str] = Field(
        "pending", description="Current status (pending, in_progress, done)."
    )


class TaskInput(BaseModel):
    """Represents an Objective/Task in the OKR framework."""
    id: Optional[int] = Field(
        None, description="Optional database ID of the task (for updates)."
    )
    job_id: str = Field(..., description="Identifier for the job or objective.")
    description: str = Field(..., description="Description of the objective/task.")
    status: Optional[str] = Field(
        "submitted", description="Current status (submitted, approved, completed)."
    )
    subtasks: List[SubtaskInput] = Field(
        default_factory=list,
        description="List of subtasks under this task.",
    )
    key_results: List['KeyResultInput'] = Field(
        default_factory=list,
        description="List of key results under this task.",
    )


class OKRInput(BaseModel):
    """Top-level model containing a list of Objectives/Tasks with nested structure."""
    objectives: List[TaskInput] = Field(
        ..., description="List of objectives/tasks with their subtasks and key results."
    )


# ---------------------------------------------------------------------------
# Task operation models (shared by the Supervisor's update_memory node and the
# okr_update tool — defined here to avoid agent/tool import cycles).
# ---------------------------------------------------------------------------

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = None  # type: ignore


class TaskOperation(BaseModel):
    """One self-contained task/subtask operation. Only the fields relevant to
    *op* need to be set."""

    op: Literal[
        "add_task",
        "add_subtask",
        "update_task_status",
        "update_subtask_status",
        "remove_task",
        "remove_subtask",
    ] = Field(..., description="Type of operation to perform.")

    job_id: Optional[str] = Field(None, description="Parent job id (add_task).")
    task_id: Optional[int] = Field(None, description="Target task id.")
    subtask_id: Optional[int] = Field(None, description="Target subtask id.")
    description: Optional[str] = Field(None, description="Task/subtask description.")
    status: Optional[str] = Field(
        None, description="New status value where applicable (e.g. 'done')."
    )


class TaskOperations(BaseModel):
    """Batch of task operations.

    An EMPTY list is valid and means "no update needed" — a schema where a
    no-op is unrepresentable drove the historical infinite update loop.
    """

    operations: List[TaskOperation] = Field(
        default_factory=list,
        description=(
            "Operations to apply in order. Return an EMPTY list if the task "
            "board already reflects reality and no change is needed."
        ),
    )


# Resolve forward references
KeyResultInput.model_rebuild()
TaskInput.model_rebuild()
