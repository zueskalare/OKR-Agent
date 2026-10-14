"""
Structured OKR tools + the single-writer README board sync.

Design: **SQL (memory.db) holds the detail; the job README is a rendered
overview with exactly one writer (this module + readme_sync).**

Previously the prompts instructed the LLM to hand-craft its own SQLite file
(`okr/okr.db`) via shell on every ACT — producing divergent schemas, stale
sizes, and unjoinable IDs — while the harness task tables stayed empty. These
in-process tools replace that: agents call them like any other tool, the data
lands in the session `memory.db`, and the README projection is re-rendered
after every mutation.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from ..models.okr_models import TaskOperation, TaskOperations
from .readme_sync import update_section, render_tasks_markdown, render_artefacts_markdown

logger = logging.getLogger(__name__)


def apply_task_operation(mem, op: TaskOperation) -> None:
    """Apply one task operation against a SQLiteRAGMemory-compatible store."""
    if op.op == "add_task":
        if not op.job_id or not op.description:
            raise ValueError("add_task requires job_id and description")
        # Dedupe: observed live — a looping run re-added the same objective
        # 44x. Adding an existing description reports the existing id instead.
        desc = op.description.strip().lower()
        for task in mem.get_all_tasks():
            if (task.get("description") or "").strip().lower() == desc:
                raise ValueError(
                    f"Task already exists as id {task['id']} "
                    f"(status: {task.get('status')}); do not re-add it — "
                    "update its status instead."
                )
        mem.create_task(op.job_id, op.description, status=op.status or "submitted")
    elif op.op == "add_subtask":
        if op.task_id is None or not op.description:
            raise ValueError("add_subtask requires task_id and description")
        if not mem.task_exists(op.task_id):
            raise ValueError(f"Task id {op.task_id} does not exist for subtask")
        desc = op.description.strip().lower()
        for sub in mem.get_all_subtasks(task_id=op.task_id):
            if (sub.get("description") or "").strip().lower() == desc:
                raise ValueError(
                    f"Subtask already exists as id {sub['id']} under task "
                    f"{op.task_id} (status: {sub.get('status')}); do not "
                    "re-add it — update its status instead."
                )
        mem.create_subtask(op.task_id, op.description, status=op.status or "pending")
    elif op.op == "update_task_status":
        if op.task_id is None or not op.status:
            raise ValueError("update_task_status requires task_id and status")
        if not mem.task_exists(op.task_id):
            raise ValueError(f"Task id {op.task_id} does not exist")
        mem.update_task_status(op.task_id, op.status)
    elif op.op == "update_subtask_status":
        if op.subtask_id is None or not op.status:
            raise ValueError("update_subtask_status requires subtask_id and status")
        if not mem.subtask_exists(op.subtask_id):
            raise ValueError(f"Subtask id {op.subtask_id} does not exist")
        mem.update_subtask_status(op.subtask_id, op.status)
    elif op.op == "remove_task":
        if op.task_id is None or not mem.task_exists(op.task_id):
            raise ValueError(f"Task id {op.task_id} does not exist")
        mem.delete_task(op.task_id)
    elif op.op == "remove_subtask":
        if op.subtask_id is None or not mem.subtask_exists(op.subtask_id):
            raise ValueError(f"Subtask id {op.subtask_id} does not exist")
        mem.delete_subtask(op.subtask_id)
    else:
        raise ValueError(f"Unknown task operation: {op.op}")


def sync_board(mem, project_root: str) -> None:
    """Re-render the README overview sections from SQL (the single writer)."""
    try:
        tasks = mem.get_all_tasks()
        subtasks = mem.get_all_subtasks()
        key_results = mem.get_all_key_results()
        update_section(
            project_root, "Tasks", render_tasks_markdown(tasks, subtasks, key_results)
        )
        artefacts = mem.get_all_artefacts()
        metrics = mem.get_all_metrics()
        if artefacts or metrics:
            update_section(
                project_root,
                "Artefacts",
                render_artefacts_markdown(artefacts, metrics),
            )
    except Exception as e:
        logger.error("Failed to sync README board: %s", e)


def build_okr_tools(mem, project_root: str, include_task_ops: bool = True) -> list:
    """Build in-process structured tools bound to this session's memory.

    Returned tools: ``register_artefact``, ``record_metric``, ``record_attempt``
    and (optionally) ``okr_update``. Import is deferred so this module stays
    importable without langchain installed.
    """
    from langchain_core.tools import tool

    tools = []

    @tool
    def register_artefact(
        path: str,
        kind: str = "",
        kr_id: Optional[int] = None,
        task_id: Optional[int] = None,
    ) -> str:
        """Register a produced artefact (file) on the task board. The file is
        verified on disk (must exist and be non-empty) — its size and sha256
        are recorded. Use this for every deliverable you produce; a task or
        key result only counts as done when its artefacts are registered.

        Args:
            path: absolute path of the produced file
            kind: short kind label, e.g. 'trajectory', 'plot', 'log', 'data'
            kr_id: optional key-result id this artefact satisfies
            task_id: optional task id this artefact belongs to
        """
        try:
            info = mem.register_artefact(path, kind=kind, kr_id=kr_id, task_id=task_id)
            sync_board(mem, project_root)
            return json.dumps({"registered": info})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def record_metric(
        metric: str,
        value: float,
        unit: str = "",
        kr_id: Optional[int] = None,
    ) -> str:
        """Record a validation metric (e.g. final_stress_GPa, energy_drift,
        solver_iterations) on the task board.

        Args:
            metric: metric name
            value: numeric value
            unit: unit string
            kr_id: optional key-result id this metric validates
        """
        try:
            mem.record_metric(metric, float(value), unit=unit, kr_id=kr_id)
            sync_board(mem, project_root)
            return json.dumps({"recorded": {"metric": metric, "value": value, "unit": unit}})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def record_attempt(
        description: str,
        result: str,
        error_excerpt: str = "",
        subtask_id: Optional[int] = None,
    ) -> str:
        """Log an attempt and its outcome to the failure ledger. ALWAYS call
        this after a failed run/fix so later retries (by you or colleagues)
        see what was already tried instead of repeating it.

        Args:
            description: what was attempted (command/approach)
            result: short outcome, e.g. 'failed', 'fixed', 'partial'
            error_excerpt: the key error line(s), if any
            subtask_id: optional subtask this attempt belongs to
        """
        try:
            mem.record_attempt(
                description, result, error_excerpt=error_excerpt, subtask_id=subtask_id
            )
            return json.dumps({"logged": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    tools.extend([register_artefact, record_metric, record_attempt])

    if include_task_ops:

        @tool(args_schema=TaskOperations)
        def okr_update(operations: List[TaskOperation]) -> str:
            """Apply task-board operations (add/update/remove tasks and
            subtasks). Pass an empty list to make no change. The README board
            is re-rendered automatically."""
            applied, errors = 0, []
            for op in operations:
                try:
                    if isinstance(op, dict):
                        op = TaskOperation(**op)
                    apply_task_operation(mem, op)
                    applied += 1
                except Exception as e:
                    errors.append(str(e))
            sync_board(mem, project_root)
            return json.dumps({"applied": applied, "errors": errors})

        tools.append(okr_update)

    return tools
