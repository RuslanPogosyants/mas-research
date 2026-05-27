"""Final task status FSM.

Decides the terminal status based on which subtasks succeeded or failed,
honouring the required vs optional split defined by the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.schemas import TaskStatus

if TYPE_CHECKING:
    from src.plan import Plan


def determine_final_status(plan: Plan, results: dict[str, Any | None]) -> TaskStatus:
    """Determine the terminal status of a task.

    A subtask is considered failed if its id maps to None in `results` or is
    missing entirely (never reported back within the deadline).

    Rules:
        - Any required failed -> FAILED
        - All required succeeded, some optional failed -> PARTIAL_READY
        - All required and all optional succeeded -> COMPLETED
    """
    failed_required = any(results.get(subtask.id) is None and subtask.required for subtask in plan.subtasks)
    if failed_required:
        return TaskStatus.FAILED
    failed_optional = any(results.get(subtask.id) is None for subtask in plan.subtasks if not subtask.required)
    if failed_optional:
        return TaskStatus.PARTIAL_READY
    return TaskStatus.COMPLETED
