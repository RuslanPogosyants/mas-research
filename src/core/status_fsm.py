"""Final task status FSM.

Decides the terminal status based on which subtasks succeeded or failed,
honouring the required vs optional split defined by the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.schemas import TaskStatus

if TYPE_CHECKING:
    from src.plan import Plan

SubtaskResults = dict[str, object | None]


def determine_final_status(plan: Plan, results: SubtaskResults) -> TaskStatus:
    """Determine the terminal status of a task.

    Failure semantics: a subtask is considered failed when its id maps to None
    in `results`, OR when its id is absent from `results` (never reported back
    within the deadline). Both cases collapse to `results.get(id) is None`.

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
