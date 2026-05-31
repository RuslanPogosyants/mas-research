"""Final task status FSM.

Decides the terminal status from how many subtasks produced a result. The task
fails only when nothing succeeded; any single success keeps the task usable.
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

    Rules (the required vs optional split no longer gates the status — a failed
    required subtask does not by itself fail the task):
        - No subtask produced a result -> FAILED
        - At least one subtask succeeded and at least one failed -> PARTIAL_READY
        - Every subtask succeeded -> COMPLETED

    An empty plan (no subtasks) has nothing to fail and is COMPLETED.
    """
    if not plan.subtasks:
        return TaskStatus.COMPLETED
    any_success = any(results.get(subtask.id) is not None for subtask in plan.subtasks)
    if not any_success:
        return TaskStatus.FAILED
    any_failure = any(results.get(subtask.id) is None for subtask in plan.subtasks)
    if any_failure:
        return TaskStatus.PARTIAL_READY
    return TaskStatus.COMPLETED
