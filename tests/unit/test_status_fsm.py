"""Property-based tests for determine_final_status.

New rule (D1): FAILED only when no subtask produced a result; otherwise a single
failure degrades to PARTIAL_READY; all-success is COMPLETED. The required vs
optional split no longer gates the terminal status.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from src.core.schemas import Operation, TaskStatus
from src.core.status_fsm import determine_final_status
from src.plan import Plan, Subtask


def _make_plan(required_ids: list[str], optional_ids: list[str]) -> Plan:
    subtasks: list[Subtask] = []
    for subtask_id in required_ids:
        subtasks.append(
            Subtask(
                id=subtask_id,
                agent="transcriber",
                operation=Operation.F1_TRANSCRIBE,
                required=True,
            )
        )
    for subtask_id in optional_ids:
        subtasks.append(
            Subtask(
                id=subtask_id,
                agent="summarizer",
                operation=Operation.F3_SUMMARIZE,
                required=False,
            )
        )
    return Plan(task_id="task-x", subtasks=subtasks)


id_strategy = st.text(alphabet="abcdef0123456789", min_size=4, max_size=8).map(lambda value: f"st-{value}")
required_strategy = st.lists(id_strategy, min_size=1, max_size=2, unique=True)
optional_strategy = st.lists(id_strategy, min_size=0, max_size=4, unique=True)


@settings(max_examples=50)
@given(required=required_strategy, optional=optional_strategy)
def test_all_successful_means_completed(required: list[str], optional: list[str]) -> None:
    assume(not (set(required) & set(optional)))
    plan = _make_plan(required, optional)
    results = {subtask_id: {"ok": True} for subtask_id in required + optional}
    assert determine_final_status(plan, results) == TaskStatus.COMPLETED


@settings(max_examples=50)
@given(required=required_strategy, optional=optional_strategy)
def test_all_subtasks_failed_means_failed(required: list[str], optional: list[str]) -> None:
    """When every subtask fails (no result at all), the task is FAILED."""
    assume(not (set(required) & set(optional)))
    plan = _make_plan(required, optional)
    results: dict[str, object | None] = dict.fromkeys(required + optional)
    assert determine_final_status(plan, results) == TaskStatus.FAILED


@settings(max_examples=50)
@given(required=required_strategy, optional=optional_strategy)
def test_failed_required_with_one_success_means_partial_ready(required: list[str], optional: list[str]) -> None:
    """A failed required op no longer forces FAILED while something else succeeded."""
    assume(optional)
    assume(not (set(required) & set(optional)))
    plan = _make_plan(required, optional)
    results: dict[str, object | None] = dict.fromkeys(required)
    results.update({subtask_id: {"ok": True} for subtask_id in optional})
    assert determine_final_status(plan, results) == TaskStatus.PARTIAL_READY


@settings(max_examples=50)
@given(required=required_strategy, optional=optional_strategy)
def test_only_optional_failed_means_partial_ready(required: list[str], optional: list[str]) -> None:
    assume(optional)
    assume(not (set(required) & set(optional)))
    plan = _make_plan(required, optional)
    results: dict[str, object] = {subtask_id: {"ok": True} for subtask_id in required}
    results[optional[0]] = None
    for subtask_id in optional[1:]:
        results[subtask_id] = {"ok": True}
    assert determine_final_status(plan, results) == TaskStatus.PARTIAL_READY


def test_empty_plan_completed() -> None:
    plan = Plan(task_id="task-x", subtasks=[])
    assert determine_final_status(plan, {}) == TaskStatus.COMPLETED


def test_zero_success_single_op_means_failed() -> None:
    """A lone requested op that fails -> nothing succeeded -> FAILED."""
    plan = _make_plan(required_ids=[], optional_ids=["st-only"])
    results: dict[str, object | None] = {"st-only": None}
    assert determine_final_status(plan, results) == TaskStatus.FAILED


def test_mixed_one_success_one_failure_means_partial_ready() -> None:
    """One op succeeds, one fails (the failed one was 'required') -> PARTIAL_READY."""
    plan = _make_plan(required_ids=["st-req"], optional_ids=["st-opt"])
    results: dict[str, object | None] = {"st-req": None, "st-opt": {"ok": True}}
    assert determine_final_status(plan, results) == TaskStatus.PARTIAL_READY


def test_all_success_means_completed() -> None:
    plan = _make_plan(required_ids=["st-req"], optional_ids=["st-opt"])
    results: dict[str, object | None] = {"st-req": {"ok": True}, "st-opt": {"ok": True}}
    assert determine_final_status(plan, results) == TaskStatus.COMPLETED


def test_missing_result_counts_as_failure() -> None:
    """A subtask absent from `results` (never reported) counts as a failure."""
    plan = _make_plan(required_ids=["st-a"], optional_ids=["st-b"])
    # st-b never reported, st-a succeeded -> one success, one missing -> PARTIAL_READY.
    results: dict[str, object | None] = {"st-a": {"ok": True}}
    assert determine_final_status(plan, results) == TaskStatus.PARTIAL_READY


def test_all_missing_means_failed() -> None:
    """No subtask reported at all -> nothing succeeded -> FAILED."""
    plan = _make_plan(required_ids=["st-a"], optional_ids=["st-b"])
    assert determine_final_status(plan, {}) == TaskStatus.FAILED
