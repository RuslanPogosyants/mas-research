"""Rebuild in-flight task state from the database for Coordinator restart.

`TaskRecovery` is the read side that mirrors the `TaskStore` write side: it loads
tasks that were still in flight (status RUNNING/PLANNING), rebuilds each `Task`
with its documents, and reconstructs the per-subtask `results` map from the
persisted output tables using the reverse mappings in `result_mapping`. Chunks are
bucketed back into F1/F2 by `source_type`. The Coordinator then resumes dispatch of
only the subtasks that have no reloaded result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus
from src.db.repos import DocumentRepo, ResultRepo, TaskRepo
from src.db.result_mapping import (
    content_from_chunk_rows,
    content_from_citation_rows,
    content_from_quiz_row,
    content_from_summary_row,
    content_from_term_rows,
)
from src.plan import subtask_id_for

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.db.models import DocumentRow, TaskRow, TextChunkRow

# Chunk source_type -> the ingestion operation that produced it.
_SOURCE_TYPE_TO_OPERATION: Final[dict[str, Operation]] = {
    "audio": Operation.F1_TRANSCRIBE,
    "pdf_extracted": Operation.F2_OCR,
    "image": Operation.F2_OCR,
}

# Statuses that mean "still in flight" — recovery resumes these.
RECOVERABLE_STATUSES: Final[set[TaskStatus]] = {TaskStatus.PLANNING, TaskStatus.RUNNING}


@dataclass(slots=True)
class RecoveredTask:
    """A task to resume plus its reloaded per-subtask results."""

    task: Task
    results: dict[str, object]


def bucket_chunks_by_operation(rows: list[TextChunkRow]) -> dict[Operation, list[TextChunkRow]]:
    """Group persisted chunks by the ingestion operation that produced them.

    Raises ValueError on an unrecognised source_type rather than silently dropping
    chunks — a future ingestion source must be wired here before recovery can carry it.
    """
    buckets: dict[Operation, list[TextChunkRow]] = defaultdict(list)
    for row in rows:
        operation = _SOURCE_TYPE_TO_OPERATION.get(row.source_type)
        if operation is None:
            raise ValueError(f"cannot bucket chunk {row.id!r}: unknown source_type {row.source_type!r}")
        buckets[operation].append(row)
    return buckets


def _task_from_row(row: TaskRow, documents: list[DocumentRow]) -> Task:
    return Task(
        id=row.id,
        user_id=row.user_id,
        status=TaskStatus(row.status),
        requested_outputs=[Operation(value) for value in row.requested_outputs],
        conversation_id=f"conv-{row.id}",
        documents=[
            Document(
                id=document.id,
                task_id=document.task_id,
                document_type=DocumentType(document.document_type),
                file_path=document.file_path,
                original_name=document.original_name,
            )
            for document in documents
        ],
    )


class TaskRecovery(Protocol):
    """Read surface the Coordinator uses to rebuild in-flight tasks at startup."""

    async def load_in_flight(self) -> list[RecoveredTask]: ...


class DbTaskRecovery:
    """TaskRecovery backed by SQLAlchemy; one session per recovery pass."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_in_flight(self) -> list[RecoveredTask]:
        async with self._session_factory() as session:
            task_rows = await TaskRepo(session).find_by_status(RECOVERABLE_STATUSES)
            recovered: list[RecoveredTask] = []
            for task_row in task_rows:
                documents = await DocumentRepo(session).list_for_task(task_row.id)
                task = _task_from_row(task_row, documents)
                results = await self._load_results(session, task_row.id)
                recovered.append(RecoveredTask(task=task, results=results))
            return recovered

    async def _load_results(self, session: AsyncSession, task_id: str) -> dict[str, object]:
        repo = ResultRepo(session)
        results: dict[str, object] = {}
        chunk_buckets = bucket_chunks_by_operation(await repo.list_chunk_rows(task_id))
        for operation, chunk_rows in chunk_buckets.items():
            results[subtask_id_for(task_id, operation)] = content_from_chunk_rows(chunk_rows)
        summary = await repo.get_summary_row(task_id)
        if summary is not None:
            results[subtask_id_for(task_id, Operation.F3_SUMMARIZE)] = content_from_summary_row(summary)
        term_rows = await repo.list_term_rows(task_id)
        if term_rows:
            results[subtask_id_for(task_id, Operation.F5_TERMS)] = content_from_term_rows(term_rows)
        quiz = await repo.get_quiz_row(task_id)
        if quiz is not None:
            results[subtask_id_for(task_id, Operation.F4_TEST)] = content_from_quiz_row(quiz)
        citation_rows = await repo.list_citation_rows(task_id)
        if citation_rows:
            results[subtask_id_for(task_id, Operation.F6_RECOMMEND)] = content_from_citation_rows(citation_rows)
        return results
