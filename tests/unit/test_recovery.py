"""Unit tests for recovery chunk bucketing and the Task reconstruction helpers."""

from __future__ import annotations

import pytest
from src.agents.coordinator import Coordinator
from src.agents.recovery import RecoveredTask, bucket_chunks_by_operation
from src.core.bus import channel_for_agent
from src.core.schemas import Document, DocumentType, Operation, Task
from src.db.models import TextChunkRow
from src.plan import subtask_id_for

from tests.support.fake_bus import FakeBus, FakeTaskStore


def _chunk(chunk_id: str, source_type: str) -> TextChunkRow:
    return TextChunkRow(
        id=chunk_id,
        task_id="t",
        document_id="doc-t-0",
        source_type=source_type,
        content="c",
        chunk_index=0,
        confidence=None,
        meta={},
    )


def test_audio_chunks_bucket_to_f1_and_extracted_to_f2() -> None:
    rows = [_chunk("a", "audio"), _chunk("b", "pdf_extracted"), _chunk("c", "image")]
    buckets = bucket_chunks_by_operation(rows)
    assert {r.id for r in buckets[Operation.F1_TRANSCRIBE]} == {"a"}
    assert {r.id for r in buckets[Operation.F2_OCR]} == {"b", "c"}


def test_unknown_source_type_fails_loud() -> None:
    with pytest.raises(ValueError, match="source_type"):
        bucket_chunks_by_operation([_chunk("x", "text")])


class _FakeRecovery:
    def __init__(self, items: list[RecoveredTask]) -> None:
        self._items = items

    async def load_in_flight(self) -> list[RecoveredTask]:
        return self._items


def _audio_task() -> Task:
    return Task(
        id="t",
        requested_outputs=[Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE, Operation.F4_TEST],
        conversation_id="conv-t",
        documents=[Document(id="doc-t-0", task_id="t", document_type=DocumentType.AUDIO, file_path="/a.mp3")],
    )


async def test_recover_redispatches_only_unfinished_subtasks() -> None:
    task = _audio_task()
    f1 = subtask_id_for("t", Operation.F1_TRANSCRIBE)
    f3 = subtask_id_for("t", Operation.F3_SUMMARIZE)
    # F1 + F3 already persisted; F4 (depends on F3) is the only unfinished subtask.
    results: dict[str, object] = {
        f1: {"chunks": [{"id": "chunk-doc-t-0-0", "content": "c"}]},
        f3: {"summary_id": "s", "sections": [{"type": "thesis", "text": "x"}], "source_chunk_ids": []},
    }
    bus = FakeBus()
    coord = Coordinator(
        bus=bus, store=FakeTaskStore(), recovery=_FakeRecovery([RecoveredTask(task=task, results=results)])
    )
    await coord._recover()
    # Only F4 (test_generator) should be (re)published; F1/F3 are already done.
    assert bus.requests_for(channel_for_agent("transcriber")) == []
    assert bus.requests_for(channel_for_agent("summarizer")) == []
    published = bus.requests_for(channel_for_agent("test_generator"))
    assert len(published) == 1
    assert published[0].subtask_id == subtask_id_for("t", Operation.F4_TEST)


async def test_recover_finalizes_task_with_all_results_present() -> None:
    task = _audio_task()
    results: dict[str, object] = {
        subtask_id_for("t", Operation.F1_TRANSCRIBE): {"chunks": [{"id": "c0", "content": "c"}]},
        subtask_id_for("t", Operation.F3_SUMMARIZE): {
            "summary_id": "s",
            "sections": [{"type": "thesis", "text": "x"}],
            "source_chunk_ids": [],
        },
        subtask_id_for("t", Operation.F4_TEST): {"quiz_id": "q", "questions": [], "difficulty": "medium"},
    }
    store = FakeTaskStore()
    coord = Coordinator(bus=FakeBus(), store=store, recovery=_FakeRecovery([RecoveredTask(task=task, results=results)]))
    await coord._recover()
    # Nothing pending -> finalised straight away; no lingering in-memory task.
    assert store.status["t"] in ("completed", "partial_ready")


async def test_recover_is_noop_without_recovery_dependency() -> None:
    coord = Coordinator(bus=FakeBus(), store=FakeTaskStore())
    await coord._recover()  # must not raise
