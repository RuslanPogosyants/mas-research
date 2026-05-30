"""Unit tests for the F1/F2 chunk identity-stamping helper."""

from __future__ import annotations

from src.agents._chunk_stamp import stamp_chunk
from src.core.schemas import TextChunk


def _adapter_chunk(content: str = "hello") -> TextChunk:
    # Mimics what an adapter returns: no task/doc identity, placeholder id.
    return TextChunk(
        id="chunk-/long/path/with/slashes.mp3-0",
        task_id="",
        document_id="",
        source_type="audio",
        content=content,
        chunk_index=0,
        confidence=0.9,
    )


def test_stamp_sets_task_document_and_deterministic_id() -> None:
    stamped = stamp_chunk(_adapter_chunk(), task_id="task-abc", document_id="doc-task-abc-0", index=0)
    assert stamped.task_id == "task-abc"
    assert stamped.document_id == "doc-task-abc-0"
    assert stamped.id == "chunk-doc-task-abc-0-0"
    assert stamped.chunk_index == 0
    assert len(stamped.id) <= 64
    # Original content/source_type/confidence are preserved.
    assert stamped.content == "hello"
    assert stamped.source_type == "audio"
    assert stamped.confidence == 0.9


def test_stamp_is_deterministic_and_index_unique() -> None:
    c0 = stamp_chunk(_adapter_chunk("a"), task_id="t", document_id="doc-t-1", index=0)
    c1 = stamp_chunk(_adapter_chunk("b"), task_id="t", document_id="doc-t-1", index=1)
    assert c0.id != c1.id
    assert (c0.chunk_index, c1.chunk_index) == (0, 1)
    # Re-stamping the same (document_id, index) is stable (idempotent persistence).
    again = stamp_chunk(_adapter_chunk("a"), task_id="t", document_id="doc-t-1", index=0)
    assert again.id == c0.id
