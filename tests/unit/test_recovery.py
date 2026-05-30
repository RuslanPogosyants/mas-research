"""Unit tests for recovery chunk bucketing and the Task reconstruction helpers."""

from __future__ import annotations

import pytest
from src.agents.recovery import bucket_chunks_by_operation
from src.core.schemas import Operation
from src.db.models import TextChunkRow


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
