"""Stamp adapter-produced TextChunks with task/document identity for persistence.

Adapters are identity-agnostic — they transcribe or OCR a file without knowing
which task or document it belongs to. The ingestion agents F1/F2 own that
linkage, so they re-stamp each chunk with the real task_id (from the request
message), document_id (from the request payload) and a deterministic, bounded id
``chunk-{document_id}-{index}``. Deterministic ids keep re-persistence an
idempotent upsert and let a restarted coordinator re-link downstream rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.schemas import TextChunk


def stamp_chunk(chunk: TextChunk, *, task_id: str, document_id: str, index: int) -> TextChunk:
    """Return a copy of ``chunk`` carrying task/document identity and a stable id."""
    return chunk.model_copy(
        update={
            "id": f"chunk-{document_id}-{index}",
            "task_id": task_id,
            "document_id": document_id,
            "chunk_index": index,
        }
    )
