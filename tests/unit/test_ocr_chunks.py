"""Unit tests for mapping extracted page texts into TextChunks."""

from __future__ import annotations

from src.adapters.pymupdf_ocr import pages_to_chunks


def test_one_chunk_per_nonempty_page_pdf() -> None:
    chunks = pages_to_chunks(["page one text", "", "  page three  "], source_type="pdf_extracted")
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.content for c in chunks] == ["page one text", "page three"]
    assert all(c.source_type == "pdf_extracted" for c in chunks)


def test_image_source_type() -> None:
    chunks = pages_to_chunks(["scanned words"], source_type="image")
    assert chunks[0].source_type == "image"
