"""Real PDF/image text extraction: PyMuPDF text layer first, EasyOCR fallback.

NOT imported in CI. `pages_to_chunks` is a pure helper (unit-tested); the F2 agent
re-stamps identity onto each chunk.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.core.metrics import MODEL_CALL_SECONDS
from src.core.schemas import TextChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

ExtractedSourceType = Literal["pdf_extracted", "image"]
_DEFAULT_MIN_CHARS = 20


def pages_to_chunks(pages: Sequence[str], *, source_type: ExtractedSourceType) -> list[TextChunk]:
    """Convert a sequence of page texts into TextChunks, skipping empty pages."""
    chunks: list[TextChunk] = []
    for page_text in pages:
        text = (page_text or "").strip()
        if not text:
            continue
        chunks.append(
            TextChunk(
                id=f"chunk-{len(chunks)}",
                task_id="",
                document_id="",
                source_type=source_type,
                content=text,
                chunk_index=len(chunks),
                confidence=None,
                meta={},
            )
        )
    return chunks


class PymupdfOcrAdapter:
    """PyMuPDF text layer with an EasyOCR fallback for low-text/scanned pages."""

    def __init__(self, *, languages: list[str], min_chars: int = _DEFAULT_MIN_CHARS) -> None:
        self._languages = languages
        self._min_chars = min_chars
        self._reader: Any | None = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import importlib

            torch = importlib.import_module("torch")
            return bool(torch.cuda.is_available())
        except ModuleNotFoundError:
            return False

    def _ensure_reader(self) -> Any:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self._languages, gpu=self._cuda_available())
        return self._reader

    def _ocr_image_bytes(self, image_bytes: bytes) -> str:
        result = self._ensure_reader().readtext(image_bytes, detail=0, paragraph=True)
        return "\n".join(str(line) for line in result)

    def _run_pdf(self, file_path: str) -> list[str]:
        import fitz

        pages: list[str] = []
        with fitz.open(file_path) as document:
            for page in document:
                text = page.get_text("text").strip()
                if len(text) < self._min_chars:
                    pixmap = page.get_pixmap(dpi=200)
                    text = self._ocr_image_bytes(pixmap.tobytes("png")).strip()
                pages.append(text)
        return pages

    def _run_image(self, file_path: str) -> list[str]:
        return [self._ocr_image_bytes(Path(file_path).read_bytes())]

    def _run(self, file_path: str, document_type: str) -> list[TextChunk]:
        if document_type == "pdf":
            return pages_to_chunks(self._run_pdf(file_path), source_type="pdf_extracted")
        return pages_to_chunks(self._run_image(file_path), source_type="image")

    async def extract(self, *, file_path: str, document_type: str) -> list[TextChunk]:
        start = time.perf_counter()
        try:
            return await asyncio.to_thread(self._run, file_path, document_type)
        finally:
            MODEL_CALL_SECONDS.labels(adapter="ocr", operation="F2").observe(time.perf_counter() - start)
