"""Real speech-to-text via faster-whisper (CTranslate2). NOT imported in CI.

The heavy WhisperModel is lazy-loaded on first transcribe and run off the event
loop. `segments_to_chunks` is a pure helper (unit-tested) that groups Whisper's
short segments into reasonably sized TextChunks; the F1 agent re-stamps identity.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from src.core.metrics import MODEL_CALL_SECONDS
from src.core.schemas import TextChunk

if TYPE_CHECKING:
    from collections.abc import Iterable

_DEFAULT_TARGET_CHARS = 1500


def segments_to_chunks(segments: Iterable[Any], *, target_chars: int = _DEFAULT_TARGET_CHARS) -> list[TextChunk]:
    """Group faster-whisper segments into TextChunks of at most target_chars.

    Greedy fill: accumulate segments; when adding the next segment would push
    total length strictly above target_chars and the buffer is non-empty, flush
    first. A single segment that alone exceeds target_chars is emitted as its
    own chunk (never truncated).
    """
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_len = 0
    start_time: float | None = None
    end_time: float = 0.0

    def flush() -> None:
        nonlocal buffer, buffer_len, start_time
        text = " ".join(buffer).strip()
        if text:
            chunks.append(
                TextChunk(
                    id=f"chunk-{len(chunks)}",
                    task_id="",
                    document_id="",
                    source_type="audio",
                    content=text,
                    chunk_index=len(chunks),
                    confidence=None,
                    meta={"start": start_time, "end": end_time},
                )
            )
        buffer = []
        buffer_len = 0
        start_time = None

    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        seg_len = len(text)
        if buffer and buffer_len + seg_len > target_chars:
            flush()
        if start_time is None:
            start_time = getattr(segment, "start", None)
        end_time = getattr(segment, "end", end_time)
        buffer.append(text)
        buffer_len += seg_len + 1
    flush()
    return chunks


class WhisperTranscriberAdapter:
    """faster-whisper backend. Lazy-loads the model; times the call."""

    def __init__(self, *, model_size: str, device: str, compute_type: str) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        return self._model

    def _run(self, file_path: str, language: str) -> list[TextChunk]:
        model = self._ensure_model()
        segments, _info = model.transcribe(file_path, language=language or None, vad_filter=True)
        return segments_to_chunks(segments)

    async def transcribe(self, *, file_path: str, language: str = "ru") -> list[TextChunk]:
        start = time.perf_counter()
        try:
            return await asyncio.to_thread(self._run, file_path, language)
        finally:
            MODEL_CALL_SECONDS.labels(adapter="whisper", operation="F1").observe(time.perf_counter() - start)
