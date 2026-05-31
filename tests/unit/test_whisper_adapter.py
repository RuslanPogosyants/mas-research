"""Unit tests for WhisperTranscriberAdapter batched inference and fallback."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Minimal segment stub (mirrors faster-whisper's NamedTuple/dataclass shape)
# ---------------------------------------------------------------------------


@dataclass
class _Seg:
    text: str
    start: float = 0.0
    end: float = 1.0


# ---------------------------------------------------------------------------
# Fake faster_whisper module factory
# ---------------------------------------------------------------------------


def _make_fake_faster_whisper(
    *,
    batched_raises: Exception | None = None,
) -> tuple[types.ModuleType, dict[str, Any]]:
    """Return (fake_module, call_log).

    call_log collects:
      - "plain_calls": list of kwargs passed to WhisperModel.transcribe
      - "batched_calls": list of kwargs passed to BatchedInferencePipeline.transcribe
    """
    log: dict[str, Any] = {"plain_calls": [], "batched_calls": []}

    class _FakeWhisperModel:
        def __init__(self, model_size: str, *, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, file_path: str, **kwargs: Any) -> tuple[list[_Seg], object]:
            log["plain_calls"].append({"file_path": file_path, **kwargs})
            return ([_Seg("hello"), _Seg("world")], object())

    if batched_raises is not None:
        exc = batched_raises

        class _FailingBatched:
            def __init__(self, model: Any) -> None:
                raise exc

        batched_cls: type = _FailingBatched
    else:

        class _FakeBatchedPipeline:
            def __init__(self, model: Any) -> None:
                pass

            def transcribe(self, file_path: str, **kwargs: Any) -> tuple[list[_Seg], object]:
                log["batched_calls"].append({"file_path": file_path, **kwargs})
                return ([_Seg("batched segment")], object())

        batched_cls = _FakeBatchedPipeline

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    module.BatchedInferencePipeline = batched_cls  # type: ignore[attr-defined]
    return module, log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_fake_fw(monkeypatch: pytest.MonkeyPatch, fake_module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    # Evict the real adapter from import cache so the fresh fixture is picked up.
    monkeypatch.delitem(sys.modules, "src.adapters.whisper_transcriber", raising=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_batched_path_uses_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """When BatchedInferencePipeline is available, _run uses it with batch_size and vad_filter."""
    fake_fw, log = _make_fake_faster_whisper()
    _inject_fake_fw(monkeypatch, fake_fw)

    from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

    adapter = WhisperTranscriberAdapter(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        batch_size=4,
    )
    chunks = adapter._run("audio.wav", "ru")

    assert log["batched_calls"], "batched pipeline was not called"
    assert not log["plain_calls"], "plain model should not be called when batched is active"

    call_kwargs = log["batched_calls"][0]
    assert call_kwargs["batch_size"] == 4
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["file_path"] == "audio.wav"
    assert call_kwargs["language"] == "ru"

    assert len(chunks) == 1
    assert "batched segment" in chunks[0].content


async def test_batched_path_language_none_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty language string is passed as None to the pipeline."""
    fake_fw, log = _make_fake_faster_whisper()
    _inject_fake_fw(monkeypatch, fake_fw)

    from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

    adapter = WhisperTranscriberAdapter(model_size="tiny", device="cpu", compute_type="int8", batch_size=8)
    adapter._run("audio.wav", "")

    assert log["batched_calls"][0]["language"] is None


async def test_fallback_to_plain_when_batched_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When BatchedInferencePipeline construction raises, the adapter falls back to the plain model."""
    fake_fw, log = _make_fake_faster_whisper(batched_raises=RuntimeError("unsupported"))
    _inject_fake_fw(monkeypatch, fake_fw)

    from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

    adapter = WhisperTranscriberAdapter(model_size="tiny", device="cpu", compute_type="int8", batch_size=4)
    chunks = adapter._run("audio.wav", "ru")

    assert not log["batched_calls"], "batched pipeline should not be called on fallback"
    assert log["plain_calls"], "plain model must be called on fallback"
    assert log["plain_calls"][0]["vad_filter"] is True
    assert log["plain_calls"][0]["language"] == "ru"

    assert len(chunks) >= 1
    assert not adapter._use_batched


async def test_fallback_to_plain_when_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When BatchedInferencePipeline is absent (ImportError), adapter falls back to plain model."""
    fake_fw, log = _make_fake_faster_whisper(batched_raises=ImportError("no batched support"))
    _inject_fake_fw(monkeypatch, fake_fw)

    from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

    adapter = WhisperTranscriberAdapter(model_size="tiny", device="cpu", compute_type="int8")
    adapter._run("audio.wav", "en")

    assert log["plain_calls"], "plain model must be called when import fails"
    assert not adapter._use_batched


async def test_segments_to_chunks_called_on_batched_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """segments_to_chunks processes the batched pipeline's segment output."""
    fake_fw, _log = _make_fake_faster_whisper()
    _inject_fake_fw(monkeypatch, fake_fw)

    from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

    adapter = WhisperTranscriberAdapter(model_size="tiny", device="cpu", compute_type="int8", batch_size=8)
    chunks = adapter._run("audio.wav", "ru")

    assert chunks
    assert all(chunk.source_type == "audio" for chunk in chunks)
    assert chunks[0].content == "batched segment"
