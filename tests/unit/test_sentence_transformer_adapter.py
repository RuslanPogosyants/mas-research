"""Unit tests for the real sentence-transformers embedding adapter (no model load)."""

from __future__ import annotations

import sys
import types

import pytest


class _RecordingModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[str] = []

    def encode(self, texts: list[str], normalize_embeddings: bool = False) -> list[list[float]]:
        self.seen.extend(texts)
        return [[0.0, 1.0] for _ in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, _RecordingModel]:
    created: dict[str, _RecordingModel] = {}

    def _factory(name: str) -> _RecordingModel:
        model = _RecordingModel(name)
        created["model"] = model
        return model

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return created


async def test_encode_prefixes_each_text_as_query(fake_sentence_transformers: dict[str, _RecordingModel]) -> None:
    from src.adapters.sentence_transformer import SentenceTransformerEmbeddingAdapter

    adapter = SentenceTransformerEmbeddingAdapter("intfloat/multilingual-e5-base")
    await adapter.encode(["graph theory", "sorting"])

    assert fake_sentence_transformers["model"].seen == ["query: graph theory", "query: sorting"]


async def test_encode_returns_one_vector_per_text(fake_sentence_transformers: dict[str, _RecordingModel]) -> None:
    from src.adapters.sentence_transformer import SentenceTransformerEmbeddingAdapter

    adapter = SentenceTransformerEmbeddingAdapter("intfloat/multilingual-e5-base")
    vectors = await adapter.encode(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(isinstance(value, float) for vector in vectors for value in vector)
