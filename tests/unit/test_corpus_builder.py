"""Unit tests for the F6 corpus builder (network + model stubbed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_fetch_papers_dedupes_and_requires_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.corpus_builder import build as builder

    payload = {
        "data": [
            {"title": "Graphs", "abstract": "about graphs", "authors": [{"name": "A"}], "year": 2021, "url": "u1"},
            {"title": "Graphs", "abstract": "duplicate title", "authors": [], "year": 2022, "url": "u2"},
            {"title": "NoAbstract", "abstract": None, "authors": [], "year": 2020, "url": "u3"},
        ]
    }

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None: ...
        def json(self) -> dict[str, Any]:
            return payload

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: Any) -> None: ...
        def get(self, *a: Any, **k: Any) -> _Resp:
            return _Resp()

    monkeypatch.setattr(builder.httpx, "Client", _Client)
    papers = builder.fetch_papers(queries=["q"], per_query=10)

    assert [p["title"] for p in papers] == ["Graphs"]  # dedup by title, abstract required
    assert papers[0]["authors"] == "A"


def test_build_prefixes_passages_and_writes_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.corpus_builder import build as builder

    monkeypatch.setattr(
        builder,
        "fetch_papers",
        lambda: [{"title": "T", "abstract": "abs", "authors": "A", "year": 2021, "url": "u"}],
    )
    seen: list[str] = []

    class _Encoder:
        def __init__(self, name: str) -> None: ...
        def encode(self, texts: list[str], normalize_embeddings: bool = False) -> Any:
            import numpy

            seen.extend(texts)
            return numpy.array([[0.1, 0.2]])

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Encoder)

    count = builder.build(out_dir=str(tmp_path))

    assert count == 1
    assert seen == ["passage: abs"]  # e5 passage prefix applied to abstracts
    meta = json.loads((tmp_path / "papers.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert meta == {"title": "T", "authors": "A", "year": 2021, "url": "u"}
    assert (tmp_path / "papers.npy").exists()
