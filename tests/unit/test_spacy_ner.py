"""Unit tests for SpacyNerAdapter — model selection by text script and pipeline caching.

All tests use a monkeypatched ``spacy.load`` so no real spaCy models are required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.adapters.spacy_ner import SpacyNerAdapter

# ---------------------------------------------------------------------------
# Minimal spaCy stubs
# ---------------------------------------------------------------------------


def _make_fake_nlp() -> Any:
    """Return a callable that mimics a spaCy pipeline returning an empty doc."""
    doc = MagicMock()
    doc.ents = []
    doc.__iter__ = MagicMock(return_value=iter([]))
    nlp = MagicMock(return_value=doc)
    return nlp


# ---------------------------------------------------------------------------
# _select_language
# ---------------------------------------------------------------------------


class TestSelectLanguage:
    def test_cyrillic_text_returns_ru(self) -> None:
        assert SpacyNerAdapter._select_language("Привет мир граф алгоритм") == "ru"

    def test_latin_text_returns_en(self) -> None:
        assert SpacyNerAdapter._select_language("Hello world graph algorithm") == "en"

    def test_mixed_latin_majority_returns_en(self) -> None:
        # 6 Latin vs 2 Cyrillic
        assert SpacyNerAdapter._select_language("Hello world Ал") == "en"

    def test_mixed_cyrillic_majority_returns_ru(self) -> None:
        # 2 Latin vs 6 Cyrillic
        assert SpacyNerAdapter._select_language("Привет мир AB") == "ru"

    def test_equal_count_returns_ru(self) -> None:
        # Equal: not strictly greater Latin -> Russian
        assert SpacyNerAdapter._select_language("ABаб") == "ru"

    def test_empty_text_returns_ru(self) -> None:
        assert SpacyNerAdapter._select_language("") == "ru"

    def test_digits_only_returns_ru(self) -> None:
        # No alphabetic chars at all -> 0 Latin, 0 Cyrillic -> ru
        assert SpacyNerAdapter._select_language("12345 !@#") == "ru"


# ---------------------------------------------------------------------------
# Model selection via _ensure_nlp (via extract)
# ---------------------------------------------------------------------------


class TestModelSelection:
    @pytest.mark.asyncio
    async def test_cyrillic_text_loads_ru_model(self) -> None:
        fake_nlp = _make_fake_nlp()
        with patch("spacy.load", return_value=fake_nlp) as mock_load:
            adapter = SpacyNerAdapter(model="ru_core_news_lg", en_model="en_core_web_sm")
            await adapter.extract("Граф алгоритм структура данных")
        mock_load.assert_called_once_with("ru_core_news_lg")

    @pytest.mark.asyncio
    async def test_latin_text_loads_en_model(self) -> None:
        fake_nlp = _make_fake_nlp()
        with patch("spacy.load", return_value=fake_nlp) as mock_load:
            adapter = SpacyNerAdapter(model="ru_core_news_lg", en_model="en_core_web_sm")
            await adapter.extract("Graph algorithm data structure")
        mock_load.assert_called_once_with("en_core_web_sm")


# ---------------------------------------------------------------------------
# Pipeline caching
# ---------------------------------------------------------------------------


class TestPipelineCaching:
    @pytest.mark.asyncio
    async def test_same_language_loads_model_only_once(self) -> None:
        fake_nlp = _make_fake_nlp()
        with patch("spacy.load", return_value=fake_nlp) as mock_load:
            adapter = SpacyNerAdapter()
            await adapter.extract("Первый вызов на русском")
            await adapter.extract("Второй вызов на русском тоже")
        # spacy.load must have been called exactly once — second call hits cache
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_languages_load_separate_models(self) -> None:
        fake_nlp_ru = _make_fake_nlp()
        fake_nlp_en = _make_fake_nlp()
        call_order: list[str] = []

        def _fake_load(name: str) -> Any:
            call_order.append(name)
            return fake_nlp_en if "en" in name else fake_nlp_ru

        with patch("spacy.load", side_effect=_fake_load):
            adapter = SpacyNerAdapter(model="ru_core_news_lg", en_model="en_core_web_sm")
            await adapter.extract("Русский текст для первого вызова")
            await adapter.extract("English text for second call")
            # Third call — same language as second: must NOT trigger another load
            await adapter.extract("More English text here")

        assert call_order == ["ru_core_news_lg", "en_core_web_sm"]

    @pytest.mark.asyncio
    async def test_pipeline_cached_across_calls(self) -> None:
        fake_nlp = _make_fake_nlp()
        with patch("spacy.load", return_value=fake_nlp):
            adapter = SpacyNerAdapter()
            await adapter.extract("Русский текст")
            ru_pipeline = adapter._pipelines.get("ru")
            await adapter.extract("Ещё один русский текст")
            # Same object in cache
            assert adapter._pipelines.get("ru") is ru_pipeline
