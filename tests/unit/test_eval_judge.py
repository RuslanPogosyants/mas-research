"""Unit tests for LLM-as-judge: prompt building + JSON parsing with a Fake LLM."""

from __future__ import annotations

import json

import pytest
from src.adapters.llm import FakeLlmAdapter
from src.evaluation.judge import (
    judge_quiz,
    judge_quiz_pedagogy,
    judge_summary,
    judge_terms_usefulness,
)


@pytest.mark.asyncio
async def test_judge_summary_parses_scores() -> None:
    payload = json.dumps({"faithfulness": 4, "coverage": 5, "coherence": 4, "comment": "хорошо"})
    result = await judge_summary(
        FakeLlmAdapter(responses=[payload]),
        summary_text="итог",
        source_excerpt="исходный текст",
    )
    assert result is not None
    assert result.faithfulness == 4 and result.coverage == 5 and result.coherence == 4
    assert result.comment == "хорошо"


@pytest.mark.asyncio
async def test_judge_quiz_parses_scores() -> None:
    payload = json.dumps({"relevance": 5, "answerability": 4, "comment": "ок"})
    result = await judge_quiz(FakeLlmAdapter(responses=[payload]), quiz_text="Q1? Q2?", summary_text="итог")
    assert result is not None
    assert result.relevance == 5 and result.answerability == 4


@pytest.mark.asyncio
async def test_judge_returns_none_on_garbage() -> None:
    result = await judge_summary(
        FakeLlmAdapter(responses=["not json", "still not json"]),
        summary_text="x",
        source_excerpt="y",
    )
    assert result is None


@pytest.mark.asyncio
async def test_judge_quiz_pedagogy_parses_scores() -> None:
    payload = json.dumps(
        {
            "tests_understanding": 4,
            "difficulty_appropriateness": 3,
            "distractor_quality": 5,
            "rationale": "вопросы требуют понимания",
        }
    )
    result = await judge_quiz_pedagogy(
        FakeLlmAdapter(responses=[payload]),
        quiz_text="Q1? Q2?",
        summary_text="итог",
    )
    assert result is not None
    assert result.tests_understanding == 4
    assert result.difficulty_appropriateness == 3
    assert result.distractor_quality == 5
    assert result.rationale == "вопросы требуют понимания"


@pytest.mark.asyncio
async def test_judge_quiz_pedagogy_returns_none_on_garbage() -> None:
    result = await judge_quiz_pedagogy(
        FakeLlmAdapter(responses=["bad", "still bad"]),
        quiz_text="Q1?",
        summary_text="итог",
    )
    assert result is None


@pytest.mark.asyncio
async def test_judge_terms_usefulness_parses_scores() -> None:
    payload = json.dumps(
        {
            "learning_value": 5,
            "key_concept_coverage": 4,
            "rationale": "термины охватывают ключевые понятия",
        }
    )
    result = await judge_terms_usefulness(
        FakeLlmAdapter(responses=[payload]),
        terms_text="нейронная сеть, градиентный спуск",
        summary_text="итог",
    )
    assert result is not None
    assert result.learning_value == 5
    assert result.key_concept_coverage == 4
    assert result.rationale == "термины охватывают ключевые понятия"


@pytest.mark.asyncio
async def test_judge_terms_usefulness_returns_none_on_garbage() -> None:
    result = await judge_terms_usefulness(
        FakeLlmAdapter(responses=["bad", "still bad"]),
        terms_text="термин",
        summary_text="итог",
    )
    assert result is None
