"""LLM-as-judge: rubric scoring of produced material via the LlmAdapter.

Scores are 1–5. Uses the shared parse_with_retry helper so a malformed model
response is retried once and ultimately yields None (caller records "unscored").
Tested with FakeLlmAdapter; uses GigaChat unchanged at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from src.agents._llm_json import parse_with_retry

if TYPE_CHECKING:
    from src.adapters.llm import LlmAdapter

_Score = Annotated[int, Field(ge=1, le=5)]
_MAX_SOURCE_CHARS: Final[int] = 6000
_JUDGE_RETRIES: Final[int] = 1


class SummaryJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithfulness: _Score
    coverage: _Score
    coherence: _Score
    comment: str = ""


class QuizJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevance: _Score
    answerability: _Score
    comment: str = ""


class QuizPedagogyJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tests_understanding: _Score
    difficulty_appropriateness: _Score
    distractor_quality: _Score
    rationale: str = ""


class TermsUsefulnessJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_value: _Score
    key_concept_coverage: _Score
    rationale: str = ""


_SUMMARY_SYSTEM = (
    "Ты — строгий эксперт по оценке учебных конспектов. Оцени конспект по исходному тексту "
    "по шкале 1–5 для каждого критерия: faithfulness (отсутствие выдумок относительно источника), "
    "coverage (полнота охвата ключевых идей), coherence (связность и читаемость). "
    'Верни строго JSON: {"faithfulness":N,"coverage":N,"coherence":N,"comment":"..."} без markdown.'
)
_QUIZ_SYSTEM = (
    "Ты — строгий эксперт по оценке тестовых вопросов. Оцени набор вопросов относительно конспекта "
    "по шкале 1–5: relevance (соответствие содержанию), answerability (можно ли ответить по конспекту). "
    'Верни строго JSON: {"relevance":N,"answerability":N,"comment":"..."} без markdown.'
)


async def judge_summary(llm: LlmAdapter, *, summary_text: str, source_excerpt: str) -> SummaryJudgement | None:
    user = f"ИСХОДНЫЙ ТЕКСТ:\n{source_excerpt[:_MAX_SOURCE_CHARS]}\n\nКОНСПЕКТ:\n{summary_text}"
    return await parse_with_retry(
        llm, system=_SUMMARY_SYSTEM, user=user, model_cls=SummaryJudgement, retries=_JUDGE_RETRIES
    )


async def judge_quiz(llm: LlmAdapter, *, quiz_text: str, summary_text: str) -> QuizJudgement | None:
    user = f"КОНСПЕКТ:\n{summary_text[:_MAX_SOURCE_CHARS]}\n\nВОПРОСЫ:\n{quiz_text}"
    return await parse_with_retry(llm, system=_QUIZ_SYSTEM, user=user, model_cls=QuizJudgement, retries=_JUDGE_RETRIES)


_QUIZ_PEDAGOGY_SYSTEM = (
    "Ты — эксперт по педагогике и разработке учебных материалов. Оцени набор тестовых вопросов "
    "с точки зрения СТУДЕНТА по шкале 1–5 по трём критериям: "
    "tests_understanding (вопросы проверяют понимание, а не зазубривание фактов), "
    "difficulty_appropriateness (сложность вопросов соответствует уровню учебного материала), "
    "distractor_quality (варианты ответов правдоподобны и не являются очевидно неверными). "
    'Верни строго JSON: {"tests_understanding":N,"difficulty_appropriateness":N,'
    '"distractor_quality":N,"rationale":"..."} без markdown.'
)
_TERMS_USEFULNESS_SYSTEM = (
    "Ты — эксперт по педагогике и разработке учебных материалов. Оцени список терминов "
    "с точки зрения СТУДЕНТА по шкале 1–5 по двум критериям: "
    "learning_value (термины являются ключевыми понятиями для изучения, а не тривиальными словами), "
    "key_concept_coverage (список охватывает главные концепции, изложенные в конспекте). "
    'Верни строго JSON: {"learning_value":N,"key_concept_coverage":N,"rationale":"..."} без markdown.'
)


async def judge_quiz_pedagogy(llm: LlmAdapter, *, quiz_text: str, summary_text: str) -> QuizPedagogyJudgement | None:
    user = f"КОНСПЕКТ:\n{summary_text[:_MAX_SOURCE_CHARS]}\n\nВОПРОСЫ:\n{quiz_text}"
    return await parse_with_retry(
        llm, system=_QUIZ_PEDAGOGY_SYSTEM, user=user, model_cls=QuizPedagogyJudgement, retries=_JUDGE_RETRIES
    )


async def judge_terms_usefulness(
    llm: LlmAdapter, *, terms_text: str, summary_text: str
) -> TermsUsefulnessJudgement | None:
    user = f"КОНСПЕКТ:\n{summary_text[:_MAX_SOURCE_CHARS]}\n\nТЕРМИНЫ:\n{terms_text}"
    return await parse_with_retry(
        llm, system=_TERMS_USEFULNESS_SYSTEM, user=user, model_cls=TermsUsefulnessJudgement, retries=_JUDGE_RETRIES
    )
