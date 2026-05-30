"""Shared LLM-JSON helper: call an LlmAdapter, parse + validate the JSON response
into a Pydantic model, retrying with a corrective re-prompt. Used by every agent
that asks the LLM for structured JSON (F3 summary, F4 quiz, ...).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from src.core.metrics import LLM_CALL_SECONDS

if TYPE_CHECKING:
    from src.adapters.llm import LlmAdapter

_RETRY_HINT = (
    "\n\nПредыдущий ответ не был валидным JSON ожидаемой структуры. "
    "Верни строго корректный JSON без markdown и пояснений."
)


async def parse_with_retry[ModelT: BaseModel](
    llm: LlmAdapter, *, system: str, user: str, model_cls: type[ModelT], retries: int
) -> ModelT | None:
    """Call `llm`, parse its JSON into `model_cls`, retrying up to `retries` times.

    Returns the validated model, or None if the response is still unparseable /
    invalid after all attempts (caller decides how to refuse).
    """
    prompt = user
    for _ in range(retries + 1):
        start = time.perf_counter()
        response = await llm.complete(system=system, user=prompt)
        parsed = _parse(response, model_cls)
        LLM_CALL_SECONDS.labels(outcome="parsed" if parsed is not None else "unparsed").observe(
            time.perf_counter() - start
        )
        if parsed is not None:
            return parsed
        prompt = user + _RETRY_HINT
    return None


def _extract_json(response: str) -> str:
    """Pull a JSON object out of a model response.

    Real LLMs (unlike the deterministic fake) intermittently wrap their JSON in a
    markdown code fence or surround it with prose despite instructions. This strips
    a fence (with an optional language tag) and otherwise falls back to the
    outermost ``{ ... }`` span, so a stray wrapper does not fail the whole parse.
    """
    text = response.strip()
    if "```" in text:
        after_open = text[text.find("```") + 3 :]
        fence_close = after_open.find("```")
        if fence_close != -1:
            inner = after_open[:fence_close].strip()
            newline = inner.find("\n")
            if newline != -1 and inner[:newline].strip().isalpha():
                inner = inner[newline + 1 :]
            text = inner.strip()
    open_brace = text.find("{")
    close_brace = text.rfind("}")
    if open_brace != -1 and close_brace > open_brace:
        return text[open_brace : close_brace + 1]
    return text


def _parse[ModelT: BaseModel](response: str, model_cls: type[ModelT]) -> ModelT | None:
    try:
        data = json.loads(_extract_json(response))
    except (json.JSONDecodeError, TypeError):
        return None
    try:
        return model_cls.model_validate(data)
    except ValidationError:
        return None
