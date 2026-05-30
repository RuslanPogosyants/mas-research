"""Intrinsic quality metrics: computable from outputs + source, no ground truth."""

from __future__ import annotations

from collections import Counter
from typing import Any

_MIN_CHOICES = 2


def summary_intrinsics(summary: dict[str, Any] | None, source_chars: int) -> dict[str, Any]:
    if not summary:
        return {"section_count": 0, "sections_present": [], "summary_chars": 0, "compression_ratio": 0.0}
    sections = summary.get("sections") or []
    present = [str(s.get("type")) for s in sections if isinstance(s, dict) and s.get("type")]
    summary_chars = sum(len(str(s.get("text", ""))) for s in sections if isinstance(s, dict))
    ratio = round(summary_chars / source_chars, 4) if source_chars > 0 else 0.0
    return {
        "section_count": len(sections),
        "sections_present": present,
        "summary_chars": summary_chars,
        "compression_ratio": ratio,
    }


def terms_intrinsics(terms: list[dict[str, Any]], source_text: str) -> dict[str, Any]:
    if not terms:
        return {"count": 0, "in_source_fraction": 0.0, "category_distribution": {}, "avg_frequency": 0.0}
    haystack = source_text.lower()
    in_source = sum(1 for t in terms if str(t.get("term", "")).lower() in haystack)
    categories = Counter(str(t.get("category", "")) for t in terms)
    freqs = [int(t.get("frequency", 0) or 0) for t in terms]
    return {
        "count": len(terms),
        "in_source_fraction": round(in_source / len(terms), 4),
        "category_distribution": dict(categories),
        "avg_frequency": round(sum(freqs) / len(freqs), 4),
    }


def _is_well_formed(question: dict[str, Any]) -> bool:
    q_type = question.get("type")
    if q_type == "open_answer":
        return bool(str(question.get("question", "")).strip())
    choices = question.get("choices") or []
    if q_type == "single_choice":
        idx = question.get("answer_idx")
        return len(choices) >= _MIN_CHOICES and isinstance(idx, int) and 0 <= idx < len(choices)
    if q_type == "multi_choice":
        idxs = question.get("answer_indices") or []
        return (
            len(choices) >= _MIN_CHOICES
            and bool(idxs)
            and all(isinstance(i, int) and 0 <= i < len(choices) for i in idxs)
        )
    return False


def quiz_intrinsics(quiz: list[dict[str, Any]]) -> dict[str, Any]:
    if not quiz:
        return {"count": 0, "type_distribution": {}, "well_formed_count": 0, "well_formed_fraction": 0.0}
    types = Counter(str(q.get("type", "")) for q in quiz)
    well_formed = sum(1 for q in quiz if _is_well_formed(q))
    return {
        "count": len(quiz),
        "type_distribution": dict(types),
        "well_formed_count": well_formed,
        "well_formed_fraction": round(well_formed / len(quiz), 4),
    }


def citations_intrinsics(citations: list[dict[str, Any]]) -> dict[str, Any]:
    if not citations:
        return {"count": 0, "mean_relevance": 0.0, "min_relevance": 0.0, "max_relevance": 0.0}
    scores = [float(c.get("relevance_score", 0.0) or 0.0) for c in citations]
    return {
        "count": len(citations),
        "mean_relevance": round(sum(scores) / len(scores), 4),
        "min_relevance": round(min(scores), 4),
        "max_relevance": round(max(scores), 4),
    }
