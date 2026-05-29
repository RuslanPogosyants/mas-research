"""Contract: the F3 inform shape equals schemas.Summary, and the raw GigaChat
JSON shape must be adapted (it does not validate directly). Pins RISK 1 before
the SummarizerAgent (M3.1) is written, so the in-agent adapter cannot drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.core.schemas import Summary

# Canonical inform content the SummarizerAgent (M3.1) must publish. assemble_payload
# calls Summary.model_validate on exactly this dict, so it must round-trip.
CANONICAL_F3_INFORM: dict[str, object] = {
    "summary_id": "sum-task-1",
    "sections": [
        {"type": "introduction", "text": "intro text"},
        {"type": "thesis", "text": "key points text"},
        {"type": "conclusion", "text": "conclusion text"},
    ],
    "source_chunk_ids": ["chunk-1", "chunk-2"],
}

# Raw GigaChat JSON per spec section 7.3. The agent adapter must map this to Summary;
# it intentionally does NOT validate as Summary on its own.
RAW_GIGACHAT_JSON: dict[str, object] = {
    "introduction": "intro text",
    "key_points": "key points text",
    "conclusions": "conclusion text",
}


def test_canonical_f3_inform_validates_as_summary() -> None:
    summary = Summary.model_validate(CANONICAL_F3_INFORM)
    assert summary.summary_id == "sum-task-1"
    assert [section.type for section in summary.sections] == ["introduction", "thesis", "conclusion"]
    assert summary.source_chunk_ids == ["chunk-1", "chunk-2"]


def test_raw_gigachat_shape_is_not_a_summary() -> None:
    # extra="forbid" + missing summary_id/sections -> the adapter is mandatory.
    with pytest.raises(ValidationError):
        Summary.model_validate(RAW_GIGACHAT_JSON)


def test_summary_section_type_is_thesis_not_key_points() -> None:
    # The schema section type is "thesis"; GigaChat/DB use "key_points".
    # The F3 adapter maps key_points -> thesis. Pin the schema vocabulary here.
    summary = Summary.model_validate(CANONICAL_F3_INFORM)
    types = {section.type for section in summary.sections}
    assert "thesis" in types
    assert "key_points" not in types
