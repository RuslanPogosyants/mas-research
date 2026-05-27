"""Retry policy tests with exponential backoff invariants."""

from __future__ import annotations

import pytest
from src.core.retry import BACKOFF_SECONDS, RETRY_MAX, compute_backoff


class TestComputeBackoff:
    def test_first_retry_returns_first_backoff(self) -> None:
        assert compute_backoff(0) == BACKOFF_SECONDS[0]

    def test_second_retry_returns_second_backoff(self) -> None:
        assert compute_backoff(1) == BACKOFF_SECONDS[1]

    def test_backoff_grows(self) -> None:
        assert BACKOFF_SECONDS[1] > BACKOFF_SECONDS[0]

    def test_retry_count_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_backoff(RETRY_MAX)

    def test_negative_retry_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_backoff(-1)


def test_retry_max_is_two() -> None:
    assert RETRY_MAX == 2


def test_backoff_lengths_match_retry_max() -> None:
    assert len(BACKOFF_SECONDS) == RETRY_MAX
