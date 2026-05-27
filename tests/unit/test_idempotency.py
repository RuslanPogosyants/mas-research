"""Property-based tests for IdempotentReceiver. RED until M1."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.idempotency import IdempotentReceiver

id_strategy = st.text(alphabet="abcdef0123456789", min_size=8, max_size=16).map(lambda value: f"msg-{value}")


@settings(max_examples=30)
@given(message_ids=st.lists(id_strategy, min_size=1, max_size=20, unique=True))
def test_first_seen_returns_true(message_ids: list[str]) -> None:
    receiver = IdempotentReceiver(cache_size=100)
    for message_id in message_ids:
        assert receiver.accept(message_id) is True


@settings(max_examples=30)
@given(message_ids=st.lists(id_strategy, min_size=1, max_size=20, unique=True))
def test_duplicates_return_false(message_ids: list[str]) -> None:
    receiver = IdempotentReceiver(cache_size=100)
    for message_id in message_ids:
        receiver.accept(message_id)
    for message_id in message_ids:
        assert receiver.accept(message_id) is False


def test_lru_eviction() -> None:
    """LRU evicts the oldest unseen message once capacity is exceeded."""
    receiver = IdempotentReceiver(cache_size=3)
    receiver.accept("a")
    receiver.accept("b")
    receiver.accept("c")
    receiver.accept("d")  # cache becomes {b, c, d}; "a" evicted
    assert receiver.accept("a") is True  # "a" was evicted, treated as new
    # cache becomes {c, d, a}; "b" evicted
    assert receiver.accept("c") is False  # "c" is still in cache
    assert receiver.accept("b") is True  # "b" was evicted by the previous insert
