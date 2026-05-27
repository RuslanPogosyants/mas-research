"""Receive-side idempotency: LRU cache of message_id values seen recently.

Protects workers from processing duplicate messages delivered by Redis Streams.
"""

from __future__ import annotations

from collections import OrderedDict


class IdempotentReceiver:
    """LRU cache over the last N message_id values."""

    def __init__(self, cache_size: int = 1000) -> None:
        self._cache_size = cache_size
        self._seen: OrderedDict[str, None] = OrderedDict()

    def accept(self, message_id: str) -> bool:
        """Accept a message. Returns True if new, False if duplicate.

        On duplicate, refreshes the entry as most-recently-used. On new and
        cache full, evicts the least-recently-used entry before inserting.
        """
        if message_id in self._seen:
            self._seen.move_to_end(message_id)
            return False
        if len(self._seen) >= self._cache_size:
            self._seen.popitem(last=False)
        self._seen[message_id] = None
        return True
