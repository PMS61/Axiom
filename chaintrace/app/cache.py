"""A tiny cache with a Redis-shaped interface.

The POC must run without a Redis server, so the default implementation is an
in-memory dict with TTL support. `RedisCache` is deliberately left as a thin
sketch: swapping `get_cache()` to return it is the only change needed later.
"""

from __future__ import annotations

import time
from typing import Any, Protocol


class Cache(Protocol):
    """The interface every cache backend implements."""

    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def clear(self) -> None: ...


class InMemoryCache:
    """Dict-backed cache. Same method names as redis-py, so it swaps cleanly."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float | None, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and expires_at < time.time():
            self._store.pop(key, None)  # expired
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


_CACHE: Cache = InMemoryCache()


def get_cache() -> Cache:
    """Return the process-wide cache. Swap the backend here to move to Redis."""
    return _CACHE
