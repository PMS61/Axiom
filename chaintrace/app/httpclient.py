"""Shared HTTP plumbing: rate limiting and retries.

The trace engine makes one request per address it walks, so a five-hop trace
can fire hundreds of calls at a public API in a few seconds. Free endpoints
answer that with 429s. Everything that talks to a blockchain API goes through
`get_json()` here, which paces requests per host and retries the failures that
are worth retrying.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import requests

from .config import HTTP_MAX_RETRIES, HTTP_TIMEOUT

# Status codes worth trying again: throttling and transient server trouble.
RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Spaces calls so they never exceed `calls_per_second`.

    Thread-safe, because Streamlit runs each session in its own thread and they
    share one limiter per host. `clock` and `sleeper` are injectable so tests do
    not have to spend real time.
    """

    def __init__(
        self,
        calls_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0.0
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> float:
        """Block until the next call is allowed. Returns how long it waited."""
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if delay > 0:
            self._sleeper(delay)
        return delay


class HttpError(RuntimeError):
    """A request failed after every retry was spent."""


def _retry_after(response: requests.Response, fallback: float) -> float:
    """Honour a Retry-After header when the server sends one."""
    header = response.headers.get("Retry-After", "")
    try:
        return max(float(header), 0.0)
    except (TypeError, ValueError):
        return fallback


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    timeout: float = HTTP_TIMEOUT,
    max_retries: int = HTTP_MAX_RETRIES,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """GET `url` and return parsed JSON, pacing and retrying as needed."""
    last_error: Exception | str | None = None

    for attempt in range(max_retries + 1):
        if limiter is not None:
            limiter.wait()

        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:   # DNS, TLS, timeout, refused
            last_error = exc
            if attempt == max_retries:
                break
            sleeper(2.0**attempt * 0.5)
            continue

        if response.status_code in RETRY_STATUS:
            last_error = f"HTTP {response.status_code}"
            if attempt == max_retries:
                break
            sleeper(_retry_after(response, 2.0**attempt * 0.5))
            continue

        if not response.ok:
            raise HttpError(f"{url} returned HTTP {response.status_code}")

        try:
            return response.json()
        except ValueError as exc:
            raise HttpError(f"{url} did not return JSON: {exc}") from exc

    raise HttpError(f"{url} failed after {max_retries + 1} attempts: {last_error}")
