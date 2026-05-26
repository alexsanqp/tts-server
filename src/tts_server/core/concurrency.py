"""Per-provider concurrency gating.

Each provider gets a :class:`ConcurrencyController` wrapping a semaphore plus
a public `queue_depth()` counter (so the API layer doesn't have to read
``asyncio.Semaphore._waiters`` to enforce backpressure).
"""

from __future__ import annotations

import asyncio
from typing import Any


class ConcurrencyController:
    """Async-context-manager around a semaphore + an in-flight + queued counter."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        self._sem = asyncio.Semaphore(limit)
        self._limit = limit
        self._pending = 0  # in-flight + queued

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def semaphore(self) -> asyncio.Semaphore:
        # Tests and some debug code touch this. Prefer queue_depth() externally.
        return self._sem

    def queue_depth(self) -> int:
        """Number of callers that haven't yet acquired the semaphore.

        ``pending - limit`` clipped at zero. A controller with limit=1 that
        currently has one holder and three waiters reports queue_depth=3.
        """
        return max(0, self._pending - self._limit)

    def in_flight(self) -> int:
        return min(self._pending, self._limit)

    async def __aenter__(self) -> "ConcurrencyController":
        self._pending += 1
        try:
            await self._sem.acquire()
        except BaseException:
            self._pending -= 1
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self._sem.release()
        finally:
            self._pending -= 1


def make_controller(limit: int) -> ConcurrencyController:
    return ConcurrencyController(limit)
