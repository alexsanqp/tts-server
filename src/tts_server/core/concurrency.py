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

    async def acquire(self, *, max_queue_depth: int) -> bool:
        """Reserve a slot and acquire the semaphore atomically.

        Returns ``True`` when the caller was admitted (and the semaphore has
        been acquired, possibly after waiting). Returns ``False`` without
        changing any state when accepting this caller would push the queue
        strictly above ``max_queue_depth``.

        Atomicity comes from the fact that the projection check and the
        ``_pending`` increment are both synchronous statements with no
        ``await`` between them — under cooperative asyncio scheduling no
        other coroutine can interleave, so two concurrent callers cannot
        both see headroom and both increment past the limit.

        Callers MUST call :meth:`release` when their work completes (or
        use the ``async with`` form, which is bookkeeping-equivalent but
        doesn't take ``max_queue_depth``).
        """
        projected_queue = max(0, (self._pending + 1) - self._limit)
        if projected_queue > max_queue_depth:
            return False
        self._pending += 1
        try:
            await self._sem.acquire()
        except BaseException:
            self._pending -= 1
            raise
        return True

    def release(self) -> None:
        """Companion to :meth:`acquire` — semaphore release + pending decrement."""
        try:
            self._sem.release()
        finally:
            self._pending -= 1

    async def __aenter__(self) -> "ConcurrencyController":
        """Unbounded-queue convenience — for callers that don't need admission control.

        Equivalent to ``acquire(max_queue_depth=2**31)``: the caller will
        wait its turn but cannot be rejected. Use :meth:`acquire` directly
        when you want backpressure.
        """
        self._pending += 1
        try:
            await self._sem.acquire()
        except BaseException:
            self._pending -= 1
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def make_controller(limit: int) -> ConcurrencyController:
    return ConcurrencyController(limit)
