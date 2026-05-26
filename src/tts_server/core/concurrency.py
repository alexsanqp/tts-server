"""Per-provider concurrency gating.

Each provider gets an asyncio.Semaphore. GPU providers default to 1
(serialize). Network/CPU providers default to many. Override per-provider
via [providers.<id>] concurrency = N in the config.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class ConcurrencyController:
    semaphore: asyncio.Semaphore
    limit: int


def make_controller(limit: int) -> ConcurrencyController:
    if limit < 1:
        raise ValueError(f"concurrency limit must be >= 1, got {limit}")
    return ConcurrencyController(semaphore=asyncio.Semaphore(limit), limit=limit)
