"""ConcurrencyController tests + API-level queue-depth backpressure."""

from __future__ import annotations

import asyncio

import pytest

from tts_server.core.concurrency import ConcurrencyController


async def test_controller_serializes_with_limit_1() -> None:
    ctl = ConcurrencyController(limit=1)
    finished: list[int] = []

    async def task(n: int) -> None:
        async with ctl:
            await asyncio.sleep(0.01)
            finished.append(n)

    await asyncio.gather(task(1), task(2), task(3))
    assert finished == [1, 2, 3]


async def test_controller_queue_depth_grows_then_drains() -> None:
    ctl = ConcurrencyController(limit=1)
    release = asyncio.Event()

    async def holder() -> None:
        async with ctl:
            await release.wait()

    async def waiter() -> None:
        async with ctl:
            pass

    h = asyncio.create_task(holder())
    # Let holder acquire the semaphore before we queue waiters.
    while ctl.in_flight() == 0:
        await asyncio.sleep(0)

    w1 = asyncio.create_task(waiter())
    w2 = asyncio.create_task(waiter())
    # Yield so the waiters enter `await self._sem.acquire()` and bump _pending.
    while ctl.queue_depth() < 2:
        await asyncio.sleep(0)

    assert ctl.in_flight() == 1
    assert ctl.queue_depth() == 2

    release.set()
    await asyncio.gather(h, w1, w2)

    assert ctl.in_flight() == 0
    assert ctl.queue_depth() == 0


def test_controller_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        ConcurrencyController(limit=0)


async def test_controller_in_flight_decremented_on_cancel() -> None:
    """Cancelling a coroutine waiting on the controller releases its slot."""
    ctl = ConcurrencyController(limit=1)
    release = asyncio.Event()

    async def holder() -> None:
        async with ctl:
            await release.wait()

    async def waiter() -> None:
        async with ctl:
            pass

    h = asyncio.create_task(holder())
    while ctl.in_flight() == 0:
        await asyncio.sleep(0)

    w = asyncio.create_task(waiter())
    while ctl.queue_depth() < 1:
        await asyncio.sleep(0)
    assert ctl.queue_depth() == 1

    w.cancel()
    try:
        await w
    except asyncio.CancelledError:
        pass

    # Pending should drop back to just the holder.
    assert ctl.queue_depth() == 0
    assert ctl.in_flight() == 1

    release.set()
    await h


async def test_api_returns_503_when_queue_depth_exceeded(monkeypatch) -> None:
    """End-to-end: max_queue_depth=0 + slow provider → 503 + Retry-After."""
    from fastapi.testclient import TestClient

    from tts_server.app import create_app
    from tts_server.providers._fake import FakeProvider
    from tts_server.settings import (
        CacheConfig,
        ProvidersConfig,
        RefsConfig,
        RoutingConfig,
        ServerConfig,
        Settings,
    )

    settings = Settings(
        server=ServerConfig(
            host="127.0.0.1", port=0, max_queue_depth=0, request_timeout_seconds=5.0,
        ),
        refs=RefsConfig(),
        cache=CacheConfig(enabled=False, max_entries=8),
        # Force concurrency=1 so a single in-flight request blocks the rest.
        providers=ProvidersConfig(enabled=["fake"], fake={"concurrency": 1}),
        routing=RoutingConfig(default="fake", by_language={"en": "fake"}),
    )

    # Slow-synthesize: holds the semaphore long enough for follower to land in-queue.
    original = FakeProvider.synthesize

    async def slow(self, req):
        await asyncio.sleep(0.3)
        return await original(self, req)

    monkeypatch.setattr(FakeProvider, "synthesize", slow)

    app = create_app(settings)
    with TestClient(app) as client:
        import threading
        results: list[tuple[int, str | None]] = []

        def fire() -> None:
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "voice": "fake-en", "language": "en"},
            )
            results.append((r.status_code, r.headers.get("retry-after")))

        threads = [threading.Thread(target=fire) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

    statuses = [r[0] for r in results]
    assert 200 in statuses
    assert 503 in statuses, f"expected at least one 503, got {statuses}"
    for status, retry in results:
        if status == 503:
            assert retry == "5", f"503 missing Retry-After: 5 (got {retry!r})"
