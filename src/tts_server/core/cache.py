"""Content-addressable synthesis cache.

The cache key is always built from a SHA-256 of the synthesis inputs
``(text, model, voice, language, speed, response_format, sample_rate)``.
A caller-supplied ``idempotency_key`` is folded in as a salt rather than
replacing the content hash; same ``(content, key)`` pair dedupes (so
network retries hit the cache) but two different inputs that happen to
share a key by accident no longer collide. The historical "the key
bypasses the content hash" semantic was a footgun — different callers
sharing a global namespace would serve each other stale audio.

In-memory LRU. Bounded by ``max_entries``. Disabled when ``enabled=False``.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CachedAudio:
    audio_bytes: bytes
    sample_rate: int
    format: str
    duration_ms: int


class SynthesisCache:
    def __init__(self, *, enabled: bool, max_entries: int) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self._lru: OrderedDict[str, CachedAudio] = OrderedDict()

    def __len__(self) -> int:
        return len(self._lru)

    def make_key(
        self,
        *,
        text: str,
        model: str,
        voice: str | None,
        language: str,
        speed: float,
        response_format: str,
        sample_rate: int | None,
        idempotency_key: str | None,
    ) -> str:
        h = hashlib.sha256()
        for part in (
            text,
            model,
            voice or "",
            language,
            f"{speed:.4f}",
            response_format,
            str(sample_rate or 0),
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        content_hash = h.hexdigest()
        if idempotency_key:
            # Salt the content hash with the caller's key. Distinct keys
            # keep distinct cache slots even for identical inputs; the
            # SAME key + SAME content reuses (retry-friendly).
            return f"idemp:{idempotency_key}:{content_hash}"
        return f"sha256:{content_hash}"

    def get(self, key: str) -> CachedAudio | None:
        if not self.enabled:
            return None
        item = self._lru.get(key)
        if item is None:
            return None
        self._lru.move_to_end(key)
        return item

    def put(self, key: str, value: CachedAudio) -> None:
        if not self.enabled:
            return
        # Always store/replace; move_to_end runs whether or not the key
        # already existed so re-puts also refresh LRU recency.
        self._lru[key] = value
        self._lru.move_to_end(key)
        while len(self._lru) > self.max_entries:
            self._lru.popitem(last=False)

    def clear(self) -> None:
        self._lru.clear()
