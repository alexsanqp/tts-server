"""Content-addressable synthesis cache.

Key = sha256 of (text, model, voice, language, speed, format, sample_rate)
when no idempotency_key is provided. If the caller passes an explicit
`idempotency_key`, we use that instead (verbatim).

In-memory LRU. Bounded by `max_entries`. Disabled when `enabled=False`.
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
        if idempotency_key:
            return f"idemp:{idempotency_key}"
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
        return f"sha256:{h.hexdigest()}"

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
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = value
            while len(self._lru) > self.max_entries:
                self._lru.popitem(last=False)

    def clear(self) -> None:
        self._lru.clear()
