"""SynthesisCache tests."""

from __future__ import annotations

from tts_server.core.cache import CachedAudio, SynthesisCache


def _cached() -> CachedAudio:
    return CachedAudio(audio_bytes=b"abc", sample_rate=24000, format="wav", duration_ms=10)


def test_disabled_cache_never_returns_hit() -> None:
    c = SynthesisCache(enabled=False, max_entries=10)
    c.put("k", _cached())
    assert c.get("k") is None


def test_get_after_put_returns_same_value() -> None:
    c = SynthesisCache(enabled=True, max_entries=10)
    v = _cached()
    c.put("k", v)
    assert c.get("k") is v


def test_make_key_idempotency_namespaces_but_doesnt_collide_on_content() -> None:
    """idempotency_key salts the content hash so retries dedupe but distinct
    inputs sharing the key by accident still get separate cache slots."""
    c = SynthesisCache(enabled=True, max_entries=10)
    base = dict(
        text="x", model="m", voice=None, language="en",
        speed=1.0, response_format="wav", sample_rate=None,
    )
    # Same content + same key → same cache slot (the dedup case retries rely on).
    same_a = c.make_key(**base, idempotency_key="abc")
    same_b = c.make_key(**base, idempotency_key="abc")
    assert same_a == same_b
    assert same_a.startswith("idemp:abc:")

    # Different content + same key → DIFFERENT slot. No more cross-content collision.
    diff_content = c.make_key(
        **{**base, "text": "totally different"},
        idempotency_key="abc",
    )
    assert diff_content != same_a

    # Same content + different key → different slot (caller-side isolation).
    diff_key = c.make_key(**base, idempotency_key="xyz")
    assert diff_key != same_a

    # No idempotency_key at all keeps the plain content hash prefix.
    no_key = c.make_key(**base, idempotency_key=None)
    assert no_key.startswith("sha256:")
    assert no_key != same_a


def test_make_key_changes_on_any_input_change() -> None:
    c = SynthesisCache(enabled=True, max_entries=10)
    base = dict(
        text="x", model="m", voice="v", language="en",
        speed=1.0, response_format="wav", sample_rate=None, idempotency_key=None,
    )
    base_key = c.make_key(**base)
    for k in ("text", "model", "voice", "language", "speed", "response_format", "sample_rate"):
        mod = dict(base)
        mod[k] = "changed" if isinstance(mod[k], str) else 2.0 if isinstance(mod[k], float) else 12345
        assert c.make_key(**mod) != base_key, f"key did not change when {k} did"


def test_put_overwriting_existing_key_moves_to_end() -> None:
    """Re-putting the same key with new value updates the slot AND
    refreshes the LRU position so it isn't evicted prematurely."""
    c = SynthesisCache(enabled=True, max_entries=2)
    v1 = _cached()
    v2 = CachedAudio(audio_bytes=b"xyz", sample_rate=24000, format="wav", duration_ms=20)
    c.put("a", v1)
    c.put("b", _cached())
    c.put("a", v2)         # overwrite — should keep value AND refresh recency
    c.put("c", _cached())  # would have evicted "a" if recency hadn't refreshed
    assert c.get("a") is v2
    assert c.get("b") is None  # "b" was the LRU after the overwrite
    assert c.get("c") is not None


def test_lru_evicts_oldest_first() -> None:
    c = SynthesisCache(enabled=True, max_entries=2)
    c.put("a", _cached())
    c.put("b", _cached())
    c.put("c", _cached())  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") is not None
    assert c.get("c") is not None


def test_get_moves_to_end_protects_from_eviction() -> None:
    c = SynthesisCache(enabled=True, max_entries=2)
    c.put("a", _cached())
    c.put("b", _cached())
    c.get("a")  # touch -> a becomes most recently used
    c.put("c", _cached())  # evicts "b"
    assert c.get("a") is not None
    assert c.get("b") is None
    assert c.get("c") is not None
