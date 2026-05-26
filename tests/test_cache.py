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


def test_make_key_idempotency_overrides_content() -> None:
    c = SynthesisCache(enabled=True, max_entries=10)
    k1 = c.make_key(
        text="x", model="m", voice=None, language="en",
        speed=1.0, response_format="wav", sample_rate=None,
        idempotency_key="abc",
    )
    k2 = c.make_key(
        text="totally different", model="other", voice="v", language="uk",
        speed=2.0, response_format="mp3", sample_rate=44100,
        idempotency_key="abc",
    )
    assert k1 == k2 == "idemp:abc"


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
