"""Smoke tests for the v1 skeleton."""

from __future__ import annotations

import base64


def test_healthz(client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_lists_providers(client) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "fake" in body["providers"]
    assert body["providers"]["fake"]["loaded"] is False  # lazy load


def test_list_models_includes_fake(client) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    ids = [m["id"] for m in body["models"]]
    assert "fake" in ids


def test_list_voices_for_fake(client) -> None:
    r = client.get("/v1/voices?model=fake")
    assert r.status_code == 200
    voices = r.json()["voices"]
    ids = {v["id"] for v in voices}
    assert {"fake-en", "fake-uk"} <= ids


def test_voices_filter_by_language(client) -> None:
    r = client.get("/v1/voices?language=uk")
    assert r.status_code == 200
    ids = {v["id"] for v in r.json()["voices"]}
    assert ids == {"fake-uk"}


def test_route_preview_auto(client) -> None:
    r = client.get("/v1/route?language=en&model=auto")
    assert r.status_code == 200
    assert r.json()["resolved_model"] == "fake"


def test_speech_returns_wav(client) -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hello world", "model": "auto", "language": "en", "voice": "fake-en"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert int(r.headers["x-sample-rate"]) > 0
    assert int(r.headers["x-duration-ms"]) > 0
    assert r.headers["x-tts-model"] == "fake"
    # Body starts with RIFF/WAVE marker
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WAVE"


def test_speech_defaults_to_wav_24khz(client) -> None:
    """Omitting response_format/sample_rate yields wav at 24 kHz (max fidelity)."""
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hello world", "model": "fake", "language": "en", "voice": "fake-en"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["x-audio-format"] == "wav"
    assert r.headers["x-sample-rate"] == "24000"


def test_speech_rejects_sample_rate_below_8khz(client) -> None:
    """sample_rate is bounded to a sensible audio range."""
    r = client.post(
        "/v1/audio/speech",
        json={
            "input": "hi", "model": "fake", "language": "en", "voice": "fake-en",
            "sample_rate": 4000,
        },
    )
    assert r.status_code == 422  # pydantic validation


def test_speech_rejects_sample_rate_above_48khz(client) -> None:
    """Upper bound (le=48000) on sample_rate is enforced."""
    r = client.post(
        "/v1/audio/speech",
        json={
            "input": "hi", "model": "fake", "language": "en", "voice": "fake-en",
            "sample_rate": 48001,
        },
    )
    assert r.status_code == 422


def test_speech_rejects_input_too_long(client) -> None:
    """Pydantic-level max_length stops pathologically long inputs cheaply."""
    r = client.post(
        "/v1/audio/speech",
        json={
            "input": "a" * 9000,  # exceeds max_length=8000
            "model": "fake", "language": "en", "voice": "fake-en",
        },
    )
    assert r.status_code == 422


def test_validation_error_uses_unified_envelope(client) -> None:
    """Pydantic 422 must surface in the same {detail: {error: {code, message, fields}}}
    shape as app-level errors so external clients only need one parser."""
    r = client.post(
        "/v1/audio/speech",
        json={"input": "", "model": "fake"},  # input violates min_length
    )
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error"]["code"] == "validation_error"
    assert "message" in body["detail"]["error"]
    assert isinstance(body["detail"]["error"]["fields"], list)
    assert len(body["detail"]["error"]["fields"]) >= 1


def test_speech_actually_resamples_when_rate_differs(client) -> None:
    """sample_rate=16000 must actually trigger ffmpeg resampling and the
    response headers/file must reflect the requested rate."""
    r = client.post(
        "/v1/audio/speech",
        json={
            "input": "hi", "model": "fake", "language": "en", "voice": "fake-en",
            "sample_rate": 16000,
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["x-sample-rate"] == "16000"
    assert r.headers["x-audio-format"] == "wav"
    # File header must report 16 kHz too
    import io
    import wave
    with wave.open(io.BytesIO(r.content), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1


def test_routing_by_language_uk_smoke(client) -> None:
    """language=uk with model=auto routes to the UK provider (fake here).

    The conftest fixture wires both 'en' and 'uk' to the fake provider;
    this exercises the by_language path end-to-end."""
    r = client.post(
        "/v1/audio/speech",
        json={"input": "Привіт", "model": "auto", "language": "uk", "voice": "fake-uk"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["x-tts-model"] == "fake"
    assert r.headers["x-route-reason"].startswith("by_language") or \
           r.headers["x-route-reason"] == "by_language"


def test_refs_upload_rejects_oversize_with_413(client_with_small_max) -> None:
    """The streaming size check aborts with 413 (not 400) and reports both
    the limit and what was received."""
    big = b"x" * (3 * 1024 * 1024)  # 3 MiB; conftest fixture caps at 1 MiB
    r = client_with_small_max.post(
        "/v1/refs",
        files={"file": ("big.wav", big, "audio/wav")},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 413
    body = r.json()
    err = body["detail"]["error"]
    assert err["code"] == "payload_too_large"
    assert err["max_bytes"] == 1 * 1024 * 1024
    assert err["received_bytes"] >= 1 * 1024 * 1024


def test_speech_envelope_json(client) -> None:
    r = client.post(
        "/v1/audio/speech?envelope=json",
        json={"input": "hello", "model": "fake", "language": "en", "voice": "fake-en"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["format"] == "wav"
    assert body["sample_rate"] > 0
    assert body["model"] == "fake"
    assert body["provider"] == "fake"
    decoded = base64.b64decode(body["audio_base64"])
    assert decoded[:4] == b"RIFF"


def test_unknown_model_returns_422(client) -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"input": "x", "model": "no-such", "language": "en"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["code"] == "unknown_model"


def test_unknown_voice_returns_422(client) -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"input": "x", "model": "fake", "language": "en", "voice": "no-such"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["code"] == "unknown_voice"
