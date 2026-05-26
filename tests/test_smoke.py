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
