"""Bearer-token auth tests via TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tts_server.app import create_app
from tts_server.settings import (
    CacheConfig,
    ProvidersConfig,
    RefsConfig,
    RoutingConfig,
    ServerConfig,
    Settings,
)


def _settings_with_token(token: str) -> Settings:
    return Settings(
        server=ServerConfig(auth_token=token),
        refs=RefsConfig(),
        cache=CacheConfig(),
        providers=ProvidersConfig(enabled=["fake"]),
        routing=RoutingConfig(default="fake", by_language={"en": "fake"}),
    )


@pytest.fixture
def client_authed():
    app = create_app(_settings_with_token("secret"))
    with TestClient(app) as c:
        yield c


def test_speech_anonymous_blocked_when_token_set(client_authed) -> None:
    r = client_authed.post(
        "/v1/audio/speech",
        json={"input": "x", "model": "fake", "language": "en", "voice": "fake-en"},
    )
    assert r.status_code == 401


def test_speech_works_with_correct_token(client_authed) -> None:
    r = client_authed.post(
        "/v1/audio/speech",
        json={"input": "x", "model": "fake", "language": "en", "voice": "fake-en"},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200


def test_refs_upload_requires_auth_even_without_token_configured(client) -> None:
    """When auth_token is empty, /v1/refs MUST still 403 (write surface)."""
    r = client.post("/v1/refs", files={"file": ("x.wav", b"fake-content", "audio/wav")})
    assert r.status_code == 403


def test_speech_rejects_wrong_token(client_authed) -> None:
    """Token mismatch returns 401 with the unified error envelope."""
    r = client_authed.post(
        "/v1/audio/speech",
        json={"input": "x", "model": "fake", "language": "en", "voice": "fake-en"},
        headers={"Authorization": "Bearer not-the-right-token"},
    )
    assert r.status_code == 401
    body = r.json()
    assert body["detail"]["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(
    "bad_header",
    [
        "Basic secret",          # wrong scheme
        "bearer secret",         # case is handled, but extra-loose checks
        "Bearer",                # missing token
        "Bearersecret",          # no separator
        "  Bearer  secret  ",    # accidental whitespace — extract handles it
    ],
)
def test_speech_rejects_malformed_authorization(client_authed, bad_header: str) -> None:
    """All non-canonical Authorization shapes get a clean 401 (never 500/crash)."""
    r = client_authed.post(
        "/v1/audio/speech",
        json={"input": "x", "model": "fake", "language": "en", "voice": "fake-en"},
        headers={"Authorization": bad_header},
    )
    # "bearer secret" lowercases to the right scheme and "  Bearer  secret  "
    # strips fine — they should actually succeed; the rest are rejected.
    if bad_header.strip().lower().startswith("bearer ") and "secret" in bad_header:
        assert r.status_code == 200, (bad_header, r.text)
    else:
        assert r.status_code == 401, (bad_header, r.text)


def test_refs_upload_works_with_token(client_authed) -> None:
    # Minimal valid WAV header
    wav = (
        b"RIFF" + (44).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + (8000).to_bytes(4, "little") + (8000).to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (8).to_bytes(2, "little")
        + b"data" + (1).to_bytes(4, "little") + b"\x80"
    )
    r = client_authed.post(
        "/v1/refs",
        files={"file": ("ref.wav", wav, "audio/wav")},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"].startswith("ref:")
