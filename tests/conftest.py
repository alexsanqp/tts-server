"""Shared test fixtures."""

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


@pytest.fixture(autouse=True)
def _disable_qwen_vram_check(monkeypatch):
    """Stub the nvidia-smi VRAM check that QwenProvider runs before spawning.

    The real implementation shells out to ``nvidia-smi`` via
    ``subprocess.run``, which uses ``subprocess.Popen`` under the hood
    and trips any test that patches ``subprocess.Popen`` to assert
    sidecar-spawn behaviour. The warning is purely informational, so
    returning None ("unknown free VRAM") in tests is the right no-op.
    """
    monkeypatch.setattr(
        "tts_server.providers.qwen._check_free_vram_mib",
        lambda device: None,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        server=ServerConfig(host="127.0.0.1", port=0),
        refs=RefsConfig(),
        cache=CacheConfig(),
        providers=ProvidersConfig(enabled=["fake"], required=[]),
        routing=RoutingConfig(default="fake", by_language={"en": "fake", "uk": "fake"}),
    )


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_small_max():
    """1 MiB upload cap + a real token so /v1/refs is reachable.

    Used by tests that exercise the streaming size guard in
    :func:`tts_server.api.refs.upload_ref`.
    """
    settings = Settings(
        server=ServerConfig(host="127.0.0.1", port=0, auth_token="secret"),
        refs=RefsConfig(max_upload_mb=1),
        cache=CacheConfig(),
        providers=ProvidersConfig(enabled=["fake"], required=[]),
        routing=RoutingConfig(default="fake", by_language={"en": "fake", "uk": "fake"}),
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
