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
