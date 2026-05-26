"""Settings loading: TOML + env-var override + defaults."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Iterator

import pytest

from tts_server.settings import load_settings


@pytest.fixture
def isolate_env(monkeypatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("TTS_"):
            monkeypatch.delenv(key, raising=False)
    yield


def _toml_with(content: str, tmp_path: Path) -> Path:
    p = tmp_path / "test.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_defaults_when_no_config(tmp_path: Path, isolate_env, monkeypatch) -> None:
    """No TOML found anywhere → bake-in defaults from the model."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TTS_CONFIG_FILE", raising=False)
    # Point at a definitely-absent path so the loader won't fall back to the
    # bundled config/tts-server.toml that ships with the repo.
    s = load_settings(tmp_path / "does-not-exist.toml")
    assert s.server.port == 8880
    assert s.server.auth_token == ""
    assert s.providers.enabled == ["fake"]


def test_toml_overrides_defaults(tmp_path: Path, isolate_env) -> None:
    cfg = _toml_with(
        """
        [server]
        port = 9000
        auth_token = "from-toml"

        [providers]
        enabled = ["fake", "edge"]
        """,
        tmp_path,
    )
    s = load_settings(cfg)
    assert s.server.port == 9000
    assert s.server.auth_token == "from-toml"
    assert s.providers.enabled == ["fake", "edge"]


def test_env_overrides_toml(tmp_path: Path, isolate_env, monkeypatch) -> None:
    cfg = _toml_with(
        """
        [server]
        port = 9000
        auth_token = "from-toml"
        """,
        tmp_path,
    )
    monkeypatch.setenv("TTS_SERVER__AUTH_TOKEN", "from-env")
    monkeypatch.setenv("TTS_SERVER__PORT", "7777")

    s = load_settings(cfg)
    assert s.server.auth_token == "from-env"
    assert s.server.port == 7777


def test_env_only_no_toml(tmp_path: Path, isolate_env, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TTS_CONFIG_FILE", raising=False)
    monkeypatch.setenv("TTS_SERVER__AUTH_TOKEN", "env-only")
    s = load_settings(tmp_path / "does-not-exist.toml")
    assert s.server.auth_token == "env-only"
