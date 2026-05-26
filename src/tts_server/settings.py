"""Application settings: TOML file + env overrides.

Loading order:
1. config/tts-server.toml (or path from $TTS_CONFIG_FILE / --config)
2. environment variables (TTS_* prefix, double-underscore for nested keys)
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8880
    auth_token: str = ""
    request_timeout_seconds: float = 120.0
    max_queue_depth: int = 32


class RefsConfig(BaseModel):
    catalog_dir: Path = Path("data/refs-catalog")
    upload_dir: Path = Path("data/refs")
    upload_ttl_hours: int = 24
    max_upload_mb: int = 10


class CacheConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 256


class ProvidersConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["fake"])
    required: list[str] = Field(default_factory=list)
    # Per-provider blocks live as extra keys; access via .provider_options(<id>)
    model_config = {"extra": "allow"}

    def provider_options(self, provider_id: str) -> dict[str, Any]:
        raw = getattr(self, provider_id, None)
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        return raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)


class RoutingConfig(BaseModel):
    default: str = "fake"
    by_language: dict[str, str] = Field(default_factory=dict)


class Settings(BaseSettings):
    server: ServerConfig = Field(default_factory=ServerConfig)
    refs: RefsConfig = Field(default_factory=RefsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_nested_delimiter="__",
        extra="ignore",
    )


def _default_config_path() -> Path | None:
    """Locate the bundled default TOML config if present."""
    candidates = [
        Path.cwd() / "config" / "tts-server.toml",
        Path(__file__).resolve().parent.parent.parent / "config" / "tts-server.toml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_settings(config_file: str | os.PathLike | None = None) -> Settings:
    """Load settings: TOML file (if any) merged with env overrides."""
    path: Path | None
    if config_file:
        path = Path(config_file)
    elif env := os.environ.get("TTS_CONFIG_FILE"):
        path = Path(env)
    else:
        path = _default_config_path()

    data: dict[str, Any] = {}
    if path and path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)

    return Settings(**data)
