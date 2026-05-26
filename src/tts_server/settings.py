"""Application settings: TOML file + env overrides.

Source priority (highest wins):

1. environment variables (``TTS_*`` prefix, ``__`` separator for nested keys —
   e.g. ``TTS_SERVER__AUTH_TOKEN=secret``)
2. TOML file (path from ``--config``, then ``$TTS_CONFIG_FILE``, then
   ``./config/tts-server.toml`` next to the package)
3. defaults baked into the model
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def _is_loopback_host(host: str) -> bool:
    """True for 127.x, ::1, or the literal name 'localhost'."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ServerConfig(BaseModel):
    # Default to loopback. Operators who want LAN/Internet exposure must
    # set both `host` AND `auth_token` explicitly — see the validator below.
    host: str = "127.0.0.1"
    port: int = 8880
    auth_token: str = ""
    request_timeout_seconds: float = 120.0
    max_queue_depth: int = 32
    # CORS allowlist for browser clients hitting /v1/audio/speech etc.
    # Empty (default) = no CORS middleware mounted, browser fetches from
    # other origins are blocked by the browser. Set to ["*"] for fully
    # open access (intra-LAN dev), or list specific origins for prod.
    cors_allow_origins: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refuse_open_synth_on_network_interface(self) -> "ServerConfig":
        """Block obvious self-foot-gun: open synth endpoint on a routable interface.

        If ``host`` is anything other than a loopback address and
        ``auth_token`` is empty, every caller on the LAN/Internet can burn
        the operator's GPU minutes unauthenticated. Refuse to start —
        either bind to 127.0.0.1 (recommended; put a reverse proxy in
        front if you need external reach) or set a token.
        """
        if not _is_loopback_host(self.host) and not self.auth_token:
            raise ValueError(
                f"server.host={self.host!r} is not a loopback address but "
                "server.auth_token is empty. Refusing to start an open "
                "synthesis endpoint on a network interface — either bind "
                "to 127.0.0.1 (and front it with a reverse proxy if you "
                "need external reach) or set a non-empty auth_token."
            )
        return self


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


# ----- pydantic-settings source for the TOML file ---------------------------


class TomlConfigSource(PydanticBaseSettingsSource):
    """Pydantic-settings source that merges values from a TOML file."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path | None) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path is not None and path.is_file():
            with path.open("rb") as fh:
                self._data = tomllib.load(fh)

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        # Returning a value here is informational; __call__ below returns the
        # full snapshot, which is what pydantic-settings actually merges.
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


# ----- Settings -------------------------------------------------------------


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Pull the TOML path out of init kwargs if the loader passed one.
        toml_path: Path | None = None
        if hasattr(init_settings, "init_kwargs"):
            toml_path = init_settings.init_kwargs.pop("__toml_path__", None)
        toml_source = TomlConfigSource(settings_cls, toml_path)

        # Priority: highest first.
        # init kwargs > env > TOML > defaults. This way an explicit Settings(server={...})
        # in tests still wins, env vars override TOML on prod, and TOML beats defaults.
        return (init_settings, env_settings, toml_source, file_secret_settings)


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
    """Load settings with priority: env > TOML > defaults."""
    path: Path | None
    if config_file:
        path = Path(config_file)
    elif env := os.environ.get("TTS_CONFIG_FILE"):
        path = Path(env)
    else:
        path = _default_config_path()

    # Pass the path through a synthetic init kwarg consumed by
    # settings_customise_sources above.
    return Settings(__toml_path__=path)  # type: ignore[arg-type]
