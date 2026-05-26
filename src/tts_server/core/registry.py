"""Provider registry — single source of truth for enabled providers.

v1: providers are registered in `BUILTIN_PROVIDERS` below. Entry-point
discovery is deferred to v2 — for three built-in providers it's overkill.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tts_server.core.concurrency import ConcurrencyController, make_controller
from tts_server.core.errors import ProviderUnavailable, UnknownModel
from tts_server.providers._fake import FakeProvider
from tts_server.providers.base import ProviderCapabilities, TTSProvider
from tts_server.settings import Settings

logger = logging.getLogger(__name__)


# Built-in providers. To add one: implement TTSProvider Protocol, add a row here.
# Each value is a zero-arg factory (or a callable taking the options dict).
BUILTIN_PROVIDERS: dict[str, Any] = {
    "fake": FakeProvider,
    "edge": lambda opts: _lazy_import("tts_server.providers.edge", "EdgeProvider")(opts),
    "qwen3-0.6b": lambda opts: _lazy_import("tts_server.providers.qwen", "QwenProvider")(opts),
    "styletts2-uk": lambda opts: _lazy_import(
        "tts_server.providers.styletts2_uk", "StyleTTS2UkProvider"
    )(opts),
}


def _lazy_import(module: str, attr: str):
    import importlib

    mod = importlib.import_module(module)
    return getattr(mod, attr)


@dataclass
class ProviderEntry:
    id: str
    instance: TTSProvider
    capabilities: ProviderCapabilities
    concurrency: ConcurrencyController
    loaded: bool = False
    required: bool = False
    load_error: str | None = None
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ProviderRegistry:
    """Holds instantiated providers and gates access to them."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._entries: dict[str, ProviderEntry] = {}

    # ---- lifecycle ----

    async def startup(self) -> None:
        """Instantiate enabled providers; eagerly call describe()."""
        for provider_id in self._settings.providers.enabled:
            factory = BUILTIN_PROVIDERS.get(provider_id)
            if factory is None:
                logger.warning("Provider %r is enabled in config but not registered; skipping.", provider_id)
                continue

            opts = self._settings.providers.provider_options(provider_id)
            try:
                instance = factory(opts) if _factory_takes_arg(factory) else factory()
            except Exception as exc:
                logger.exception("Failed to instantiate provider %r: %s", provider_id, exc)
                if provider_id in self._settings.providers.required:
                    raise
                continue

            caps = await instance.describe()
            concurrency_limit = int(opts.get("concurrency", 1 if caps.is_gpu else 16))
            entry = ProviderEntry(
                id=provider_id,
                instance=instance,
                capabilities=caps,
                concurrency=make_controller(concurrency_limit),
                required=provider_id in self._settings.providers.required,
            )
            self._entries[provider_id] = entry
            logger.info("Registered provider %r (family=%s, languages=%s)", provider_id, caps.provider_family, caps.languages)

    async def shutdown(self) -> None:
        """Tear down providers, calling each one's `terminate()` if it has one.

        Failures are logged, not raised — shutdown must always complete.
        """
        for entry in self._entries.values():
            terminate = getattr(entry.instance, "terminate", None)
            if terminate is None:
                continue
            try:
                result = terminate()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning(
                    "Provider %r terminate() failed during shutdown: %s",
                    entry.id, exc,
                )
        self._entries.clear()

    # ---- access ----

    def list_entries(self) -> Iterable[ProviderEntry]:
        return self._entries.values()

    def get(self, provider_id: str) -> ProviderEntry:
        entry = self._entries.get(provider_id)
        if entry is None:
            raise UnknownModel(f"Unknown or disabled model: {provider_id!r}")
        return entry

    async def ensure_loaded(self, entry: ProviderEntry) -> None:
        """Idempotent provider warm-up. Concurrent callers serialize on load_lock."""
        if entry.loaded:
            return
        async with entry.load_lock:
            if entry.loaded:
                return
            try:
                await entry.instance.load()
            except Exception as exc:
                entry.load_error = str(exc)
                logger.exception("Provider %r failed to load: %s", entry.id, exc)
                raise ProviderUnavailable(f"Provider {entry.id!r} failed to load: {exc}") from exc
            entry.loaded = True
            logger.info("Provider %r loaded", entry.id)

    def readiness(self) -> dict[str, Any]:
        """Snapshot of provider load state for /readyz."""
        return {
            "providers": {
                entry.id: {
                    "loaded": entry.loaded,
                    "required": entry.required,
                    "load_error": entry.load_error,
                }
                for entry in self._entries.values()
            }
        }


def _factory_takes_arg(factory: Any) -> bool:
    """Cheap heuristic: classes with __init__ taking >0 positional args, or lambdas."""
    import inspect

    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"]
    return len(params) >= 1
