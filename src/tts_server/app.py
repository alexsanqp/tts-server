"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tts_server.api import health, models, refs, routing, speech
from tts_server.core.cache import SynthesisCache
from tts_server.core.probing import probe_provider
from tts_server.core.refs import RefStore
from tts_server.core.registry import ProviderRegistry
from tts_server.settings import Settings, load_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    registry: ProviderRegistry = app.state.registry
    ref_store: RefStore = app.state.ref_store

    logger.info("Starting tts-server with providers=%s", settings.providers.enabled)
    await registry.startup()

    # Probe voices in the background — don't block startup. Edge can be slow.
    probe_task = asyncio.create_task(_background_probe(registry))

    # TTL sweep for ref uploads.
    await ref_store.start_background_sweep()

    try:
        yield
    finally:
        probe_task.cancel()
        try:
            await probe_task
        except asyncio.CancelledError:
            pass
        await ref_store.stop()
        await registry.shutdown()
        logger.info("tts-server stopped")


async def _background_probe(registry: ProviderRegistry) -> None:
    for entry in registry.list_entries():
        try:
            await probe_provider(entry)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Voice probing for %r failed: %s", entry.id, exc)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    registry = ProviderRegistry(settings)
    ref_store = RefStore(
        catalog_dir=settings.refs.catalog_dir,
        upload_dir=settings.refs.upload_dir,
        upload_ttl_hours=settings.refs.upload_ttl_hours,
        max_upload_mb=settings.refs.max_upload_mb,
    )
    cache = SynthesisCache(enabled=settings.cache.enabled, max_entries=settings.cache.max_entries)

    app = FastAPI(
        title="tts-server",
        version="0.1.0",
        description="Self-hosted pluggable TTS HTTP service",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.ref_store = ref_store
    app.state.cache = cache

    app.include_router(health.router)
    app.include_router(models.router, prefix="/v1")
    app.include_router(speech.router, prefix="/v1")
    app.include_router(refs.router, prefix="/v1")
    app.include_router(routing.router, prefix="/v1")

    return app
