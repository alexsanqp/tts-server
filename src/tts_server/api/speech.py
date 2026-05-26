"""POST /v1/audio/speech — the synthesis endpoint."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tts_server.api.routing import resolve_model
from tts_server.core.auth import optional_bearer_token
from tts_server.core.cache import CachedAudio, SynthesisCache
from tts_server.core.errors import (
    CapacityExceeded,
    InputTooLong,
    ProviderFailure,
    ProviderUnavailable,
    TTSError,
    UnknownVoice,
)
from tts_server.core.refs import RefStore
from tts_server.core.registry import ProviderEntry, ProviderRegistry
from tts_server.providers.base import SynthesisRequest
from tts_server.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech"])


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Text to synthesize")
    model: str = Field(default="auto")
    language: str = Field(default="en", description="BCP-47 tag, e.g. en, en-US, uk")
    voice: str | None = Field(default=None, description="Voice id or 'ref:<id>' for cloning")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal["wav", "mp3"] = Field(default="wav")
    sample_rate: int | None = Field(default=None, ge=8000, le=48000)
    idempotency_key: str | None = Field(default=None)


@router.post("/audio/speech", dependencies=[Depends(optional_bearer_token)])
async def create_speech(
    request: Request,
    body: SpeechRequest,
    envelope: str | None = None,
) -> Response:
    settings: Settings = request.app.state.settings
    registry: ProviderRegistry = request.app.state.registry
    ref_store: RefStore = request.app.state.ref_store
    cache: SynthesisCache = request.app.state.cache

    try:
        resolved_model, _reason = resolve_model(
            settings, registry, model=body.model, language=body.language
        )
    except TTSError as exc:
        raise _http_from_tts(exc) from exc

    entry = registry.get(resolved_model)
    caps = entry.capabilities

    if len(body.input) > caps.max_text_length:
        raise _http_from_tts(
            InputTooLong(
                f"Input length {len(body.input)} exceeds max {caps.max_text_length} for model {entry.id!r}"
            )
        )

    try:
        voice_id, voice_kind, ref_text = _resolve_voice(body.voice, caps, ref_store)
    except TTSError as exc:
        raise _http_from_tts(exc) from exc

    request_id = uuid.uuid4().hex
    headers_meta = {
        "X-Request-Id": request_id,
        "X-TTS-Provider": caps.provider_family,
        "X-TTS-Model": entry.id,
    }

    cache_key = cache.make_key(
        text=body.input,
        model=entry.id,
        voice=voice_id,
        language=body.language,
        speed=body.speed,
        response_format=body.response_format,
        sample_rate=body.sample_rate,
        idempotency_key=body.idempotency_key,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        headers_meta["X-Cache"] = "hit"
        return _build_response(envelope, cached, headers_meta, request_id, caps, entry.id)

    headers_meta["X-Cache"] = "miss"

    if entry.concurrency.semaphore.locked() and _queue_depth(entry) >= settings.server.max_queue_depth:
        raise _http_from_tts(CapacityExceeded("Server is at capacity, retry later"))

    try:
        await asyncio.wait_for(
            registry.ensure_loaded(entry),
            timeout=settings.server.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise _http_from_tts(ProviderUnavailable("Provider load timed out")) from exc
    except TTSError as exc:
        raise _http_from_tts(exc) from exc

    synth_req = SynthesisRequest(
        text=body.input,
        language=body.language,
        voice=voice_id,
        voice_kind=voice_kind,
        ref_text=ref_text,
        speed=body.speed,
        target_sample_rate=body.sample_rate,
        target_format=body.response_format,
    )

    try:
        async with entry.concurrency.semaphore:
            stream = await asyncio.wait_for(
                entry.instance.synthesize(synth_req),
                timeout=settings.server.request_timeout_seconds,
            )
            audio_bytes = b"".join([chunk async for chunk in stream.chunks])
    except asyncio.TimeoutError as exc:
        raise _http_from_tts(ProviderUnavailable("Synthesis timed out")) from exc
    except TTSError as exc:
        raise _http_from_tts(exc) from exc
    except Exception as exc:
        logger.exception("Provider %r raised during synthesize: %s", entry.id, exc)
        raise _http_from_tts(ProviderFailure(str(exc))) from exc

    cached_value = CachedAudio(
        audio_bytes=audio_bytes,
        sample_rate=stream.sample_rate,
        format=stream.format,
        duration_ms=stream.duration_ms,
    )
    cache.put(cache_key, cached_value)
    return _build_response(envelope, cached_value, headers_meta, request_id, caps, entry.id)


def _build_response(
    envelope: str | None,
    audio: CachedAudio,
    headers_meta: dict[str, str],
    request_id: str,
    caps,
    model_id: str,
) -> Response:
    headers_meta = dict(headers_meta)
    headers_meta["X-Sample-Rate"] = str(audio.sample_rate)
    headers_meta["X-Duration-Ms"] = str(audio.duration_ms)

    if envelope == "json":
        return JSONResponse(
            {
                "audio_base64": base64.b64encode(audio.audio_bytes).decode("ascii"),
                "format": audio.format,
                "sample_rate": audio.sample_rate,
                "duration_ms": audio.duration_ms,
                "provider": caps.provider_family,
                "model": model_id,
                "request_id": request_id,
            },
            headers=headers_meta,
        )

    media_type = "audio/wav" if audio.format == "wav" else "audio/mpeg"

    async def _one_chunk():
        yield audio.audio_bytes

    return StreamingResponse(_one_chunk(), media_type=media_type, headers=headers_meta)


def _resolve_voice(
    voice: str | None, caps, ref_store: RefStore
) -> tuple[str | None, str, str | None]:
    """Resolve the `voice` field into (id_or_path, kind, ref_text).

    Three cases:
    1. None — no voice specified; provider's default applies.
    2. "ref:<id>" — voice cloning. Resolve against RefStore; pull ref_text
       from catalog VoiceInfo metadata if known.
    3. Plain id — must be in the provider's catalog AND accept voice ids.
    """
    if voice is None:
        return None, "none", None

    if voice.startswith("ref:"):
        if not caps.supports_voice_cloning:
            raise UnknownVoice("This model does not support voice cloning")

        # Look up resolved local path via RefStore.
        local_path = ref_store.resolve(voice)
        if local_path is None:
            raise UnknownVoice(f"Unknown ref voice {voice!r} (not in catalog and no matching upload)")

        # If this ref is a curated catalog voice, pull its ref_text from VoiceInfo metadata.
        ref_text = None
        for v in caps.voices:
            if v.id == voice and v.accepts_clone_ref:
                ref_text = v.metadata.get("ref_text")
                break

        return str(local_path), "clone_ref", ref_text

    if not caps.supports_voice_id:
        raise UnknownVoice("Model does not accept voice ids; pass 'ref:<id>' instead")
    for v in caps.voices:
        if v.id == voice and v.accepts_voice_id:
            return voice, "id", None
    raise UnknownVoice(f"Unknown voice {voice!r} for this model")


def _queue_depth(entry: ProviderEntry) -> int:
    sem = entry.concurrency.semaphore
    waiters = getattr(sem, "_waiters", None)
    return len(waiters) if waiters else 0


def _http_from_tts(exc: TTSError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={"error": {"code": exc.code, "message": exc.message}},
    )
