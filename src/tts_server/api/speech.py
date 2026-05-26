"""POST /v1/audio/speech — the synthesis endpoint."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
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
from tts_server.core.transcode import (
    TranscoderError,
    TranscoderUnavailable,
    transcode,
)
from tts_server.providers.base import SynthesisRequest, SynthesisStream
from tts_server.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech"])


class SpeechRequest(BaseModel):
    """Request body for ``POST /v1/audio/speech``.

    Defaults are chosen for maximum fidelity with the current providers
    (Qwen3-TTS, StyleTTS2-UK, edge-tts — all native 24 kHz):

    * ``response_format = "wav"`` — lossless container, no codec noise.
    * ``sample_rate = 24000`` — matches the providers' native rate so no
      ffmpeg resampling happens on the hot path.

    Override either when you'd rather have a smaller payload (``mp3``,
    or a lower ``sample_rate`` like ``16000``). Asking for ``48000``
    works but is pure interpolation — the model has no information
    above 24 kHz.
    """
    input: str = Field(..., min_length=1, description="Text to synthesize")
    model: str = Field(default="auto", description="Provider id, or 'auto' to route by language")
    language: str = Field(default="en", description="BCP-47 tag, e.g. en, en-US, uk")
    voice: str | None = Field(default=None, description="Voice id or 'ref:<id>' for cloning")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal["wav", "mp3"] = Field(
        default="wav",
        description="Audio container. 'wav' is lossless; 'mp3' for smaller payloads.",
    )
    sample_rate: int = Field(
        default=24000,
        ge=8000,
        le=48000,
        description=(
            "Output sample rate in Hz. Default 24000 matches every "
            "current provider's native rate (no resampling). Values "
            "above 24000 only upsample — no extra fidelity from the model."
        ),
    )
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
        resolved_model, route_reason = resolve_model(
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
        "X-Route-Reason": route_reason,
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

    # Admission control: refuse the request before it joins the queue if
    # there's already no in-flight headroom AND the queue is at its cap.
    ctl = entry.concurrency
    if ctl.in_flight() >= ctl.limit and ctl.queue_depth() >= settings.server.max_queue_depth:
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

    # Whole-synthesis timeout (load → semaphore wait → synth → chunk drain).
    # Without this, a slow provider stream could hold the GPU lane silently.
    timeout = settings.server.request_timeout_seconds
    try:
        async with entry.concurrency:
            stream, audio_bytes = await asyncio.wait_for(
                _drain_synthesis(entry, synth_req),
                timeout=timeout,
            )
    except asyncio.TimeoutError as exc:
        raise _http_from_tts(ProviderUnavailable("Synthesis timed out")) from exc
    except TTSError as exc:
        raise _http_from_tts(exc) from exc
    except Exception as exc:
        logger.exception("Provider %r raised during synthesize: %s", entry.id, exc)
        raise _http_from_tts(ProviderFailure(str(exc))) from exc

    # Server-side transcoding: honor body.response_format and sample_rate.
    # If both match the provider's output, this is a fast no-op.
    final_audio, final_format, final_sample_rate, final_duration_ms = await _maybe_transcode(
        audio=audio_bytes,
        source_format=stream.format,
        source_sample_rate=stream.sample_rate,
        source_duration_ms=stream.duration_ms,
        target_format=body.response_format,
        target_sample_rate=body.sample_rate,
    )

    cached_value = CachedAudio(
        audio_bytes=final_audio,
        sample_rate=final_sample_rate,
        format=final_format,
        duration_ms=final_duration_ms,
    )
    cache.put(cache_key, cached_value)
    return _build_response(envelope, cached_value, headers_meta, request_id, caps, entry.id)


async def _drain_synthesis(
    entry: ProviderEntry, synth_req: SynthesisRequest
) -> tuple[SynthesisStream, bytes]:
    """Run synthesis and fully drain the chunk iterator into bytes."""
    stream = await entry.instance.synthesize(synth_req)
    audio_bytes = b"".join([chunk async for chunk in stream.chunks])
    return stream, audio_bytes


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
    headers_meta["X-Audio-Format"] = audio.format

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

    media_type = _media_type_for(audio.format)
    # For fully-materialized bodies, plain Response is cheaper than StreamingResponse.
    return Response(content=audio.audio_bytes, media_type=media_type, headers=headers_meta)


def _media_type_for(fmt: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "opus": "audio/opus",
        "flac": "audio/flac",
    }.get(fmt, "application/octet-stream")


async def _maybe_transcode(
    *,
    audio: bytes,
    source_format: str,
    source_sample_rate: int,
    source_duration_ms: int,
    target_format: str,
    target_sample_rate: int | None,
) -> tuple[bytes, str, int, int]:
    """Run ffmpeg only when needed; otherwise return source unchanged."""
    needs_format = target_format != source_format
    needs_resample = bool(target_sample_rate and target_sample_rate != source_sample_rate)
    if not needs_format and not needs_resample:
        return audio, source_format, source_sample_rate, source_duration_ms

    try:
        result = await transcode(
            audio=audio,
            source_format=source_format,
            target_format=target_format,
            target_sample_rate=target_sample_rate,
        )
    except TranscoderUnavailable as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "transcoder_unavailable", "message": str(exc)}},
        ) from exc
    except TranscoderError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": {"code": "transcoder_failed", "message": str(exc)}},
        ) from exc

    new_rate = target_sample_rate or source_sample_rate
    # Transcoding preserves wall-clock duration regardless of rate / format.
    return result.audio, target_format, new_rate, source_duration_ms


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

        local_path = ref_store.resolve(voice)
        if local_path is None:
            raise UnknownVoice(
                f"Unknown ref voice {voice!r} (not in catalog and no matching upload)"
            )

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


def _http_from_tts(exc: TTSError) -> HTTPException:
    headers = {"Retry-After": "5"} if exc.http_status == 503 else None
    return HTTPException(
        status_code=exc.http_status,
        detail={"error": {"code": exc.code, "message": exc.message}},
        headers=headers,
    )
