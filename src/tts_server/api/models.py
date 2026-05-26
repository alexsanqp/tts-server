"""GET /v1/models, GET /v1/voices — capability introspection.

Both endpoints gate behind :func:`optional_bearer_token`: when the
server runs without a configured token they're fully public (intra-LAN
dev), but the moment an operator sets one, the introspection surface is
locked down too — otherwise the routing/voice catalog leaks even though
synthesis itself requires the token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tts_server.core.auth import optional_bearer_token
from tts_server.core.probing import voice_available
from tts_server.core.registry import ProviderRegistry

router = APIRouter(tags=["models"])


@router.get("/models", dependencies=[Depends(optional_bearer_token)])
async def list_models(request: Request) -> dict:
    registry: ProviderRegistry = request.app.state.registry
    items = []
    for entry in registry.list_entries():
        caps = entry.capabilities
        items.append(
            {
                "id": entry.id,
                "provider": caps.provider_family,
                "languages": list(caps.languages),
                "supports_voice_id": caps.supports_voice_id,
                "supports_voice_cloning": caps.supports_voice_cloning,
                "native_sample_rate": caps.native_sample_rate,
                "native_format": caps.native_format,
                "max_text_length": caps.max_text_length,
                "accepts_speed": caps.accepts_speed,
                "loaded": entry.loaded,
                "voices_endpoint": f"/v1/voices?model={entry.id}",
            }
        )
    return {"models": items}


@router.get("/voices", dependencies=[Depends(optional_bearer_token)])
async def list_voices(
    request: Request,
    model: str | None = None,
    language: str | None = None,
    include_unavailable: bool = False,
) -> dict:
    registry: ProviderRegistry = request.app.state.registry
    voices = []
    for entry in registry.list_entries():
        if model and entry.id != model:
            continue
        for v in entry.capabilities.voices:
            if language and language.split("-")[0] not in {l.split("-")[0] for l in v.languages}:
                continue
            available = voice_available(entry, v.id)
            if not available and not include_unavailable:
                continue
            voices.append(
                {
                    "id": v.id,
                    "model": entry.id,
                    "languages": list(v.languages),
                    "gender": v.gender,
                    "accepts_voice_id": v.accepts_voice_id,
                    "accepts_clone_ref": v.accepts_clone_ref,
                    "available": available,
                    "metadata": v.metadata,
                }
            )
    return {"voices": voices}
