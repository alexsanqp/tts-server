"""GET /v1/route — preview language→model routing without synthesizing."""

from __future__ import annotations

from fastapi import APIRouter, Request

from tts_server.core.errors import UnknownModel
from tts_server.core.registry import ProviderRegistry
from tts_server.settings import Settings

router = APIRouter(tags=["routing"])


def resolve_model(settings: Settings, registry: ProviderRegistry, *, model: str, language: str) -> tuple[str, str]:
    """Return (resolved_model_id, reason).

    `model` may be "auto" (or empty) to trigger language-based routing,
    or a concrete provider id. Routing table lives in [routing.by_language].
    """
    if model and model != "auto":
        registry.get(model)  # raises UnknownModel if missing
        return model, "explicit"

    primary = language.split("-")[0].lower() if language else ""
    by_lang = settings.routing.by_language
    if primary in by_lang:
        resolved = by_lang[primary]
        registry.get(resolved)
        return resolved, f"by_language[{primary}]"

    default = settings.routing.default
    if not default:
        raise UnknownModel(f"No route for language={language!r} and no default configured")
    registry.get(default)
    return default, "default"


@router.get("/route")
async def preview_route(request: Request, language: str = "", model: str = "auto") -> dict:
    settings: Settings = request.app.state.settings
    registry: ProviderRegistry = request.app.state.registry
    resolved, reason = resolve_model(settings, registry, model=model, language=language)
    return {"resolved_model": resolved, "reason": reason, "input": {"language": language, "model": model}}
