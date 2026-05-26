"""Liveness + readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from tts_server.core.registry import ProviderRegistry

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request, response: Response) -> dict:
    registry: ProviderRegistry = request.app.state.registry
    snapshot = registry.readiness()

    # Required providers must be loaded *and* error-free.
    all_required_ok = all(
        info["loaded"] and not info["load_error"]
        for info in snapshot["providers"].values()
        if info["required"]
    )
    if not all_required_ok:
        response.status_code = 503
        snapshot["status"] = "not_ready"
    else:
        snapshot["status"] = "ready"
    return snapshot
