"""Bearer-token auth helpers.

* If `settings.server.auth_token` is empty, the optional dependency lets
  any request through (intra-LAN default).
* `require_bearer_token` always enforces — used by /v1/refs which is the
  only write surface and never runs anonymously.

Token comparison uses :func:`hmac.compare_digest` so the time taken to
reject a wrong token does not leak its prefix length over the network.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request

from tts_server.settings import Settings


def _configured_token(request: Request) -> str:
    settings: Settings = request.app.state.settings
    return settings.server.auth_token or ""


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def _matches(token: str | None, configured: str) -> bool:
    """Constant-time equality check; rejects empty/missing token upfront."""
    if not token or not configured:
        return False
    return hmac.compare_digest(token, configured)


def optional_bearer_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Allow anonymous when no token is configured; otherwise enforce."""
    configured = _configured_token(request)
    if not configured:
        return
    token = _extract_token(authorization)
    if not _matches(token, configured):
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "unauthorized", "message": "invalid or missing bearer token"}},
        )


def require_bearer_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Always enforce. Used for write endpoints (uploads)."""
    configured = _configured_token(request)
    if not configured:
        raise HTTPException(
            status_code=403,
            detail={"error": {"code": "auth_not_configured", "message": "this endpoint requires server.auth_token to be configured"}},
        )
    token = _extract_token(authorization)
    if not _matches(token, configured):
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "unauthorized", "message": "invalid or missing bearer token"}},
        )
