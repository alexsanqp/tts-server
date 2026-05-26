"""POST /v1/refs — upload reference audio for voice cloning."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from tts_server.core.auth import require_bearer_token
from tts_server.core.refs import RefStore, RefStoreError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["refs"])


@router.post("/refs", dependencies=[Depends(require_bearer_token)])
async def upload_ref(request: Request, file: UploadFile = File(...)) -> dict:
    ref_store: RefStore = request.app.state.ref_store
    content = await file.read()
    try:
        stored = await ref_store.store(
            content=content,
            filename=file.filename,
            content_type=file.content_type,
        )
    except RefStoreError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "invalid_upload", "message": str(exc)}},
        ) from exc

    logger.info("Stored ref %s (%d bytes)", stored.ref_id, stored.size_bytes)
    return {
        "id": stored.ref_id,
        "size_bytes": stored.size_bytes,
    }


@router.get("/refs/catalog")
async def list_catalog(request: Request) -> dict:
    """Read-only listing of the baked-in ref catalog (no upload TTL)."""
    ref_store: RefStore = request.app.state.ref_store
    return {"ids": ref_store.catalog_ids()}
