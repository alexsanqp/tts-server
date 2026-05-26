"""POST /v1/refs — upload reference audio for voice cloning.

The body is streamed in 64 KiB chunks so an oversize payload aborts as
soon as it crosses ``refs.max_upload_mb`` instead of buffering the whole
file in RAM (an unauthenticated 10 GB POST would otherwise OOM uvicorn
even before any size check ran).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from tts_server.core.auth import optional_bearer_token, require_bearer_token
from tts_server.core.refs import RefStore, RefStoreError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["refs"])

# 64 KiB matches the default Starlette UploadFile chunk and keeps memory
# bounded even when many concurrent uploads share the worker.
_UPLOAD_CHUNK_SIZE = 64 * 1024


@router.post("/refs", dependencies=[Depends(require_bearer_token)])
async def upload_ref(request: Request, file: UploadFile = File(...)) -> dict:
    ref_store: RefStore = request.app.state.ref_store
    max_bytes = ref_store.max_upload_bytes

    # Fast-fail on Content-Length when the client honestly declares an
    # oversize payload. Missing / bogus header just falls through to the
    # streaming check below.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "code": "payload_too_large",
                    "message": f"upload exceeds {max_bytes // 1024 // 1024} MB limit",
                    "max_bytes": max_bytes,
                    "received_bytes": int(declared),
                }
            },
        )

    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        received += len(chunk)
        if received > max_bytes:
            # Drain and abort: stop accumulating, return 413 immediately.
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "code": "payload_too_large",
                        "message": f"upload exceeds {max_bytes // 1024 // 1024} MB limit",
                        "max_bytes": max_bytes,
                        "received_bytes": received,
                    }
                },
            )
        chunks.append(chunk)

    content = b"".join(chunks)
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


@router.get("/refs/catalog", dependencies=[Depends(optional_bearer_token)])
async def list_catalog(request: Request) -> dict:
    """Read-only listing of the baked-in ref catalog (no upload TTL).

    Gated by :func:`optional_bearer_token` so that when an operator
    configures ``auth_token`` the catalog ids are locked down alongside
    synthesis — listing them without auth would leak voice inventory.
    """
    ref_store: RefStore = request.app.state.ref_store
    return {"ids": ref_store.catalog_ids()}
