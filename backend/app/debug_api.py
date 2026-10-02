from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.auth import CurrentUserId

router = APIRouter()
MAX_DEBUG_UPLOAD_BYTES = int(os.getenv("DEBUG_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))


@router.post("/debug/echo")
async def echo(payload: dict[str, object], user_id: CurrentUserId) -> dict[str, object]:
    return {"received": payload}


@router.post("/debug/upload_test")
async def upload_test(user_id: CurrentUserId, file: UploadFile = File(...)) -> dict[str, str | int | None]:
    """Accept a single file and return its name and size for debugging browser uploads."""
    size = 0
    chunk_size = 64 * 1024
    while chunk := await file.read(chunk_size):
        size += len(chunk)
        if size > MAX_DEBUG_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Debug upload exceeds the configured size limit")
    return {"filename": file.filename, "size": size}
