from __future__ import annotations

import mimetypes
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - dependency is required in deployment
    Client = Any  # type: ignore[misc,assignment]
    create_client = None


def _client() -> Client | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key or create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception as exc:
        logger.warning("Supabase Storage is unavailable; using local file storage: %s", exc)
        return None


def storage_enabled() -> bool:
    return _client() is not None


def upload_file(path: str | Path, object_key: str) -> None:
    client = _client()
    if client is None:
        return
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "ocr-documents")
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    with Path(path).open("rb") as handle:
        client.storage.from_(bucket).upload(
            object_key,
            handle,
            {"upsert": "true", "content-type": content_type},
        )


def download_file(object_key: str) -> bytes:
    client = _client()
    if client is None:
        raise RuntimeError("Supabase Storage is not configured")
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "ocr-documents")
    return bytes(client.storage.from_(bucket).download(object_key))


def delete_file(object_key: str) -> None:
    client = _client()
    if client is None:
        return
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "ocr-documents")
    client.storage.from_(bucket).remove([object_key])
