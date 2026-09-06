from __future__ import annotations

import os
import re
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import ColumnElement

try:
    import jwt
except ImportError:  # pragma: no cover - dependency is required in deployment
    jwt = None

bearer_scheme = HTTPBearer(auto_error=False)
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}


def owner_scope(column: Any, user_id: str) -> ColumnElement[bool]:
    return column == user_id


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    device_id: Annotated[str | None, Header(alias="X-Device-ID")] = None,
    query_device_id: Annotated[str | None, Query(alias="device_id")] = None,
) -> str:
    if credentials is None:
        if auth_required():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        candidate = device_id or query_device_id
        if candidate:
            if not DEVICE_ID_PATTERN.fullmatch(candidate):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid device id")
            return f"device:{candidate}"
        return os.getenv("DEV_USER_ID", "local-development-user")

    if jwt is None:
        raise HTTPException(status_code=500, detail="JWT authentication dependency is unavailable")

    token = credentials.credentials
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET is not configured")
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication token has no user id")
    return user_id


CurrentUserId = Annotated[str, Depends(get_current_user_id)]
