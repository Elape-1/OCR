from __future__ import annotations

from fastapi.testclient import TestClient

from app.auth import auth_required, owner_scope
from app.main import app
from app.models import Document


def test_authentication_and_owner_scope_default_to_secure(monkeypatch) -> None:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    assert auth_required() is True
    assert "IS NULL" not in str(owner_scope(Document.owner_id, "user-1"))


def test_protected_routes_reject_missing_and_invalid_tokens(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")
    client = TestClient(app)
    missing = client.get("/api/v1/documents")
    invalid = client.get("/api/v1/documents", headers={"Authorization": "Bearer invalid"})
    assert missing.status_code == 401
    assert invalid.status_code == 401