from __future__ import annotations

import os

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Attribute, Base, Document, Page


def test_attribute_cannot_reference_page_from_another_document() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    session = session_factory()
    first = Document(name="first.pdf", format="pdf", status="PROCESSED")
    second = Document(name="second.pdf", format="pdf", status="PROCESSED")
    session.add_all([first, second])
    session.flush()
    page = Page(document_id=first.id, page_number=1, image_path="page-001.png")
    session.add(page)
    session.flush()

    session.add(
        Attribute(
            page_id=page.id,
            document_id=second.id,
            entity_type_label="TOTAL",
            extracted_value="10.00",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.close()


def test_users_cannot_access_each_others_documents(monkeypatch) -> None:
    secret = "test-secret"
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    session = session_factory()
    first = Document(name="first.pdf", format="pdf", owner_id="user-1", status="PROCESSED")
    second = Document(name="second.pdf", format="pdf", owner_id="user-2", status="PROCESSED")
    session.add_all([first, second])
    session.commit()
    session.refresh(first)
    session.refresh(second)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    token = lambda user_id: jwt.encode({"sub": user_id, "aud": "authenticated"}, secret, algorithm="HS256")
    try:
        first_headers = {"Authorization": f"Bearer {token('user-1')}"}
        second_headers = {"Authorization": f"Bearer {token('user-2')}"}
        limited_response = client.get("/api/v1/documents?limit=1", headers=first_headers)
        assert limited_response.status_code == 200
        assert len(limited_response.json()["documents"]) == 1
        assert client.get("/api/v1/documents", headers=first_headers).json()["documents"][0]["name"] == "first.pdf"
        assert client.get("/api/v1/documents", headers=second_headers).json()["documents"][0]["name"] == "second.pdf"
        assert client.get(f"/api/v1/documents/{second.id}", headers=first_headers).status_code == 404
        assert client.get(f"/api/v1/documents/{first.id}", headers=second_headers).status_code == 404
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_raw_page_directory_is_not_public() -> None:
    client = TestClient(app)
    response = client.get("/pages/1/page-001.png")
    assert response.status_code == 404
