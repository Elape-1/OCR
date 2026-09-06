from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Attribute, Base, Document, Page



def _build_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def _override_get_db(session):
    def _dependency():
        try:
            yield session
        finally:
            pass

    return _dependency


def make_document(db, name="search-test-doc", doc_type="invoice", status="PROCESSED"):
    doc = Document(name=name, format="pdf", document_type=doc_type, page_count=1, status=status)
    db.add(doc)
    db.flush()
    page = Page(document_id=doc.id, page_number=1, image_path=f"assets/{doc.id}/1.png")
    db.add(page)
    db.flush()
    return doc, page


def test_search_documents_basic():
    factory = _build_session_factory()
    db = factory()
    try:
        doc, page = make_document(db, name="unique-search-doc-123", doc_type="report", status="PROCESSED")
        db.commit()
    finally:
        db.close()

    session = factory()
    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        resp = client.get("/api/v1/search/documents", params={"q": "unique-search-doc-123", "page": 1, "page_size": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data and data["total"] >= 1
        assert any(d["name"] == "unique-search-doc-123" for d in data.get("documents", []))
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_search_attributes_basic():
    factory = _build_session_factory()
    db = factory()
    try:
        doc, page = make_document(db, name="attr-doc-1", doc_type="invoice", status="PROCESSED")
        attr = Attribute(page_id=page.id, document_id=doc.id, entity_type_label="Invoice_Total", extracted_value="123.45", bounding_boxes=[], confidence_score=0.92, validation_status="APPROVED")
        db.add(attr)
        db.commit()
    finally:
        db.close()

    session = factory()
    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        resp = client.get("/api/v1/search/attributes", params={"q": "Invoice_Total", "page": 1, "page_size": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data and data["total"] >= 1
        assert any(a["entity_type_label"] == "Invoice_Total" for a in data.get("attributes", []))
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_search_documents_explicit_statuses() -> None:
    factory = _build_session_factory()
    db = factory()
    try:
        document_ids = {
            status: make_document(db, name=f"{status.lower()}-doc", status=status)[0].id
            for status in ("QUEUED", "PROCESSING", "FAILED")
        }
        db.commit()
    finally:
        db.close()

    session = factory()
    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        for status, document_id in document_ids.items():
            response = client.get("/api/v1/search/documents", params={"status": status})
            assert response.status_code == 200
            payload = response.json()
            assert payload["total"] == 1
            assert payload["documents"][0]["id"] == document_id
            assert payload["documents"][0]["status"] == status
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_search_documents_end_date_includes_the_entire_day() -> None:
    factory = _build_session_factory()
    db = factory()
    try:
        included, _ = make_document(db, name="included.pdf", status="PROCESSED")
        included.timestamp = datetime(2026, 9, 5, 23, 59, 59)
        excluded, _ = make_document(db, name="excluded.pdf", status="PROCESSED")
        excluded.timestamp = datetime(2026, 9, 6, 0, 0, 0)
        db.commit()
    finally:
        db.close()

    session = factory()
    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        response = client.get("/api/v1/search/documents", params={"end_date": "2026-09-05"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["documents"][0]["name"] == "included.pdf"
    finally:
        app.dependency_overrides.clear()
        session.close()
