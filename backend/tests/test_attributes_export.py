from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Attribute, Base, Document, Page


def _build_test_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)()


def test_export_validated_attributes_json_and_csv() -> None:
    session = _build_test_session()
    document = Document(name="test.pdf", format="pdf", page_count=1, status="PROCESSED")
    session.add(document)
    session.flush()

    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()

    session.add_all(
        [
            Attribute(
                page_id=page.id,
                document_id=document.id,
                entity_type_label="TOTAL",
                extracted_value="42.00",
                bounding_boxes=[[0, 0, 10, 10]],
                confidence_score=0.97,
                validation_status="APPROVED",
            ),
            Attribute(
                page_id=page.id,
                document_id=document.id,
                entity_type_label="IGNORED",
                extracted_value="pending",
                bounding_boxes=[[1, 1, 2, 2]],
                confidence_score=0.12,
                validation_status="PENDING",
            ),
        ]
    )
    session.commit()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        json_response = client.get(f"/api/v1/attributes/export/{document.id}/json")
        assert json_response.status_code == 200
        payload = json_response.json()
        assert payload["exported_count"] == 1
        assert payload["rows"][0]["entity_type_label"] == "TOTAL"

        csv_response = client.get(f"/api/v1/attributes/export/{document.id}/csv")
        assert csv_response.status_code == 200
        assert csv_response.headers["content-type"].startswith("text/csv")
        assert "TOTAL" in csv_response.text
        assert "PENDING" not in csv_response.text
    finally:
        app.dependency_overrides.clear()
        session.close()


