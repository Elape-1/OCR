from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import ingestion_pipeline
from app.db import get_db
from app.main import app
from app.models import Attribute, Base, Correction, Document, Page


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


def test_document_status_endpoint_validates_transitions_and_prerequisites() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(name="status.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    session.add(document)
    session.commit()
    document_id = document.id

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        assert client.patch(f"/api/v1/documents/{document_id}/status?status=unknown").status_code == 400
        assert client.patch(f"/api/v1/documents/{document_id}/status?status=PROCESSED").status_code == 400
        assert client.patch(f"/api/v1/documents/{document_id}/status?status=PROCESSING").status_code == 400
        assert session.get(Document, document_id).status == "QUEUED"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_document_status_endpoint_persists_canceled_state() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(name="cancel.pdf", format="pdf", owner_id="local-development-user", status="QUEUED")
    session.add(document)
    session.commit()
    document_id = document.id

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        response = client.patch(f"/api/v1/documents/{document_id}/status?status=CANCELED")
        assert response.status_code == 200
        assert response.json()["status"] == "CANCELED"
        assert session.get(Document, document_id).status == "CANCELED"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_processing_stages_attributes_without_persisting(tmp_path, monkeypatch) -> None:
    factory = _build_session_factory()
    seed_session = factory()
    document = Document(name="sample.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    seed_session.add(document)
    seed_session.commit()
    document_id = document.id
    seed_session.close()

    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    rasterized_page = tmp_path / "page-1.png"
    rasterized_page.write_bytes(b"fake-png")

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, doc_id: [rasterized_page])
    monkeypatch.setattr(
        ingestion_pipeline,
        "_detect_document_type",
        lambda page_paths, original_name: {"document_type": "invoice"},
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "_build_attribute_payloads_for_page",
        lambda page_id, doc_id, image_path, document_type=None: [
            {
                "entity_type_label": "TOTAL",
                "extracted_value": "123.45",
                "bounding_boxes": [[1, 1, 2, 2]],
                "confidence_score": 0.91,
                "validation_status": "APPROVED",
            }
        ],
    )

    result = ingestion_pipeline.process_document_job(document_id, str(source_path), "sample.pdf", "local-development-user")
    assert result["status"] == "processed"

    verify_session = factory()
    try:
        saved_attributes = verify_session.query(Attribute).filter(Attribute.document_id == document_id).all()
        assert saved_attributes == []

        refreshed_document = verify_session.get(Document, document_id)
        assert refreshed_document is not None
        assert refreshed_document.status == "PROCESSED"
        assert refreshed_document.processing_started_at is not None
        assert refreshed_document.processing_finished_at is not None
        assert refreshed_document.processing_duration_seconds is not None
        assert refreshed_document.processing_duration_seconds >= 0
        assert refreshed_document.extracted_attributes
        assert refreshed_document.extracted_attributes[0]["entity_type_label"] == "TOTAL"
        assert refreshed_document.extracted_attributes[0]["saved"] is False
    finally:
        verify_session.close()


def test_processing_worker_downloads_source_storage_key(tmp_path, monkeypatch) -> None:
    factory = _build_session_factory()
    seed_session = factory()
    document = Document(name="remote.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    seed_session.add(document)
    seed_session.commit()
    document_id = document.id
    seed_session.close()

    rasterized_page = tmp_path / "page-001.png"
    rasterized_page.write_bytes(b"fake-png")
    downloaded_keys = []

    def fake_download(key):
        downloaded_keys.append(key)
        return b"downloaded-pdf"

    def fake_rasterize(path, doc_id):
        assert path.read_bytes() == b"downloaded-pdf"
        return [rasterized_page]

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "storage_enabled", lambda: True)
    monkeypatch.setattr(ingestion_pipeline, "download_file", fake_download)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", fake_rasterize)
    monkeypatch.setattr(
        ingestion_pipeline,
        "_detect_document_type",
        lambda page_paths, original_name: {"document_type": "invoice"},
    )
    monkeypatch.setattr(ingestion_pipeline, "_build_attribute_payloads_for_page", lambda *args, **kwargs: [])
    monkeypatch.setattr(ingestion_pipeline, "upload_file", lambda *args, **kwargs: None)

    result = ingestion_pipeline.process_document_job(
        document_id,
        "owner/document/source.pdf",
        "remote.pdf",
        "local-development-user",
    )

    assert result["status"] == "processed"
    assert downloaded_keys == ["owner/document/source.pdf"]


def test_reprocessing_failure_preserves_previous_success(tmp_path, monkeypatch) -> None:
    factory = _build_session_factory()
    old_page_path = tmp_path / "old-page.png"
    old_page_path.write_bytes(b"old-page")
    new_page_path = tmp_path / "new-page.png"
    new_page_path.write_bytes(b"new-page")
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"source")

    seed_session = factory()
    document = Document(
        name="reprocess.pdf",
        format="pdf",
        owner_id="local-development-user",
        page_count=1,
        status="PROCESSED",
        processing_finished_at=datetime.now(timezone.utc),
        processing_duration_seconds=1.0,
    )
    seed_session.add(document)
    seed_session.flush()
    old_page = Page(document_id=document.id, page_number=1, image_path=str(old_page_path))
    seed_session.add(old_page)
    seed_session.commit()
    document_id = document.id
    old_page_id = old_page.id
    seed_session.close()

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, artifact_id: [new_page_path])
    monkeypatch.setattr(
        ingestion_pipeline,
        "_detect_document_type",
        lambda page_paths, original_name: {"document_type": "invoice"},
    )
    monkeypatch.setattr(ingestion_pipeline, "upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingestion_pipeline,
        "_build_attribute_payloads_for_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        ingestion_pipeline.process_document_job(document_id, str(source_path), "reprocess.pdf", "local-development-user")

    verify_session = factory()
    try:
        refreshed = verify_session.get(Document, document_id)
        assert refreshed.status == "PROCESSED"
        assert refreshed.page_count == 1
        assert verify_session.get(Page, old_page_id) is not None
        assert old_page_path.read_bytes() == b"old-page"
        assert not new_page_path.exists()
    finally:
        verify_session.close()


def test_docx_processing_uses_page_aware_pdf_rasterization(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "multi-page.docx"
    source_path.write_bytes(b"not a real docx")
    rasterized_pages = [tmp_path / "page-001.png", tmp_path / "page-002.png"]
    convert_calls = []

    def fake_convert(path, output_dir):
        convert_calls.append((path, output_dir))
        (output_dir / "multi-page.pdf").write_bytes(b"fake-pdf")

    monkeypatch.setattr(ingestion_pipeline, "_convert_with_libreoffice", fake_convert)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, document_id: rasterized_pages)

    result = ingestion_pipeline._rasterize_office_document(source_path, "document-1")

    assert result == rasterized_pages
    assert len(convert_calls) == 1
    assert convert_calls[0][0] == source_path


def test_docx_conversion_failure_is_explicit(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "unrenderable.docx"
    source_path.write_bytes(b"not a real docx")

    def fail_conversion(path, output_dir):
        raise RuntimeError("LibreOffice unavailable")

    monkeypatch.setattr(ingestion_pipeline, "_convert_with_libreoffice", fail_conversion)

    with pytest.raises(RuntimeError, match="Failed to rasterize office document"):
        ingestion_pipeline._rasterize_office_document(source_path, "document-1")


def test_processing_missing_source_marks_document_failed(tmp_path, monkeypatch) -> None:
    factory = _build_session_factory()
    seed_session = factory()
    document = Document(name="missing.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    seed_session.add(document)
    seed_session.commit()
    document_id = document.id
    seed_session.close()

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)

    with pytest.raises(FileNotFoundError, match="Source file missing"):
        ingestion_pipeline.process_document_job(
            document_id,
            str(tmp_path / "missing.pdf"),
            "missing.pdf",
            "local-development-user",
        )

    verify_session = factory()
    try:
        refreshed_document = verify_session.get(Document, document_id)
        assert refreshed_document is not None
        assert refreshed_document.status == "FAILED"
        assert refreshed_document.processing_started_at is not None
        assert refreshed_document.processing_finished_at is not None
        assert refreshed_document.processing_duration_seconds is not None
        assert refreshed_document.processing_duration_seconds >= 0
    finally:
        verify_session.close()


def test_save_selected_persists_only_requested_attributes_and_manual_entries() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(
        name="review.pdf",
        owner_id="local-development-user",
        format="pdf",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-1",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "TOTAL",
                "extracted_value": "123.45",
                "original_prediction": "123.45",
                "bounding_boxes": [[10, 10, 20, 20]],
                "confidence_score": 0.93,
                "validation_status": "APPROVED",
                "source": "model",
                "saved": False,
            }
        ],
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()
    document.extracted_attributes = [
        {
            **document.extracted_attributes[0],
            "page_id": page.id,
        }
    ]
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [
                    {
                        "temp_id": "model-1",
                        "source": "model",
                        "entity_type_label": "TOTAL",
                        "extracted_value": "150.00",
                        "page_number": 1,
                        "page_id": page.id,
                        "bounding_boxes": [[10, 10, 20, 20]],
                        "confidence_score": 0.93,
                        "validation_status": "APPROVED",
                    },
                    {
                        "temp_id": "manual-1",
                        "source": "manual",
                        "entity_type_label": "NOTES",
                        "extracted_value": "Customer requested paper copy",
                        "page_number": 1,
                        "bounding_boxes": [],
                        "confidence_score": 0.0,
                        "validation_status": "PENDING",
                    },
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["saved_count"] == 2

        saved_attributes = session.query(Attribute).order_by(Attribute.id).all()
        assert len(saved_attributes) == 2
        assert {attribute.source for attribute in saved_attributes} == {"model", "manual"}
        assert session.get(Document, document.id).extracted_attributes == []
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_corrections_still_apply_to_saved_attributes() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(name="corrected.pdf", format="pdf", owner_id="local-development-user", page_count=1, status="PROCESSED")
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()

    attribute = Attribute(
        page_id=page.id,
        document_id=document.id,
        entity_type_label="TOTAL",
        extracted_value="123.45",
        bounding_boxes=[[0, 0, 10, 10]],
        confidence_score=0.88,
        validation_status="APPROVED",
        source="manual",
    )
    session.add(attribute)
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/attributes/correct",
            json={
                "document_id": document.id,
                "corrections": [
                    {
                        "attribute_id": attribute.id,
                        "original_prediction": "123.45",
                        "corrected_value": "150.00",
                        "validation_status": "APPROVED",
                        "operator_user": "tester",
                        "corrected_label": "TOTAL",
                    }
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["correction_count"] == 1

        refreshed_attribute = session.get(Attribute, attribute.id)
        assert refreshed_attribute is not None
        assert refreshed_attribute.extracted_value == "150.00"
        assert refreshed_attribute.source == "manual"
        corrections = session.query(Correction).all()
        assert len(corrections) == 1
        assert corrections[0].operator_user == "local-development-user"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_saved_attributes_are_searchable_but_staged_ones_are_not() -> None:
    factory = _build_session_factory()
    session = factory()
    staged_document = Document(
        owner_id="local-development-user",
        name="staged.pdf",
        format="pdf",
        document_type="invoice",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-1",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "UNSAVED_TOTAL",
                "extracted_value": "999.99",
                "original_prediction": "999.99",
                "bounding_boxes": [[1, 1, 2, 2]],
                "confidence_score": 0.91,
                "validation_status": "APPROVED",
                "source": "model",
                "saved": False,
            }
        ],
    )
    session.add(staged_document)
    session.flush()
    staged_page = Page(document_id=staged_document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(staged_page)
    session.flush()
    session.add(
        Attribute(
            page_id=staged_page.id,
            document_id=staged_document.id,
            entity_type_label="UNSAVED_TOTAL",
            extracted_value="999.99",
            bounding_boxes=[[1, 1, 2, 2]],
            confidence_score=0.91,
            validation_status="PENDING",
            source="model",
        )
    )
    staged_document.extracted_attributes = [
        {
            **staged_document.extracted_attributes[0],
            "page_id": staged_page.id,
        }
    ]

    saved_document = Document(name="saved.pdf", format="pdf", owner_id="local-development-user", document_type="invoice", page_count=1, status="PROCESSED")
    session.add(saved_document)
    session.flush()
    saved_page = Page(document_id=saved_document.id, page_number=1, image_path="datasets/raw-pages/2/page-001.png")
    session.add(saved_page)
    session.flush()
    session.add(
        Attribute(
            page_id=saved_page.id,
            document_id=saved_document.id,
            entity_type_label="SAVED_TOTAL",
            extracted_value="111.11",
            bounding_boxes=[[0, 0, 10, 10]],
            confidence_score=0.95,
            validation_status="APPROVED",
            source="model",
        )
    )
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        staged_response = client.get("/api/v1/search/attributes", params={"q": "UNSAVED_TOTAL"})
        assert staged_response.status_code == 200
        assert staged_response.json()["total"] == 0

        staged_document_response = client.get("/api/v1/search/documents", params={"q": "UNSAVED_TOTAL"})
        assert staged_document_response.status_code == 200
        assert staged_document_response.json()["total"] == 1

        saved_response = client.get("/api/v1/search/attributes", params={"q": "SAVED_TOTAL"})
        assert saved_response.status_code == 200
        assert saved_response.json()["total"] >= 1

        saved_document_response = client.get("/api/v1/search/documents", params={"q": "111.11"})
        assert saved_document_response.status_code == 200
        assert saved_document_response.json()["total"] == 1
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_partial_save_keeps_unselected_staged_attributes() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(
        owner_id="local-development-user",
        name="partial.pdf",
        format="pdf",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-a",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "A",
                "extracted_value": "one",
                "original_prediction": "one",
                "bounding_boxes": [],
                "confidence_score": 0.8,
                "validation_status": "PENDING",
                "source": "model",
                "saved": False,
            },
            {
                "temp_id": "model-b",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "B",
                "extracted_value": "two",
                "original_prediction": "two",
                "bounding_boxes": [],
                "confidence_score": 0.8,
                "validation_status": "PENDING",
                "source": "model",
                "saved": False,
            },
        ],
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()
    document.extracted_attributes = [
        {**document.extracted_attributes[0], "page_id": page.id},
        {**document.extracted_attributes[1], "page_id": page.id},
    ]
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [
                    {
                        "temp_id": "model-a",
                        "source": "model",
                        "entity_type_label": "A",
                        "extracted_value": "one-edited",
                        "page_number": 1,
                        "page_id": page.id,
                        "bounding_boxes": [],
                        "confidence_score": 0.8,
                        "validation_status": "APPROVED",
                    }
                ],
            },
        )
        assert response.status_code == 200

        refreshed_document = session.get(Document, document.id)
        assert refreshed_document is not None
        assert len(refreshed_document.extracted_attributes) == 1
        assert refreshed_document.extracted_attributes[0]["temp_id"] == "model-b"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_save_selected_returns_attribute_details() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(
        owner_id="local-development-user",
        name="details.pdf",
        format="pdf",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-x",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "REF",
                "extracted_value": "ABC",
                "original_prediction": "ABC",
                "bounding_boxes": [[5, 5, 15, 15]],
                "confidence_score": 0.85,
                "validation_status": "PENDING",
                "source": "model",
                "saved": False,
            }
        ],
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()
    document.extracted_attributes = [
        {**document.extracted_attributes[0], "page_id": page.id},
    ]
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [
                    {
                        "temp_id": "model-x",
                        "source": "model",
                        "entity_type_label": "REF",
                        "extracted_value": "XYZ",
                        "page_number": 1,
                        "page_id": page.id,
                        "bounding_boxes": [[5, 5, 15, 15]],
                        "confidence_score": 0.85,
                        "validation_status": "APPROVED",
                    }
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["saved_count"] == 1
        saved_ids = payload.get("saved_attribute_ids") or []
        assert len(saved_ids) == 1

        # Fetch attribute detail and validate values persisted
        attr_id = saved_ids[0]
        detail_resp = client.get(f"/api/v1/attributes/{attr_id}")
        assert detail_resp.status_code == 200
        attr_payload = detail_resp.json()
        assert attr_payload["attribute"]["extracted_value"] == "XYZ"
        assert attr_payload["attribute"]["entity_type_label"] == "REF"

    finally:
        app.dependency_overrides.clear()
        session.close()
