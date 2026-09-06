from __future__ import annotations

from pathlib import Path

from app.ingestion_pipeline import _store_raw_dataset_pages


def test_store_raw_dataset_pages_creates_document_folder(tmp_path, monkeypatch) -> None:
    source_one = tmp_path / "asset-1.png"
    source_two = tmp_path / "asset-2.png"
    source_one.write_bytes(b"page one")
    source_two.write_bytes(b"page two")
    dataset_root = tmp_path / "datasets" / "raw-pages"
    monkeypatch.setattr("app.ingestion_pipeline.RAW_DATASET_ROOT", dataset_root)

    stored = _store_raw_dataset_pages([source_one, source_two], document_id=42)

    assert stored == [
        dataset_root / "42" / "page-001.png",
        dataset_root / "42" / "page-002.png",
    ]
    assert stored[0].read_bytes() == b"page one"
    assert stored[1].read_bytes() == b"page two"
