from __future__ import annotations

from app.document_schemas import classify_document_type, normalize_prediction_label


def test_classify_document_type_prefers_invoice_keywords() -> None:
    result = classify_document_type(
        text="Invoice number 1042 total amount due 42.50 bill to Acme Corp",
        filename="acme_invoice.pdf",
        layout_hints={"page_count": 1, "token_count": 18, "line_count": 6},
    )

    assert result["document_type"] == "invoice"
    assert result["confidence"] > 0


def test_normalize_prediction_label_uses_type_specific_aliases() -> None:
    assert normalize_prediction_label("invoice", "DATE") == "invoice_date"
    assert normalize_prediction_label("student_id", "NAME") == "student_name"
    assert normalize_prediction_label(None, "RAW_LABEL") == "raw_label"
