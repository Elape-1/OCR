from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentSchema:
    document_type: str
    display_name: str
    keywords: tuple[str, ...]
    expected_fields: tuple[str, ...]
    label_aliases: dict[str, str]


DOCUMENT_SCHEMAS: dict[str, DocumentSchema] = {
    "invoice": DocumentSchema(
        document_type="invoice",
        display_name="Invoice",
        keywords=("invoice", "bill to", "subtotal", "amount due", "invoice number", "due date"),
        expected_fields=("invoice_number", "invoice_date", "vendor_name", "bill_to", "total_amount"),
        label_aliases={
            "invoice number": "invoice_number",
            "invoice_no": "invoice_number",
            "invoice no": "invoice_number",
            "number": "invoice_number",
            "date": "invoice_date",
            "name": "vendor_name",
            "total": "total_amount",
            "amount": "total_amount",
            "due date": "due_date",
            "address": "bill_to",
        },
    ),
    "receipt": DocumentSchema(
        document_type="receipt",
        display_name="Receipt",
        keywords=("receipt", "cashier", "subtotal", "tax", "change", "payment"),
        expected_fields=("merchant_name", "purchase_date", "total_amount", "tax_amount"),
        label_aliases={
            "merchant": "merchant_name",
            "name": "merchant_name",
            "date": "purchase_date",
            "total": "total_amount",
            "amount": "total_amount",
            "tax": "tax_amount",
        },
    ),
    "student_id": DocumentSchema(
        document_type="student_id",
        display_name="Student ID",
        keywords=("student id", "student card", "id card", "school", "university", "campus"),
        expected_fields=("student_name", "student_id", "issue_date", "expiry_date"),
        label_aliases={
            "name": "student_name",
            "student name": "student_name",
            "id": "student_id",
            "number": "student_id",
            "date": "issue_date",
            "expiry": "expiry_date",
        },
    ),
    "transcript": DocumentSchema(
        document_type="transcript",
        display_name="Transcript",
        keywords=("transcript", "course", "semester", "gpa", "credits", "grades"),
        expected_fields=("student_name", "student_id", "program", "gpa"),
        label_aliases={
            "name": "student_name",
            "student id": "student_id",
            "id": "student_id",
            "program": "program",
            "gpa": "gpa",
        },
    ),
    "form": DocumentSchema(
        document_type="form",
        display_name="Form",
        keywords=("form", "please fill", "signature", "checkbox", "check box", "applicant"),
        expected_fields=("applicant_name", "date", "signature"),
        label_aliases={
            "name": "applicant_name",
            "applicant": "applicant_name",
            "date": "date",
            "signature": "signature",
        },
    ),
}

DEFAULT_DOCUMENT_SCHEMA = DocumentSchema(
    document_type="unknown",
    display_name="Unknown",
    keywords=(),
    expected_fields=(),
    label_aliases={},
)


def get_document_schema(document_type: str | None) -> DocumentSchema:
    if not document_type:
        return DEFAULT_DOCUMENT_SCHEMA
    return DOCUMENT_SCHEMAS.get(document_type.lower(), DEFAULT_DOCUMENT_SCHEMA)


def classify_document_type(
    text: str,
    filename: str = "",
    layout_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    haystack = " ".join([filename, text]).lower()
    scores: dict[str, int] = {}

    for document_type, schema in DOCUMENT_SCHEMAS.items():
        score = 0
        for keyword in schema.keywords:
            if keyword in haystack:
                score += 2 if " " in keyword else 1
        if layout_hints:
            token_count = int(layout_hints.get("token_count", 0) or 0)
            line_count = int(layout_hints.get("line_count", 0) or 0)
            page_count = int(layout_hints.get("page_count", 0) or 0)
            if document_type == "invoice" and token_count > 10:
                score += 1
            if document_type == "receipt" and page_count <= 2:
                score += 1
            if document_type == "transcript" and line_count > 10:
                score += 1
        scores[document_type] = score

    if not scores:
        return {
            "document_type": DEFAULT_DOCUMENT_SCHEMA.document_type,
            "confidence": 0.0,
            "scores": {},
        }

    best_document_type, best_score = max(scores.items(), key=lambda item: item[1])
    total_score = sum(scores.values())
    confidence = float(best_score / total_score) if total_score > 0 else 0.0
    if best_score <= 0:
        best_document_type = DEFAULT_DOCUMENT_SCHEMA.document_type

    return {
        "document_type": best_document_type,
        "confidence": confidence,
        "scores": scores,
        "display_name": get_document_schema(best_document_type).display_name,
    }


def normalize_prediction_label(document_type: str | None, raw_label: str) -> str:
    schema = get_document_schema(document_type)
    normalized_label = raw_label.strip().lower()
    return schema.label_aliases.get(normalized_label, normalized_label or raw_label)
