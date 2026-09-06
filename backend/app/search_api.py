from __future__ import annotations

from typing import Any
from datetime import date, datetime, time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import String, and_, cast, func, or_, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import CurrentUserId, owner_scope
from app.models import Document, Attribute, Correction

router = APIRouter()


def _parse_search_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    if end_of_day:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            parsed_date = None
        if parsed_date is not None:
            return datetime.combine(parsed_date, time.max)
    return datetime.fromisoformat(value)


@router.get("/search/document-types")
def list_document_types(db: Session = Depends(get_db), user_id: CurrentUserId = None) -> dict[str, Any]:
    statement = select(func.distinct(Document.document_type)).where(owner_scope(Document.owner_id, user_id)).order_by(Document.document_type)
    rows = db.execute(statement).scalars().all()
    return {"document_types": [r for r in rows if r is not None]}


@router.get("/search/attribute-names")
def list_attribute_names(db: Session = Depends(get_db), user_id: CurrentUserId = None) -> dict[str, Any]:
    statement = select(func.distinct(Attribute.entity_type_label)).join(Document).where(owner_scope(Document.owner_id, user_id)).order_by(Attribute.entity_type_label)
    rows = db.execute(statement).scalars().all()
    return {"attribute_names": [r for r in rows if r is not None]}


@router.get("/search/documents")
def search_documents(
    q: str | None = Query(None, description="Search text for filename or attributes"),
    document_type: str | None = Query(None),
    status: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
    user_id: CurrentUserId = None,
):
    stmt = select(Document)
    filters = []
    # Default searches show completed documents; an explicit status searches that
    # lifecycle state instead of intersecting with the completed-state filter.
    if status:
        filters.append(Document.status == status)
    else:
        filters.append(Document.status.in_(["APPROVED", "PROCESSED"]))
    filters.append(owner_scope(Document.owner_id, user_id))
    if q:
        like = f"%{q}%"
        persisted_attribute_match = select(Attribute.id).where(
            Attribute.document_id == Document.id,
            or_(Attribute.entity_type_label.ilike(like), Attribute.extracted_value.ilike(like)),
        ).exists()
        staged_attribute_match = cast(Document.extracted_attributes, String).ilike(like)
        filters.append(
            or_(
                Document.name.ilike(like),
                Document.format.ilike(like),
                persisted_attribute_match,
                staged_attribute_match,
            )
        )
    if document_type:
        filters.append(Document.document_type == document_type)
    if start_date:
        try:
            start = _parse_search_datetime(start_date)
            filters.append(Document.timestamp >= start)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid start_date format; use ISO format")
    if end_date:
        try:
            end = _parse_search_datetime(end_date, end_of_day=True)
            filters.append(Document.timestamp <= end)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid end_date format; use ISO format")

    if filters:
        stmt = stmt.where(and_(*filters))

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.order_by(Document.timestamp.desc()).offset(offset).limit(page_size)

    rows = db.execute(stmt).scalars().all()
    documents = [
        {
            "id": d.id,
            "name": d.name,
            "format": d.format,
            "document_type": d.document_type,
            "page_count": d.page_count,
            "timestamp": d.timestamp.isoformat() if d.timestamp else None,
            "status": d.status,
            "processing_started_at": d.processing_started_at.isoformat() if d.processing_started_at else None,
            "processing_finished_at": d.processing_finished_at.isoformat() if d.processing_finished_at else None,
            "processing_duration_seconds": d.processing_duration_seconds,
        }
        for d in rows
    ]

    return {"total": int(total), "page": page, "page_size": page_size, "documents": documents}


@router.get("/search/attributes")
def search_attributes(
    q: str | None = Query(None, description="Search text for attribute name or value"),
    attribute_name: str | None = Query(None),
    min_conf: float | None = Query(None, ge=0.0, le=1.0),
    max_conf: float | None = Query(None, ge=0.0, le=1.0),
    document_type: str | None = Query(None),
    hitl_corrected: bool | None = Query(None, description="Filter attributes that have been corrected via HITL"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
    user_id: CurrentUserId = None,
):
    # consider attributes that are validated/approved (or processed)
    # note: do not require the parent document to be approved so saved attributes
    # on documents that may still be transitioning are discoverable in search results
    stmt = select(Attribute).join(Document, Attribute.document_id == Document.id)
    filters = []
    filters.append(Attribute.validation_status.in_( ["APPROVED", "PROCESSED"]))
    filters.append(owner_scope(Document.owner_id, user_id))

    if q:
        like = f"%{q}%"
        filters.append(or_(Attribute.entity_type_label.ilike(like), Attribute.extracted_value.ilike(like)))

    if attribute_name:
        filters.append(Attribute.entity_type_label == attribute_name)

    if min_conf is not None:
        filters.append(Attribute.confidence_score >= min_conf)
    if max_conf is not None:
        filters.append(Attribute.confidence_score <= max_conf)

    if document_type:
        filters.append(Document.document_type == document_type)

    if hitl_corrected is not None:
        # left join with corrections and filter existence
        if hitl_corrected:
            stmt = stmt.join(Correction, Attribute.id == Correction.attribute_id)
        else:
            # exclude attributes that have any corrections
            subq = select(Correction.attribute_id).where(Correction.attribute_id == Attribute.id)
            filters.append(~Attribute.id.in_(subq))

    if filters:
        stmt = stmt.where(and_(*filters))

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.order_by(Attribute.confidence_score.desc()).offset(offset).limit(page_size)

    rows = db.execute(stmt).scalars().all()
    attributes = []
    for a in rows:
        # Determine if corrected
        corrected = db.execute(select(func.count()).select_from(Correction).where(Correction.attribute_id == a.id)).scalar_one() > 0
        attributes.append(
            {
                "id": a.id,
                "document_id": a.document_id,
                "entity_type_label": a.entity_type_label,
                "extracted_value": a.extracted_value,
                "confidence_score": a.confidence_score,
                "validation_status": a.validation_status,
                "corrected": corrected,
            }
        )

    return {"total": int(total), "page": page, "page_size": page_size, "attributes": attributes, "included_unapproved": False}
