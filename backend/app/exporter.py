from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .db import SessionLocal
from .models import Document, Page, Attribute
from .ocr_processor import extract_tokens_with_bboxes
from .document_schemas import normalize_prediction_label

logger = logging.getLogger(__name__)
RAW_DATASET_ROOT = Path(os.getenv("RAW_DATASET_PAGES_DIR", "datasets/raw-pages"))


def _resolve_page_image_path(page: Page) -> Path:
    image_path = Path(page.image_path)
    if image_path.exists():
        return image_path
    return RAW_DATASET_ROOT / str(page.document_id) / f"page-{page.page_number:03d}.png"


def _iou(box_a: List[int], box_b: List[int]) -> float:
    # boxes are [x0,y0,x1,y1] in same scale
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter_area
    return float(inter_area) / union if union > 0 else 0.0


def _normalize_attr_boxes(boxes: List[List[float]], page_width: int, page_height: int) -> List[List[int]]:
    """Return boxes in normalized 0..1000 integer coordinates.

    Accepts boxes that may be:
    - absolute pixel coordinates (values > 1000 or > page dims)
    - fractional coordinates in [0,1]
    - already normalized 0..1000
    """
    normalized: List[List[int]] = []
    for b in boxes:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            logger.warning("Skipping malformed attribute bounding box: expected four coordinates")
            continue
        try:
            vals = [float(v) for v in b]
        except Exception:
            logger.warning("Skipping malformed attribute bounding box: coordinates must be numeric")
            continue
        if not all(math.isfinite(value) for value in vals):
            logger.warning("Skipping malformed attribute bounding box: coordinates must be finite")
            continue
        # heuristics
        max_val = max(vals)
        if max_val <= 1.0:
            # fractional [0,1] relative to page pixels -> convert to pixels then normalized
            x0 = int(round(1000 * (vals[0] * page_width) / page_width))
            y0 = int(round(1000 * (vals[1] * page_height) / page_height))
            x1 = int(round(1000 * (vals[2] * page_width) / page_width))
            y1 = int(round(1000 * (vals[3] * page_height) / page_height))
        elif max_val > 1000 or max_val > max(page_width, page_height):
            # assume absolute pixel coords -> convert to normalized 0..1000
            x0 = int(round(1000 * vals[0] / page_width))
            y0 = int(round(1000 * vals[1] / page_height))
            x1 = int(round(1000 * vals[2] / page_width))
            y1 = int(round(1000 * vals[3] / page_height))
        else:
            # assume already in 0..1000
            x0 = int(round(vals[0]))
            y0 = int(round(vals[1]))
            x1 = int(round(vals[2]))
            y1 = int(round(vals[3]))

        # clamp
        x0 = max(0, min(1000, x0))
        y0 = max(0, min(1000, y0))
        x1 = max(0, min(1000, x1))
        y1 = max(0, min(1000, y1))
        normalized.append([x0, y0, x1, y1])
    return normalized


def export_approved_attributes_jsonl(output_path: str | Path, min_iou: float = 0.1) -> int:
    """Export approved `Attribute` rows into token-level JSONL consumable by token-classifier.

    Each line is a JSON record with: `document_id`, `page_id`, `image`, `tokens`, `bboxes`, `labels`.
    Only `Attribute` rows with `validation_status == 'APPROVED'` are used to assign token labels.
    Labels are normalized via `normalize_prediction_label(document_type, raw_label)` to match model id2label.
    Returns number of records written.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    session = SessionLocal()
    try:
        # Query pages that have approved attributes
        stmt = select(Page).join(Attribute).where(Attribute.validation_status == "APPROVED")
        pages = session.scalars(stmt).unique().all()

        with output.open("w", encoding="utf-8") as handle:
            for page in pages:
                document = page.document
                image_path = _resolve_page_image_path(page)
                if not image_path.exists():
                    logger.warning("Page image missing, skipping: %s", image_path)
                    continue

                tokens = extract_tokens_with_bboxes(image_path)
                if not tokens:
                    continue

                token_bboxes = [t["bbox"] for t in tokens]
                token_texts = [t["token"] for t in tokens]

                # approved attributes on this page
                approved_attrs = [a for a in page.attributes if a.validation_status == "APPROVED"]
                if not approved_attrs:
                    continue

                # prepare attribute boxes and labels
                attr_boxes_list: List[List[List[int]]] = []
                attr_labels: List[str] = []
                for attr in approved_attrs:
                    # bounding_boxes may be list of lists or list of dicts
                    boxes = []
                    raw_boxes = getattr(attr, "bounding_boxes", []) or []
                    if not isinstance(raw_boxes, list):
                        logger.warning("Skipping malformed bounding_boxes for attribute %s", attr.id)
                        continue
                    for b in raw_boxes:
                        if isinstance(b, dict):
                            boxes.append([b.get(k, 0) for k in ("x0", "y0", "x1", "y1")])
                        elif isinstance(b, (list, tuple)):
                            boxes.append(list(b))
                        else:
                            logger.warning("Skipping malformed bounding box for attribute %s", attr.id)
                    if not boxes:
                        continue
                    # normalize attribute boxes into 0..1000 ints using page dimensions
                    try:
                        from PIL import Image

                        with Image.open(image_path) as im:
                            page_width, page_height = im.size
                    except Exception:
                        page_width, page_height = 1, 1

                    boxes_normalized = _normalize_attr_boxes(boxes, page_width, page_height)
                    if not boxes_normalized:
                        continue
                    normalized_label = normalize_prediction_label(document.document_type, attr.entity_type_label)
                    attr_boxes_list.append(boxes_normalized)
                    attr_labels.append(normalized_label)

                if not attr_boxes_list:
                    continue

                # assign tokens to attributes by IoU
                labels_for_tokens: List[str] = ["O"] * len(token_texts)
                for ti, tb in enumerate(token_bboxes):
                    best_iou = 0.0
                    best_attr = None
                    for ai, boxes in enumerate(attr_boxes_list):
                        for ab in boxes:
                            iou = _iou(tb, ab)
                            if iou > best_iou:
                                best_iou = iou
                                best_attr = ai
                    if best_attr is not None and best_iou >= min_iou:
                        labels_for_tokens[ti] = attr_labels[best_attr]

                record = {
                    "document_id": document.id,
                    "page_id": page.id,
                    "image": str(image_path),
                    "tokens": token_texts,
                    "bboxes": token_bboxes,
                    "labels": labels_for_tokens,
                }
                handle.write(json.dumps(record) + "\n")
                written += 1
    finally:
        session.close()

    return written


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: exporter.py <output.jsonl>")
        raise SystemExit(2)
    out = sys.argv[1]
    n = export_approved_attributes_jsonl(out)
    print(f"Wrote {n} records to {out}")
