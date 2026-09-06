from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.document_schemas import normalize_prediction_label


SAME_LINE_HORIZONTAL_GAP_MULTIPLIER = 1.5
SAME_BLOCK_VERTICAL_GAP_MULTIPLIER = 1.75
SAME_BLOCK_LEFT_ALIGNMENT_MULTIPLIER = 2.0
SAME_LINE_VERTICAL_OVERLAP_RATIO = 0.25


@dataclass
class RoutedAttribute:
    entity_type_label: str
    extracted_value: str
    bounding_boxes: List[List[int]]
    confidence_score: float
    validation_status: str


def _bbox_metrics(bbox: Sequence[int]) -> dict[str, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return {
        "left": x0,
        "top": y0,
        "right": x1,
        "bottom": y1,
        "width": width,
        "height": height,
        "center_x": x0 + width / 2.0,
        "center_y": y0 + height / 2.0,
    }


def _reading_order_key(prediction: Dict[str, Any]) -> tuple[int, int, int, float, float]:
    block_num = int(prediction.get("block_num", 0) or 0)
    paragraph_num = int(prediction.get("paragraph_num", 0) or 0)
    line_num = int(prediction.get("line_num", 0) or 0)
    metrics = _bbox_metrics(prediction.get("bbox", [0, 0, 0, 0]))
    return (block_num, paragraph_num, line_num, metrics["top"], metrics["left"])


def _should_merge_spatially(current_cluster: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    """Merge only when the next token is plausibly part of the same field span.

    The current heuristic is intentionally conservative:
    - same-line spans require matching block/paragraph metadata and a horizontal gap
      no larger than 1.5x the wider token box
    - same-block fallback merges require a vertical gap no larger than 1.75x the
      taller box and a left-edge delta no larger than 2.0x the wider box
    - when line metadata is missing, same-line detection falls back to a 25% vertical
      overlap ratio between the two boxes
    """
    current_metrics = current_cluster["metrics"]
    candidate_metrics = candidate["metrics"]

    same_block = current_cluster["block_num"] == candidate["block_num"] if current_cluster["block_num"] and candidate["block_num"] else True
    same_paragraph = (
        current_cluster["paragraph_num"] == candidate["paragraph_num"]
        if current_cluster["paragraph_num"] and candidate["paragraph_num"]
        else True
    )

    vertical_overlap = min(current_metrics["bottom"], candidate_metrics["bottom"]) - max(
        current_metrics["top"], candidate_metrics["top"]
    )
    horizontal_gap = candidate_metrics["left"] - current_metrics["right"]
    vertical_gap = candidate_metrics["top"] - current_metrics["bottom"]

    same_line = False
    if current_cluster["line_num"] and candidate["line_num"]:
        same_line = current_cluster["line_num"] == candidate["line_num"]
    elif vertical_overlap > 0:
        same_line = vertical_overlap >= SAME_LINE_VERTICAL_OVERLAP_RATIO * max(
            1.0,
            min(current_metrics["height"], candidate_metrics["height"]),
        )

    if same_line and same_block and same_paragraph:
        return horizontal_gap <= max(current_metrics["width"], candidate_metrics["width"]) * SAME_LINE_HORIZONTAL_GAP_MULTIPLIER

    if same_block and same_paragraph and vertical_gap >= 0:
        return vertical_gap <= max(current_metrics["height"], candidate_metrics["height"]) * SAME_BLOCK_VERTICAL_GAP_MULTIPLIER and abs(
            candidate_metrics["left"] - current_metrics["left"]
        ) <= max(current_metrics["width"], candidate_metrics["width"]) * SAME_BLOCK_LEFT_ALIGNMENT_MULTIPLIER

    return False


def cluster_entity_tokens(token_predictions: Sequence[Dict[str, Any]], threshold: float = 0.75) -> List[RoutedAttribute]:
    labels_to_predictions: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for prediction in token_predictions:
        label = str(prediction.get("label", "")).strip()
        token = str(prediction.get("token", "")).strip()
        bbox = list(prediction.get("bbox", []))
        if not label or not token or len(bbox) != 4:
            continue

        metrics = _bbox_metrics(bbox)
        labels_to_predictions[label].append(
            {
                "label": label,
                "token": token,
                "confidence": float(prediction.get("confidence", 0.0)),
                "bbox": bbox,
                "metrics": metrics,
                "sort_key": _reading_order_key(prediction),
                "block_num": int(prediction.get("block_num", 0) or 0),
                "paragraph_num": int(prediction.get("paragraph_num", 0) or 0),
                "line_num": int(prediction.get("line_num", 0) or 0),
            }
        )

    clusters: List[Dict[str, Any]] = []
    for label, label_predictions in labels_to_predictions.items():
        ordered_predictions = sorted(label_predictions, key=lambda item: item["sort_key"])
        current: Dict[str, Any] | None = None

        for prediction in ordered_predictions:
            if current is None:
                current = {
                    "label": label,
                    "tokens": [prediction["token"]],
                    "boxes": [prediction["bbox"]],
                    "confidences": [prediction["confidence"]],
                    "sort_key": prediction["sort_key"],
                    "metrics": prediction["metrics"],
                    "block_num": prediction["block_num"],
                    "paragraph_num": prediction["paragraph_num"],
                    "line_num": prediction["line_num"],
                }
                continue

            if _should_merge_spatially(current, prediction):
                current["tokens"].append(prediction["token"])
                current["boxes"].append(prediction["bbox"])
                current["confidences"].append(prediction["confidence"])
                current["metrics"] = prediction["metrics"]
                current["sort_key"] = min(current["sort_key"], prediction["sort_key"])
            else:
                clusters.append(current)
                current = {
                    "label": label,
                    "tokens": [prediction["token"]],
                    "boxes": [prediction["bbox"]],
                    "confidences": [prediction["confidence"]],
                    "sort_key": prediction["sort_key"],
                    "metrics": prediction["metrics"],
                    "block_num": prediction["block_num"],
                    "paragraph_num": prediction["paragraph_num"],
                    "line_num": prediction["line_num"],
                }

        if current:
            clusters.append(current)

    clusters.sort(key=lambda cluster: cluster["sort_key"])

    routed: List[RoutedAttribute] = []
    for cluster in clusters:
        mean_confidence = sum(cluster["confidences"]) / max(len(cluster["confidences"]), 1)
        validation_status = "APPROVED" if mean_confidence >= threshold else "HITL"
        routed.append(
            RoutedAttribute(
                entity_type_label=str(cluster["label"]),
                extracted_value=" ".join(cluster["tokens"]).strip(),
                bounding_boxes=[list(box) for box in cluster["boxes"]],
                confidence_score=mean_confidence,
                validation_status=validation_status,
            )
        )
    return routed


def build_attribute_payloads(
    page_id: int,
    document_id: int,
    token_predictions: Sequence[Dict[str, Any]],
    document_type: str | None = None,
    threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    normalized_predictions: list[Dict[str, Any]] = []
    for prediction in token_predictions:
        normalized_prediction = dict(prediction)
        normalized_prediction["label"] = normalize_prediction_label(document_type, str(prediction.get("label", "")))
        normalized_predictions.append(normalized_prediction)

    routed_attributes = cluster_entity_tokens(normalized_predictions, threshold=threshold)
    payloads: List[Dict[str, Any]] = []
    for attribute in routed_attributes:
        payloads.append(
            {
                "page_id": page_id,
                "document_id": document_id,
                "entity_type_label": attribute.entity_type_label,
                "extracted_value": attribute.extracted_value,
                "bounding_boxes": attribute.bounding_boxes,
                "confidence_score": attribute.confidence_score,
                "validation_status": attribute.validation_status,
                "document_type": document_type or "unknown",
            }
        )
    return payloads
