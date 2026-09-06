from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image

from app.document_schemas import get_document_schema, normalize_prediction_label

logger = logging.getLogger(__name__)
DEFAULT_LAYOUTLMV3_MODEL_SOURCE = "microsoft/layoutlmv3-base"

try:
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
except Exception as exc:  # pragma: no cover - exercised in lightweight local environments
    LayoutLMv3ForTokenClassification = None
    LayoutLMv3Processor = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass
class PredictionResult:
    token: str
    label: str
    confidence: float
    bbox: List[int]


class LayoutLMv3InferenceService:
    def __init__(self, model_source: str | None = None, ocr_only: bool | None = None) -> None:
        self.model_source = model_source or os.getenv("LAYOUTLMV3_MODEL_SOURCE", DEFAULT_LAYOUTLMV3_MODEL_SOURCE)
        self.ocr_only = ocr_only if ocr_only is not None else os.getenv("LAYOUTLMV3_OCR_ONLY", "true").lower() in {"1", "true", "yes", "on"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def _ensure_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        if self.ocr_only:
            return
        if self.model_source == DEFAULT_LAYOUTLMV3_MODEL_SOURCE:
            raise RuntimeError(
                "LayoutLMv3 extraction requires a trained token-classification checkpoint. "
                "Set LAYOUTLMV3_MODEL_SOURCE to the trained checkpoint or set LAYOUTLMV3_OCR_ONLY=true."
            )
        if LayoutLMv3ForTokenClassification is None or LayoutLMv3Processor is None:
            raise RuntimeError("LayoutLMv3 dependencies are unavailable") from _TRANSFORMERS_IMPORT_ERROR
        # We supply OCR words and normalized boxes from our own OCR pipeline.
        # apply_ocr must be disabled to avoid transformer-side OCR conflicts.
        self.processor = LayoutLMv3Processor.from_pretrained(self.model_source, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.model_source)
        self.model.to(self.device)
        self.model.eval()

    def infer_page(
        self,
        image_path: str | Path,
        tokens: Sequence[str],
        normalized_bboxes: Sequence[Sequence[int]],
        document_type: str | None = None,
    ) -> Dict[str, Any]:
        if self.ocr_only:
            return self._build_ocr_only_result(image_path, tokens, normalized_bboxes, document_type)
        self._ensure_loaded()
        id2label = self.model.config.id2label
        schema = get_document_schema(document_type)

        with Image.open(image_path) as page_image:
            image = page_image.convert("RGB")

        chunk_size = self._max_chunk_size()
        overlap = self._chunk_overlap(chunk_size)
        chunked_inference = len(tokens) > chunk_size
        if chunked_inference:
            logger.warning(
                "LayoutLMv3 input for %s exceeds model window (%s tokens > %s); using overlapping chunks",
                image_path,
                len(tokens),
                chunk_size,
            )

        stitched_predictions: dict[int, PredictionResult] = {}
        chunk_count = 0
        for chunk_start, chunk_end in self._iter_chunk_ranges(len(tokens), chunk_size, overlap):
            chunk_count += 1
            chunk_predictions = self._infer_chunk(
                image=image,
                chunk_tokens=tokens[chunk_start:chunk_end],
                chunk_bboxes=normalized_bboxes[chunk_start:chunk_end],
                document_type=document_type,
                id2label=id2label,
            )
            for local_index, prediction in chunk_predictions:
                global_index = chunk_start + local_index
                stitched_predictions[global_index] = self._select_preferred_prediction(
                    existing=stitched_predictions.get(global_index),
                    candidate=prediction,
                )

        if len(stitched_predictions) != len(tokens):
            missing_indices = [index for index in range(len(tokens)) if index not in stitched_predictions]
            logger.warning(
                "LayoutLMv3 chunk stitching missed %s token(s) for %s; filling from OCR tokens",
                len(missing_indices),
                image_path,
            )
            for index in missing_indices:
                stitched_predictions[index] = PredictionResult(
                    token=str(tokens[index]),
                    label=normalize_prediction_label(document_type, "HITL"),
                    confidence=0.0,
                    bbox=list(normalized_bboxes[index]),
                )

        predictions: List[PredictionResult] = [stitched_predictions[index] for index in sorted(stitched_predictions)]

        grouped = self._build_layout_dictionary(predictions)
        return {
            "page_image": str(Path(image_path)),
            "document_type": document_type or schema.document_type,
            "schema_display_name": schema.display_name,
            "schema_expected_fields": list(schema.expected_fields),
            "tokens": [prediction.__dict__ for prediction in predictions],
            "chunked_inference": chunked_inference,
            "chunk_count": chunk_count,
            "max_tokens_per_chunk": chunk_size,
            "layout": grouped,
        }

    def _build_ocr_only_result(
        self,
        image_path: str | Path,
        tokens: Sequence[str],
        normalized_bboxes: Sequence[Sequence[int]],
        document_type: str | None,
    ) -> Dict[str, Any]:
        schema = get_document_schema(document_type)
        predictions = [
            PredictionResult(
                token=str(token),
                label="OCR",
                confidence=0.0,
                bbox=list(normalized_bboxes[index]),
            )
            for index, token in enumerate(tokens)
        ]
        return {
            "page_image": str(Path(image_path)),
            "document_type": document_type or schema.document_type,
            "schema_display_name": schema.display_name,
            "schema_expected_fields": list(schema.expected_fields),
            "tokens": [prediction.__dict__ for prediction in predictions],
            "chunked_inference": False,
            "chunk_count": 0,
            "max_tokens_per_chunk": 0,
            "layout": self._build_layout_dictionary(predictions),
        }

    def _max_chunk_size(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        model_max_length = getattr(tokenizer, "model_max_length", 512)
        try:
            model_max_length = int(model_max_length)
        except Exception:
            model_max_length = 512
        return max(1, model_max_length - 2)

    def _chunk_overlap(self, chunk_size: int) -> int:
        return max(1, min(64, chunk_size // 4)) if chunk_size > 1 else 0

    def _iter_chunk_ranges(self, total_tokens: int, chunk_size: int, overlap: int) -> Iterable[Tuple[int, int]]:
        if total_tokens <= chunk_size:
            yield (0, total_tokens)
            return

        step = max(1, chunk_size - overlap)
        start = 0
        while start < total_tokens:
            end = min(total_tokens, start + chunk_size)
            yield (start, end)
            if end >= total_tokens:
                break
            start += step

    def _infer_chunk(
        self,
        image: Image.Image,
        chunk_tokens: Sequence[str],
        chunk_bboxes: Sequence[Sequence[int]],
        document_type: str | None,
        id2label: Dict[int, str],
    ) -> List[Tuple[int, PredictionResult]]:
        encoding = self.processor(
            images=image,
            text=list(chunk_tokens),
            boxes=[list(box) for box in chunk_bboxes],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_ids = probabilities.argmax(dim=-1)
            confidences, _ = probabilities.max(dim=-1)

        chunk_predictions: List[Tuple[int, PredictionResult]] = []
        for index, word_id in enumerate(word_ids):
            if word_id is None or word_id >= len(chunk_tokens):
                continue
            label_id = int(predicted_ids[index].item())
            raw_label = id2label.get(label_id, str(label_id))
            chunk_predictions.append(
                (
                    int(word_id),
                    PredictionResult(
                        token=str(chunk_tokens[word_id]),
                        label=normalize_prediction_label(document_type, raw_label),
                        confidence=float(confidences[index].item()),
                        bbox=list(chunk_bboxes[word_id]),
                    ),
                )
            )
        return chunk_predictions

    def _select_preferred_prediction(
        self,
        existing: PredictionResult | None,
        candidate: PredictionResult,
    ) -> PredictionResult:
        """Resolve overlap duplicates by confidence, keeping the earlier chunk on ties.

        Overlap regions are not averaged or voted on. The chunk stitcher keeps the
        prediction with the higher confidence score and falls back to the first
        prediction encountered when scores are equal.
        """
        if existing is None:
            return candidate
        if candidate.confidence > existing.confidence:
            return candidate
        return existing

    def _build_layout_dictionary(self, predictions: Sequence[PredictionResult]) -> Dict[str, Any]:
        entities: Dict[str, List[Dict[str, Any]]] = {}
        for prediction in predictions:
            entities.setdefault(prediction.label, []).append(
                {
                    "token": prediction.token,
                    "confidence": prediction.confidence,
                    "bbox": prediction.bbox,
                }
            )
        averages = {
            label: sum(item["confidence"] for item in items) / max(len(items), 1)
            for label, items in entities.items()
        }
        return {"entities": entities, "mean_confidence_by_label": averages}
