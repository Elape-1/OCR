from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from PIL import Image

from app.model_inference import LayoutLMv3InferenceService


class FakeEncoding(dict):
    def __init__(self, word_ids: list[int | None], sequence_length: int) -> None:
        super().__init__(
            {
                "input_ids": torch.arange(sequence_length, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
            }
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return list(self._word_ids)


class FakeProcessor:
    def __init__(self, max_length: int = 6) -> None:
        self.tokenizer = SimpleNamespace(model_max_length=max_length)

    def __call__(self, images, text, boxes, return_tensors, truncation, padding):
        sequence_length = len(text) + 2
        return FakeEncoding([None] + list(range(len(text))) + [None], sequence_length)


class FakeModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "other", 1: "date"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros((batch_size, sequence_length, 2), dtype=torch.float)
        logits[:, :, 1] = 10.0
        return FakeModelOutput(logits)


def test_base_layoutlm_checkpoint_is_rejected_for_extraction(monkeypatch) -> None:
    from app.model_inference import DEFAULT_LAYOUTLMV3_MODEL_SOURCE

    service = LayoutLMv3InferenceService(model_source=DEFAULT_LAYOUTLMV3_MODEL_SOURCE)

    with pytest.raises(RuntimeError, match="trained token-classification checkpoint"):
        service._ensure_loaded()


def test_ocr_only_mode_does_not_load_layoutlm(tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), "white").save(image_path)

    service = LayoutLMv3InferenceService(ocr_only=True)
    result = service.infer_page(image_path, ["Invoice"], [[10, 10, 100, 40]], document_type="invoice")

    assert service.model is None
    assert result["tokens"] == [{
        "token": "Invoice",
        "label": "OCR",
        "confidence": 0.0,
        "bbox": [10, 10, 100, 40],
    }]


def test_chunked_inference_preserves_all_tokens(tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), "white").save(image_path)

    service = LayoutLMv3InferenceService()
    service.processor = FakeProcessor(max_length=6)
    service.model = FakeModel()
    service._ensure_loaded = lambda: None

    tokens = [f"token-{index}" for index in range(10)]
    bboxes = [[index * 5, 10, index * 5 + 4, 20] for index in range(10)]

    result = service.infer_page(
        image_path=image_path,
        tokens=tokens,
        normalized_bboxes=bboxes,
        document_type="invoice",
    )

    assert result["chunked_inference"] is True
    assert result["chunk_count"] > 1
    assert len(result["tokens"]) == len(tokens)
    assert result["tokens"][0]["token"] == "token-0"
    assert result["tokens"][-1]["token"] == "token-9"
    assert all(item["label"] == "invoice_date" for item in result["tokens"])


def test_overlap_resolution_prefers_higher_confidence_prediction(tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), "white").save(image_path)

    service = LayoutLMv3InferenceService()
    service.processor = FakeProcessor(max_length=6)
    service.model = SimpleNamespace(config=SimpleNamespace(id2label={0: "other", 1: "date", 2: "vendor_name"}))
    service._ensure_loaded = lambda: None

    tokens = ["t0", "t1", "t2", "overlap", "t4", "t5"]
    bboxes = [[index * 5, 10, index * 5 + 4, 20] for index in range(len(tokens))]
    # Replace the private helper directly for this test to simulate conflicting
    # predictions for the overlap token across two adjacent chunks.
    def fake_infer_chunk(image, chunk_tokens, chunk_bboxes, document_type, id2label):
        from app.model_inference import PredictionResult

        if chunk_tokens[0] == "t0":
            return [
                (0, PredictionResult(token="t0", label="vendor_name", confidence=0.95, bbox=list(chunk_bboxes[0]))),
                (1, PredictionResult(token="t1", label="vendor_name", confidence=0.95, bbox=list(chunk_bboxes[1]))),
                (2, PredictionResult(token="t2", label="vendor_name", confidence=0.95, bbox=list(chunk_bboxes[2]))),
                (3, PredictionResult(token="overlap", label="date", confidence=0.40, bbox=list(chunk_bboxes[3]))),
            ]
        return [
            (0, PredictionResult(token="overlap", label="vendor_name", confidence=0.92, bbox=list(chunk_bboxes[0]))),
            (1, PredictionResult(token="t4", label="vendor_name", confidence=0.92, bbox=list(chunk_bboxes[1]))),
            (2, PredictionResult(token="t5", label="vendor_name", confidence=0.92, bbox=list(chunk_bboxes[2]))),
        ]

    service._infer_chunk = fake_infer_chunk

    result = service.infer_page(
        image_path=image_path,
        tokens=tokens,
        normalized_bboxes=bboxes,
        document_type="invoice",
    )

    overlap_token = result["tokens"][3]
    assert overlap_token["token"] == "overlap"
    assert overlap_token["label"] == "vendor_name"
    assert overlap_token["confidence"] == 0.92