from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.training import (
    TokenExample,
    compute_entity_metrics,
    load_token_examples,
    set_reproducible_seed,
    split_examples,
    smoke_test_token_training,
)


def test_training_smoke_chain_uses_seeded_split_and_metrics(tmp_path) -> None:
    set_reproducible_seed(7)

    synthetic_rows = [
        {
            "document_id": 1,
            "page_id": 1,
            "image": "sample.png",
            "tokens": ["Invoice", "total", "is", "42", "."],
            "bboxes": [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10], [30, 0, 40, 10], [40, 0, 50, 10]],
            "labels": ["O", "TOTAL", "O", "TOTAL", "O"],
        },
        {
            "document_id": 1,
            "page_id": 2,
            "image": "sample-page-2.png",
            "tokens": ["Invoice", "page", "two"],
            "bboxes": [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]],
            "labels": ["O", "O", "O"],
        },
        {
            "document_id": 2,
            "page_id": 1,
            "image": "sample2.png",
            "tokens": ["Ship", "to", "Acme"],
            "bboxes": [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]],
            "labels": ["O", "O", "VENDOR"],
        },
        {
            "document_id": 3,
            "page_id": 1,
            "image": "sample3.png",
            "tokens": ["Paid", "32", "."],
            "bboxes": [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]],
            "labels": ["O", "TOTAL", "O"],
        },
    ]
    export_path = tmp_path / "export.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for row in synthetic_rows:
            handle.write(json.dumps(row) + "\n")

    examples = load_token_examples(export_path)
    assert len(examples) == 4

    split = split_examples(examples, seed=7, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    assert set(split.keys()) == {"train", "val", "test"}
    assert sum(len(v) for v in split.values()) == len(examples)
    split_documents = {
        name: {example.document_id for example in group}
        for name, group in split.items()
    }
    assert not (split_documents["train"] & split_documents["val"])
    assert not (split_documents["train"] & split_documents["test"])
    assert not (split_documents["val"] & split_documents["test"])

    forward = smoke_test_token_training(num_labels=3, seq_len=16)
    assert "logits_shape" in forward
    assert tuple(forward["logits_shape"]) == (1, 16, 3)

    logits = np.array([[[10.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]], dtype=float)
    labels = np.array([[0, 1, 2]], dtype=int)
    metrics = compute_entity_metrics(logits, labels, ["O", "TOTAL", "VENDOR"])
    assert metrics["f1"] >= 0.0
    assert set(metrics) >= {"precision", "recall", "f1"}
