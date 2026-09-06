from __future__ import annotations

import json

from app.training import TextExample, build_label_list, compute_classification_metrics, load_examples


def test_load_examples_reads_jsonl(tmp_path) -> None:
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        "\n".join(
            [
                json.dumps({"text": "Invoice 42", "label": "invoice_total"}),
                json.dumps({"text": "Ship to Acme", "label": "shipping_name"}),
            ]
        ),
        encoding="utf-8",
    )

    examples = load_examples(data_file)

    assert len(examples) == 2
    assert examples[0].text == "Invoice 42"


def test_build_label_list_sorts_unique_labels() -> None:
    labels = build_label_list(
        [
            TextExample(text="beta text", label="beta"),
            TextExample(text="alpha text", label="alpha"),
        ],
        [
            TextExample(text="beta text 2", label="beta"),
        ],
    )

    assert labels == ["alpha", "beta"]


def test_compute_classification_metrics_returns_expected_values() -> None:
    metrics = compute_classification_metrics(
        predictions=[0, 1, 1],
        references=[0, 1, 0],
        label_names=["negative", "positive"],
    )

    assert metrics["accuracy"] == 2 / 3
    assert "macro_f1" in metrics
    assert "classification_report" in metrics