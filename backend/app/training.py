from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image


def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_reproducible_seed()

try:
    from datasets import Dataset
    from transformers import (
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor,
        LayoutLMv3Config,
        Trainer,
        TrainingArguments,
    )
except Exception as exc:  # pragma: no cover - handled at runtime with a clear error
    Dataset = None
    LayoutLMv3ForTokenClassification = None
    LayoutLMv3Processor = None
    LayoutLMv3Config = None
    Trainer = None
    TrainingArguments = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class TokenExample:
    tokens: list[str]
    bboxes: list[list[int]]
    labels: list[str]
    document_id: str | int | None = None
    image: str | None = None


@dataclass(frozen=True)
class TextExample:
    text: str
    label: str


def _require_training_dependencies() -> None:
    if Dataset is None or LayoutLMv3ForTokenClassification is None or LayoutLMv3Processor is None or Trainer is None:
        raise RuntimeError("Training dependencies are unavailable") from _IMPORT_ERROR


def load_token_examples(file_path: str | Path) -> list[TokenExample]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[TokenExample] = []
    suffix = path.suffix.lower()

    if suffix == ".csv":
        raise ValueError("CSV format not supported for token-level training; use JSONL with tokens/bboxes/labels")

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            tokens = record.get("tokens")
            bboxes = record.get("bboxes")
            labels = record.get("labels")
            if not tokens or not labels or not bboxes:
                continue
            if not (len(tokens) == len(labels) == len(bboxes)):
                raise ValueError(f"Token/example length mismatch in {path}: tokens={len(tokens)} labels={len(labels)} bboxes={len(bboxes)}")
            document_id = record.get("document_id")
            if document_id is None:
                document_id = record.get("image") or record.get("id")
            image = record.get("image")
            if image:
                image_path = Path(str(image))
                candidates = [image_path, path.parent / image_path]
                image = str(next((candidate for candidate in candidates if candidate.exists()), image_path))
            examples.append(TokenExample(tokens=tokens, bboxes=bboxes, labels=labels, document_id=document_id, image=image))

    if not examples:
        raise ValueError(f"No token-labeled examples found in {path}")
    return examples


def build_label_list(*example_groups: Sequence[Any]) -> list[str]:
    labels: set[str] = set()
    for group in example_groups:
        for ex in group:
            if hasattr(ex, "labels"):
                labels.update(label for label in getattr(ex, "labels") if label)
            elif hasattr(ex, "label"):
                label = getattr(ex, "label")
                if label:
                    labels.add(str(label))
    if not labels:
        raise ValueError("At least one label is required")
    return sorted(labels)


def load_examples(file_path: str | Path) -> list[TextExample]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[TextExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            text = record.get("text")
            label = record.get("label")
            if text is None or label is None:
                continue
            examples.append(TextExample(text=str(text), label=str(label)))

    if not examples:
        raise ValueError(f"No labeled examples found in {path}")
    return examples


def compute_classification_metrics(
    predictions: Sequence[int],
    references: Sequence[int],
    label_names: Sequence[str],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    labels = list(range(len(label_names)))
    accuracy = float(accuracy_score(list(references), list(predictions)))
    macro_f1 = float(f1_score(list(references), list(predictions), labels=labels, average="macro", zero_division=0))
    report = classification_report(
        list(references),
        list(predictions),
        labels=labels,
        target_names=list(label_names),
        zero_division=0,
        output_dict=True,
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1, "classification_report": report}


def split_examples(
    examples: Sequence[TokenExample],
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[TokenExample]]:
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")
    rng = random.Random(seed)
    grouped: dict[str, list[TokenExample]] = {}
    for index, example in enumerate(examples):
        group_id = str(example.document_id) if example.document_id is not None else f"example-{index}"
        grouped.setdefault(group_id, []).append(example)

    group_items = list(grouped.items())
    rng.shuffle(group_items)
    fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    targets = {name: len(examples) * fraction for name, fraction in fractions.items()}
    assignments: dict[str, list[TokenExample]] = {name: [] for name in fractions}

    for group_id, group_examples in group_items:
        destination = min(
            fractions,
            key=lambda name: len(assignments[name]) / targets[name] if targets[name] else float("inf"),
        )
        assignments[destination].extend(group_examples)

    return assignments


def _examples_to_dataset(examples: Sequence[TokenExample], label2id: dict[str, int]) -> Any:
    _require_training_dependencies()
    rows = []
    for ex in examples:
        rows.append({"tokens": ex.tokens, "bboxes": ex.bboxes, "labels": [label2id[l] for l in ex.labels], "image": ex.image})
    return Dataset.from_list(rows)


def _tokenize_and_align_labels(dataset: Any, processor: Any, max_length: int) -> Any:
    # dataset items: {tokens: list[str], bboxes: list[list[int]], labels: list[int]}
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        images = []
        for image_path in batch.get("image", []):
            if image_path:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            else:
                images.append(Image.new("RGB", (224, 224), "white"))
        tokenized_inputs = processor(
            images=images,
            text=batch["tokens"],
            boxes=batch["bboxes"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        all_labels = batch["labels"]
        aligned_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(int(labels[word_idx]))
                else:
                    # for sub-tokens, ignore
                    label_ids.append(-100)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        # convert lists to plain Python lists for datasets
        return {k: [v[i] for i in range(len(batch["tokens"]))] for k, v in tokenized_inputs.items()}

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["tokens", "bboxes", "labels", "image"])
    tokenized.set_format(type="torch")
    return tokenized


def _ids_to_label_sequence(ids: Sequence[int], id2label: dict[int, str]) -> list[str]:
    return [id2label[int(i)] if isinstance(i, int) and i in id2label else "O" for i in ids]


def _to_bio(labels: list[str]) -> list[str]:
    bio: list[str] = []
    prev: str | None = None
    for lbl in labels:
        if not lbl or lbl == "O":
            bio.append("O")
            prev = None
        else:
            if prev == lbl:
                bio.append(f"I-{lbl}")
            else:
                bio.append(f"B-{lbl}")
            prev = lbl
    return bio


def compute_entity_metrics(predictions: np.ndarray, references: np.ndarray, label_list: list[str]) -> dict[str, Any]:
    # predictions: (batch, seq_len, num_labels)
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
    except Exception as exc:
        raise RuntimeError("seqeval is required for entity-level metrics") from exc

    id2label = {i: label for i, label in enumerate(label_list)}

    pred_ids = np.argmax(predictions, axis=-1)
    batch_preds: list[list[str]] = []
    batch_refs: list[list[str]] = []

    for i in range(pred_ids.shape[0]):
        preds_i = []
        refs_i = []
        for j in range(pred_ids.shape[1]):
            ref_id = int(references[i, j])
            if ref_id == -100:
                # ignored token
                continue
            pred_label = id2label.get(int(pred_ids[i, j]), "O")
            ref_label = id2label.get(ref_id, "O")
            preds_i.append(pred_label)
            refs_i.append(ref_label)

        if not refs_i:
            continue

        batch_preds.append(_to_bio(preds_i))
        batch_refs.append(_to_bio(refs_i))

    if not batch_refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = float(precision_score(batch_refs, batch_preds))
    recall = float(recall_score(batch_refs, batch_preds))
    f1 = float(f1_score(batch_refs, batch_preds))
    return {"precision": precision, "recall": recall, "f1": f1}


def train_token_classifier(
    train_file: str | Path,
    eval_file: str | Path,
    output_dir: str | Path,
    model_name: str = "microsoft/layoutlmv3-base",
    max_length: int = 512,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    num_train_epochs: float = 1.0,
    learning_rate: float = 5e-5,
    seed: int = 42,
) -> dict[str, Any]:
    _require_training_dependencies()
    set_reproducible_seed(seed)

    train_examples = load_token_examples(train_file)
    eval_examples = load_token_examples(eval_file)
    label_list = build_label_list(train_examples, eval_examples)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Training on CPU fallback (CUDA not available); this will be slower but functional.")

    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    train_dataset = _examples_to_dataset(train_examples, label2id)
    eval_dataset = _examples_to_dataset(eval_examples, label2id)

    tokenized_train = _tokenize_and_align_labels(train_dataset, processor, max_length)
    tokenized_eval = _tokenize_and_align_labels(eval_dataset, processor, max_length)

    run_config = {
        "seed": seed,
        "model_name": model_name,
        "max_length": max_length,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "device": str(device),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    training_arguments = TrainingArguments(
        output_dir=str(output_path),
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, Any]:
        logits, labels = eval_pred
        return compute_entity_metrics(logits, labels, label_list)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model(str(output_path))
    processor.save_pretrained(str(output_path))
    (output_path / "eval_metrics.json").write_text(json.dumps(eval_result, indent=2, default=str), encoding="utf-8")

    return {
        "labels": label_list,
        "run_config": run_config,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "output_dir": str(output_path),
    }


def smoke_test_token_training(num_labels: int = 3, seq_len: int = 128) -> dict[str, Any]:
    """Create a tiny LayoutLMv3 token-classification model and run a forward pass to validate shapes.

    This does not require downloading pretrained weights if transformers is available;
    it builds a config-backed model to avoid network fetches.
    """
    if LayoutLMv3ForTokenClassification is None or LayoutLMv3Config is None:
        raise RuntimeError("Transformers LayoutLMv3 classes unavailable")

    config = LayoutLMv3Config(num_labels=num_labels)
    model = LayoutLMv3ForTokenClassification(config)

    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    bbox = torch.zeros((1, seq_len, 4), dtype=torch.long)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
    logits = outputs.logits
    return {"logits_shape": list(logits.shape)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Token-classification training and evaluation workflow for LayoutLMv3.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a token classifier and evaluate it")
    train_parser.add_argument("--train-file", required=True)
    train_parser.add_argument("--eval-file", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    train_parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    train_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved token classifier")
    eval_parser.add_argument("--model-dir", required=True)
    eval_parser.add_argument("--eval-file", required=True)
    eval_parser.add_argument("--max-length", type=int, default=512)
    eval_parser.add_argument("--per-device-eval-batch-size", type=int, default=2)

    test_parser = subparsers.add_parser("smoke", help="Run a smoke test that checks model forward pass shapes")
    test_parser.add_argument("--num-labels", type=int, default=3)
    test_parser.add_argument("--seq-len", type=int, default=128)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        result = train_token_classifier(
            train_file=args.train_file,
            eval_file=args.eval_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_length=args.max_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
        )
    elif args.command == "evaluate":
        # For evaluation, reuse train_token_classifier loading/eval pathway is sufficient
        # to keep this script minimal; users should load the model and run evaluation separately.
        raise NotImplementedError("Use the Trainer API or the `train` command with eval data to evaluate.")
    else:
        result = smoke_test_token_training(num_labels=args.num_labels, seq_len=args.seq_len)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())