# OCR Attribute Extraction System

Production-oriented scaffold for a deep learning-driven OCR pipeline that ingests non-standard documents, rasterizes them at 300 DPI, classifies the document type, extracts text with Tesseract, applies document-type-specific attribute mapping with LayoutLMv3, and routes low-confidence fields to human review.

## Status

The current codebase focuses on ingestion, OCR, document-type classification, chunked LayoutLMv3 inference, spatial same-label routing, human review, and a minimal experimental token-classification training/evaluation workflow for LayoutLMv3.

## Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- LayoutLMv3
- Tesseract OCR v5
- OpenCV
- PyMuPDF (fitz)
- python-docx + LibreOffice for `.doc` conversion, with a direct PNG fallback for `.docx`
- Pillow (PIL)
- scikit-learn metrics
- FastAPI backend
- Celery worker with Redis broker
- PostgreSQL database
- React + Vite + Tailwind frontend
- Alembic migrations

## Project Layout

- `backend/app/` FastAPI app, ingestion pipeline, OCR processor, inference, models, and routing
- `backend/alembic/` database migration environment
- `frontend/` HITL dashboard UI
- `datasets/raw-pages/` rasterized page storage for application, CVAT, and training
- `uploads/` raw upload staging directory

## Local Development

1. Copy [.env.example](.env.example) to `.env` and adjust values if needed, or use the provided local `.env` for SQLite.
2. Install Python dependencies with `pip install -r requirements.txt`.
3. Install frontend dependencies with `cd frontend && npm install`.
4. If you want the full Docker stack, start infrastructure with `docker compose up redis`. Set `DATABASE_URL` to PostgreSQL when using a PostgreSQL service; otherwise the backend uses the explicit local SQLite default.
5. Run migrations with `alembic -c backend/alembic.ini upgrade head` if using PostgreSQL. SQLite will create tables automatically.
6. Start the backend with `python scripts/check_port_free.py --port 8000 && python -m uvicorn app.main:app --reload --app-dir backend`.
7. Start the Celery worker with `celery -A app.tasks.celery_app worker --loglevel=info --app-dir backend`.
8. Start the frontend with `cd frontend && npm run dev`.

If port 8000 is already occupied, the preflight check exits with a clear message: `Port 8000 already in use by PID X — kill it first or choose another port.`

## Training requirements and reproducibility

The LayoutLMv3 training path is intentionally CPU-safe; if CUDA is unavailable, the code falls back to CPU and logs a warning instead of failing hard. This is slower but still allows smoke tests and small runs to complete.

Recommended training hardware:

- CPU fallback: works for tiny smoke tests and debugging, but training is slow and not recommended for real production training.
- Minimum GPU VRAM for a small LayoutLMv3 token-classification run: approximately 8 GB VRAM.
- Recommended GPU VRAM: 16-24 GB for comfortable batch sizes and stable training on real annotation sets.
- For very large batch sizes or longer sequences, 24+ GB is preferred.

This repository has been smoke-tested in the current active environment, using versions confirmed by `pip show`:

- `torch 2.13.0`
- `transformers 5.13.1`
- `accelerate 1.14.0`
- `datasets 5.0.0`
- `seqeval 1.2.2`

`cuda_available == False` and `cuda_device_count == 0` in this environment.

The pinned install target in `requirements.txt` remains `torch==2.3.1`, `transformers==4.43.0`, `datasets==2.20.0`, and `accelerate==0.33.0`; the smoke-test environment above reflects the actual packages installed in the current venv and should be treated as the verified runtime state for this session.

Reproducibility controls:

- A fixed global seed is set via `set_reproducible_seed()` and applied to Python, NumPy, and PyTorch before training starts.
- The same seed is also passed to `TrainingArguments(seed=...)` so dataset shuffling and model initialization are deterministic.
- Training runs save a `run_config.json` file in the output directory and `eval_metrics.json` after evaluation, which makes the run settings and metrics traceable later.
- The code splits datasets with a seeded `random.Random(seed)` shuffle before creating train/val/test partitions, rather than relying on random behavior only during model initialization.

A dedicated training Docker image is not required for the current dev workflow; the existing development environment is sufficient for CPU smoke tests and small-scale experiments, while larger GPU jobs benefit from a GPU-enabled environment.

## Run The System

To run the full system locally:

1. Start the database and Redis if you are using Docker-based infrastructure: `docker compose up db redis`.
2. Start the backend API from the repository root: `uvicorn app.main:app --reload --app-dir backend`.
3. Start the Celery worker in a second terminal: `celery -A app.tasks.celery_app worker --loglevel=info --app-dir backend`.
4. Start the frontend in a third terminal: `cd frontend && npm run dev`.
5. Open the dashboard at `http://localhost:5173` and the API docs at `http://localhost:8000/docs`.

Set `DATABASE_URL` explicitly to select PostgreSQL or SQLite. A configured PostgreSQL connection failure stops startup instead of switching to a different database, preventing split state and apparent data loss.

Celery runs in asynchronous mode by default. Set `CELERY_TASK_ALWAYS_EAGER=true` in your environment if you want synchronous execution for local debugging or tests.

DOCX files are currently rasterized with a direct text-to-PNG fallback in the ingestion pipeline. LibreOffice/PDF conversion is used for `.doc` inputs.

Documents are classified before LayoutLMv3 inference so the active attribute schema can be selected by type instead of applying one generic label set to every document. Long pages are processed with overlapping inference chunks so tokens are not silently dropped, and same-label spans are merged by spatial reading order instead of strict token adjacency.

`microsoft/layoutlmv3-base` is a pretrained LayoutLMv3 encoder, not an attribute-extraction checkpoint. Keep `LAYOUTLMV3_OCR_ONLY=true` when using that base model. For attribute extraction, set `LAYOUTLMV3_MODEL_SOURCE` to a checkpoint produced by the token-classification training workflow and set `LAYOUTLMV3_OCR_ONLY=false`.

When the same token appears in two adjacent overlap chunks, the stitched result keeps the higher-confidence prediction and uses the first-seen chunk only as the tie-breaker. No averaging or voting is applied.

Spatial clustering in `backend/app/routing_engine.py` uses these explicit thresholds: same-line merges require matching block/paragraph metadata and a horizontal gap no larger than 1.5x the wider box; the fallback same-block merge path allows a vertical gap no larger than 1.75x the taller box plus a left-edge delta no larger than 2.0x the wider box; and when line metadata is missing, the boxes must overlap vertically by at least 25% of the shorter box height to count as the same line. These values are conservative enough to join multi-token values while keeping nearby same-label fields on the same line separate.

## Training And Evaluation

The current training workflow in `backend/app/training.py` is a LayoutLMv3 token-classification pipeline, not a text-classification workflow. It consumes token-level JSONL records exported from approved attributes and uses `tokens`, `bboxes`, and `labels` for each example.

The format is the same as the output produced by `backend/app/exporter.py`:

```jsonl
{"document_id": 7, "page_id": 13, "image": "datasets/raw-pages/7/page-013.png", "tokens": ["Invoice", "total", "is", "42", "."], "bboxes": [[0, 0, 100, 30], [110, 0, 170, 30], [180, 0, 220, 30], [220, 0, 260, 30], [270, 0, 285, 30]], "labels": ["O", "TOTAL", "O", "TOTAL", "O"]}
{"document_id": 7, "page_id": 13, "image": "datasets/raw-pages/7/page-013.png", "tokens": ["Ship", "to", "Acme"], "bboxes": [[0, 0, 80, 25], [90, 0, 120, 25], [130, 0, 210, 25]], "labels": ["O", "O", "VENDOR"]}
```

Typical workflow:

1. Export approved attributes to token-level JSONL:

```bash
cd backend
python -m app.exporter output/train-export.jsonl
```

2. Split the exported data into train/val/test partitions with a fixed seed in the Python training pipeline.
3. Train and evaluate in one command:

```bash
cd backend
python -m app.training train --train-file output/train.jsonl --eval-file output/eval.jsonl --output-dir outputs/layoutlmv3-token-clf --seed 42
```

The current training code writes `run_config.json` and `eval_metrics.json` to the output directory and uses the `LayoutLMv3ForTokenClassification` model with `seqeval` metrics for entity-level evaluation.

## Ground Truth And Evaluation

The training loop is supervised: it must learn from independently verified labels, never from its own predictions. The dataset tools in `backend/app/dataset_tools.py` provide the repeatable workflow:

```bash
# From the backend directory
python -m app.dataset_tools cvat-to-jsonl --xml ../datasets/cvat/annotations.xml --image-root ../datasets/raw-pages --output ../datasets/ground-truth/ground_truth.jsonl
python -m app.dataset_tools predict --raw ../datasets/raw-pages --output ../datasets/predictions/predictions.jsonl
python -m app.dataset_tools evaluate --predictions ../datasets/predictions/predictions.jsonl --ground-truth ../datasets/ground-truth/ground_truth.jsonl
python -m app.dataset_tools convert --ground-truth ../datasets/ground-truth/source --output ../datasets/ground-truth/train.jsonl
python -m app.training train --train-file ../datasets/ground-truth/train.jsonl --eval-file ../datasets/ground-truth/validation.jsonl --output-dir ../datasets/models/layoutlmv3
```

Each ground-truth JSON file must contain `tokens`, normalized `bboxes` in 0..1000 coordinates, and `labels`; optional `fields` values enable exact field-match reporting. Keep train, validation, and test documents separate before training to prevent data leakage. The evaluation command reports token accuracy, bounding-box IoU match rate, per-label precision/recall/F1, and optional field exact-match rate.

CVAT rectangle exports can be converted into the same token-level JSONL format. See [cvat/README.md](cvat/README.md) for the self-hosted CVAT workflow. CVAT is an annotation tool here, not the model or training engine.

The `evaluate` subcommand is not currently implemented as a standalone CLI in this version; the `train` command performs both training and evaluation in the same run. For a real project run, create a separate train/eval dataset split and keep the run seed fixed so the trace remains reproducible.

## Docker Compose

The backend container runs Alembic migrations automatically before launching the API. Use `docker compose up --build` from the repository root to start the full stack.

## API

- `GET /health`
- `POST /api/v1/ingest/validate`
- `POST /api/v1/ingest`
- `POST /api/v1/attributes/correct`
- `GET /api/v1/attributes/export/{document_id}/json`
- `GET /api/v1/attributes/export/{document_id}/csv`
