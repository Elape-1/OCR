# OCR Learning Workflow

This guide explains exactly where files go and the order to follow when creating ground truth, evaluating predictions, and training the OCR model.

## Important

- The system uses LayoutLMv3 for token classification.
- CVAT is a separate annotation application. It is not part of this project's Docker Compose stack.
- CVAT creates ground-truth annotations. It does not train the OCR model.
- Never use the model's predictions as ground truth.
- Use the same generated page images for both CVAT and prediction.

## Folder Layout

```text
D:\OCR\
  datasets/
    raw-pages/                 Generated PNG pages used by CVAT and prediction
      <document_id>/
        page-001.png
        page-002.png
    cvat/                      CVAT export files
      annotations.xml
    ground-truth/
      ground_truth.jsonl       Converted CVAT annotations
      train.jsonl              Training split
      validation.jsonl         Validation split
      test.jsonl               Final holdout split
    predictions/
      predictions.jsonl        Model predictions
    models/
      layoutlmv3/              Trained model output
  uploads/                     Original uploaded PDFs; runtime storage
  samples/                     Sample files for development/tests
```

The old `assets/` folder was removed. Page images are now served by the backend from `datasets/raw-pages/` through `/pages/...`.

## Complete Workflow

### 1. Start the system

Start the backend:

```powershell
cd D:\OCR
python -m uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
cd D:\OCR\frontend
npm run dev -- --host localhost --port 5173
```

Open:

- Dashboard: http://localhost:5173/
- API: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs/

### 2. Upload a PDF once

Upload the PDF through the dashboard. The system automatically creates a document ID, for example `42`.

It stores the original file here:

```text
uploads/42/source.pdf
```

It creates the page images here:

```text
datasets/raw-pages/42/page-001.png
datasets/raw-pages/42/page-002.png
```

The page images are generated in the correct page order. Do not rename, edit, or reorder them before CVAT annotation.

### 3. Annotate the page images in CVAT

CVAT runs separately. Follow [cvat/README.md](cvat/README.md) to install and start it.

In CVAT:

1. Create a task.
2. Upload the exact PNG files from `datasets/raw-pages/<document_id>/`.
3. Create labels such as `TOTAL`, `DATE`, `VENDOR`, or other fields in your documents.
4. Draw rectangles around complete field values.
5. Finish all pages.
6. Export as **CVAT for images 1.1 XML**.

Place the exported file here:

```text
datasets/cvat/annotations.xml
```

The CVAT XML file is the annotation result. You do not need to create a JSON file manually for every page.

### 4. Convert the CVAT result

From the backend directory:

```powershell
cd D:\OCR\backend

python -m app.dataset_tools cvat-to-jsonl `
  --xml ..\datasets\cvat\annotations.xml `
  --image-root ..\datasets\raw-pages\42 `
  --output ..\datasets\ground-truth\ground_truth.jsonl
```

The converter runs the project's OCR over the same page images and assigns each OCR token the label of the CVAT rectangle it overlaps.

The result is:

```text
datasets/ground-truth/ground_truth.jsonl
```

Review this file before training. Pay special attention to words that cross annotation-box boundaries.

### 5. Generate model predictions

You do not need to upload the PDF again. Use the saved page-image folder:

```powershell
cd D:\OCR\backend

python -m app.dataset_tools predict `
  --raw ..\datasets\raw-pages\42 `
  --output ..\datasets\predictions\predictions.jsonl
```

The model guesses are saved here:

```text
datasets/predictions/predictions.jsonl
```

### 6. Compare predictions with ground truth

```powershell
cd D:\OCR\backend

python -m app.dataset_tools evaluate `
  --predictions ..\datasets\predictions\predictions.jsonl `
  --ground-truth ..\datasets\ground-truth\ground_truth.jsonl
```

The evaluation reports:

- Token accuracy
- Bounding-box IoU match rate
- Per-label precision
- Per-label recall
- Per-label F1 score
- Optional field exact-match rate

Comparison measures the model. It does not train the model.

### 7. Prepare training, validation, and test data

Do not train and evaluate on the exact same documents. Separate the ground truth into three document groups:

```text
ground-truth/train.jsonl
 ground-truth/validation.jsonl
 ground-truth/test.jsonl
```

Recommended split:

- 70% training documents
- 15% validation documents
- 15% test documents

The test documents must remain unseen until final evaluation.

### 8. Train LayoutLMv3

```powershell
cd D:\OCR\backend

python -m app.training train `
  --train-file ..\datasets\ground-truth\train.jsonl `
  --eval-file ..\datasets\ground-truth\validation.jsonl `
  --output-dir ..\datasets\models\layoutlmv3 `
  --seed 42
```

Training uses the verified ground-truth labels to update the model weights. It saves the trained model and processor in:

```text
datasets/models/layoutlmv3/
```

### 9. Use the trained model

Set the model source in `.env` to the trained model directory:

```text
LAYOUTLMV3_MODEL_SOURCE=datasets/models/layoutlmv3
```

Restart the backend after changing `.env`. New predictions will then use the trained model instead of the base `microsoft/layoutlmv3-base` model.

## Simple Process Summary

```text
PDF uploaded once
    -> datasets/raw-pages/<document_id>/page-###.png

Same page PNGs
    -> CVAT annotation
    -> CVAT annotations.xml
    -> ground-truth/ground_truth.jsonl

Same page PNGs
    -> LayoutLMv3 prediction
    -> predictions/predictions.jsonl

Ground truth + predictions
    -> evaluation metrics

Ground truth training split
    -> LayoutLMv3 fine-tuning
    -> datasets/models/layoutlmv3/
```

## What Goes Where

| Item | Location |
|---|---|
| Original PDF | `uploads/<document_id>/source.pdf` |
| Generated page images | `datasets/raw-pages/<document_id>/` |
| CVAT export | `datasets/cvat/annotations.xml` |
| Converted ground truth | `datasets/ground-truth/ground_truth.jsonl` |
| Training data | `datasets/ground-truth/train.jsonl` |
| Validation data | `datasets/ground-truth/validation.jsonl` |
| Test data | `datasets/ground-truth/test.jsonl` |
| Model guesses | `datasets/predictions/predictions.jsonl` |
| Trained model | `datasets/models/layoutlmv3/` |

## Fresh Start Cleanup

To begin again, stop the backend and remove only generated/user data:

```powershell
Remove-Item -Recurse -Force datasets\raw-pages\*
Remove-Item -Recurse -Force datasets\cvat\*
Remove-Item -Recurse -Force datasets\ground-truth\source\*
Remove-Item -Recurse -Force datasets\predictions\*
Remove-Item -Recurse -Force datasets\models\*
Remove-Item -Recurse -Force uploads\*
Remove-Item -Force ocr_system.db, backend\ocr_system.db -ErrorAction SilentlyContinue
```

Keep the source code, `samples/`, and the README files.

## Online Multi-User Deployment

Before exposing the system online, configure these server-only values in the deployment environment:

```text
AUTH_REQUIRED=true
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_JWT_SECRET=<server-only JWT signing secret>
SUPABASE_SERVICE_ROLE_KEY=<server-only service-role key>
SUPABASE_STORAGE_BUCKET=ocr-documents
```

Create a private Supabase Storage bucket named `ocr-documents`. Never put the service-role key in the frontend. Set these public frontend values separately:

```text
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=<publishable-or-anon-key>
```

The backend and Celery worker use the Supabase database, Redis, and private Storage. The browser must sign in with Supabase Auth. Every document is assigned to the authenticated user's ID, and document, page, attribute, correction, search, export, and image requests are owner-scoped. RLS is also enabled in Supabase as defense in depth.

Test deployment isolation with two Supabase Auth accounts: user A must not be able to list, open, search, edit, export, delete, or retrieve page images belonging to user B.
