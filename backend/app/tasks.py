from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ocr_system",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.ingestion_pipeline"],
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_always_eager=os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() in {"1", "true", "yes", "on"},
)
