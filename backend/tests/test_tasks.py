from __future__ import annotations

import importlib

from backend.app import tasks


def test_celery_defaults_to_async_mode(monkeypatch) -> None:
    monkeypatch.delenv("CELERY_TASK_ALWAYS_EAGER", raising=False)

    reloaded = importlib.reload(tasks)

    assert reloaded.celery_app.conf.task_always_eager is False


def test_celery_honors_eager_override(monkeypatch) -> None:
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")

    reloaded = importlib.reload(tasks)

    assert reloaded.celery_app.conf.task_always_eager is True