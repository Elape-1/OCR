from __future__ import annotations

import importlib
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

import app.main as main_module


def test_default_cors_is_restricted_without_explicit_dev_mode(monkeypatch) -> None:
    monkeypatch.delenv("DEV_ALLOW_ALL_CORS", raising=False)
    monkeypatch.delenv("ENV", raising=False)

    reloaded = importlib.reload(main_module)

    assert reloaded.allow_all_cors is False
    assert reloaded.merged_local_origins
    assert "*" not in reloaded.merged_local_origins
    assert "http://localhost:5173" in reloaded.merged_local_origins


def test_development_env_enables_permissive_cors_only_when_explicit(monkeypatch) -> None:
    monkeypatch.delenv("DEV_ALLOW_ALL_CORS", raising=False)
    monkeypatch.setenv("ENV", "development")

    reloaded = importlib.reload(main_module)

    assert reloaded.allow_all_cors is True
    assert reloaded.app.user_middleware
