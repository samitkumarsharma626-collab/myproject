"""Application package initialization."""

from __future__ import annotations

from .config import AppSettings, SettingsError, get_settings
from .main import create_app

try:
    app = create_app()
except SettingsError as exc:  # pragma: no cover - fail fast during startup
    raise SystemExit(str(exc)) from exc

__all__ = [
    "AppSettings",
    "SettingsError",
    "app",
    "create_app",
    "get_settings",
]
