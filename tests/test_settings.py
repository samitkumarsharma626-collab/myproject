import os
from pathlib import Path

from app_settings import Settings


def test_settings_defaults_and_validation(tmp_path: Path, monkeypatch):
    # Ensure minimal env
    monkeypatch.delenv("DELTA_API_KEY", raising=False)
    monkeypatch.delenv("DELTA_API_SECRET", raising=False)
    monkeypatch.setenv("STATIC_DIR", str(tmp_path))

    s = Settings()

    assert s.app_name
    assert s.candle_resolution in {"1m","3m","5m","15m","30m","1h","2h","4h","1d"}
    assert s.static_dir == tmp_path
    assert s.request_timeout_seconds > 0
    assert s.is_valid() is True

    # Missing creds should raise when explicitly required
    try:
        s.require_api_credentials()
        raised = False
    except ValueError:
        raised = True
    assert raised is True


def test_prod_defaults_secure():
    s = Settings()
    assert s.security.force_https is True
    assert s.security.secure_cookies is True
