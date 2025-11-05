import pytest

from app.config import AppEnvironment, SettingsError, get_settings, reload_settings


def test_settings_parse_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "[\"https://example.com\",\"http://localhost:3000\"]")
    monkeypatch.setenv("TRUSTED_HOSTS", "[\"localhost\",\"testserver\"]")
    monkeypatch.delenv("DELTA_API_KEY", raising=False)
    monkeypatch.delenv("DELTA_API_SECRET", raising=False)

    settings = reload_settings()

    assert settings.cors_allow_origins == ["https://example.com", "http://localhost:3000"]
    assert settings.trusted_hosts == ["localhost", "testserver"]
    assert not settings.is_production


def test_production_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", AppEnvironment.PRODUCTION.value)
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "[\"https://prod.example.com\"]")
    monkeypatch.setenv("TRUSTED_HOSTS", "[\"prod.example.com\"]")
    monkeypatch.delenv("DELTA_API_KEY", raising=False)
    monkeypatch.delenv("DELTA_API_SECRET", raising=False)

    with pytest.raises(SettingsError) as exc:
        _ = get_settings()

    assert "Missing required environment variables" in str(exc.value)


def test_production_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", AppEnvironment.PRODUCTION.value)
    monkeypatch.setenv("DELTA_API_KEY", "key")
    monkeypatch.setenv("DELTA_API_SECRET", "secret")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "[\"https://prod.example.com\"]")
    monkeypatch.setenv("TRUSTED_HOSTS", "[\"prod.example.com\"]")

    settings = reload_settings()

    assert settings.is_production
    assert settings.get_api_key() == "key"
    assert settings.get_api_secret() == "secret"
    assert settings.enforce_https is True


def test_reload_settings_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "[\"https://example.com\"]")
    monkeypatch.setenv("TRUSTED_HOSTS", "[\"localhost\"]")
    monkeypatch.setenv("APP_ENV", AppEnvironment.DEVELOPMENT.value)

    reload_settings()

    messages = "".join(record.message for record in caplog.records)
    assert "Settings reloaded" in messages
