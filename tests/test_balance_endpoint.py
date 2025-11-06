import pytest
from fastapi.testclient import TestClient

from app.config import AppEnvironment, reload_settings
from app.main import create_app


def _configure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", AppEnvironment.DEVELOPMENT.value)
    monkeypatch.setenv("DELTA_API_KEY", "testkey")
    monkeypatch.setenv("DELTA_API_SECRET", "testsecret")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "[\"http://localhost:3000\"]")
    monkeypatch.setenv("TRUSTED_HOSTS", "[\"testserver\",\"localhost\"]")
    monkeypatch.setenv("ENFORCE_HTTPS", "false")
    monkeypatch.setenv("CONTENT_SECURITY_POLICY", "default-src 'self'")


def test_api_balance_returns_usdt(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    def fake_safe_api_request(*_: object, **__: object) -> dict[str, object]:
        return {
            "result": [
                {
                    "asset_symbol": "USDC",
                    "available_balance": "50.00",
                },
                {
                    "asset_symbol": "USDT",
                    "available_balance": "123.45",
                },
            ]
        }

    monkeypatch.setattr("app.main.safe_api_request", fake_safe_api_request)

    settings = reload_settings()
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/api/balance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset"] == "USDT"
    assert payload["balance"] == 123.45
    assert payload["available"] is True


def test_api_balance_handles_missing_usdt(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    def fake_safe_api_request(*_: object, **__: object) -> dict[str, object]:
        return {
            "result": [
                {
                    "asset_symbol": "USDC",
                    "available_balance": "50.00",
                }
            ]
        }

    monkeypatch.setattr("app.main.safe_api_request", fake_safe_api_request)

    settings = reload_settings()
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/api/balance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset"] is None
    assert payload["balance"] == 0.0
    assert payload["available"] is False
    assert payload["message"] == "USDT balance not available"
