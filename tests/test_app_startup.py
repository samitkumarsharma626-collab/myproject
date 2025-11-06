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


def test_app_uses_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    settings = reload_settings()
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["mode"] == AppEnvironment.DEVELOPMENT.value


def test_validation_errors_are_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    settings = reload_settings()
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/api/v1/candles", params={"limit": 0})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "Validation Error"
    assert "details" in payload


def test_balance_endpoint_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    monkeypatch.delenv("DELTA_API_KEY", raising=False)
    monkeypatch.delenv("DELTA_API_SECRET", raising=False)

    settings = reload_settings()
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/api/balance")
    assert response.status_code == 401
    assert response.json()["detail"] == "API credentials not configured"


def test_balance_endpoint_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    fake_response = {
        "result": [
            {
                "asset_symbol": "USDT",
                "available_balance": "1500.12",
            }
        ]
    }

    def fake_safe_api_request(*args, **kwargs):  # type: ignore[unused-arg]
        return fake_response

    monkeypatch.setattr("app.main.safe_api_request", fake_safe_api_request)

    settings = reload_settings()
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/api/balance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["balance"] == 1500.12
    assert payload["asset"] == "USDT"
    assert payload["available"] is True
