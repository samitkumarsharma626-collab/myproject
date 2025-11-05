"""
Production-ready Flask application for ETH price and wallet balance.

Key improvements over the previous implementation:
- Migrated to Flask (as requested) with clean, modern structure
- Environment-driven configuration (no hard-coded secrets or paths)
- Robust error handling and timeouts for outbound HTTP calls
- Safe static file serving from the local `static` directory
- Minimal, dependency-light code (removed unused imports like pandas)
"""

from __future__ import annotations

import hmac
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory


# Load environment variables early
load_dotenv()


# Configuration with sensible defaults and environment overrides
class Config:
    # External API configuration
    DELTA_API_KEY: Optional[str] = os.getenv("DELTA_API_KEY")
    DELTA_API_SECRET: Optional[str] = os.getenv("DELTA_API_SECRET")
    DELTA_API_BASE_URL: str = os.getenv(
        "DELTA_API_BASE_URL", "https://api.delta.exchange"
    ).rstrip("/")

    # Application settings
    SYMBOL: str = os.getenv("SYMBOL", "ETHUSD")
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")


# Initialize Flask app
app = Flask(__name__, static_folder=None)
app.config.from_object(Config)


# Logging: production-friendly basic configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ethbot")


# Resolve static directory relative to this file (workspace/static)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def _json_request(url: str, *, timeout: int) -> Dict[str, Any]:
    """Perform a GET request and return JSON with consistent error handling."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.warning("Request timed out: %s", url)
        return {"success": False, "error": "timeout"}
    except requests.RequestException as exc:
        logger.exception("Request failed: %s", url)
        return {"success": False, "error": str(exc)}
    except ValueError:
        logger.exception("Invalid JSON from: %s", url)
        return {"success": False, "error": "invalid_json"}


def _sign_delta_request(method: str, path: str, timestamp: str, secret: str) -> str:
    """Create HMAC SHA-256 signature used by Delta Exchange.

    Message format: method + timestamp + path
    """
    message = f"{method}{timestamp}{path}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


@app.get("/")
def index():
    if STATIC_DIR.exists():
        return send_from_directory(STATIC_DIR.as_posix(), "index.html")
    # Fallback minimal response if static file is missing
    return jsonify({"ok": True, "msg": "ETH Bot API", "static": False})


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "ethbot", "env": app.config["ENVIRONMENT"]})


@app.get("/price")
def get_price():
    symbol = app.config["SYMBOL"]
    base_url = app.config["DELTA_API_BASE_URL"]
    timeout = app.config["REQUEST_TIMEOUT_SECONDS"]

    url = f"{base_url}/v2/tickers/{symbol}"
    data = _json_request(url, timeout=timeout)

    # Handle error forms and normalize output
    if not isinstance(data, dict) or not data.get("success"):
        return jsonify({"symbol": symbol, "price": 0.0, "msg": "Failed to fetch"}), 502

    result = data.get("result", {}) or {}
    # Prefer mark_price, then index_price, then spot_price
    price_value = (
        result.get("mark_price")
        or result.get("index_price")
        or result.get("spot_price")
        or 0
    )

    try:
        price = round(float(price_value), 6)
    except (TypeError, ValueError):
        price = 0.0

    return jsonify({"symbol": symbol, "price": price, "msg": "OK"})


@app.get("/balance")
def get_balance():
    api_key = app.config["DELTA_API_KEY"]
    api_secret = app.config["DELTA_API_SECRET"]
    base_url = app.config["DELTA_API_BASE_URL"]
    timeout = app.config["REQUEST_TIMEOUT_SECONDS"]

    if not api_key or not api_secret:
        return (
            jsonify({"error": "Missing DELTA_API_KEY/DELTA_API_SECRET"}),
            500,
        )

    method = "GET"
    path = "/v2/wallet/balances"
    timestamp = str(int(time.time()))
    signature = _sign_delta_request(method, path, timestamp, api_secret)

    headers = {
        "Accept": "application/json",
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "ethbot-flask-client",
    }

    url = f"{base_url}{path}"
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        return jsonify({"error": "timeout"}), 504
    except requests.RequestException as exc:
        logger.exception("Balance request failed")
        return jsonify({"error": str(exc)}), 502
    except ValueError:
        return jsonify({"error": "invalid_json"}), 502

    assets = data.get("result", []) if isinstance(data, dict) else []
    # Prefer USD-like assets
    preferred = {"USD", "USDT", "USDC", "MARGIN", "CASH"}
    asset = next((x for x in assets if x.get("asset_symbol") in preferred), None)

    if not asset:
        return jsonify({"balance": 0.0, "symbol": "", "msg": "No preferred asset found"})

    balance_raw = asset.get("available_balance", asset.get("balance", 0))
    try:
        balance = float(balance_raw)
    except (TypeError, ValueError):
        balance = 0.0

    return jsonify(
        {
            "balance": balance,
            "symbol": asset.get("asset_symbol", ""),
            "msg": "OK",
        }
    )


# Expose `app` for WSGI servers (gunicorn, uWSGI)
__all__ = ["app"]

