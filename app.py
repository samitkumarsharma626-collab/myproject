from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, abort, current_app, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("ethbot.app")

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "ethbot/1.0",
})


class DeltaAPIError(RuntimeError):
    """Raised when the Delta Exchange API returns an error or invalid response."""


def create_app() -> Flask:
    """Application factory for the trading helper service."""

    static_root = _determine_static_root()

    app = Flask(__name__, static_folder=static_root, static_url_path="/static")

    price_base_url = os.getenv("DELTA_PRICE_BASE_URL")
    api_base_url = os.getenv("DELTA_API_BASE_URL", "https://api.delta.exchange")

    asset_pref_raw = os.getenv("BALANCE_ASSET_PREFERENCE", "USD,USDT,USDC,MARGIN,CASH")
    asset_preferences = tuple(
        asset.strip().upper() for asset in asset_pref_raw.split(",") if asset.strip()
    ) or ("USD", "USDT", "USDC", "MARGIN", "CASH")

    app.config.update(
        DELTA_API_KEY=os.getenv("DELTA_API_KEY"),
        DELTA_API_SECRET=os.getenv("DELTA_API_SECRET"),
        DELTA_API_BASE_URL=api_base_url.rstrip("/"),
        DELTA_PRICE_BASE_URL=(price_base_url or api_base_url).rstrip("/"),
        DELTA_PRICE_SYMBOL=os.getenv("DELTA_PRICE_SYMBOL", "ETHUSD"),
        REQUEST_TIMEOUT=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
        BALANCE_ASSET_PREFERENCE=asset_preferences,
    )

    register_routes(app)
    register_error_handlers(app)

    app.logger.debug("Application configured with static root %s", static_root)
    return app


def register_routes(app: Flask) -> None:
    """Attach HTTP routes to the Flask application."""

    @app.route("/", methods=["GET"])
    def serve_index() -> Any:
        try:
            return send_from_directory(app.static_folder, "index.html")
        except FileNotFoundError:
            abort(404, description="Static index file not found")

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify(
            {
                "ok": True,
                "message": "Service healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.route("/price", methods=["GET"])
    def price() -> Any:
        symbol = current_app.config["DELTA_PRICE_SYMBOL"]
        base_url = current_app.config["DELTA_PRICE_BASE_URL"]
        url = f"{base_url}/v2/tickers/{symbol}"

        try:
            payload = _fetch_json("GET", url)
        except DeltaAPIError as exc:
            current_app.logger.warning("Failed to fetch price for %s: %s", symbol, exc)
            return (
                jsonify({"symbol": symbol, "price": None, "message": str(exc)}),
                502,
            )

        if not payload.get("success", False):
            message = _extract_error_message(payload)
            current_app.logger.warning("Delta API returned failure for price: %s", message)
            return (
                jsonify({"symbol": symbol, "price": None, "message": message}),
                502,
            )

        price_value = _extract_price(payload.get("result", {}))
        if price_value is None:
            current_app.logger.warning("No price field present in response for %s", symbol)
            return (
                jsonify({"symbol": symbol, "price": None, "message": "Price unavailable"}),
                502,
            )

        return jsonify(
            {
                "symbol": symbol,
                "price": round(price_value, 2),
                "message": "ETH live price fetched successfully",
            }
        )

    @app.route("/balance", methods=["GET"])
    def balance() -> Any:
        path = "/v2/wallet/balances"
        base_url = current_app.config["DELTA_API_BASE_URL"]
        url = f"{base_url}{path}"

        try:
            headers = _build_auth_headers("GET", path)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive programming
            current_app.logger.exception("Failed to prepare auth headers: %s", exc)
            return jsonify({"message": "Unable to prepare request"}), 500

        try:
            payload = _fetch_json("GET", url, headers=headers)
        except DeltaAPIError as exc:
            current_app.logger.warning("Failed to fetch balance: %s", exc)
            return (
                jsonify({"asset": None, "balance": None, "message": str(exc)}),
                502,
            )

        assets = payload.get("result") or []
        asset_preferences = current_app.config["BALANCE_ASSET_PREFERENCE"]
        selected_asset = _select_asset(assets, asset_preferences)

        if not selected_asset:
            current_app.logger.info("No asset matched preferences %s", asset_preferences)
            return (
                jsonify(
                    {
                        "asset": None,
                        "balance": 0.0,
                        "message": "No preferred asset found",
                    }
                ),
                404,
            )

        available_balance = _coerce_float(
            selected_asset.get("available_balance", selected_asset.get("balance"))
        )
        total_balance = _coerce_float(selected_asset.get("balance", available_balance))

        return jsonify(
            {
                "asset": selected_asset.get("asset_symbol"),
                "available_balance": available_balance,
                "balance": total_balance,
                "message": "Balance fetched successfully",
            }
        )


def register_error_handlers(app: Flask) -> None:
    """Ensure all errors return JSON responses."""

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException) -> Any:
        response_payload = {
            "message": exc.description or exc.name,
            "status": exc.code,
        }
        return jsonify(response_payload), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected_exception(exc: Exception) -> Any:
        app.logger.exception("Unhandled server error: %s", exc)
        return jsonify({"message": "Internal server error"}), 500


def _determine_static_root() -> str:
    configured_root = os.getenv("STATIC_ROOT")
    candidate_paths = []
    if configured_root:
        candidate_paths.append(os.path.abspath(configured_root))

    default_root = os.path.join(os.path.dirname(__file__), "static")
    candidate_paths.append(default_root)

    for path in candidate_paths:
        if os.path.isdir(path):
            return path

    LOGGER.warning(
        "Static root directories %s were not found; using first candidate %s",
        candidate_paths,
        candidate_paths[0],
    )
    return candidate_paths[0]


def _fetch_json(method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    timeout = current_app.config["REQUEST_TIMEOUT"]

    try:
        response = SESSION.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise DeltaAPIError("Delta API request timed out") from exc
    except requests.RequestException as exc:
        raise DeltaAPIError("Delta API request failed") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise DeltaAPIError("Delta API returned invalid JSON") from exc


def _build_auth_headers(method: str, path: str) -> Dict[str, str]:
    api_key, api_secret = _get_credentials()
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    message = f"{method.upper()}{timestamp}{path}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "Accept": "application/json",
        "User-Agent": SESSION.headers.get("User-Agent", "ethbot/1.0"),
    }
    return headers


def _get_credentials() -> Tuple[str, str]:
    api_key = current_app.config.get("DELTA_API_KEY")
    api_secret = current_app.config.get("DELTA_API_SECRET")

    if not api_key or not api_secret:
        abort(503, description="Delta API credentials are not configured")

    return api_key, api_secret


def _extract_price(result: Dict[str, Any]) -> Optional[float]:
    price_fields = ("mark_price", "index_price", "spot_price", "price", "last_price")

    for field in price_fields:
        value = result.get(field)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _extract_error_message(payload: Dict[str, Any]) -> str:
    error = payload.get("error") or {}
    message = error.get("message") or payload.get("message")
    if message:
        return str(message)
    return "Delta API reported failure"


def _select_asset(assets: Iterable[Dict[str, Any]], preferences: Iterable[str]) -> Optional[Dict[str, Any]]:
    assets_list = list(assets)
    if not assets_list:
        return None

    pref_upper = [pref.upper() for pref in preferences]

    for pref in pref_upper:
        for asset in assets_list:
            symbol = (asset.get("asset_symbol") or "").upper()
            if symbol == pref:
                return asset

    return assets_list[0]


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
