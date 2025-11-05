"""FastAPI application entry point."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import AppSettings, get_settings
from .logging_config import setup_logging
from .middleware import ExceptionHandlingMiddleware
from .schemas import (
    BalanceResponse,
    Candle,
    CandleResponse,
    CurrentCandle,
    Filters,
    HealthResponse,
    Indicators,
    PriceBoxResponse,
    PriceResponse,
    StrategyFiltersResponse,
    SuggestedBox,
    Thresholds,
)
from .security import configure_security

logger = logging.getLogger(__name__)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure a FastAPI application instance."""

    settings = settings or get_settings()
    setup_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description="Real-time ETH strategy monitoring and trading API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        debug=settings.debug,
    )

    app.state.settings = settings

    configure_security(app, settings)
    app.add_middleware(ExceptionHandlingMiddleware, settings=settings)

    if settings.static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
        logger.info("Static files mounted", extra={"event": "static_mount", "path": str(settings.static_dir)})
    else:
        logger.warning(
            "Static directory missing",
            extra={"event": "static_missing", "path": str(settings.static_dir)},
        )

    register_routes(app)
    register_exception_handlers(app)
    register_events(app)

    logger.info("Application configured", extra={"config": settings.masked()})

    return app


def get_app_settings(request: Request) -> AppSettings:
    """Retrieve settings stored in application state."""

    return cast(AppSettings, request.app.state.settings)


def register_routes(app: FastAPI) -> None:
    """Register API routes with the FastAPI application."""

    register_root_routes(app)
    register_price_routes(app)
    register_balance_routes(app)
    register_strategy_routes(app)
    register_candle_routes(app)


def register_root_routes(app: FastAPI) -> None:
    @app.get("/", response_class=FileResponse)
    async def root(settings: AppSettings = Depends(get_app_settings)) -> FileResponse:
        index_file = settings.static_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(str(index_file))

    @app.get("/health", response_model=HealthResponse)
    async def health_check(settings: AppSettings = Depends(get_app_settings)) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            mode=settings.environment,
        )


def register_price_routes(app: FastAPI) -> None:
    @app.get("/api/v1/price", response_model=PriceResponse)
    async def get_current_price(settings: AppSettings = Depends(get_app_settings)) -> PriceResponse:
        url = f"{settings.delta_base_url}/v2/tickers/{settings.trading_symbol}"
        logger.info(
            "Fetching current price",
            extra={"event": "price_lookup", "symbol": settings.trading_symbol},
        )
        data = safe_api_request(url=url, settings=settings)
        result = data.get("result", {})
        price = float(
            result.get("mark_price")
            or result.get("index_price")
            or result.get("spot_price")
            or 0
        )

        if price == 0:
            logger.warning(
                "Price returned as zero",
                extra={"event": "price_zero", "symbol": settings.trading_symbol},
            )

        return PriceResponse(
            symbol=settings.trading_symbol,
            price=round(price, 2),
            timestamp=datetime.now(timezone.utc),
            source="delta_exchange",
        )

    @app.get("/price", response_model=PriceResponse)
    async def get_price_legacy(settings: AppSettings = Depends(get_app_settings)) -> PriceResponse:
        return await get_current_price(settings)

    @app.get("/api/v1/box", response_model=PriceBoxResponse)
    async def get_price_box(settings: AppSettings = Depends(get_app_settings)) -> PriceBoxResponse:
        url = f"{settings.delta_base_url}/v2/tickers/{settings.trading_symbol}"
        data = safe_api_request(url=url, settings=settings)

        result = data.get("result", {})
        high_24h = float(result.get("high", 0)) or None
        low_24h = float(result.get("low", 0)) or None

        box_range = (
            f"{round(low_24h, 2)} - {round(high_24h, 2)}"
            if high_24h is not None and low_24h is not None
            else "—"
        )

        return PriceBoxResponse(
            box=box_range,
            high=round(high_24h, 2) if high_24h is not None else None,
            low=round(low_24h, 2) if low_24h is not None else None,
            timestamp=datetime.now(timezone.utc),
        )

    @app.get("/box", response_model=PriceBoxResponse)
    async def get_box_legacy(settings: AppSettings = Depends(get_app_settings)) -> PriceBoxResponse:
        return await get_price_box(settings)


def register_balance_routes(app: FastAPI) -> None:
    @app.get("/api/v1/balance", response_model=BalanceResponse)
    async def get_wallet_balance(settings: AppSettings = Depends(get_app_settings)) -> BalanceResponse:
        api_key = settings.get_api_key()
        api_secret = settings.get_api_secret()

        if not api_key or not api_secret:
            raise HTTPException(status_code=401, detail="API credentials not configured")

        method = "GET"
        path = "/v2/wallet/balances"
        headers = get_auth_headers(method, path, settings)
        url = f"{settings.delta_base_url}{path}"

        logger.info("Fetching wallet balance", extra={"event": "balance_lookup"})
        data = safe_api_request(url=url, headers=headers, settings=settings, authenticated=True)

        assets = data.get("result", [])
        supported_assets = {"USD", "USDT", "USDC", "MARGIN", "CASH"}

        asset = next((item for item in assets if item.get("asset_symbol") in supported_assets), None)

        timestamp = datetime.now(timezone.utc)

        if not asset:
            logger.warning(
                "No supported asset found",
                extra={
                    "event": "balance_missing",
                    "assets": [item.get("asset_symbol") for item in assets],
                },
            )
            return BalanceResponse(
                balance=0.0,
                asset=None,
                timestamp=timestamp,
                available=False,
                message=f"No {', '.join(sorted(supported_assets))} asset found",
            )

        balance_value = float(asset.get("available_balance") or asset.get("balance") or 0)
        asset_symbol = asset.get("asset_symbol")

        logger.info(
            "Wallet balance fetched",
            extra={"event": "balance_success", "asset": asset_symbol},
        )

        return BalanceResponse(
            balance=round(balance_value, 2),
            asset=asset_symbol,
            timestamp=timestamp,
            available=True,
        )

    @app.get("/balance", response_model=BalanceResponse)
    async def get_balance_legacy(settings: AppSettings = Depends(get_app_settings)) -> BalanceResponse:
        return await get_wallet_balance(settings)


def register_strategy_routes(app: FastAPI) -> None:
    @app.get("/api/strategy/filters", response_model=StrategyFiltersResponse)
    async def get_strategy_filters(settings: AppSettings = Depends(get_app_settings)) -> StrategyFiltersResponse:
        df_15m = fetch_candle_data(settings.trading_symbol, "15m", limit=200, settings=settings)
        if df_15m.empty or len(df_15m) < 50:
            raise HTTPException(status_code=404, detail="Insufficient candle data for calculations")

        df_1h = fetch_candle_data(settings.trading_symbol, "1h", limit=250, settings=settings)

        current = df_15m.iloc[-1]
        candle_range = float(current["high"] - current["low"])

        df_15m["atr_14"] = calculate_atr(df_15m["high"], df_15m["low"], df_15m["close"], period=14)
        df_15m["adx_14"] = calculate_adx(df_15m["high"], df_15m["low"], df_15m["close"], period=14)
        df_15m["volume_sma20"] = calculate_sma(df_15m["volume"], period=20)

        atr_value = float(df_15m["atr_14"].iloc[-1]) if not pd.isna(df_15m["atr_14"].iloc[-1]) else 0.0
        adx_value = float(df_15m["adx_14"].iloc[-1]) if not pd.isna(df_15m["adx_14"].iloc[-1]) else 0.0
        current_volume = float(current["volume"])
        average_volume = (
            float(df_15m["volume_sma20"].iloc[-1])
            if not pd.isna(df_15m["volume_sma20"].iloc[-1])
            else current_volume
        )
        volume_ratio = (current_volume / average_volume * 100) if average_volume > 0 else 100.0

        trend_status = "neutral"
        trend_ok = False
        if not df_1h.empty and len(df_1h) >= 200:
            df_1h["sma50"] = calculate_sma(df_1h["close"], period=50)
            df_1h["sma200"] = calculate_sma(df_1h["close"], period=200)
            sma50 = df_1h["sma50"].iloc[-1]
            sma200 = df_1h["sma200"].iloc[-1]
            if not pd.isna(sma50) and not pd.isna(sma200):
                if sma50 > sma200:
                    trend_status = "bullish"
                    trend_ok = True
                elif sma50 < sma200:
                    trend_status = "bearish"
                    trend_ok = False
                else:
                    trend_status = "neutral"

        current_close = float(current["close"])
        atr_multiplier = 1.5
        suggested_high = current_close + (atr_value * atr_multiplier)
        suggested_low = current_close - (atr_value * atr_multiplier)

        RANGE_MIN = 5.0
        ADX_MIN = 20.0
        ATR_MIN = 10.0
        VOLUME_MIN = 80.0

        range_ok = candle_range >= RANGE_MIN
        adx_ok = adx_value >= ADX_MIN
        atr_ok = atr_value >= ATR_MIN
        volume_ok = volume_ratio >= VOLUME_MIN

        filters_passed = all([range_ok, adx_ok, atr_ok, volume_ok, trend_ok])

        logger.info(
            "Strategy filters calculated",
            extra={"event": "strategy_filters", "filters_passed": filters_passed},
        )

        return StrategyFiltersResponse(
            symbol=settings.trading_symbol,
            timestamp=datetime.now(timezone.utc),
            current_candle=CurrentCandle(
                open=round(float(current["open"]), 2),
                high=round(float(current["high"]), 2),
                low=round(float(current["low"]), 2),
                close=round(float(current_close), 2),
                volume=round(current_volume, 2),
            ),
            indicators=Indicators(
                range=round(candle_range, 2),
                adx_14=round(adx_value, 2),
                atr_14=round(atr_value, 2),
                volume_vs_sma20=round(volume_ratio, 2),
                trend_1h=trend_status,
            ),
            suggested_box=SuggestedBox(
                high=round(suggested_high, 2),
                low=round(suggested_low, 2),
                range=round(suggested_high - suggested_low, 2),
            ),
            filters=Filters(
                range_ok=range_ok,
                adx_ok=adx_ok,
                atr_ok=atr_ok,
                volume_ok=volume_ok,
                trend_ok=trend_ok,
            ),
            filters_passed=filters_passed,
            thresholds=Thresholds(
                range_min=RANGE_MIN,
                adx_min=ADX_MIN,
                atr_min=ATR_MIN,
                volume_min_pct=VOLUME_MIN,
            ),
        )

    @app.get("/api/v1/strategy-filters", response_model=StrategyFiltersResponse)
    async def get_strategy_filters_v1(settings: AppSettings = Depends(get_app_settings)) -> StrategyFiltersResponse:
        return await get_strategy_filters(settings)

    @app.get("/strategy-filters", response_model=StrategyFiltersResponse)
    async def get_strategy_filters_legacy(settings: AppSettings = Depends(get_app_settings)) -> StrategyFiltersResponse:
        return await get_strategy_filters(settings)


def register_candle_routes(app: FastAPI) -> None:
    @app.get("/api/v1/candles", response_model=CandleResponse)
    async def get_candles(
        limit: int = Query(default=100, ge=1, le=500),
        settings: AppSettings = Depends(get_app_settings),
    ) -> CandleResponse:
        end_time = int(time.time())
        start_time = end_time - (limit * 15 * 60)

        url = f"{settings.delta_base_url}/v2/history/candles"
        params = {
            "resolution": settings.candle_resolution,
            "symbol": settings.trading_symbol,
            "start": start_time,
            "end": end_time,
        }
        query_string = "&".join([f"{key}={value}" for key, value in params.items()])
        data = safe_api_request(url=f"{url}?{query_string}", settings=settings)

        candles = [Candle(**candle) for candle in data.get("result", [])]
        logger.info(
            "Candles fetched",
            extra={"event": "candles_fetched", "count": len(candles)},
        )

        return CandleResponse(
            symbol=settings.trading_symbol,
            resolution=settings.candle_resolution,
            candles=candles,
            count=len(candles),
            timestamp=datetime.now(timezone.utc),
        )


def register_events(app: FastAPI) -> None:
    """Register lifecycle event handlers."""

    @app.on_event("startup")
    async def on_startup() -> None:
        settings: AppSettings = app.state.settings
        logger.info(
            "Application starting",
            extra={
                "event": "startup",
                "config": settings.masked(),
            },
        )

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Application shutting down", extra={"event": "shutdown"})


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers for validation errors."""

    exception_logger = logging.getLogger("app.exceptions")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        correlation_id = request.headers.get("X-Request-ID", str(uuid4()))
        exception_logger.warning(
            "Validation failed",
            extra={
                "event": "validation_error",
                "path": str(request.url.path),
                "correlation_id": correlation_id,
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "correlation_id": correlation_id,
                "details": exc.errors(),
            },
        )


def create_signature(method: str, timestamp: str, path: str, body: str, settings: AppSettings) -> str:
    message = method + timestamp + path + body
    secret = settings.get_api_secret()
    if not secret:
        raise HTTPException(status_code=401, detail="API secret not configured")
    signature = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return signature


def get_auth_headers(method: str, path: str, settings: AppSettings, body: str = "") -> Dict[str, str]:
    timestamp = str(int(time.time()))
    signature = create_signature(method, timestamp, path, body, settings)
    api_key = settings.get_api_key()
    if not api_key:
        raise HTTPException(status_code=401, detail="API key not configured")
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "ethbot-fastapi/1.0",
    }


def safe_api_request(
    url: str,
    settings: AppSettings,
    headers: Optional[Dict[str, str]] = None,
    authenticated: bool = False,
) -> Dict[str, Any]:
    try:
        request_headers = headers or {"Accept": "application/json"}
        response = requests.get(url, headers=request_headers, timeout=settings.request_timeout)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()

        if data.get("success") is False:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            logger.error(
                "API reported failure",
                extra={
                    "event": "api_error",
                    "url": url,
                    "authenticated": authenticated,
                    "error": error_msg,
                },
            )
            raise HTTPException(status_code=400, detail=error_msg)

        return data
    except requests.exceptions.Timeout as exc:
        logger.error(
            "API request timed out",
            extra={"event": "timeout", "url": url},
        )
        raise HTTPException(status_code=504, detail="API request timeout") from exc
    except requests.exceptions.ConnectionError as exc:
        logger.error(
            "API connection error",
            extra={"event": "connection_error", "url": url},
        )
        raise HTTPException(status_code=503, detail="Cannot connect to exchange API") from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else 500
        logger.error(
            "HTTP error from upstream",
            extra={"event": "http_error", "url": url, "status_code": status_code},
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "Unexpected API error",
            extra={"event": "unexpected_error", "url": url},
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    return data.rolling(window=period, min_periods=period).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_diff = high.diff()
    low_diff = -low.diff()
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    pos_di = 100 * pos_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr
    neg_di = 100 * neg_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
    return dx.ewm(span=period, adjust=False, min_periods=period).mean()


def fetch_candle_data(
    symbol: str,
    resolution: str,
    limit: int,
    settings: AppSettings,
) -> pd.DataFrame:
    end_time = int(time.time())
    resolution_map = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "1d": 86400,
    }
    seconds_per_candle = resolution_map.get(resolution, 900)
    start_time = end_time - (limit * seconds_per_candle)

    url = f"{settings.delta_base_url}/v2/history/candles"
    params = {
        "resolution": resolution,
        "symbol": symbol,
        "start": start_time,
        "end": end_time,
    }
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    data = safe_api_request(url=f"{url}?{query_string}", settings=settings)
    candles = data.get("result", [])

    if not candles:
        raise HTTPException(status_code=404, detail="No candle data available")

    df = pd.DataFrame(candles)

    column_map = {
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    for old, new in column_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df
