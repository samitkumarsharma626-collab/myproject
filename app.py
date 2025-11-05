"""
ETH Strategy Dashboard - FastAPI Backend
Production-ready, secure, and fully validated implementation
"""

import time
import hmac
import hashlib
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
import logging
import logging.config

# =============================================================================
# STRUCTURED LOGGING CONFIGURATION
# =============================================================================

class SecureFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive information"""
    
    SENSITIVE_KEYS = {
        'api_key', 'api-key', 'signature', 'password', 'secret', 
        'token', 'authorization', 'auth'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Redact sensitive data from log message
        if hasattr(record, 'args') and record.args:
            record.args = self._redact_sensitive(record.args)
        
        return super().format(record)
    
    def _redact_sensitive(self, data: Any) -> Any:
        """Recursively redact sensitive information"""
        if isinstance(data, dict):
            return {
                k: '***REDACTED***' if k.lower() in self.SENSITIVE_KEYS else self._redact_sensitive(v)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return type(data)(self._redact_sensitive(item) for item in data)
        return data


def setup_logging():
    """Configure structured logging based on environment"""
    log_level = getattr(logging, settings.log_level)
    
    if settings.log_format == "json":
        # JSON logging for production
        import logging.config
        
        LOGGING_CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "json",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"]
            }
        }
        
        try:
            logging.config.dictConfig(LOGGING_CONFIG)
        except ImportError:
            # Fallback if pythonjsonlogger not installed
            logging.basicConfig(
                level=log_level,
                format='{"timestamp":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}',
            )
    else:
        # Text logging for development
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    # Apply secure formatter
    for handler in logging.root.handlers:
        if not settings.is_production():
            handler.setFormatter(SecureFormatter(handler.formatter._fmt if hasattr(handler, 'formatter') else '%(message)s'))


setup_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # Remove server header
        response.headers.pop("Server", None)
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error(
                f"Unhandled exception: {exc}",
                exc_info=settings.is_development(),
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "client": request.client.host if request.client else "unknown"
                }
            )
            
            # Include stack trace only in development
            detail = {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }
            
            if settings.is_development():
                detail["detail"] = str(exc)
                detail["traceback"] = traceback.format_exc()
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=detail
            )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="ETH Strategy Dashboard API",
    description="Real-time ETH strategy monitoring and trading API",
    version="1.0.0",
    docs_url="/api/docs" if not settings.is_production() else None,
    redoc_url="/api/redoc" if not settings.is_production() else None,
    openapi_url="/api/openapi.json" if not settings.is_production() else None,
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Add CORS middleware with strict settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Add trusted host middleware for production
if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts
    )

# Mount static files if directory exists
if settings.static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    logger.info(f"Static files mounted from {settings.static_dir}")
else:
    logger.warning(f"Static directory not found: {settings.static_dir}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_signature(method: str, timestamp: str, path: str, body: str = "") -> str:
    """
    Create HMAC signature for Delta Exchange API authentication
    
    Args:
        method: HTTP method (GET, POST, etc.)
        timestamp: Unix timestamp as string
        path: API endpoint path
        body: Request body (for POST/PUT requests)
    
    Returns:
        HMAC signature as hexadecimal string
    """
    message = method + timestamp + path + body
    signature = hmac.new(
        settings.get_api_secret().encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature


def get_auth_headers(method: str, path: str, body: str = "") -> Dict[str, str]:
    """
    Generate authenticated headers for Delta Exchange API
    
    Args:
        method: HTTP method
        path: API endpoint path
        body: Request body
    
    Returns:
        Dictionary of HTTP headers (secrets are not logged)
    """
    timestamp = str(int(time.time()))
    signature = create_signature(method, timestamp, path, body)
    
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "api-key": settings.get_api_key(),
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "ethbot-fastapi/1.0"
    }


def safe_api_request(url: str, headers: Optional[Dict[str, str]] = None, 
                     authenticated: bool = False) -> Dict[str, Any]:
    """
    Make a safe API request with proper error handling
    
    Args:
        url: Full API URL
        headers: Optional HTTP headers
        authenticated: Whether to use authenticated request
    
    Returns:
        JSON response data
    
    Raises:
        HTTPException: On request failure
    """
    try:
        if headers is None:
            headers = {"Accept": "application/json"}
        
        # Log request (without sensitive headers)
        safe_headers = {k: v for k, v in headers.items() if k.lower() not in ['api-key', 'signature', 'authorization']}
        logger.debug(f"API request to {url}", extra={"headers": safe_headers})
        
        response = requests.get(url, headers=headers, timeout=settings.request_timeout)
        response.raise_for_status()
        
        data = response.json()
        
        # Check Delta Exchange API success field
        if "success" in data and not data["success"]:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            logger.error(f"API returned error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        return data
    
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout for URL: {url}")
        raise HTTPException(status_code=504, detail="API request timeout")
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error for URL: {url}")
        raise HTTPException(status_code=503, detail="Cannot connect to exchange API")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in API request: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =============================================================================

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period, min_periods=period).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return atr


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)"""
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
    adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return adx


def fetch_candle_data(symbol: str, resolution: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch candle data from Delta Exchange and return as DataFrame
    
    Args:
        symbol: Trading symbol
        resolution: Candle resolution (e.g., '15m', '1h')
        limit: Number of candles to fetch
    
    Returns:
        DataFrame with OHLCV data
    
    Raises:
        HTTPException: On request failure
    """
    try:
        end_time = int(time.time())
        
        resolution_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
            '1d': 86400
        }
        
        seconds_per_candle = resolution_map.get(resolution, 900)
        start_time = end_time - (limit * seconds_per_candle)
        
        url = f"{settings.delta_base_url}/v2/history/candles"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "start": start_time,
            "end": end_time
        }
        
        full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        data = safe_api_request(full_url)
        
        candles = data.get("result", [])
        
        if not candles or len(candles) == 0:
            raise HTTPException(status_code=404, detail="No candle data available")
        
        df = pd.DataFrame(candles)
        
        column_map = {
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching candle data: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail=f"Failed to fetch candle data: {str(e)}")


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/")
async def root():
    """Serve the main dashboard HTML"""
    index_file = settings.static_dir / "index.html"
    
    if not index_file.exists():
        logger.error(f"index.html not found at {index_file}")
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return FileResponse(str(index_file))


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    
    Returns:
        JSON with health status
    """
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "environment": settings.environment,
        "config_valid": settings.is_valid()
    })


@app.get("/api/v1/price")
async def get_current_price():
    """
    Get current ETH price from Delta Exchange
    
    Returns:
        JSON with symbol and current price
    """
    try:
        url = f"{settings.delta_base_url}/v2/tickers/{settings.trading_symbol}"
        logger.info(f"Fetching price for {settings.trading_symbol}")
        
        data = safe_api_request(url)
        
        result = data.get("result", {})
        price = float(
            result.get("mark_price") or 
            result.get("index_price") or 
            result.get("spot_price") or 
            0
        )
        
        if price == 0:
            logger.warning("Price is 0, check API response")
        
        logger.info(f"Price fetched successfully: {price}")
        
        return JSONResponse({
            "symbol": settings.trading_symbol,
            "price": round(price, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "delta_exchange"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail=f"Failed to fetch price: {str(e)}")


@app.get("/price")
async def get_price_legacy():
    """Legacy price endpoint - redirects to new API"""
    return await get_current_price()


@app.get("/api/v1/balance")
async def get_wallet_balance():
    """
    Get wallet balance from Delta Exchange (requires API credentials)
    
    Returns:
        JSON with balance information
    """
    # Validate credentials
    if not settings.get_api_key() or settings.get_api_key() == "YOUR_API_KEY":
        raise HTTPException(
            status_code=401, 
            detail="API credentials not configured. Set DELTA_API_KEY environment variable."
        )
    
    if not settings.get_api_secret() or settings.get_api_secret() == "YOUR_API_SECRET":
        raise HTTPException(
            status_code=401,
            detail="API secret not configured. Set DELTA_API_SECRET environment variable."
        )
    
    try:
        method = "GET"
        path = "/v2/wallet/balances"
        
        headers = get_auth_headers(method, path)
        url = settings.delta_base_url + path
        
        logger.info("Fetching wallet balance")
        data = safe_api_request(url, headers=headers, authenticated=True)
        
        assets = data.get("result", [])
        supported_assets = ["USD", "USDT", "USDC", "MARGIN", "CASH"]
        
        asset = next(
            (x for x in assets if x.get("asset_symbol") in supported_assets),
            None
        )
        
        if asset:
            balance = float(asset.get("available_balance", asset.get("balance", 0)))
            symbol = asset.get("asset_symbol", "")
            
            logger.info(f"Balance fetched successfully")
            
            return JSONResponse({
                "balance": round(balance, 2),
                "asset": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available": True
            })
        else:
            logger.warning(f"No supported asset found")
            return JSONResponse({
                "balance": 0,
                "asset": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available": False,
                "message": f"No {', '.join(supported_assets)} asset found"
            })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching balance: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail=f"Failed to fetch balance: {str(e)}")


@app.get("/balance")
async def get_balance_legacy():
    """Legacy balance endpoint"""
    return await get_wallet_balance()


@app.get("/api/v1/box")
async def get_price_box():
    """
    Get high/low price box for current trading session
    
    Returns:
        JSON with high/low price range
    """
    try:
        url = f"{settings.delta_base_url}/v2/tickers/{settings.trading_symbol}"
        data = safe_api_request(url)
        
        result = data.get("result", {})
        high_24h = float(result.get("high", 0))
        low_24h = float(result.get("low", 0))
        
        if high_24h and low_24h:
            box_range = f"{round(low_24h, 2)} - {round(high_24h, 2)}"
        else:
            box_range = "—"
        
        logger.info(f"Price box: {box_range}")
        
        return JSONResponse({
            "box": box_range,
            "high": round(high_24h, 2) if high_24h else None,
            "low": round(low_24h, 2) if low_24h else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price box: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail=f"Failed to fetch price box: {str(e)}")


@app.get("/box")
async def get_box_legacy():
    """Legacy box endpoint"""
    return await get_price_box()


@app.get("/api/strategy/filters")
async def get_strategy_filters():
    """
    Get current strategy filter values with technical indicators
    
    Returns:
        JSON with comprehensive filter status and technical indicators
    """
    try:
        logger.info("Calculating strategy filters...")
        
        df_15m = fetch_candle_data(settings.trading_symbol, '15m', limit=200)
        
        if df_15m.empty or len(df_15m) < 50:
            raise HTTPException(status_code=404, detail="Insufficient candle data for calculations")
        
        df_1h = fetch_candle_data(settings.trading_symbol, '1h', limit=250)
        
        current = df_15m.iloc[-1]
        
        candle_range = float(current['high'] - current['low'])
        
        df_15m['atr_14'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=14)
        df_15m['adx_14'] = calculate_adx(df_15m['high'], df_15m['low'], df_15m['close'], period=14)
        df_15m['volume_sma20'] = calculate_sma(df_15m['volume'], period=20)
        
        atr_value = float(df_15m['atr_14'].iloc[-1]) if not pd.isna(df_15m['atr_14'].iloc[-1]) else 0
        adx_value = float(df_15m['adx_14'].iloc[-1]) if not pd.isna(df_15m['adx_14'].iloc[-1]) else 0
        current_volume = float(current['volume'])
        avg_volume = float(df_15m['volume_sma20'].iloc[-1]) if not pd.isna(df_15m['volume_sma20'].iloc[-1]) else current_volume
        
        volume_ratio = (current_volume / avg_volume * 100) if avg_volume > 0 else 100
        
        trend_status = "neutral"
        trend_ok = False
        
        if not df_1h.empty and len(df_1h) >= 200:
            df_1h['sma50'] = calculate_sma(df_1h['close'], period=50)
            df_1h['sma200'] = calculate_sma(df_1h['close'], period=200)
            
            sma50 = df_1h['sma50'].iloc[-1]
            sma200 = df_1h['sma200'].iloc[-1]
            
            if not pd.isna(sma50) and not pd.isna(sma200):
                if sma50 > sma200:
                    trend_status = "bullish"
                    trend_ok = True
                elif sma50 < sma200:
                    trend_status = "bearish"
                    trend_ok = False
        
        current_close = float(current['close'])
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
        
        response = {
            "symbol": settings.trading_symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_candle": {
                "open": round(float(current['open']), 2),
                "high": round(float(current['high']), 2),
                "low": round(float(current['low']), 2),
                "close": round(float(current['close']), 2),
                "volume": round(float(current['volume']), 2)
            },
            "indicators": {
                "range": round(candle_range, 2),
                "adx_14": round(adx_value, 2),
                "atr_14": round(atr_value, 2),
                "volume_vs_sma20": round(volume_ratio, 2),
                "trend_1h": trend_status
            },
            "suggested_box": {
                "high": round(suggested_high, 2),
                "low": round(suggested_low, 2),
                "range": round(suggested_high - suggested_low, 2)
            },
            "filters": {
                "range_ok": range_ok,
                "adx_ok": adx_ok,
                "atr_ok": atr_ok,
                "volume_ok": volume_ok,
                "trend_ok": trend_ok
            },
            "filters_passed": filters_passed,
            "thresholds": {
                "range_min": RANGE_MIN,
                "adx_min": ADX_MIN,
                "atr_min": ATR_MIN,
                "volume_min_pct": VOLUME_MIN
            }
        }
        
        logger.info(f"Strategy filters calculated - Passed: {filters_passed}")
        return JSONResponse(response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating strategy filters: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail=f"Failed to calculate strategy filters: {str(e)}")


@app.get("/api/v1/strategy-filters")
async def get_strategy_filters_v1():
    """Legacy strategy filters endpoint"""
    return await get_strategy_filters()


@app.get("/strategy-filters")
async def get_strategy_filters_legacy():
    """Legacy strategy filters endpoint"""
    return await get_strategy_filters()


@app.get("/api/v1/candles")
async def get_candles(limit: int = 100):
    """
    Get historical candle data
    
    Args:
        limit: Number of candles to fetch (1-500)
    
    Returns:
        JSON with candle data
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    
    try:
        end_time = int(time.time())
        start_time = end_time - (limit * 15 * 60)
        
        url = f"{settings.delta_base_url}/v2/history/candles"
        params = {
            "resolution": settings.candle_resolution,
            "symbol": settings.trading_symbol,
            "start": start_time,
            "end": end_time
        }
        
        full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        data = safe_api_request(full_url)
        
        candles = data.get("result", [])
        
        logger.info(f"Fetched {len(candles)} candles")
        
        return JSONResponse({
            "symbol": settings.trading_symbol,
            "resolution": settings.candle_resolution,
            "candles": candles,
            "count": len(candles),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching candles: {e}", exc_info=settings.is_development())
        raise HTTPException(status_code=500, detail=f"Failed to fetch candles: {str(e)}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Custom 500 handler"""
    logger.error(f"Internal error: {exc}", exc_info=settings.is_development())
    
    content = {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Include details only in development
    if settings.is_development():
        content["detail"] = str(exc)
    
    return JSONResponse(
        status_code=500,
        content=content
    )


# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 50)
    logger.info("ETH Strategy Dashboard API Starting")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Symbol: {settings.trading_symbol}")
    logger.info(f"Resolution: {settings.candle_resolution}")
    logger.info(f"Base URL: {settings.delta_base_url}")
    logger.info(f"Static Dir: {settings.static_dir}")
    logger.info(f"API Key configured: {'Yes' if settings.get_api_key() and settings.get_api_key() != 'YOUR_API_KEY' else 'No'}")
    logger.info(f"Configuration valid: {settings.is_valid()}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ETH Strategy Dashboard API Shutting Down")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
