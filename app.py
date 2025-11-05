"""
ETH Strategy Dashboard - FastAPI Backend
Production-ready, secure, and maintainable implementation
"""

import os
import time
import hmac
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import secure configuration and utilities
from src.config import settings, get_settings
from src.logging_config import setup_logging
from src.middleware import SecurityHeadersMiddleware, HTTPSRedirectMiddleware
from src.error_handler import ErrorHandlerMiddleware
import logging

# Setup logging first
setup_logging(
    log_level=settings.LOG_LEVEL,
    log_format=settings.LOG_FORMAT,
    show_stack_traces=settings.is_development
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ETH Strategy Dashboard API",
    description="Real-time ETH strategy monitoring and trading API",
    version="1.0.0",
    docs_url="/api/docs" if settings.is_development else None,
    redoc_url="/api/redoc" if settings.is_development else None,
    openapi_url="/api/openapi.json" if settings.is_development else None,
)

# Add error handling middleware first (to catch all errors)
app.add_middleware(
    ErrorHandlerMiddleware,
    show_details=settings.is_development
)

# Add security headers middleware
app.add_middleware(
    SecurityHeadersMiddleware,
    force_https=settings.FORCE_HTTPS and settings.is_production,
    secure_cookies=settings.SECURE_COOKIES
)

# Add HTTPS redirect middleware in production
if settings.is_production and settings.FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

# Add CORS middleware
cors_origins = settings.cors_origins_list
if settings.is_development:
    # Allow all origins in development
    cors_origins = ["*"] if not cors_origins else cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True if "*" not in cors_origins else False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files if directory exists
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
    logger.info(f"Static files mounted from {settings.STATIC_DIR}")
else:
    logger.warning(f"Static directory not found: {settings.STATIC_DIR}")

# Helper Functions
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
        settings.DELTA_API_SECRET.encode('utf-8'),
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
        Dictionary of HTTP headers
    """
    timestamp = str(int(time.time()))
    signature = create_signature(method, timestamp, path, body)
    
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "api-key": settings.DELTA_API_KEY,
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
        
        response = requests.get(url, headers=headers, timeout=settings.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Check Delta Exchange API success field
        if "success" in data and not data["success"]:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            logger.error("API returned error", extra={"error_message": error_msg})
            raise HTTPException(status_code=400, detail=error_msg)
        
        return data
    
    except requests.exceptions.Timeout:
        logger.error("Request timeout", extra={"url": url})
        raise HTTPException(status_code=504, detail="API request timeout")
    
    except requests.exceptions.ConnectionError:
        logger.error("Connection error", extra={"url": url})
        raise HTTPException(status_code=503, detail="Cannot connect to exchange API")
    
    except requests.exceptions.HTTPError as e:
        logger.error("HTTP error", extra={"status_code": e.response.status_code, "url": url})
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    
    except Exception as e:
        logger.error("Unexpected error in API request", exc_info=e, extra={"url": url})
        raise HTTPException(status_code=500, detail="Internal server error")

# Technical Indicator Functions

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        data: Price series
        period: SMA period
    
    Returns:
        SMA series
    """
    return data.rolling(window=period, min_periods=period).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
    
    Returns:
        ATR series
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR using exponential moving average
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return atr

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default 14)
    
    Returns:
        ADX series
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Smooth using Wilder's method (exponential moving average)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    pos_di = 100 * pos_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr
    neg_di = 100 * neg_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr
    
    # Calculate DX and ADX
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
        # Calculate time range
        end_time = int(time.time())
        
        # Convert resolution to seconds
        resolution_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
            '1d': 86400
        }
        
        seconds_per_candle = resolution_map.get(resolution, 900)
        start_time = end_time - (limit * seconds_per_candle)
        
        url = f"{settings.DELTA_BASE_URL}/v2/history/candles"
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
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Rename columns to standard OHLCV format
        column_map = {
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Check which columns are present
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching candle data", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch candle data: {str(e)}")

# API Routes

@app.get("/")
async def root():
    """Serve the main dashboard HTML"""
    index_file = settings.STATIC_DIR / "index.html"
    
    if not index_file.exists():
        logger.error("index.html not found", extra={"path": str(index_file)})
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
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG
    })

@app.get("/api/v1/price")
async def get_current_price():
    """
    Get current ETH price from Delta Exchange
    
    Returns:
        JSON with symbol and current price
    """
    try:
        url = f"{settings.DELTA_BASE_URL}/v2/tickers/{settings.TRADING_SYMBOL}"
        logger.info("Fetching price", extra={"symbol": settings.TRADING_SYMBOL})
        
        data = safe_api_request(url)
        
        # Extract price from result
        result = data.get("result", {})
        price = float(
            result.get("mark_price") or 
            result.get("index_price") or 
            result.get("spot_price") or 
            0
        )
        
        if price == 0:
            logger.warning("Price is 0", extra={"symbol": settings.TRADING_SYMBOL})
        
        logger.info("Price fetched successfully", extra={"symbol": settings.TRADING_SYMBOL, "price": price})
        
        return JSONResponse({
            "symbol": settings.TRADING_SYMBOL,
            "price": round(price, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "delta_exchange"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching price", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch price: {str(e)}")

# Legacy endpoint for backward compatibility
@app.get("/price")
async def get_price_legacy():
    """Legacy price endpoint - redirects to new API"""
    return await get_current_price()

@app.get("/api/v1/balance")
async def get_wallet_balance():
    """
    Get wallet balance from Delta Exchange
    
    Returns:
        JSON with balance information
    """
    # Validate credentials
    if not settings.DELTA_API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="API credentials not configured. Set DELTA_API_KEY environment variable."
        )
    
    if not settings.DELTA_API_SECRET:
        raise HTTPException(
            status_code=401,
            detail="API secret not configured. Set DELTA_API_SECRET environment variable."
        )
    
    try:
        method = "GET"
        path = "/v2/wallet/balances"
        
        headers = get_auth_headers(method, path)
        url = str(settings.DELTA_BASE_URL) + path
        
        logger.info("Fetching wallet balance")
        data = safe_api_request(url, headers=headers, authenticated=True)
        
        # Find USD/USDT/USDC balance
        assets = data.get("result", [])
        supported_assets = ["USD", "USDT", "USDC", "MARGIN", "CASH"]
        
        asset = next(
            (x for x in assets if x.get("asset_symbol") in supported_assets),
            None
        )
        
        if asset:
            balance = float(asset.get("available_balance", asset.get("balance", 0)))
            symbol = asset.get("asset_symbol", "")
            
            logger.info("Balance fetched", extra={"balance": balance, "asset": symbol})
            
            return JSONResponse({
                "balance": round(balance, 2),
                "asset": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available": True
            })
        else:
            logger.warning("No supported asset found", extra={"available_assets": [a.get('asset_symbol') for a in assets]})
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
        logger.error("Error fetching balance", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch balance: {str(e)}")

# Legacy endpoint for backward compatibility
@app.get("/balance")
async def get_balance_legacy():
    """Legacy balance endpoint - redirects to new API"""
    return await get_wallet_balance()

@app.get("/api/v1/box")
async def get_price_box():
    """
    Get high/low price box for current trading session
    
    Returns:
        JSON with high/low price range
    """
    try:
        url = f"{settings.DELTA_BASE_URL}/v2/tickers/{settings.TRADING_SYMBOL}"
        data = safe_api_request(url)
        
        result = data.get("result", {})
        high_24h = float(result.get("high", 0))
        low_24h = float(result.get("low", 0))
        
        if high_24h and low_24h:
            box_range = f"{round(low_24h, 2)} - {round(high_24h, 2)}"
        else:
            box_range = "—"
        
        logger.info("Price box fetched", extra={"box_range": box_range})
        
        return JSONResponse({
            "box": box_range,
            "high": round(high_24h, 2) if high_24h else None,
            "low": round(low_24h, 2) if low_24h else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching price box", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch price box: {str(e)}")

# Legacy endpoint
@app.get("/box")
async def get_box_legacy():
    """Legacy box endpoint"""
    return await get_price_box()

@app.get("/api/strategy/filters")
async def get_strategy_filters():
    """
    Get current strategy filter values with technical indicators
    
    This endpoint calculates various technical indicators used for trading decisions:
    - Range (high - low) of current candle
    - ADX(14) for trend strength
    - ATR(14) for volatility
    - Volume vs SMA20
    - 1H trend (SMA50 vs SMA200)
    - Suggested high/low box using ATR
    - Boolean filters for each indicator
    - Overall filters_passed status
    
    Returns:
        JSON with comprehensive filter status and technical indicators
    """
    try:
        logger.info("Calculating strategy filters...")
        
        # Fetch 15m candle data (need enough for calculations)
        df_15m = fetch_candle_data(settings.TRADING_SYMBOL, '15m', limit=200)
        
        if df_15m.empty or len(df_15m) < 50:
            raise HTTPException(status_code=404, detail="Insufficient candle data for calculations")
        
        # Fetch 1h candle data for trend analysis
        df_1h = fetch_candle_data(settings.TRADING_SYMBOL, '1h', limit=250)
        
        # Get current (latest) candle data
        current = df_15m.iloc[-1]
        
        # Calculate Range (high - low) of current candle
        candle_range = float(current['high'] - current['low'])
        
        # Calculate technical indicators on 15m timeframe
        df_15m['atr_14'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=14)
        df_15m['adx_14'] = calculate_adx(df_15m['high'], df_15m['low'], df_15m['close'], period=14)
        df_15m['volume_sma20'] = calculate_sma(df_15m['volume'], period=20)
        
        # Get current indicator values
        atr_value = float(df_15m['atr_14'].iloc[-1]) if not pd.isna(df_15m['atr_14'].iloc[-1]) else 0
        adx_value = float(df_15m['adx_14'].iloc[-1]) if not pd.isna(df_15m['adx_14'].iloc[-1]) else 0
        current_volume = float(current['volume'])
        avg_volume = float(df_15m['volume_sma20'].iloc[-1]) if not pd.isna(df_15m['volume_sma20'].iloc[-1]) else current_volume
        
        # Calculate volume ratio
        volume_ratio = (current_volume / avg_volume * 100) if avg_volume > 0 else 100
        
        # Calculate 1H trend (SMA50 vs SMA200)
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
                else:
                    trend_status = "neutral"
                    trend_ok = False
        
        # Calculate suggested high/low box using ATR
        current_close = float(current['close'])
        atr_multiplier = 1.5  # Standard multiplier for box calculation
        suggested_high = current_close + (atr_value * atr_multiplier)
        suggested_low = current_close - (atr_value * atr_multiplier)
        
        # Define filter thresholds (adjust based on strategy)
        RANGE_MIN = 5.0  # Minimum range threshold
        ADX_MIN = 20.0   # Minimum ADX for trending market
        ATR_MIN = 10.0   # Minimum ATR threshold
        VOLUME_MIN = 80.0  # Minimum volume percentage
        
        # Calculate boolean filters
        range_ok = candle_range >= RANGE_MIN
        adx_ok = adx_value >= ADX_MIN
        atr_ok = atr_value >= ATR_MIN
        volume_ok = volume_ratio >= VOLUME_MIN
        # trend_ok already calculated above
        
        # Overall filter status
        filters_passed = all([range_ok, adx_ok, atr_ok, volume_ok, trend_ok])
        
        # Prepare response
        response = {
            "symbol": settings.TRADING_SYMBOL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            # Current candle info
            "current_candle": {
                "open": round(float(current['open']), 2),
                "high": round(float(current['high']), 2),
                "low": round(float(current['low']), 2),
                "close": round(float(current['close']), 2),
                "volume": round(float(current['volume']), 2)
            },
            
            # Technical indicators
            "indicators": {
                "range": round(candle_range, 2),
                "adx_14": round(adx_value, 2),
                "atr_14": round(atr_value, 2),
                "volume_vs_sma20": round(volume_ratio, 2),
                "trend_1h": trend_status
            },
            
            # Suggested box levels
            "suggested_box": {
                "high": round(suggested_high, 2),
                "low": round(suggested_low, 2),
                "range": round(suggested_high - suggested_low, 2)
            },
            
            # Boolean filters
            "filters": {
                "range_ok": range_ok,
                "adx_ok": adx_ok,
                "atr_ok": atr_ok,
                "volume_ok": volume_ok,
                "trend_ok": trend_ok
            },
            
            # Overall status
            "filters_passed": filters_passed,
            
            # Filter thresholds (for transparency)
            "thresholds": {
                "range_min": RANGE_MIN,
                "adx_min": ADX_MIN,
                "atr_min": ATR_MIN,
                "volume_min_pct": VOLUME_MIN
            }
        }
        
        logger.info("Strategy filters calculated", extra={"filters_passed": filters_passed})
        return JSONResponse(response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error calculating strategy filters", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Failed to calculate strategy filters: {str(e)}")

@app.get("/api/v1/strategy-filters")
async def get_strategy_filters_v1():
    """Legacy strategy filters endpoint - redirects to new API"""
    return await get_strategy_filters()

# Legacy endpoint
@app.get("/strategy-filters")
async def get_strategy_filters_legacy():
    """Legacy strategy filters endpoint"""
    return await get_strategy_filters()

@app.get("/api/v1/candles")
async def get_candles(limit: int = 100):
    """
    Get historical candle data
    
    Args:
        limit: Number of candles to fetch (max 500)
    
    Returns:
        JSON with candle data
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    
    try:
        # Calculate time range (assuming 15m candles)
        end_time = int(time.time())
        start_time = end_time - (limit * 15 * 60)  # 15 minutes per candle
        
        url = f"{settings.DELTA_BASE_URL}/v2/history/candles"
        params = {
            "resolution": settings.CANDLE_RESOLUTION,
            "symbol": settings.TRADING_SYMBOL,
            "start": start_time,
            "end": end_time
        }
        
        full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        data = safe_api_request(full_url)
        
        candles = data.get("result", [])
        
        logger.info("Candles fetched", extra={"count": len(candles)})
        
        return JSONResponse({
            "symbol": settings.TRADING_SYMBOL,
            "resolution": settings.CANDLE_RESOLUTION,
            "candles": candles,
            "count": len(candles),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching candles", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch candles: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 50)
    logger.info("ETH Strategy Dashboard API Starting")
    logger.info(f"Symbol: {settings.TRADING_SYMBOL}")
    logger.info(f"Resolution: {settings.CANDLE_RESOLUTION}")
    logger.info(f"Base URL: {settings.DELTA_BASE_URL}")
    logger.info(f"Static Dir: {settings.STATIC_DIR}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"API Key configured: {'Yes' if settings.DELTA_API_KEY else 'No'}")
    logger.info(f"API Secret configured: {'Yes' if settings.DELTA_API_SECRET else 'No'}")
    logger.info("=" * 50)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ETH Strategy Dashboard API Shutting Down")

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )
