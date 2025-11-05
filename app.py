"""
ETH Strategy Dashboard - FastAPI Backend
Production-ready, secure, and maintainable implementation
"""

import os
import time
import hmac
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    API_KEY: str = os.getenv("DELTA_API_KEY", "")
    API_SECRET: str = os.getenv("DELTA_API_SECRET", "")
    BASE_URL: str = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
    SYMBOL: str = os.getenv("TRADING_SYMBOL", "ETHUSD")
    RESOLUTION: str = os.getenv("CANDLE_RESOLUTION", "15m")
    
    # File paths - support both dev and production
    STATIC_DIR: Path = Path(os.getenv("STATIC_DIR", "./static"))
    
    # API settings
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    
    # Validate critical settings
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        if not cls.API_KEY or cls.API_KEY == "YOUR_API_KEY":
            logger.warning("API_KEY not configured - some endpoints will not work")
        if not cls.API_SECRET or cls.API_SECRET == "YOUR_API_SECRET":
            logger.warning("API_SECRET not configured - some endpoints will not work")
        if not cls.STATIC_DIR.exists():
            logger.warning(f"Static directory not found: {cls.STATIC_DIR}")

# Validate configuration on startup
Config.validate()

# Initialize FastAPI app
app = FastAPI(
    title="ETH Strategy Dashboard API",
    description="Real-time ETH strategy monitoring and trading API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if Config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(Config.STATIC_DIR)), name="static")
    logger.info(f"Static files mounted from {Config.STATIC_DIR}")
else:
    logger.warning(f"Static directory not found: {Config.STATIC_DIR}")

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
        Config.API_SECRET.encode('utf-8'),
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
        "api-key": Config.API_KEY,
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
        
        response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
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
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def require_bearer_token(request: Request) -> None:
    """
    Enforce optional bearer token auth for sensitive endpoints.
    If STRATEGY_API_TOKEN is set, require Authorization: Bearer <token> header.
    """
    token = os.getenv("STRATEGY_API_TOKEN", "").strip()
    if not token:
        return  # No token configured; allow access
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = auth_header.split(" ", 1)[1].strip()
    if not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Invalid token")


def fetch_candles(symbol: str, resolution: str, limit: int) -> pd.DataFrame:
    """
    Fetch historical candles from Delta Exchange and return as DataFrame.
    Expects fields: time, open, high, low, close, volume
    """
    end_time = int(time.time())
    # Map resolution to seconds per candle
    res_to_sec = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }
    sec = res_to_sec.get(resolution, 900)
    start_time = end_time - (limit * sec)

    url = f"{Config.BASE_URL}/v2/history/candles"
    params = {
        "resolution": resolution,
        "symbol": symbol,
        "start": start_time,
        "end": end_time,
    }
    full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    data = safe_api_request(full_url)
    candles = data.get("result", [])
    if not isinstance(candles, list) or len(candles) == 0:
        raise HTTPException(status_code=502, detail="No candle data returned")

    # Delta typically returns list of dicts. Normalize into DataFrame
    df = pd.DataFrame(candles)
    # Try common field names
    rename_map = {
        "t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "timestamp": "time"
    }
    df = df.rename(columns=rename_map)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=502, detail=f"Candle data missing fields: {missing}")
    # Ensure correct dtypes and sort by time
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df.sort_values("time").reset_index(drop=True)
    return df


def compute_atr_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Compute ATR and ADX using Wilder's smoothing via EWM as approximation.
    Returns (atr, adx) as pandas Series aligned with df.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr).replace([np.inf, -np.inf], np.nan)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr).replace([np.inf, -np.inf], np.nan)
    dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return atr, adx


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

# API Routes

@app.get("/")
async def root():
    """Serve the main dashboard HTML"""
    index_file = Config.STATIC_DIR / "index.html"
    
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
        "mode": os.getenv("APP_MODE", "PRODUCTION")
    })

@app.get("/api/v1/price")
async def get_current_price():
    """
    Get current ETH price from Delta Exchange
    
    Returns:
        JSON with symbol and current price
    """
    try:
        url = f"{Config.BASE_URL}/v2/tickers/{Config.SYMBOL}"
        logger.info(f"Fetching price for {Config.SYMBOL}")
        
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
            logger.warning("Price is 0, check API response")
        
        logger.info(f"Price fetched successfully: {price}")
        
        return JSONResponse({
            "symbol": Config.SYMBOL,
            "price": round(price, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "delta_exchange"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price: {e}")
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
    if not Config.API_KEY or Config.API_KEY == "YOUR_API_KEY":
        raise HTTPException(
            status_code=401, 
            detail="API credentials not configured. Set DELTA_API_KEY environment variable."
        )
    
    if not Config.API_SECRET or Config.API_SECRET == "YOUR_API_SECRET":
        raise HTTPException(
            status_code=401,
            detail="API secret not configured. Set DELTA_API_SECRET environment variable."
        )
    
    try:
        method = "GET"
        path = "/v2/wallet/balances"
        
        headers = get_auth_headers(method, path)
        url = Config.BASE_URL + path
        
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
            
            logger.info(f"Balance fetched: {balance} {symbol}")
            
            return JSONResponse({
                "balance": round(balance, 2),
                "asset": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available": True
            })
        else:
            logger.warning(f"No supported asset found. Available assets: {[a.get('asset_symbol') for a in assets]}")
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
        logger.error(f"Error fetching balance: {e}")
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
        url = f"{Config.BASE_URL}/v2/tickers/{Config.SYMBOL}"
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
        logger.error(f"Error fetching price box: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch price box: {str(e)}")

# Legacy endpoint
@app.get("/box")
async def get_box_legacy():
    """Legacy box endpoint"""
    return await get_price_box()

@app.get("/api/v1/strategy-filters")
async def get_strategy_filters():
    """
    Get current strategy filter values
    
    This endpoint calculates various technical indicators used for trading decisions.
    Note: This is a placeholder implementation. In production, you would calculate
    these values from actual historical candle data.
    
    Returns:
        JSON with filter status
    """
    try:
        # For a production implementation, you would:
        # 1. Fetch historical candle data
        # 2. Calculate technical indicators (ADX, ATR, SMA, etc.)
        # 3. Return actual computed values
        
        # Placeholder response
        return JSONResponse({
            "range": "—",
            "adx_proxy": "—",
            "atr_14": "—",
            "volume_status": "—",
            "one_hour_trend": "—",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Strategy filters require historical data implementation"
        })
    
    except Exception as e:
        logger.error(f"Error in strategy filters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategy filters: {str(e)}")


@app.get("/api/strategy/filters")
async def strategy_filters(request: Request, resolution: Optional[str] = None):
    """
    Secure endpoint: compute strategy filters for current candle using Delta Exchange.

    Query params:
    - resolution: candle timeframe (default from config, e.g., 15m)

    Security: if STRATEGY_API_TOKEN is set, requires Authorization: Bearer <token>.
    """
    # Security check
    require_bearer_token(request)

    # Parameters and thresholds
    res = (resolution or Config.RESOLUTION).strip()
    min_adx = float(os.getenv("MIN_ADX", "20"))
    min_volume_ratio = float(os.getenv("MIN_VOLUME_RATIO", "1.0"))
    range_atr_multiplier = float(os.getenv("RANGE_ATR_MULTIPLIER", "1.0"))

    try:
        # Fetch base resolution candles (need at least 200 for robust calcs)
        base_limit = 300
        df = fetch_candles(Config.SYMBOL, res, base_limit)
        if len(df) < 50:
            raise HTTPException(status_code=502, detail="Insufficient candles for calculation")

        # Compute indicators on base timeframe
        atr14, adx14 = compute_atr_adx(df, period=14)
        vol_sma20 = sma(df["volume"], 20)

        # Current candle metrics (latest row)
        last = df.iloc[-1]
        last_atr = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else None
        last_adx = float(adx14.iloc[-1]) if not np.isnan(adx14.iloc[-1]) else None
        last_vol_sma20 = float(vol_sma20.iloc[-1]) if not np.isnan(vol_sma20.iloc[-1]) else None

        last_range = float(abs(last["high"] - last["low"]))
        last_volume = float(last["volume"]) if not np.isnan(last["volume"]) else None
        last_close = float(last["close"]) if not np.isnan(last["close"]) else None

        volume_ratio = None
        if last_volume is not None and last_vol_sma20 and last_vol_sma20 > 0:
            volume_ratio = last_volume / last_vol_sma20

        # 1H trend via SMA50 vs SMA200 on 1h candles
        trend = None
        trend_ok = None
        try:
            df_1h = fetch_candles(Config.SYMBOL, "1h", 260)
            sma50 = sma(df_1h["close"], 50)
            sma200 = sma(df_1h["close"], 200)
            v50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None
            v200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None
            if v50 is not None and v200 is not None:
                trend = "bull" if v50 > v200 else "bear"
                trend_ok = v50 > v200
        except HTTPException:
            # Bubble up later as partial data
            pass
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")

        # Suggested box using ATR (± ATR around close)
        suggested_high = suggested_low = None
        if last_close is not None and last_atr is not None:
            suggested_high = last_close + last_atr
            suggested_low = last_close - last_atr

        # Boolean filters
        range_ok = None
        adx_ok = None
        atr_ok = None
        volume_ok = None

        if last_atr is not None:
            range_ok = last_range >= (range_atr_multiplier * last_atr)
            atr_ok = last_atr > 0
        if last_adx is not None:
            adx_ok = last_adx >= min_adx
        if volume_ratio is not None:
            volume_ok = volume_ratio >= min_volume_ratio

        bools = [b for b in [range_ok, adx_ok, atr_ok, volume_ok, trend_ok] if b is not None]
        filters_passed = (len(bools) > 0 and all(bools))

        # Build response
        response = {
            "symbol": Config.SYMBOL,
            "resolution": res,
            "range": round(last_range, 6) if last_range is not None else None,
            "adx_14": round(last_adx, 6) if last_adx is not None else None,
            "atr_14": round(last_atr, 6) if last_atr is not None else None,
            "volume": round(last_volume, 6) if last_volume is not None else None,
            "volume_sma20": round(last_vol_sma20, 6) if last_vol_sma20 is not None else None,
            "volume_ratio": round(volume_ratio, 6) if volume_ratio is not None else None,
            "one_hour_trend": trend,
            "suggested_box": {
                "high": round(suggested_high, 6) if suggested_high is not None else None,
                "low": round(suggested_low, 6) if suggested_low is not None else None,
            },
            "range_ok": range_ok,
            "adx_ok": adx_ok,
            "atr_ok": atr_ok,
            "volume_ok": volume_ok,
            "trend_ok": trend_ok,
            "filters_passed": filters_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "delta_exchange",
        }
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing strategy filters: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute strategy filters")

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
        
        url = f"{Config.BASE_URL}/v2/history/candles"
        params = {
            "resolution": Config.RESOLUTION,
            "symbol": Config.SYMBOL,
            "start": start_time,
            "end": end_time
        }
        
        full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        data = safe_api_request(full_url)
        
        candles = data.get("result", [])
        
        logger.info(f"Fetched {len(candles)} candles")
        
        return JSONResponse({
            "symbol": Config.SYMBOL,
            "resolution": Config.RESOLUTION,
            "candles": candles,
            "count": len(candles),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch candles: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Custom 500 handler"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 50)
    logger.info("ETH Strategy Dashboard API Starting")
    logger.info(f"Symbol: {Config.SYMBOL}")
    logger.info(f"Resolution: {Config.RESOLUTION}")
    logger.info(f"Base URL: {Config.BASE_URL}")
    logger.info(f"Static Dir: {Config.STATIC_DIR}")
    logger.info(f"API Key configured: {'Yes' if Config.API_KEY and Config.API_KEY != 'YOUR_API_KEY' else 'No'}")
    logger.info("=" * 50)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ETH Strategy Dashboard API Shutting Down")

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )
