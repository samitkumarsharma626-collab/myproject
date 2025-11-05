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
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import requests
import pandas as pd
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
