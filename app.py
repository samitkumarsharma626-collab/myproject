"""
ETH Trading Bot API - Production Ready Flask Application

This application provides REST API endpoints for ETH trading strategy monitoring.
All endpoints follow Flask best practices and include proper error handling,
security measures, and logging.
"""

import os
import logging
import time
import hmac
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder=None)

# Enable CORS for production (configure origins as needed)
CORS(app, resources={
    r"/*": {
        "origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration from environment variables
API_KEY = os.getenv("DELTA_API_KEY")
API_SECRET = os.getenv("DELTA_API_SECRET")
BASE_URL = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
STATIC_DIR = os.getenv("STATIC_DIR", os.path.join(os.path.dirname(__file__), "static"))
SYMBOL = os.getenv("TRADING_SYMBOL", "ETHUSD")
RESOLUTION = os.getenv("CANDLE_RESOLUTION", "15m")

# Request timeout configuration
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

# Validate required environment variables
if not API_KEY or not API_SECRET:
    logger.warning("API credentials not found in environment variables. "
                   "Balance endpoint will not function properly.")


def generate_signature(api_secret: str, method: str, timestamp: str, path: str) -> str:
    """
    Generate HMAC SHA256 signature for Delta Exchange API authentication.
    
    Args:
        api_secret: API secret key
        method: HTTP method (GET, POST, etc.)
        timestamp: Unix timestamp as string
        path: API endpoint path
        
    Returns:
        Hexadecimal signature string
    """
    msg = f"{method}{timestamp}{path}"
    return hmac.new(
        api_secret.encode('utf-8'),
        msg.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def get_auth_headers(method: str, path: str) -> Dict[str, str]:
    """
    Generate authentication headers for Delta Exchange API.
    
    Args:
        method: HTTP method
        path: API endpoint path
        
    Returns:
        Dictionary of headers including authentication
    """
    if not API_KEY or not API_SECRET:
        raise ValueError("API credentials not configured")
    
    timestamp = str(int(time.time()))
    signature = generate_signature(API_SECRET, method, timestamp, path)
    
    return {
        "Accept": "application/json",
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "flask-ethbot-client/1.0"
    }


@app.route("/", methods=["GET"])
def index():
    """Serve the main dashboard HTML page."""
    try:
        static_path = Path(STATIC_DIR)
        if not static_path.exists():
            logger.error(f"Static directory not found: {STATIC_DIR}")
            return jsonify({"error": "Static files not found"}), 500
        
        return send_from_directory(static_path, "index.html")
    except Exception as e:
        logger.error(f"Error serving index page: {e}", exc_info=True)
        return jsonify({"error": "Failed to serve index page"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "message": "Flask application running",
        "mode": os.getenv("APP_MODE", "PRODUCTION")
    }), 200


@app.route("/price", methods=["GET"])
def get_price():
    """
    Get current ETH price from Delta Exchange.
    
    Returns:
        JSON response with symbol, price, and status message
    """
    try:
        symbol = request.args.get("symbol", SYMBOL)
        url = f"{BASE_URL}/v2/tickers/{symbol}"
        
        logger.info(f"Fetching price for symbol: {symbol}")
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Log response for debugging (remove sensitive data in production)
        logger.debug(f"Price API response: {data}")
        
        # Check response success
        if not data.get("success"):
            logger.warning(f"Price API returned unsuccessful response: {data}")
            return jsonify({
                "symbol": symbol,
                "price": 0.0,
                "message": "Failed to fetch price from exchange"
            }), 200
        
        # Safely extract price from response
        result = data.get("result", {})
        price = float(
            result.get("mark_price") or
            result.get("index_price") or
            result.get("spot_price") or
            0.0
        )
        
        if price <= 0:
            logger.warning(f"Invalid price received: {price}")
            return jsonify({
                "symbol": symbol,
                "price": 0.0,
                "message": "Invalid price data received"
            }), 200
        
        logger.info(f"Successfully fetched price: {price} for {symbol}")
        return jsonify({
            "symbol": symbol,
            "price": round(price, 2),
            "message": "Price fetched successfully"
        }), 200
        
    except requests.exceptions.Timeout:
        logger.error("Price API request timed out")
        return jsonify({
            "symbol": SYMBOL,
            "price": 0.0,
            "message": "Request timeout"
        }), 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Price API request failed: {e}", exc_info=True)
        return jsonify({
            "symbol": SYMBOL,
            "price": 0.0,
            "message": f"API request failed: {str(e)}"
        }), 200
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing price response: {e}", exc_info=True)
        return jsonify({
            "symbol": SYMBOL,
            "price": 0.0,
            "message": "Error parsing response"
        }), 200
    except Exception as e:
        logger.error(f"Unexpected error in get_price: {e}", exc_info=True)
        return jsonify({
            "symbol": SYMBOL,
            "price": 0.0,
            "message": "Internal server error"
        }), 500


@app.route("/balance", methods=["GET"])
def get_balance():
    """
    Get account balance from Delta Exchange.
    
    Requires DELTA_API_KEY and DELTA_API_SECRET environment variables.
    
    Returns:
        JSON response with balance, symbol, and status message
    """
    try:
        if not API_KEY or not API_SECRET:
            logger.error("API credentials not configured for balance endpoint")
            return jsonify({
                "error": "API credentials not configured",
                "balance": 0.0,
                "symbol": "",
                "message": "Balance endpoint not available"
            }), 503
        
        method = "GET"
        path = "/v2/wallet/balances"
        url = f"{BASE_URL}{path}"
        
        headers = get_auth_headers(method, path)
        
        logger.info("Fetching account balance")
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Check response success
        if not data.get("success"):
            logger.warning(f"Balance API returned unsuccessful response: {data}")
            return jsonify({
                "balance": 0.0,
                "symbol": "",
                "message": "Failed to fetch balance from exchange"
            }), 200
        
        # Find USD/USDT/USDC/MARGIN/CASH asset
        assets = data.get("result", [])
        if not isinstance(assets, list):
            logger.error(f"Invalid assets format: {assets}")
            return jsonify({
                "balance": 0.0,
                "symbol": "",
                "message": "Invalid response format"
            }), 200
        
        asset_symbols = ["USD", "USDT", "USDC", "MARGIN", "CASH"]
        asset = next(
            (x for x in assets if x.get("asset_symbol") in asset_symbols),
            None
        )
        
        if asset:
            balance = float(
                asset.get("available_balance") or
                asset.get("balance") or
                0.0
            )
            symbol = asset.get("asset_symbol", "")
            
            logger.info(f"Successfully fetched balance: {balance} {symbol}")
            return jsonify({
                "balance": round(balance, 2),
                "symbol": symbol,
                "message": "Balance fetched successfully"
            }), 200
        else:
            logger.warning("No USD/USDT/USDC/MARGIN/CASH asset found")
            return jsonify({
                "balance": 0.0,
                "symbol": "",
                "message": "No USD/USDT/USDC/MARGIN/CASH asset found"
            }), 200
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({
            "error": "Configuration error",
            "balance": 0.0,
            "symbol": "",
            "message": str(e)
        }), 503
    except requests.exceptions.Timeout:
        logger.error("Balance API request timed out")
        return jsonify({
            "error": "Request timeout",
            "balance": 0.0,
            "symbol": "",
            "message": "Exchange API timeout"
        }), 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Balance API request failed: {e}", exc_info=True)
        return jsonify({
            "error": "API request failed",
            "balance": 0.0,
            "symbol": "",
            "message": f"Exchange API error: {str(e)}"
        }), 200
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing balance response: {e}", exc_info=True)
        return jsonify({
            "error": "Response parsing error",
            "balance": 0.0,
            "symbol": "",
            "message": "Invalid response format"
        }), 200
    except Exception as e:
        logger.error(f"Unexpected error in get_balance: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "balance": 0.0,
            "symbol": "",
            "message": "An unexpected error occurred"
        }), 500


@app.route("/box", methods=["GET"])
def get_box():
    """
    Get high/low box values for strategy.
    
    This is a placeholder endpoint. Implement based on your strategy requirements.
    
    Returns:
        JSON response with box information
    """
    try:
        # TODO: Implement actual box calculation logic
        # This is a placeholder response
        return jsonify({
            "box": "—",
            "high": None,
            "low": None,
            "message": "Box calculation not implemented"
        }), 200
    except Exception as e:
        logger.error(f"Error in get_box: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "box": "—",
            "message": "Failed to calculate box"
        }), 500


@app.route("/strategy-filters", methods=["GET"])
def get_strategy_filters():
    """
    Get current strategy filter values.
    
    This is a placeholder endpoint. Implement based on your strategy requirements.
    
    Returns:
        JSON response with filter values
    """
    try:
        # TODO: Implement actual strategy filter calculation logic
        # This is a placeholder response
        return jsonify({
            "range": "—",
            "adx_proxy": "—",
            "atr_14": "—",
            "volume_status": "—",
            "one_hour_trend": "—",
            "message": "Strategy filters not implemented"
        }), 200
    except Exception as e:
        logger.error(f"Error in get_strategy_filters: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": "Failed to calculate strategy filters"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Development server configuration
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    
    logger.info(f"Starting Flask application on {host}:{port}")
    app.run(host=host, port=port, debug=debug_mode)
