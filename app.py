from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import time
import hmac
import hashlib
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

API_KEY = os.getenv("DELTA_API_KEY")
API_SECRET = os.getenv("DELTA_API_SECRET")
BASE_URL = "https://api.delta.exchange"

# Confirmed Symbol and Resolution for Delta Perp
SYMBOL = "ETHUSD"        # Yeh tumhare dashboard/trading me market ka symbol hai
RESOLUTION = "15m"       # allowed_candle_resolutions list me se, 15m recommended

app = FastAPI()
app.mount("/static", StaticFiles(directory="/var/www/ethbot/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("/var/www/ethbot/static/index.html")

@app.get("/health")
async def health():
    return {"ok": True, "msg": "FastAPI running", "mode": "LIVE"}






@app.get("/price")
async def get_price():
    try:
        import requests

        # ✅ ETH perpetual pair
        SYMBOL = "ETHUSD"
        BASE_URL = "https://api.india.delta.exchange"

        # ✅ API request
        url = f"{BASE_URL}/v2/tickers/{SYMBOL}"
        r = requests.get(url, timeout=10)
        data = r.json()

        print("\n====== DEBUG PRICE RESPONSE ======\n", data, "\n==================================\n")

        # ✅ Check response success
        if not data.get("success"):
            return {"symbol": SYMBOL, "price": 0, "msg": "Failed to fetch"}

        # ✅ Safely pick correct price field
        result = data.get("result", {})
        price = float(
            result.get("mark_price")
            or result.get("index_price")
            or result.get("spot_price")
            or 0
        )

        # ✅ Return ETH live price
        return {
            "symbol": SYMBOL,
            "price": round(price, 2),
            "msg": "ETH live price fetched successfully"
        }

    except Exception as e:
        print("Price error:", e)
        return {"symbol": "ETHUSD", "price": 0, "msg": str(e)}



@app.get("/balance")
async def get_balance():
    try:
        import time, hmac, hashlib, os, requests
        from fastapi.responses import JSONResponse

        API_KEY = "YOUR_API_KEY"
        API_SECRET = "YOUR_API_SECRET"
        BASE_URL = "https://api.india.delta.exchange"

        method = "GET"
        ts = str(int(time.time()))
        path = "/v2/wallet/balances"
        msg = method + ts + path
        signature = hmac.new(
            API_SECRET.encode(), msg.encode(), hashlib.sha256
        ).hexdigest()

        headers = {
            "Accept": "application/json",
            "api-key": API_KEY,
            "signature": signature,
            "timestamp": ts,
            "User-Agent": "python-fastapi-client"
        }

        url = BASE_URL + path
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        assets = data.get("result", [])
        asset = next(
            (x for x in assets if x.get("asset_symbol") in ["USD", "USDT", "USDC", "MARGIN", "CASH"]),
            None
        )

        if asset:
            balance = asset.get("available_balance", asset.get("balance", 0))
            symbol = asset.get("asset_symbol", "")
            return {
                "balance": float(balance),
                "symbol": symbol,
                "msg": "Balance fetched successfully"
            }
        else:
            return {
                "balance": 0,
                "symbol": "",
                "msg": "No USD/USDT/USDC/MARGIN/CASH asset found"
            }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
