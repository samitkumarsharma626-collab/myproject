"""Pydantic schemas used for FastAPI responses."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    timestamp: datetime
    version: str
    mode: str


class PriceResponse(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    source: str


class BalanceResponse(BaseModel):
    balance: float
    asset: str | None
    timestamp: datetime
    available: bool
    message: str | None = None


class PriceBoxResponse(BaseModel):
    box: str
    high: float | None
    low: float | None
    timestamp: datetime


class CurrentCandle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float


class Indicators(BaseModel):
    range: float
    adx_14: float
    atr_14: float
    volume_vs_sma20: float
    trend_1h: str


class SuggestedBox(BaseModel):
    high: float
    low: float
    range: float


class Filters(BaseModel):
    range_ok: bool
    adx_ok: bool
    atr_ok: bool
    volume_ok: bool
    trend_ok: bool


class Thresholds(BaseModel):
    range_min: float
    adx_min: float
    atr_min: float
    volume_min_pct: float


class StrategyFiltersResponse(BaseModel):
    symbol: str
    timestamp: datetime
    current_candle: CurrentCandle
    indicators: Indicators
    suggested_box: SuggestedBox
    filters: Filters
    filters_passed: bool
    thresholds: Thresholds


class Candle(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: int | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None


class CandleResponse(BaseModel):
    symbol: str
    resolution: str
    candles: list[Candle]
    count: int
    timestamp: datetime
