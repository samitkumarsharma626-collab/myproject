from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, BaseSettings, Field, AnyHttpUrl, validator

# Load .env if present
load_dotenv()

Environment = Literal["development", "production", "staging", "test"]

class SecuritySettings(BaseModel):
    cors_origins: List[str] = Field(default_factory=lambda: ["https://example.com"])  # whitelist
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"]) 
    force_https: bool = True
    secure_cookies: bool = True
    session_cookie_name: str = "session"
    csp_default_src: str = "'self'"
    csp_script_src: List[str] = Field(default_factory=lambda: ["'self'"])
    csp_style_src: List[str] = Field(default_factory=lambda: ["'self'", "'unsafe-inline'"])  # keep minimal inline for static demo
    csp_img_src: List[str] = Field(default_factory=lambda: ["'self'", "data:"])

class Settings(BaseSettings):
    # App
    app_name: str = "ETH Strategy Dashboard API"
    environment: Environment = Field("production", env="APP_ENV")
    debug: bool = Field(False, env="DEBUG")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")

    # External API
    delta_api_key: Optional[str] = Field(None, env="DELTA_API_KEY")
    delta_api_secret: Optional[str] = Field(None, env="DELTA_API_SECRET")
    delta_base_url: AnyHttpUrl = Field("https://api.india.delta.exchange", env="DELTA_BASE_URL")

    # Trading params
    trading_symbol: str = Field("ETHUSD", env="TRADING_SYMBOL")
    candle_resolution: str = Field("15m", env="CANDLE_RESOLUTION")

    # Paths
    static_dir: Path = Field(Path("./static"), env="STATIC_DIR")

    # HTTP
    request_timeout_seconds: int = Field(10, env="REQUEST_TIMEOUT")

    # Security
    security: SecuritySettings = SecuritySettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("static_dir", pre=True)
    def _coerce_static_dir(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @validator("candle_resolution")
    def _validate_resolution(cls, v: str) -> str:
        allowed = {"1m","3m","5m","15m","30m","1h","2h","4h","1d"}
        if v not in allowed:
            raise ValueError(f"CANDLE_RESOLUTION must be one of {sorted(allowed)}")
        return v

    def require_api_credentials(self) -> None:
        if not self.delta_api_key or not self.delta_api_secret:
            raise ValueError("Exchange API credentials are required for authenticated endpoints")

    def is_production(self) -> bool:
        return self.environment == "production"

    def is_valid(self) -> bool:
        try:
            # basic sanity checks
            _ = self.delta_base_url
            _ = self.static_dir
            _ = self.trading_symbol
            return True
        except Exception:
            return False

# Singleton settings instance
settings = Settings()