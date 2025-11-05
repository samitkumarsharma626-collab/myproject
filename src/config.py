"""
Secure configuration management with validation
Uses pydantic BaseSettings for type-safe environment variable parsing
"""

import os
from pathlib import Path
from typing import List, Optional, Set
from functools import lru_cache

from pydantic import Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Credentials (required in production)
    DELTA_API_KEY: str = Field(
        default="",
        description="Delta Exchange API key"
    )
    DELTA_API_SECRET: str = Field(
        default="",
        description="Delta Exchange API secret"
    )
    
    # API Configuration
    DELTA_BASE_URL: HttpUrl = Field(
        default="https://api.india.delta.exchange",
        description="Delta Exchange API base URL"
    )
    TRADING_SYMBOL: str = Field(
        default="ETHUSD",
        description="Trading symbol"
    )
    CANDLE_RESOLUTION: str = Field(
        default="15m",
        description="Candle resolution (e.g., '15m', '1h')"
    )
    REQUEST_TIMEOUT: int = Field(
        default=10,
        ge=1,
        le=60,
        description="API request timeout in seconds"
    )
    
    # Server Configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    
    # Security Settings
    DEBUG: bool = Field(
        default=False,
        description="Debug mode (disables security features)"
    )
    ENVIRONMENT: str = Field(
        default="production",
        description="Environment: production, development, testing"
    )
    
    # CORS Configuration
    CORS_ORIGINS: str = Field(
        default="",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # Security Headers
    FORCE_HTTPS: bool = Field(
        default=True,
        description="Force HTTPS redirects"
    )
    SECURE_COOKIES: bool = Field(
        default=True,
        description="Use secure cookies"
    )
    
    # Paths
    STATIC_DIR: Path = Field(
        default=Path("./static"),
        description="Static files directory"
    )
    
    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json or text"
    )
    
    @field_validator("CANDLE_RESOLUTION")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        """Validate candle resolution"""
        valid_resolutions = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
        if v not in valid_resolutions:
            raise ValueError(f"Invalid resolution: {v}. Must be one of {valid_resolutions}")
        return v
    
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment"""
        valid_envs = {"production", "development", "testing"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("STATIC_DIR", mode="before")
    @classmethod
    def validate_static_dir(cls, v: str | Path) -> Path:
        """Convert and validate static directory"""
        path = Path(v) if isinstance(v, str) else v
        return path.resolve()
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as list"""
        if not self.CORS_ORIGINS:
            return [] if self.DEBUG else []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production" and not self.DEBUG
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development" or self.DEBUG
    
    def validate_required(self) -> None:
        """Validate required settings for production"""
        if self.is_production:
            if not self.DELTA_API_KEY:
                raise ValueError("DELTA_API_KEY is required in production")
            if not self.DELTA_API_SECRET:
                raise ValueError("DELTA_API_SECRET is required in production")
            if not self.CORS_ORIGINS:
                raise ValueError("CORS_ORIGINS must be set in production (cannot be wildcard)")
            if "*" in self.cors_origins_list:
                raise ValueError("CORS_ORIGINS cannot contain '*' in production")
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        try:
            self.validate_required()
            return True
        except ValueError:
            return False
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.validate_required()
    return settings


# Export settings instance
settings = get_settings()
