"""
Secure Configuration Management
Uses Pydantic for validation and environment variables for all settings
"""

from pathlib import Path
from typing import Literal, Optional, List
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and security best practices
    All sensitive data loaded from environment variables
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Mode
    debug: bool = Field(default=False, description="Debug mode (never enable in production)")
    environment: Literal["development", "staging", "production"] = Field(
        default="production",
        description="Application environment"
    )
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    
    # Delta Exchange API Credentials (REQUIRED for trading features)
    delta_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Delta Exchange API Key"
    )
    delta_api_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Delta Exchange API Secret"
    )
    delta_base_url: str = Field(
        default="https://api.india.delta.exchange",
        description="Delta Exchange API base URL"
    )
    
    # Trading Configuration
    trading_symbol: str = Field(default="ETHUSD", description="Trading symbol")
    candle_resolution: str = Field(default="15m", description="Candle resolution")
    request_timeout: int = Field(default=10, ge=1, le=60, description="API request timeout in seconds")
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins (comma-separated in env)"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    
    # Security Headers
    trusted_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Trusted hosts for security middleware"
    )
    
    # File Paths
    static_dir: Path = Field(default=Path("./static"), description="Static files directory")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log format (json for production, text for dev)"
    )
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @field_validator("trusted_hosts", mode="before")
    @classmethod
    def parse_trusted_hosts(cls, v):
        """Parse trusted hosts from comma-separated string or list"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v
    
    @field_validator("static_dir", mode="before")
    @classmethod
    def parse_static_dir(cls, v):
        """Ensure static_dir is a Path object"""
        if isinstance(v, str):
            return Path(v)
        return v
    
    def validate_credentials(self) -> None:
        """
        Validate that required credentials are configured
        Raises ValueError if critical settings are missing in production
        """
        errors = []
        
        # Check API credentials
        api_key = self.delta_api_key.get_secret_value()
        api_secret = self.delta_api_secret.get_secret_value()
        
        if not api_key or api_key == "YOUR_API_KEY":
            errors.append("DELTA_API_KEY is not configured")
        
        if not api_secret or api_secret == "YOUR_API_SECRET":
            errors.append("DELTA_API_SECRET is not configured")
        
        # Check static directory
        if not self.static_dir.exists():
            errors.append(f"Static directory does not exist: {self.static_dir}")
        
        # In production, fail fast
        if self.environment == "production" and errors:
            raise ValueError(
                f"Critical configuration errors in production:\n" + 
                "\n".join(f"  - {err}" for err in errors)
            )
        
        # In development, just warn
        if errors and self.environment != "production":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Configuration warnings:\n" + "\n".join(f"  - {err}" for err in errors))
    
    def get_api_key(self) -> str:
        """Safely get API key (for internal use only)"""
        return self.delta_api_key.get_secret_value()
    
    def get_api_secret(self) -> str:
        """Safely get API secret (for internal use only)"""
        return self.delta_api_secret.get_secret_value()
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    def is_valid(self) -> bool:
        """
        Check if configuration is valid
        Returns True if all required settings are configured
        """
        try:
            self.validate_credentials()
            return True
        except ValueError:
            return False


# Global settings instance
settings = Settings()


# Validate on import
try:
    settings.validate_credentials()
except ValueError as e:
    import sys
    import logging
    
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Configuration validation failed: {e}")
    
    # Only exit in production
    if settings.environment == "production":
        sys.exit(1)
