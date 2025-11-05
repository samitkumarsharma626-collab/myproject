"""Application configuration powered by Pydantic settings."""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


load_dotenv()


class SettingsError(RuntimeError):
    """Raised when configuration validation fails."""


class AppEnvironment(str, Enum):
    """Allowed application environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="ETH Strategy Dashboard API", alias="APP_NAME", min_length=3)
    environment: AppEnvironment = Field(default=AppEnvironment.DEVELOPMENT, alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")

    delta_api_key: SecretStr | None = Field(default=None, alias="DELTA_API_KEY")
    delta_api_secret: SecretStr | None = Field(default=None, alias="DELTA_API_SECRET")
    delta_base_url: AnyHttpUrl = Field(
        default="https://api.india.delta.exchange",
        alias="DELTA_BASE_URL",
    )
    trading_symbol: str = Field(default="ETHUSD", alias="TRADING_SYMBOL", min_length=1, max_length=20)
    candle_resolution: str = Field(default="15m", alias="CANDLE_RESOLUTION", min_length=2, max_length=5)

    static_dir: Path = Field(default=Path("./static"), alias="STATIC_DIR")
    request_timeout: int = Field(default=10, alias="REQUEST_TIMEOUT", ge=1, le=60)

    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["https://example.com"],
        alias="CORS_ALLOW_ORIGINS",
    )
    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"],
        alias="TRUSTED_HOSTS",
    )
    allow_credentials: bool = Field(default=False, alias="ALLOW_CREDENTIALS")

    enforce_https: bool = Field(default=True, alias="ENFORCE_HTTPS")
    hsts_max_age: int = Field(default=31536000, alias="HSTS_MAX_AGE", ge=0)
    content_security_policy: str = Field(
        default="default-src 'self'; frame-ancestors 'none'; script-src 'self'; object-src 'none'",
        alias="CONTENT_SECURITY_POLICY",
        min_length=10,
    )

    expose_stack_traces: bool = Field(default=False, alias="EXPOSE_STACKTRACES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def is_production(self) -> bool:
        return self.environment == AppEnvironment.PRODUCTION

    @property
    def api_key_required(self) -> bool:
        return self.is_production

    def get_api_key(self) -> str | None:
        return self.delta_api_key.get_secret_value() if self.delta_api_key else None

    def get_api_secret(self) -> str | None:
        return self.delta_api_secret.get_secret_value() if self.delta_api_secret else None

    @field_validator("static_dir", mode="after")
    @classmethod
    def _ensure_static_dir(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.exists():
            logger.warning("Static directory %s does not exist; continuing", resolved)
        return resolved

    @model_validator(mode="after")
    def _validate_security(self) -> "AppSettings":
        missing: list[str] = []
        if self.api_key_required:
            if not self.delta_api_key:
                missing.append("DELTA_API_KEY")
            if not self.delta_api_secret:
                missing.append("DELTA_API_SECRET")

        if missing:
            raise ValueError(
                "Missing required environment variables for production: " + ", ".join(missing)
            )

        if self.is_production:
            if any(origin == "*" for origin in self.cors_allow_origins):
                raise ValueError("Wildcard CORS origins are not permitted in production")
            if any(self._is_disallowed_host(host) for host in self.trusted_hosts):
                raise ValueError("Trusted hosts cannot contain '*' or unspecified addresses in production")

        if not self.cors_allow_origins:
            raise ValueError("At least one CORS origin must be configured")

        if not self.trusted_hosts:
            raise ValueError("At least one trusted host must be configured")

        return self

    def masked(self) -> dict[str, str | int | bool | list[str]]:
        """Return non-sensitive settings for logging or debugging purposes."""

        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "debug": self.debug,
            "delta_base_url": str(self.delta_base_url),
            "trading_symbol": self.trading_symbol,
            "candle_resolution": self.candle_resolution,
            "static_dir": str(self.static_dir),
            "request_timeout": self.request_timeout,
            "cors_allow_origins": self.cors_allow_origins,
            "trusted_hosts": self.trusted_hosts,
            "enforce_https": self.enforce_https,
            "hsts_max_age": self.hsts_max_age,
            "content_security_policy": self.content_security_policy,
            "expose_stack_traces": self.expose_stack_traces,
            "log_level": self.log_level,
        }

    @staticmethod
    def _is_disallowed_host(host: str) -> bool:
        if host == "*":
            return True
        try:
            return ip_address(host).is_unspecified
        except ValueError:
            return False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Instantiate and cache validated application settings."""

    try:
        settings = AppSettings()
    except ValidationError as exc:  # pragma: no cover - formatting error message
        rendered = "\n".join(
            f"- {' -> '.join(str(loc) for loc in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        message = f"Configuration validation failed:\n{rendered}"
        raise SettingsError(message) from exc

    return settings


def reload_settings() -> AppSettings:
    """Clear the cached settings and reload them (useful for tests)."""

    get_settings.cache_clear()
    settings = get_settings()
    logger.info("Settings reloaded", extra={"config": settings.masked()})
    return settings
