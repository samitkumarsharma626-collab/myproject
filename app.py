"""ASGI entry point ensuring production-safe execution with Uvicorn and systemd."""

from __future__ import annotations

import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI

from app.config import AppSettings, SettingsError, get_settings
from app.main import create_app

logger = logging.getLogger("uvicorn.error")

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _get_settings() -> AppSettings:
    """Load application settings and surface configuration issues early."""

    try:
        return get_settings()
    except SettingsError as exc:
        logger.critical("Failed to load application settings: %s", exc, extra={"event": "settings_error"})
        raise


def get_application() -> FastAPI:
    """Create a FastAPI application instance using validated settings."""

    settings = _get_settings()
    return create_app(settings)


try:
    app: FastAPI = get_application()
except SettingsError as exc:
    raise SystemExit(2) from exc


def _get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean from environment variables."""

    value = os.getenv(name)
    if value is None:
        return default

    normalised = value.strip().lower()
    if normalised in _TRUE_VALUES:
        return True
    if normalised in _FALSE_VALUES:
        return False

    logger.warning(
        "Invalid boolean value '%s' for %s; falling back to default=%s",
        value,
        name,
        default,
        extra={"event": "invalid_env_bool"},
    )
    return default


def _get_env_int(name: str, default: int | None) -> int | None:
    """Read an integer from environment variables."""

    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Invalid integer value '%s' for %s; falling back to default=%s",
            value,
            name,
            default,
            extra={"event": "invalid_env_int"},
        )
        return default


def build_uvicorn_config(asgi_app: FastAPI) -> uvicorn.Config:
    """Build a configured Uvicorn Config instance honouring environment overrides."""

    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = _get_env_int("UVICORN_PORT", 8000) or 8000
    uds = os.getenv("UVICORN_UDS")

    reload_enabled = _get_env_bool("UVICORN_RELOAD", default=False)
    workers = _get_env_int("UVICORN_WORKERS", default=None)
    if reload_enabled and workers:
        logger.warning(
            "Both reload and workers requested; ignoring workers because reload implies a single process",
            extra={"event": "reload_workers_conflict"},
        )
        workers = None

    log_level = os.getenv("UVICORN_LOG_LEVEL", os.getenv("LOG_LEVEL", "info")).lower()
    access_log = _get_env_bool("UVICORN_ACCESS_LOG", default=False)
    proxy_headers = _get_env_bool("UVICORN_PROXY_HEADERS", default=True)
    forwarded_allow_ips = os.getenv("UVICORN_FORWARDED_ALLOW_IPS", "127.0.0.1")

    limit_concurrency = _get_env_int("UVICORN_LIMIT_CONCURRENCY", default=None)
    limit_max_requests = _get_env_int("UVICORN_LIMIT_MAX_REQUESTS", default=None)
    timeout_keep_alive = _get_env_int("UVICORN_TIMEOUT_KEEP_ALIVE", default=5) or 5
    timeout_graceful_shutdown = _get_env_int("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", default=30) or 30

    loop = os.getenv("UVICORN_LOOP", "auto")
    http = os.getenv("UVICORN_HTTP", "auto")
    ws = os.getenv("UVICORN_WS", "auto")
    lifespan = os.getenv("UVICORN_LIFESPAN", "auto")
    root_path = os.getenv("UVICORN_ROOT_PATH", "")

    ssl_keyfile = os.getenv("UVICORN_SSL_KEYFILE")
    ssl_certfile = os.getenv("UVICORN_SSL_CERTFILE")
    ssl_ca_certs = os.getenv("UVICORN_SSL_CA_CERTS")
    ssl_keyfile_password = os.getenv("UVICORN_SSL_KEYFILE_PASSWORD")

    config_kwargs: dict[str, Any] = {
        "app": asgi_app,
        "reload": reload_enabled,
        "workers": workers,
        "log_level": log_level,
        "access_log": access_log,
        "proxy_headers": proxy_headers,
        "forwarded_allow_ips": forwarded_allow_ips,
        "limit_concurrency": limit_concurrency,
        "limit_max_requests": limit_max_requests,
        "timeout_keep_alive": timeout_keep_alive,
        "timeout_graceful_shutdown": timeout_graceful_shutdown,
        "loop": loop,
        "http": http,
        "ws": ws,
        "lifespan": lifespan,
        "root_path": root_path,
        "log_config": None,  # respect application logging configuration
    }

    if uds:
        config_kwargs["uds"] = uds
    else:
        config_kwargs["host"] = host
        config_kwargs["port"] = port

    if ssl_keyfile:
        config_kwargs["ssl_keyfile"] = ssl_keyfile
    if ssl_certfile:
        config_kwargs["ssl_certfile"] = ssl_certfile
    if ssl_ca_certs:
        config_kwargs["ssl_ca_certs"] = ssl_ca_certs
    if ssl_keyfile_password:
        config_kwargs["ssl_keyfile_password"] = ssl_keyfile_password

    # Remove keys with None values to avoid passing unexpected parameters.
    filtered_kwargs = {key: value for key, value in config_kwargs.items() if value is not None}
    return uvicorn.Config(**filtered_kwargs)


def run() -> None:
    """Launch the application using Uvicorn (useful for systemd ExecStart)."""

    config = build_uvicorn_config(app)
    server = uvicorn.Server(config)

    logger.info(
        "Starting Uvicorn server",
        extra={
            "event": "uvicorn_start",
            "host": getattr(config, "host", None),
            "port": getattr(config, "port", None),
            "uds": getattr(config, "uds", None),
            "workers": config.workers,
            "reload": config.reload,
        },
    )

    try:
        server.run()
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Uvicorn server terminated unexpectedly", extra={"event": "uvicorn_failure"})
        raise


def main() -> None:
    """Entry point for command-line execution."""

    run()


if __name__ == "__main__":
    main()


__all__ = ["app", "get_application", "build_uvicorn_config", "run", "main"]
