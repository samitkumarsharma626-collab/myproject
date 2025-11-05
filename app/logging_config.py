"""Structured logging configuration for the application."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .config import AppSettings

LOGGING_CONFIGURED = False


class SecretsFilter(logging.Filter):
    """Filter that redacts configured secrets from log records."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [secret for secret in secrets if secret]

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - defensive
        if not self._secrets:
            return True

        record.msg = self._redact(record.msg)

        if isinstance(record.args, tuple):
            record.args = tuple(self._redact(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: self._redact(value) for key, value in record.args.items()}

        for key, value in list(record.__dict__.items()):
            if isinstance(value, str):
                record.__dict__[key] = self._redact(value)

        return True

    def _redact(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        redacted = value
        for secret in self._secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted


class JsonFormatter(logging.Formatter):
    """Render log records as JSON."""

    def __init__(self, include_stack_traces: bool) -> None:
        super().__init__()
        self._include_stack_traces = include_stack_traces

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting only
        log_record: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            if self._include_stack_traces:
                log_record["stacktrace"] = self.formatException(record.exc_info)
            else:
                log_record["error"] = str(record.exc_info[1])

        for attribute in ("event", "config", "environment", "path"):
            value = getattr(record, attribute, None)
            if value is not None:
                log_record[attribute] = value

        return json.dumps(log_record, default=str)


def setup_logging(settings: AppSettings) -> logging.Logger:
    """Configure application logging once and return the application logger."""

    global LOGGING_CONFIGURED
    if LOGGING_CONFIGURED:
        return logging.getLogger("app")

    handler = logging.StreamHandler()
    formatter = JsonFormatter(include_stack_traces=settings.debug or settings.expose_stack_traces)
    handler.setFormatter(formatter)

    secrets = [value for value in (settings.get_api_key(), settings.get_api_secret()) if value]
    handler.addFilter(SecretsFilter(secrets))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    LOGGING_CONFIGURED = True

    logger = logging.getLogger("app")
    logger.debug("Structured logging initialized", extra={"config": settings.masked()})

    return logger
