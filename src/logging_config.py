"""
Structured logging configuration
"""

import logging
import json
import sys
from typing import Any, Dict
from datetime import datetime, timezone
from pathlib import Path


class SecureJSONFormatter(logging.Formatter):
    """JSON formatter that excludes sensitive data"""
    
    # Fields that should never be logged
    SENSITIVE_FIELDS = {
        "api_key", "api_secret", "password", "secret", "token",
        "authorization", "auth", "credential", "private_key",
        "access_token", "refresh_token", "delta_api_key", "delta_api_secret"
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        # Add extra fields, but filter sensitive data
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key.startswith("_") or key in ["name", "msg", "args", "created", "filename", "levelname", "levelno", "lineno", "module", "msecs", "message", "pathname", "process", "processName", "relativeCreated", "thread", "threadName", "exc_info", "exc_text", "stack_info"]:
                    continue
                
                # Check if key contains sensitive information
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in self.SENSITIVE_FIELDS):
                    log_data[key] = "***REDACTED***"
                else:
                    log_data[key] = self._sanitize_value(value)
        
        return json.dumps(log_data)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize values to remove sensitive data"""
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) if k.lower() not in self.SENSITIVE_FIELDS else "***REDACTED***" 
                   for k, v in value.items()}
        elif isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        elif isinstance(value, str):
            # Check if string contains sensitive patterns
            if any(sensitive in value.lower() for sensitive in self.SENSITIVE_FIELDS):
                return "***REDACTED***"
        return value


class TextFormatter(logging.Formatter):
    """Simple text formatter for development"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text"""
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        level = record.levelname.ljust(8)
        logger_name = record.name
        message = record.getMessage()
        
        log_line = f"{timestamp} | {level} | {logger_name} | {message}"
        
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    show_stack_traces: bool = False
) -> None:
    """
    Setup structured logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type (json or text)
        show_stack_traces: Whether to show full stack traces (dev only)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Set formatter
    if log_format.lower() == "json":
        formatter = SecureJSONFormatter()
    else:
        formatter = TextFormatter()
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
