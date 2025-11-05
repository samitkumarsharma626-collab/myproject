"""Custom middleware for error handling and security headers."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from .config import AppSettings


class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware providing consistent error responses."""

    def __init__(self, app: ASGIApp, settings: AppSettings) -> None:
        super().__init__(app)
        self._settings = settings
        self._logger = logging.getLogger("app.exceptions")

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        correlation_id = request.headers.get("X-Request-ID", str(uuid4()))
        try:
            response = await call_next(request)
            return response
        except RequestValidationError as exc:
            validation_payload: dict[str, Any] = {
                "error": "Validation Error",
                "correlation_id": correlation_id,
                "details": exc.errors(),
            }
            self._logger.warning(
                "Validation failed",
                extra={
                    "event": "validation_error",
                    "path": str(request.url.path),
                    "correlation_id": correlation_id,
                },
            )
            return JSONResponse(status_code=422, content=validation_payload)
        except HTTPException as exc:
            http_payload: dict[str, Any] = {
                "error": _normalize_detail(exc.detail),
                "status_code": exc.status_code,
                "correlation_id": correlation_id,
            }
            self._logger.info(
                "Handled HTTPException",
                extra={
                    "event": "http_exception",
                    "status_code": exc.status_code,
                    "path": str(request.url.path),
                    "correlation_id": correlation_id,
                },
            )
            return JSONResponse(status_code=exc.status_code, content=http_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            error_payload: dict[str, Any] = {
                "error": "Internal Server Error",
                "correlation_id": correlation_id,
            }
            if self._settings.debug or self._settings.expose_stack_traces:
                error_payload["detail"] = repr(exc)

            self._logger.error(
                "Unhandled exception",
                exc_info=self._settings.debug or self._settings.expose_stack_traces,
                extra={
                    "event": "unhandled_exception",
                    "path": str(request.url.path),
                    "correlation_id": correlation_id,
                },
            )
            return JSONResponse(status_code=500, content=error_payload)


def _normalize_detail(detail: Any) -> Any:
    if isinstance(detail, (dict, list)):
        return detail
    return str(detail)
