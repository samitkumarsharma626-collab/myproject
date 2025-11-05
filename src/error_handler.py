"""
Centralized error handling middleware
"""

import logging
from typing import Callable
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Centralized error handling"""
    
    def __init__(self, app, show_details: bool = False):
        super().__init__(app)
        self.show_details = show_details
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Handle errors"""
        try:
            response = await call_next(request)
            return response
        except StarletteHTTPException as exc:
            return self._handle_http_exception(request, exc)
        except RequestValidationError as exc:
            return self._handle_validation_error(request, exc)
        except ValidationError as exc:
            return self._handle_validation_error(request, exc)
        except Exception as exc:
            return self._handle_unexpected_error(request, exc)
    
    def _handle_http_exception(self, request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle HTTP exceptions"""
        status_code = exc.status_code
        detail = exc.detail
        
        # Log error
        logger.warning(
            f"HTTP {status_code} error",
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "status_code": status_code,
                "detail": detail
            }
        )
        
        # Determine error type
        if 400 <= status_code < 500:
            error_type = "client_error"
        elif 500 <= status_code < 600:
            error_type = "server_error"
        else:
            error_type = "http_error"
        
        response_data = {
            "error": error_type,
            "status_code": status_code,
            "message": detail if isinstance(detail, str) else "An error occurred",
        }
        
        if self.show_details and isinstance(detail, dict):
            response_data["details"] = detail
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def _handle_validation_error(self, request: Request, exc: RequestValidationError | ValidationError) -> JSONResponse:
        """Handle validation errors"""
        errors = exc.errors() if hasattr(exc, "errors") else []
        
        logger.warning(
            "Validation error",
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "errors": errors
            }
        )
        
        response_data = {
            "error": "validation_error",
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "message": "Validation failed",
        }
        
        if self.show_details:
            response_data["details"] = errors
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_data
        )
    
    def _handle_unexpected_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors"""
        # Log full error with stack trace
        logger.error(
            "Unexpected error",
            exc_info=exc,
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        )
        
        response_data = {
            "error": "internal_server_error",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "An unexpected error occurred",
        }
        
        if self.show_details:
            response_data["details"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_data
        )
