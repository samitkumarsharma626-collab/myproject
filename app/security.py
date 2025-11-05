"""Security-related middleware helpers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .config import AppSettings


def configure_security(app: FastAPI, settings: AppSettings) -> None:
    """Attach security-related middleware."""

    if settings.enforce_https:
        app.add_middleware(HTTPSRedirectMiddleware)

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
        max_age=86400,
    )

    app.add_middleware(SecureHeadersMiddleware, settings=settings)


class SecureHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce secure HTTP response headers and cookies."""

    def __init__(self, app: ASGIApp, settings: AppSettings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        headers = MutableHeaders(response.headers)

        headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={self._settings.hsts_max_age}; includeSubDomains; preload",
        )
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer-when-downgrade")
        headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        headers["Content-Security-Policy"] = self._settings.content_security_policy

        self._secure_cookies(headers)

        return response

    def _secure_cookies(self, headers: MutableHeaders) -> None:
        cookies = headers.getlist("set-cookie")
        if not cookies:
            return

        while "set-cookie" in headers:
            del headers["set-cookie"]

        for cookie in cookies:
            secured = self._ensure_cookie_secure(cookie)
            headers.append("set-cookie", secured)

    def _ensure_cookie_secure(self, cookie_header: str) -> str:
        directives = [segment.strip() for segment in cookie_header.split(";") if segment.strip()]
        flags = {directive.lower() for directive in directives}

        if "secure" not in flags:
            directives.append("Secure")
        if "httponly" not in flags:
            directives.append("HttpOnly")
        if not any(flag.startswith("samesite") for flag in flags):
            directives.append("SameSite=Strict")

        return "; ".join(directives)
