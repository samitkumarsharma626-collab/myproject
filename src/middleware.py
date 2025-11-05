"""
Secure middleware for FastAPI application
"""

import logging
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    def __init__(self, app, force_https: bool = True, secure_cookies: bool = True):
        super().__init__(app)
        self.force_https = force_https
        self.secure_cookies = secure_cookies
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers"""
        response = await call_next(request)
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https://api.india.delta.exchange; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HTTPS enforcement
        if self.force_https:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Secure cookies
        if self.secure_cookies and "Set-Cookie" in response.headers:
            cookies = response.headers.get_list("Set-Cookie")
            response.headers.pop("Set-Cookie", None)
            for cookie in cookies:
                secure_cookie = cookie
                if "Secure" not in cookie:
                    secure_cookie += "; Secure"
                if "HttpOnly" not in cookie:
                    secure_cookie += "; HttpOnly"
                if "SameSite=None" not in cookie and "SameSite=" not in cookie:
                    secure_cookie += "; SameSite=Lax"
                response.headers.append("Set-Cookie", secure_cookie)
        
        return response


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP to HTTPS in production"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Redirect to HTTPS if needed"""
        if request.url.scheme == "http":
            # Check if X-Forwarded-Proto header exists (from reverse proxy)
            forwarded_proto = request.headers.get("X-Forwarded-Proto")
            if forwarded_proto == "https":
                # Already HTTPS behind proxy, continue
                return await call_next(request)
            
            # Redirect to HTTPS
            https_url = request.url.replace(scheme="https")
            return Response(
                status_code=status.HTTP_301_MOVED_PERMANENTLY,
                headers={"Location": str(https_url)}
            )
        
        return await call_next(request)
