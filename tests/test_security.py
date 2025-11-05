"""
Unit tests for security middleware
"""

import pytest
from fastapi.testclient import TestClient

from app import app


class TestSecurityMiddleware:
    """Test security middleware"""
    
    def test_security_headers_present(self):
        """Test that security headers are present"""
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check security headers
        assert "Content-Security-Policy" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers
    
    def test_csp_header_content(self):
        """Test CSP header content"""
        client = TestClient(app)
        
        response = client.get("/health")
        csp = response.headers.get("Content-Security-Policy", "")
        
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
