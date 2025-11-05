"""
Integration tests for the FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os


@pytest.fixture
def test_env(monkeypatch, tmp_path):
    """Set up test environment variables"""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    
    # Create a test index.html
    (static_dir / "index.html").write_text("<html><body>Test</body></html>")
    
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DELTA_API_KEY", "test_key")
    monkeypatch.setenv("DELTA_API_SECRET", "test_secret")
    monkeypatch.setenv("STATIC_DIR", str(static_dir))
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    
    # Force reload of config module to pick up new env vars
    import sys
    if 'config' in sys.modules:
        del sys.modules['config']
    if 'app' in sys.modules:
        del sys.modules['app']


@pytest.fixture
def client(test_env):
    """Create test client"""
    from app import app
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "config_valid" in data


class TestSecurityHeaders:
    """Test security headers are applied"""
    
    def test_security_headers_present(self, client):
        """Test that security headers are present in responses"""
        response = client.get("/health")
        
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in response.headers
        
        assert "Strict-Transport-Security" in response.headers
        
        assert "Content-Security-Policy" in response.headers
        
        # Server header should be removed
        assert "Server" not in response.headers or response.headers.get("Server") != "uvicorn"


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_handler(self, client):
        """Test custom 404 handler"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] == "Not Found"
        assert "message" in data
        assert "path" in data
        assert "timestamp" in data
    
    def test_validation_error(self, client):
        """Test validation error handling"""
        response = client.get("/api/v1/candles?limit=9999")
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data


class TestAPIEndpoints:
    """Test API endpoints basic functionality"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint serves index.html"""
        response = client.get("/")
        assert response.status_code == 200
        assert b"<html>" in response.content
    
    def test_price_endpoint_structure(self, client):
        """Test price endpoint returns proper structure (may fail if API is down)"""
        response = client.get("/api/v1/price")
        
        # Endpoint should return either success or error, but not crash
        assert response.status_code in [200, 500, 503, 504]
        
        # If successful, check structure
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert "timestamp" in data
    
    def test_legacy_endpoints(self, client):
        """Test legacy endpoints are accessible"""
        # These should exist and not return 404
        for endpoint in ["/price", "/box", "/balance", "/strategy-filters"]:
            response = client.get(endpoint)
            # May fail due to API issues, but should not be 404
            assert response.status_code != 404


class TestConfigIntegration:
    """Integration test for environment-driven startup"""
    
    def test_app_starts_with_env_config(self, test_env):
        """Test that app starts successfully with environment configuration"""
        from app import app
        from config import settings
        
        # App should be created without errors
        assert app is not None
        assert app.title == "ETH Strategy Dashboard API"
        
        # Settings should be loaded
        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.get_api_key() == "test_key"
    
    def test_config_validation_on_startup(self, test_env):
        """Test that config validation happens on startup"""
        from config import settings
        
        # Should be valid with test env
        assert settings.is_valid() is True
        
        # Settings should have required values
        assert settings.get_api_key() != ""
        assert settings.static_dir.exists()


class TestMiddleware:
    """Test middleware functionality"""
    
    def test_error_handling_middleware(self, client):
        """Test that unhandled errors are caught by middleware"""
        # This tests the error handling middleware
        response = client.get("/health")
        
        # Should not raise unhandled exceptions
        assert response.status_code in [200, 500]
