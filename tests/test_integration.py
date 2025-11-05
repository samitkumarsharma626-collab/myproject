"""
Integration test for environment-driven startup
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.config import get_settings
from app import app


class TestEnvDrivenStartup:
    """Test environment-driven application startup"""
    
    def test_startup_with_minimal_env(self):
        """Test startup with minimal environment variables"""
        # Clear cache
        get_settings.cache_clear()
        
        # Set minimal env for development
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"
        os.environ.pop("DELTA_API_KEY", None)
        os.environ.pop("DELTA_API_SECRET", None)
        
        # Should not raise
        client = TestClient(app)
        
        # Health check should work
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["environment"] == "development"
    
    def test_startup_with_production_env(self):
        """Test startup with production environment"""
        # Clear cache
        get_settings.cache_clear()
        
        # Set production env
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "false"
        os.environ["DELTA_API_KEY"] = "test_key"
        os.environ["DELTA_API_SECRET"] = "test_secret"
        os.environ["CORS_ORIGINS"] = "https://example.com"
        
        # Should not raise
        client = TestClient(app)
        
        # Health check should work
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["environment"] == "production"
        assert data["debug"] is False
    
    def test_api_docs_disabled_in_production(self):
        """Test that API docs are disabled in production"""
        # Clear cache
        get_settings.cache_clear()
        
        # Set production env
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "false"
        os.environ["DELTA_API_KEY"] = "test_key"
        os.environ["DELTA_API_SECRET"] = "test_secret"
        os.environ["CORS_ORIGINS"] = "https://example.com"
        
        client = TestClient(app)
        
        # API docs should not be available
        response = client.get("/api/docs")
        assert response.status_code == 404
        
        response = client.get("/api/openapi.json")
        assert response.status_code == 404
    
    def test_api_docs_enabled_in_development(self):
        """Test that API docs are enabled in development"""
        # Clear cache
        get_settings.cache_clear()
        
        # Set development env
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"
        os.environ.pop("DELTA_API_KEY", None)
        os.environ.pop("DELTA_API_SECRET", None)
        
        client = TestClient(app)
        
        # API docs should be available
        response = client.get("/api/docs")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
