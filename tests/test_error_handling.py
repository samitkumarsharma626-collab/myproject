"""
Unit tests for error handling middleware
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.error_handler import ErrorHandlerMiddleware
from app import app


class TestErrorHandling:
    """Test error handling middleware"""
    
    def test_404_error(self):
        """Test 404 error handling"""
        client = TestClient(app)
        
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "status_code" in data
    
    def test_validation_error(self):
        """Test validation error handling"""
        client = TestClient(app)
        
        # Invalid limit parameter
        response = client.get("/api/v1/candles?limit=1000")
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "validation_error" in data["error"] or "status_code" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
