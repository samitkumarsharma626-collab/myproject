"""
Unit tests for configuration parsing
"""

import os
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config import Settings, get_settings


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_default_values(self):
        """Test default configuration values"""
        # Clear cache to test fresh instance
        get_settings.cache_clear()
        
        # Set minimal env vars
        os.environ.pop("DELTA_API_KEY", None)
        os.environ.pop("DELTA_API_SECRET", None)
        os.environ.pop("DEBUG", None)
        os.environ.pop("ENVIRONMENT", None)
        
        settings = Settings()
        
        assert settings.DELTA_API_KEY == ""
        assert settings.DELTA_API_SECRET == ""
        assert settings.DELTA_BASE_URL == "https://api.india.delta.exchange"
        assert settings.TRADING_SYMBOL == "ETHUSD"
        assert settings.CANDLE_RESOLUTION == "15m"
        assert settings.DEBUG is False
        assert settings.ENVIRONMENT == "production"
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
    
    def test_resolution_validation(self):
        """Test candle resolution validation"""
        valid_resolutions = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
        
        for resolution in valid_resolutions:
            settings = Settings(CANDLE_RESOLUTION=resolution)
            assert settings.CANDLE_RESOLUTION == resolution
        
        # Invalid resolution should raise error
        with pytest.raises(ValidationError):
            Settings(CANDLE_RESOLUTION="invalid")
    
    def test_environment_validation(self):
        """Test environment validation"""
        valid_envs = ["production", "development", "testing"]
        
        for env in valid_envs:
            settings = Settings(ENVIRONMENT=env)
            assert settings.ENVIRONMENT == env.lower()
        
        # Invalid environment should raise error
        with pytest.raises(ValidationError):
            Settings(ENVIRONMENT="invalid")
    
    def test_log_level_validation(self):
        """Test log level validation"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            settings = Settings(LOG_LEVEL=level)
            assert settings.LOG_LEVEL == level.upper()
        
        # Invalid log level should raise error
        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="invalid")
    
    def test_port_validation(self):
        """Test port range validation"""
        # Valid ports
        assert Settings(PORT=1).PORT == 1
        assert Settings(PORT=8000).PORT == 8000
        assert Settings(PORT=65535).PORT == 65535
        
        # Invalid ports should raise error
        with pytest.raises(ValidationError):
            Settings(PORT=0)
        
        with pytest.raises(ValidationError):
            Settings(PORT=65536)
    
    def test_request_timeout_validation(self):
        """Test request timeout validation"""
        # Valid timeouts
        assert Settings(REQUEST_TIMEOUT=1).REQUEST_TIMEOUT == 1
        assert Settings(REQUEST_TIMEOUT=60).REQUEST_TIMEOUT == 60
        
        # Invalid timeouts should raise error
        with pytest.raises(ValidationError):
            Settings(REQUEST_TIMEOUT=0)
        
        with pytest.raises(ValidationError):
            Settings(REQUEST_TIMEOUT=61)
    
    def test_cors_origins_list(self):
        """Test CORS origins list parsing"""
        settings = Settings(CORS_ORIGINS="https://example.com,https://app.example.com")
        origins = settings.cors_origins_list
        assert len(origins) == 2
        assert "https://example.com" in origins
        assert "https://app.example.com" in origins
        
        # Empty CORS origins
        settings = Settings(CORS_ORIGINS="")
        assert settings.cors_origins_list == []
    
    def test_production_flags(self):
        """Test production/development flags"""
        # Production mode
        settings = Settings(ENVIRONMENT="production", DEBUG=False)
        assert settings.is_production is True
        assert settings.is_development is False
        
        # Development mode
        settings = Settings(ENVIRONMENT="development", DEBUG=True)
        assert settings.is_production is False
        assert settings.is_development is True
        
        # Production with DEBUG=True is not production
        settings = Settings(ENVIRONMENT="production", DEBUG=True)
        assert settings.is_production is False
        assert settings.is_development is True
    
    def test_production_validation(self):
        """Test production validation requirements"""
        # Production without API key should fail
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            DELTA_API_KEY="",
            DELTA_API_SECRET=""
        )
        
        with pytest.raises(ValueError, match="DELTA_API_KEY"):
            settings.validate_required()
        
        # Production without API secret should fail
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            DELTA_API_KEY="test_key",
            DELTA_API_SECRET=""
        )
        
        with pytest.raises(ValueError, match="DELTA_API_SECRET"):
            settings.validate_required()
        
        # Production with wildcard CORS should fail
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            DELTA_API_KEY="test_key",
            DELTA_API_SECRET="test_secret",
            CORS_ORIGINS="*"
        )
        
        with pytest.raises(ValueError, match="CORS_ORIGINS"):
            settings.validate_required()
        
        # Valid production config
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            DELTA_API_KEY="test_key",
            DELTA_API_SECRET="test_secret",
            CORS_ORIGINS="https://example.com"
        )
        
        # Should not raise
        settings.validate_required()
        assert settings.is_valid() is True
    
    def test_static_dir_validation(self):
        """Test static directory validation"""
        # Should convert string to Path
        settings = Settings(STATIC_DIR="./static")
        assert isinstance(settings.STATIC_DIR, Path)
        
        # Should resolve path
        settings = Settings(STATIC_DIR="./static")
        assert settings.STATIC_DIR.is_absolute()
    
    def test_settings_caching(self):
        """Test settings caching"""
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should return same instance
        assert settings1 is settings2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
