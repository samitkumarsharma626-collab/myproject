"""
Unit tests for configuration management
"""

import pytest
import os
from pathlib import Path
from pydantic import ValidationError
from config import Settings


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_default_settings(self):
        """Test default settings are valid"""
        settings = Settings()
        assert settings.environment == "production"
        assert settings.debug is False
        assert settings.port == 8000
        assert settings.trading_symbol == "ETHUSD"
        assert settings.candle_resolution == "15m"
    
    def test_port_validation(self):
        """Test port number validation"""
        # Valid port
        settings = Settings(port=8080)
        assert settings.port == 8080
        
        # Invalid port (too high)
        with pytest.raises(ValidationError):
            Settings(port=99999)
        
        # Invalid port (too low)
        with pytest.raises(ValidationError):
            Settings(port=0)
    
    def test_environment_validation(self):
        """Test environment validation"""
        # Valid environments
        for env in ["development", "staging", "production"]:
            settings = Settings(environment=env)
            assert settings.environment == env
        
        # Invalid environment
        with pytest.raises(ValidationError):
            Settings(environment="invalid")
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string"""
        # Comma-separated string
        settings = Settings(cors_origins="http://localhost:3000,https://example.com")
        assert len(settings.cors_origins) == 2
        assert "http://localhost:3000" in settings.cors_origins
        assert "https://example.com" in settings.cors_origins
        
        # List input
        settings = Settings(cors_origins=["http://localhost:3000"])
        assert len(settings.cors_origins) == 1
    
    def test_trusted_hosts_parsing(self):
        """Test trusted hosts parsing from string"""
        settings = Settings(trusted_hosts="localhost,example.com,192.168.1.1")
        assert len(settings.trusted_hosts) == 3
        assert "localhost" in settings.trusted_hosts
        assert "example.com" in settings.trusted_hosts
    
    def test_static_dir_path_conversion(self):
        """Test static_dir is converted to Path"""
        settings = Settings(static_dir="./static")
        assert isinstance(settings.static_dir, Path)
        assert str(settings.static_dir) == "static"
    
    def test_log_level_validation(self):
        """Test log level validation"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level
        
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")
    
    def test_secret_fields(self):
        """Test that secret fields are properly masked"""
        settings = Settings(
            delta_api_key="test_key_123",
            delta_api_secret="test_secret_456"
        )
        
        # Secrets should be SecretStr objects
        assert str(settings.delta_api_key) != "test_key_123"
        
        # Can retrieve actual value
        assert settings.get_api_key() == "test_key_123"
        assert settings.get_api_secret() == "test_secret_456"
    
    def test_is_production(self):
        """Test production environment detection"""
        settings = Settings(environment="production")
        assert settings.is_production() is True
        assert settings.is_development() is False
        
        settings = Settings(environment="development")
        assert settings.is_production() is False
        assert settings.is_development() is True
    
    def test_validate_credentials_missing(self, tmp_path):
        """Test credential validation with missing values"""
        # Create a temporary static directory
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        
        settings = Settings(
            environment="development",
            delta_api_key="",
            delta_api_secret="",
            static_dir=str(static_dir)
        )
        
        # Should not raise in development
        settings.validate_credentials()
    
    def test_validate_credentials_production_fails(self, tmp_path):
        """Test credential validation fails in production with missing credentials"""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        
        settings = Settings(
            environment="production",
            delta_api_key="",
            delta_api_secret="",
            static_dir=str(static_dir)
        )
        
        # Should raise ValueError in production
        with pytest.raises(ValueError, match="Critical configuration errors"):
            settings.validate_credentials()
    
    def test_validate_credentials_success(self, tmp_path):
        """Test credential validation succeeds with all values set"""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        
        settings = Settings(
            environment="production",
            delta_api_key="valid_key",
            delta_api_secret="valid_secret",
            static_dir=str(static_dir)
        )
        
        # Should not raise
        settings.validate_credentials()
        assert settings.is_valid() is True
    
    def test_is_valid(self, tmp_path):
        """Test is_valid() method"""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        
        # Valid configuration
        settings = Settings(
            environment="development",
            delta_api_key="key",
            delta_api_secret="secret",
            static_dir=str(static_dir)
        )
        assert settings.is_valid() is True
        
        # Invalid configuration (missing static dir)
        settings = Settings(
            environment="development",
            delta_api_key="key",
            delta_api_secret="secret",
            static_dir="/nonexistent/path"
        )
        assert settings.is_valid() is False


class TestConfigEnvironmentVariables:
    """Test loading from environment variables"""
    
    def test_load_from_env(self, monkeypatch):
        """Test loading configuration from environment variables"""
        monkeypatch.setenv("ENVIRONMENT", "staging")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("DELTA_API_KEY", "env_key")
        monkeypatch.setenv("TRADING_SYMBOL", "BTCUSD")
        
        settings = Settings()
        
        assert settings.environment == "staging"
        assert settings.debug is True
        assert settings.port == 9000
        assert settings.get_api_key() == "env_key"
        assert settings.trading_symbol == "BTCUSD"
    
    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case-insensitive"""
        monkeypatch.setenv("debug", "true")
        monkeypatch.setenv("ENVIRONMENT", "development")
        
        settings = Settings()
        
        assert settings.debug is True
        assert settings.environment == "development"
