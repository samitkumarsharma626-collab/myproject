# ETH Strategy Dashboard

A production-ready, secure FastAPI application for real-time ETH strategy monitoring and trading with Delta Exchange.

## Features

- 🔒 **Secure Configuration**: Environment-based configuration with Pydantic validation
- 🛡️ **Security Headers**: CSP, HSTS, X-Frame-Options, and more
- 📊 **Real-time Data**: Live price feeds, strategy filters, and technical indicators
- 🔐 **API Authentication**: HMAC-based authentication for Delta Exchange API
- 📝 **Structured Logging**: JSON logging with sensitive data redaction
- ✅ **Input Validation**: Type-safe configuration with fail-fast validation
- 🧪 **Comprehensive Tests**: Unit and integration tests with 80%+ coverage
- 🚀 **CI/CD Ready**: GitHub Actions pipeline with linting, type-checking, and security scans

## Project Structure

```
.
├── app.py                  # Main FastAPI application
├── config.py               # Secure configuration management with Pydantic
├── requirements.txt        # Pinned production dependencies
├── requirements-dev.txt    # Development and testing dependencies
├── .env.example            # Environment variables template (NO SECRETS)
├── .gitignore              # Git ignore rules for sensitive files
├── pyproject.toml          # Tool configurations (ruff, black, mypy, pytest)
├── static/                 # Frontend files
│   └── index.html
├── tests/                  # Test suite
│   ├── test_config.py      # Configuration tests
│   └── test_app.py         # Application integration tests
├── config/                 # Server configurations
│   ├── nginx.conf          # Nginx reverse proxy config
│   └── ethbot.service      # Systemd service file
├── scripts/                # Deployment and utility scripts
│   └── deploy.sh           # Safe deployment script
└── .github/
    └── workflows/
        └── ci.yml          # CI/CD pipeline
```

## Prerequisites

- Python 3.11+
- pip
- virtualenv (recommended)
- Delta Exchange API credentials (for trading features)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (for testing/linting)
pip install -r requirements-dev.txt
```

### 4. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```bash
nano .env  # or use your preferred editor
```

**IMPORTANT**: Never commit `.env` to version control!

## Environment Configuration

### Required Settings for Production

Create a `.env` file with the following variables:

```bash
# Application Environment
ENVIRONMENT=production
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000

# Delta Exchange API Credentials (REQUIRED for trading)
DELTA_API_KEY=your_actual_api_key_here
DELTA_API_SECRET=your_actual_api_secret_here
DELTA_BASE_URL=https://api.india.delta.exchange

# Trading Configuration
TRADING_SYMBOL=ETHUSD
CANDLE_RESOLUTION=15m
REQUEST_TIMEOUT=10

# Security Settings
# IMPORTANT: Replace with your actual domain(s) in production
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_ALLOW_CREDENTIALS=true
TRUSTED_HOSTS=yourdomain.com,www.yourdomain.com

# Paths
STATIC_DIR=./static

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Development Settings

For local development:

```bash
ENVIRONMENT=development
DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1
LOG_FORMAT=text
```

### Configuration Validation

The application validates all configuration on startup and will:
- ✅ **Warn** in development mode if credentials are missing
- ❌ **Fail fast** in production mode if critical settings are missing

## Running the Application

### Development Mode

```bash
# With auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python app.py
```

### Production Mode

```bash
# Using Gunicorn with Uvicorn workers (recommended)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000

# Or using systemd service (see deployment section)
sudo systemctl start ethbot
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=term --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_config.py

# Integration tests only
pytest tests/test_app.py
```

## Code Quality

### Linting

```bash
# Check code with Ruff
ruff check .

# Auto-fix issues
ruff check . --fix
```

### Formatting

```bash
# Check formatting
black --check .

# Apply formatting
black .
```

### Type Checking

```bash
mypy app.py config.py --ignore-missing-imports
```

### Security Scanning

```bash
# Scan for common security issues
bandit -r . -f json

# Check for known vulnerabilities in dependencies
safety check
```

### Run All Checks (CI Pipeline)

```bash
ruff check . && \
black --check . && \
mypy app.py config.py --ignore-missing-imports && \
bandit -r . && \
pytest --cov=. && \
echo "✓ All checks passed!"
```

## API Endpoints

### Public Endpoints

- `GET /` - Main dashboard
- `GET /health` - Health check (includes config validation status)

### Trading Endpoints

- `GET /api/v1/price` - Current ETH price
- `GET /api/v1/balance` - Wallet balance (requires API credentials)
- `GET /api/v1/box` - High/low price box
- `GET /api/strategy/filters` - Strategy filters with technical indicators
- `GET /api/v1/candles?limit=100` - Historical candle data

### Legacy Endpoints (backward compatibility)

- `GET /price`
- `GET /balance`
- `GET /box`
- `GET /strategy-filters`

### API Documentation

When running in development mode:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

**Note**: API docs are disabled in production for security.

## Deployment to VPS/Hostinger

### Prerequisites

- Ubuntu/Debian VPS with sudo access
- Domain name pointed to your server IP
- SSH access to the server

### 1. Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip nginx

# Install and configure firewall
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable
```

### 2. Setup Application

```bash
# Create application directory
sudo mkdir -p /var/www/ethbot
sudo chown $USER:$USER /var/www/ethbot

# Clone repository
cd /var/www/ethbot
git clone <your-repo-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy and edit .env file
cp .env.example .env
nano .env

# Set production values
# CRITICAL: Set ENVIRONMENT=production and DEBUG=false
```

### 4. Setup Systemd Service

```bash
# Copy service file
sudo cp config/ethbot.service /etc/systemd/system/

# Edit service file with correct paths
sudo nano /etc/systemd/system/ethbot.service

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ethbot
sudo systemctl start ethbot

# Check status
sudo systemctl status ethbot
```

### 5. Configure Nginx

```bash
# Copy nginx config
sudo cp config/nginx.conf /etc/nginx/sites-available/ethbot

# Update server_name with your domain
sudo nano /etc/nginx/sites-available/ethbot

# Enable site
sudo ln -s /etc/nginx/sites-available/ethbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Setup SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal is configured automatically
```

### 7. Deploy Script

For subsequent deployments, use the provided deploy script:

```bash
# Make script executable
chmod +x scripts/deploy.sh

# Run deployment
./scripts/deploy.sh
```

The deploy script will:
1. Pull latest code from git
2. Install/update dependencies
3. Run database migrations (if any)
4. Restart the service
5. Check service health

## Monitoring and Logs

### View Application Logs

```bash
# Follow logs in real-time
sudo journalctl -u ethbot -f

# View recent logs
sudo journalctl -u ethbot -n 100

# View logs from specific time
sudo journalctl -u ethbot --since "1 hour ago"
```

### Health Check

```bash
# Check if service is running
systemctl status ethbot

# Check application health endpoint
curl http://localhost:8000/health

# Check from internet
curl https://yourdomain.com/health
```

### Monitoring Recommendations

- Setup log aggregation (e.g., ELK stack, Datadog)
- Configure alerts for 5xx errors
- Monitor API response times
- Track memory and CPU usage
- Setup uptime monitoring (e.g., UptimeRobot)

## Security Best Practices

### ✅ Implemented

- [x] Environment-based configuration (no hardcoded secrets)
- [x] Pydantic validation with fail-fast in production
- [x] .gitignore excludes .env and sensitive files
- [x] Type hints and input validation on all endpoints
- [x] Safe defaults (DEBUG=false, ENVIRONMENT=production)
- [x] Security headers (CSP, HSTS, X-Frame-Options, etc.)
- [x] CORS whitelist (no wildcards in production)
- [x] Structured logging with secret redaction
- [x] Graceful error handling with appropriate HTTP status codes
- [x] Pinned dependencies with version control
- [x] Comprehensive test coverage
- [x] CI pipeline with linting, type-checking, and security scans

### 🔐 Additional Recommendations

1. **Rate Limiting**: Consider adding rate limiting middleware
2. **API Keys**: Implement API key authentication for endpoints
3. **Database Encryption**: Encrypt sensitive data at rest
4. **Regular Updates**: Keep dependencies updated (use Dependabot)
5. **Secrets Management**: Use a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault)
6. **Backup Strategy**: Regular backups of configuration and data
7. **DDoS Protection**: Use Cloudflare or similar CDN
8. **Monitoring**: Setup 24/7 monitoring and alerting

## Troubleshooting

### Configuration Errors

```bash
# Validate configuration
python -c "from config import settings; settings.validate_credentials(); print('✓ Config valid')"
```

### Service Won't Start

```bash
# Check service logs
sudo journalctl -u ethbot -n 50

# Check if port is already in use
sudo lsof -i :8000

# Test application manually
cd /var/www/ethbot
source venv/bin/activate
python app.py
```

### API Connection Issues

```bash
# Test external API connection
curl https://api.india.delta.exchange/v2/tickers/ETHUSD

# Check firewall rules
sudo ufw status

# Test local endpoint
curl http://localhost:8000/health
```

## Development

### Project Setup for Contributors

```bash
# Clone and setup
git clone <repo-url>
cd <project>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Setup pre-commit hooks (optional)
pre-commit install

# Run tests before committing
pytest && ruff check . && black --check .
```

### Making Changes

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Commit with clear messages
5. Push and create a pull request

## Verification Command

Verify everything is working:

```bash
pytest && ruff check . && python -c "from config import settings; print('Config valid:', settings.is_valid())"
```

## License

[Your License Here]

## Support

For issues and questions:
- GitHub Issues: [your-repo]/issues
- Email: your-email@example.com

## Changelog

### v1.0.0 (2025-11-05)
- Initial secure, production-ready release
- Pydantic-based configuration management
- Comprehensive test coverage
- CI/CD pipeline
- Security hardening
- Deployment automation
