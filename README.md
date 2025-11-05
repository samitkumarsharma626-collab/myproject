# ETH Strategy Dashboard

Production-ready FastAPI backend for real-time ETH strategy monitoring and trading API.

## Features

- 🔒 **Secure Configuration**: Environment-based configuration with validation
- 🛡️ **Security Hardened**: CSP headers, secure cookies, HTTPS enforcement
- 📊 **Structured Logging**: JSON logging with sensitive data redaction
- ✅ **Type Safety**: Full type hints and pydantic validation
- 🧪 **Tested**: Unit and integration tests
- 🚀 **CI/CD Ready**: Automated linting, type-checking, and security scanning

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd eth-strategy-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## Environment Configuration

### How to Set Environment Variables for Production

The application uses environment variables for all configuration. **Never commit `.env` files to version control.**

#### 1. Create `.env` file

Copy the example file:
```bash
cp .env.example .env
```

#### 2. Configure Required Variables

Edit `.env` and set the following **required** variables for production:

```bash
# API Credentials (REQUIRED for production)
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here

# Security Settings (REQUIRED for production)
DEBUG=false
ENVIRONMENT=production

# CORS Configuration (REQUIRED for production)
# Set specific origins, NEVER use "*" in production
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Server Configuration
HOST=0.0.0.0
PORT=8000

# API Configuration
DELTA_BASE_URL=https://api.india.delta.exchange
TRADING_SYMBOL=ETHUSD
CANDLE_RESOLUTION=15m
REQUEST_TIMEOUT=10

# Security Headers
FORCE_HTTPS=true
SECURE_COOKIES=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

#### 3. Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DELTA_API_KEY` | Yes (prod) | `""` | Delta Exchange API key |
| `DELTA_API_SECRET` | Yes (prod) | `""` | Delta Exchange API secret |
| `DEBUG` | No | `false` | Enable debug mode |
| `ENVIRONMENT` | No | `production` | Environment: `production`, `development`, `testing` |
| `CORS_ORIGINS` | Yes (prod) | `""` | Comma-separated allowed origins (no wildcards in prod) |
| `HOST` | No | `0.0.0.0` | Server host |
| `PORT` | No | `8000` | Server port |
| `FORCE_HTTPS` | No | `true` | Force HTTPS redirects |
| `SECURE_COOKIES` | No | `true` | Use secure cookies |

#### 4. Validation

The application validates configuration at startup. In production:
- `DELTA_API_KEY` and `DELTA_API_SECRET` must be set
- `CORS_ORIGINS` must be set and cannot contain `*`
- `DEBUG` must be `false`

Test your configuration:
```bash
python -c "from app.config import settings; settings.validate_required(); print('✓ Configuration valid')"
```

## Deployment

### Hostinger/VPS Deployment

The project includes a deployment script for easy deployment on Hostinger or other VPS providers.

#### Prerequisites

1. Systemd service file (see `config/ethbot.service`)
2. Nginx configuration (see `config/nginx.conf`)
3. `.env` file configured

#### Deployment Steps

1. **Copy service file**
   ```bash
   sudo cp config/ethbot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Set up Nginx** (optional, for reverse proxy)
   ```bash
   sudo cp config/nginx.conf /etc/nginx/sites-available/ethbot
   sudo ln -s /etc/nginx/sites-available/ethbot /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

3. **Run deployment script**
   ```bash
   ./deploy.sh
   ```

The deployment script will:
- Pull latest changes from git
- Install/update dependencies
- Validate configuration
- Stop the service
- Start the service
- Perform health check

#### Manual Deployment

If you prefer manual deployment:

```bash
# Stop service
sudo systemctl stop ethbot.service

# Pull changes
git pull

# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Validate configuration
python -c "from app.config import settings; settings.validate_required()"

# Start service
sudo systemctl start ethbot.service

# Check status
sudo systemctl status ethbot.service
```

#### Service Management

```bash
# Start service
sudo systemctl start ethbot.service

# Stop service
sudo systemctl stop ethbot.service

# Restart service
sudo systemctl restart ethbot.service

# View logs
sudo journalctl -u ethbot.service -f

# View recent logs
sudo journalctl -u ethbot.service -n 50
```

### Health Check Endpoint

The application provides a health check endpoint for monitoring:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "environment": "production",
  "debug": false
}
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy src/

# Security scanning
bandit -r src/ -ll
safety check --file requirements.txt
```

### CI/CD

The project includes GitHub Actions workflow (`.github/workflows/ci.yml`) that runs:
- Linting (ruff, flake8)
- Type checking (mypy)
- Tests (pytest)
- Security scanning (bandit, safety)

## Security Features

### Configuration Security

- ✅ No hard-coded secrets
- ✅ Environment variable validation
- ✅ Fail-fast on missing required config
- ✅ Type-safe configuration with pydantic

### Application Security

- ✅ Content Security Policy (CSP) headers
- ✅ HTTPS enforcement in production
- ✅ Secure cookies (HttpOnly, Secure, SameSite)
- ✅ CORS whitelist (no wildcards in production)
- ✅ XSS protection headers
- ✅ Frame protection (X-Frame-Options)

### Logging Security

- ✅ Structured JSON logging
- ✅ Automatic redaction of sensitive data
- ✅ Stack traces only in development mode
- ✅ No secrets in logs

### Error Handling

- ✅ Centralized error handling middleware
- ✅ Graceful error responses
- ✅ No sensitive data in error messages
- ✅ Proper HTTP status codes

## API Endpoints

- `GET /` - Dashboard HTML
- `GET /health` - Health check endpoint
- `GET /api/v1/price` - Get current ETH price
- `GET /api/v1/balance` - Get wallet balance
- `GET /api/v1/box` - Get price box (high/low)
- `GET /api/strategy/filters` - Get strategy filters
- `GET /api/v1/candles` - Get historical candles

## Project Structure

```
.
├── app/
│   ├── __init__.py          # Main application
│   ├── config.py            # Secure configuration
│   ├── logging_config.py    # Structured logging
│   ├── middleware.py        # Security middleware
│   └── error_handler.py     # Error handling
├── tests/
│   ├── test_config.py       # Config tests
│   ├── test_integration.py  # Integration tests
│   ├── test_error_handling.py
│   └── test_security.py
├── config/
│   ├── ethbot.service       # Systemd service
│   └── nginx.conf           # Nginx config
├── static/
│   └── index.html           # Frontend
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Production dependencies
├── requirements-test.txt    # Test dependencies
├── deploy.sh                # Deployment script
└── README.md                # This file
```

## Troubleshooting

### Configuration Errors

If you see configuration validation errors:

1. Check your `.env` file exists
2. Verify all required variables are set
3. Run validation: `python -c "from app.config import settings; settings.validate_required()"`

### Service Won't Start

1. Check logs: `sudo journalctl -u ethbot.service -n 50`
2. Verify `.env` file is readable
3. Check Python path in service file
4. Verify virtual environment path

### API Authentication Errors

1. Verify `DELTA_API_KEY` and `DELTA_API_SECRET` are set correctly
2. Check API credentials are valid
3. Ensure no extra spaces in `.env` file

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.
