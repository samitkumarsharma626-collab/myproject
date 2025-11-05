# ETH Strategy Dashboard API

A FastAPI backend for monitoring ETH strategy signals with secure, validated configuration.

### How to set env for production

1. Copy the example env and edit values (no secrets committed):

```bash
cp .env.example .env
```

2. Set required secrets via environment (recommended) or in `.env` (not committed):
- `DELTA_API_KEY`
- `DELTA_API_SECRET`

3. Configure security and hosts:
- `APP_ENV=production`
- `DEBUG=false`
- `CORS_ORIGINS=https://yourdomain.com`
- `ALLOWED_HOSTS=yourdomain.com,localhost,127.0.0.1`

4. Start server (example):
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

5. Healthcheck:
```bash
curl -fsSL http://127.0.0.1:8000/health
```

### Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest && ruff check . && mypy .
```

### Deployment on Hostinger/VPS (systemd)

- Install dependencies and set up service file similar to `config/ethbot.service`.
- Use the helper script:

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh /var/www/ethbot /var/www/ethbot/.venv ethbot
```

The script performs a safe `git pull`, installs deps, restarts the service, and runs a healthcheck at `/health`.
