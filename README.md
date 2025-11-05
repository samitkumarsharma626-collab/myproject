## ETH Strategy Dashboard API

FastAPI backend for monitoring ETH trading strategies securely. The service uses validated environment-driven configuration, structured logging, and strict security middleware by default.

### Prerequisites

- Python 3.12+
- Recommended: virtual environment (`python -m venv .venv`)

### Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health` (returns JSON status and build metadata).

### Quality Checks

```bash
ruff check .
mypy .
pytest
bandit -r app
safety check --full-report --file=requirements.txt
```

### How to Set Env for Production

1. Copy `.env.example` to `.env` and edit the values.
2. Set `APP_ENV=production` and keep `DEBUG=false`.
3. Generate and set strong values for `DELTA_API_KEY` and `DELTA_API_SECRET` (do not store them in version control).
4. Configure strict origins and hosts (as JSON arrays):
   - `CORS_ALLOW_ORIGINS` to the exact domain(s) serving the frontend, e.g. `["https://app.example.com"]`.
   - `TRUSTED_HOSTS` to the load balancer or CDN hostnames, e.g. `["app.example.com","lb.example.net"]`.
5. Keep `ENFORCE_HTTPS=true` and adjust `HSTS_MAX_AGE` only if you understand HSTS implications.
6. Deploy the `.env` file securely on the server (e.g., via secret manager) and ensure it is owned by the service user with `0600` permissions.
7. Restart the service after any configuration change to trigger validation.

Refer to `.env.example` for the complete list of supported variables.

### Deployment on Hostinger/VPS

Use the provided script `scripts/deploy_hostinger.sh` to safely update the service. It performs:

- Git fetch & pull from the tracked branch
- Dependency installation with pinned requirements
- Database migrations placeholder (extend as needed)
- Graceful service restart via systemd
- Post-deploy health check against `GET /health`

```bash
bash scripts/deploy_hostinger.sh
```

Ensure the script runs under a non-root user with sudo privileges configured for `systemctl` where required.
