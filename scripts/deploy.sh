#!/usr/bin/env bash
set -euo pipefail

# Safe deploy script for VPS/Hostinger
# Usage: ./scripts/deploy.sh /var/www/ethbot /path/to/venv service_name

APP_DIR=${1:-/var/www/ethbot}
VENV_DIR=${2:-/var/www/ethbot/.venv}
SERVICE_NAME=${3:-ethbot}

if [ ! -d "$APP_DIR" ]; then
  echo "App directory not found: $APP_DIR" >&2
  exit 1
fi

cd "$APP_DIR"

# Fetch latest
git fetch --all --prune
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git pull --ff-only origin "$CURRENT_BRANCH"

# Setup venv
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: run migrations if applicable (placeholder)
# e.g., alembic upgrade head

# Restart service
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop "$SERVICE_NAME" || true
  sudo systemctl start "$SERVICE_NAME"
  sudo systemctl status "$SERVICE_NAME" --no-pager || true
fi

# Healthcheck
sleep 2
HEALTH_URL=${HEALTH_URL:-"http://127.0.0.1:8000/health"}
set +e
curl -fsSL "$HEALTH_URL" | jq . || curl -fsSL "$HEALTH_URL"
RC=$?
set -e
exit $RC
