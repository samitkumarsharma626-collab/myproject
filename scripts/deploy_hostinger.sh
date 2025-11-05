#!/usr/bin/env bash
# Safe deployment script for Hostinger/VPS environments.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/var/www/eth-strategy-api}"
SERVICE_NAME="${SERVICE_NAME:-ethbot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-http://127.0.0.1:8000/health}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Repository directory '${REPO_DIR}' does not exist" >&2
  exit 1
fi

cd "${REPO_DIR}"

echo "[deploy] Fetching latest code..."
git fetch --prune
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "[deploy] Working tree not clean. Commit or stash changes before deploying." >&2
  exit 1
fi

git pull --ff-only

echo "[deploy] Setting up virtual environment..."
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install --requirement requirements.txt
pip install --requirement requirements-dev.txt

if [[ -x "manage.py" ]]; then
  echo "[deploy] Running database migrations..."
  ./manage.py migrate
fi

echo "[deploy] Restarting service ${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl reload-or-restart "${SERVICE_NAME}.service"

sleep 2

echo "[deploy] Running health check: ${HEALTHCHECK_URL}"
if ! curl --fail --silent "${HEALTHCHECK_URL}" > /dev/null; then
  echo "[deploy] Health check failed" >&2
  journalctl -u "${SERVICE_NAME}.service" -n 100 --no-pager
  exit 1
fi

echo "[deploy] Deployment completed successfully."
