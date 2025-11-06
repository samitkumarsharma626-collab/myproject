#!/bin/bash
set -e

# Deployment script for Hostinger VPS
# This script runs inside /root/myproject on the VPS

echo "=========================================="
echo "ETH Strategy Dashboard - Hostinger Deployment"
echo "=========================================="

# Navigate to project directory
cd /root/myproject

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    print_info "Please ensure .env file exists in /root/myproject"
    exit 1
fi

# Check if service is running
SERVICE_NAME="ethbot"
if systemctl is-active --quiet "$SERVICE_NAME.service"; then
    print_info "Stopping $SERVICE_NAME service..."
    systemctl stop "$SERVICE_NAME.service"
    print_success "Service stopped"
fi

# Git pull
print_info "Pulling latest changes from git..."
if git pull; then
    print_success "Git pull successful"
else
    print_error "Git pull failed!"
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_PATH="${VENV_PATH:-venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$VENV_PATH" ]; then
    print_info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_PATH"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
print_info "Installing dependencies..."
if pip install -r requirements.txt --quiet; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies!"
    exit 1
fi

# Validate configuration
print_info "Validating configuration..."
if python -c "from src.config import settings; settings.validate_required(); print('Configuration valid')" 2>/dev/null; then
    print_success "Configuration validated"
else
    print_error "Configuration validation failed!"
    print_info "Please check your .env file"
    exit 1
fi

# Start service
print_info "Starting $SERVICE_NAME service..."
if systemctl start "$SERVICE_NAME.service"; then
    print_success "Service started"
else
    print_error "Failed to start service!"
    print_info "Check logs with: journalctl -u $SERVICE_NAME.service -n 50"
    exit 1
fi

# Wait a moment for service to start
sleep 2

# Check service status
if systemctl is-active --quiet "$SERVICE_NAME.service"; then
    print_success "Service is running"
else
    print_error "Service failed to start!"
    print_info "Check logs with: journalctl -u $SERVICE_NAME.service -n 50"
    exit 1
fi

# Health check
print_info "Performing health check..."
if command -v curl &> /dev/null; then
    HEALTH_URL="http://localhost:8000/health"
    if curl -f -s "$HEALTH_URL" > /dev/null; then
        print_success "Health check passed"
    else
        print_error "Health check failed!"
        print_info "Check logs with: journalctl -u $SERVICE_NAME.service -n 50"
        exit 1
    fi
else
    print_info "curl not available, skipping health check"
fi

echo ""
echo "=========================================="
print_success "✅ Deployed Successfully!"
echo "=========================================="
