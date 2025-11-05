#!/bin/bash
set -e

# ============================================================================
# Safe Deployment Script for ETH Strategy Dashboard
# ============================================================================
# This script safely deploys updates to the production server
# Usage: ./scripts/deploy.sh
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/var/www/ethbot"
SERVICE_NAME="ethbot"
VENV_DIR="$APP_DIR/venv"
BACKUP_DIR="$APP_DIR/backups"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check if running as correct user (not root)
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
    
    # Check if app directory exists
    if [ ! -d "$APP_DIR" ]; then
        log_error "Application directory not found: $APP_DIR"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found: $VENV_DIR"
        exit 1
    fi
    
    # Check if .env exists
    if [ ! -f "$APP_DIR/.env" ]; then
        log_error ".env file not found. Please create it from .env.example"
        exit 1
    fi
    
    log_info "✓ All requirements met"
}

create_backup() {
    log_info "Creating backup..."
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Create timestamped backup
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"
    
    cd "$APP_DIR"
    tar -czf "$BACKUP_FILE" \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='backups' \
        . 2>/dev/null || true
    
    log_info "✓ Backup created: $BACKUP_FILE"
    
    # Keep only last 5 backups
    cd "$BACKUP_DIR"
    ls -t backup_*.tar.gz | tail -n +6 | xargs -r rm --
}

pull_changes() {
    log_info "Pulling latest changes from git..."
    
    cd "$APP_DIR"
    
    # Stash any local changes (shouldn't be any in production)
    if ! git diff-index --quiet HEAD --; then
        log_warn "Local changes detected, stashing..."
        git stash
    fi
    
    # Get current branch
    CURRENT_BRANCH=$(git branch --show-current)
    log_info "Current branch: $CURRENT_BRANCH"
    
    # Pull changes
    git pull origin "$CURRENT_BRANCH"
    
    log_info "✓ Git pull completed"
}

install_dependencies() {
    log_info "Installing/updating dependencies..."
    
    cd "$APP_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip -q
    
    # Install requirements
    pip install -r requirements.txt -q
    
    log_info "✓ Dependencies installed"
}

run_migrations() {
    log_info "Running migrations (if any)..."
    
    cd "$APP_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Add migration commands here if needed
    # Example: alembic upgrade head
    
    log_info "✓ Migrations completed"
}

validate_config() {
    log_info "Validating configuration..."
    
    cd "$APP_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Check if config is valid
    python3 -c "from config import settings; settings.validate_credentials(); print('Config valid')" || {
        log_error "Configuration validation failed!"
        exit 1
    }
    
    log_info "✓ Configuration valid"
}

restart_service() {
    log_info "Restarting service..."
    
    # Stop service
    sudo systemctl stop "$SERVICE_NAME" || log_warn "Service was not running"
    
    # Wait a moment
    sleep 2
    
    # Start service
    sudo systemctl start "$SERVICE_NAME"
    
    # Check if service started successfully
    sleep 3
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "✓ Service restarted successfully"
    else
        log_error "Service failed to start!"
        sudo journalctl -u "$SERVICE_NAME" -n 20
        exit 1
    fi
}

health_check() {
    log_info "Performing health check..."
    
    # Wait for service to be ready
    sleep 5
    
    # Check health endpoint
    HEALTH_URL="http://localhost:8000/health"
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || echo "000")
    
    if [ "$RESPONSE" = "200" ]; then
        log_info "✓ Health check passed (HTTP $RESPONSE)"
    else
        log_error "Health check failed (HTTP $RESPONSE)"
        log_error "Check logs: sudo journalctl -u $SERVICE_NAME -n 50"
        exit 1
    fi
}

show_status() {
    log_info "Current service status:"
    sudo systemctl status "$SERVICE_NAME" --no-pager -l
}

# ============================================================================
# Main Deployment Flow
# ============================================================================

main() {
    log_info "Starting deployment..."
    echo ""
    
    # Step 1: Check requirements
    check_requirements
    echo ""
    
    # Step 2: Create backup
    create_backup
    echo ""
    
    # Step 3: Pull changes
    pull_changes
    echo ""
    
    # Step 4: Install dependencies
    install_dependencies
    echo ""
    
    # Step 5: Run migrations
    run_migrations
    echo ""
    
    # Step 6: Validate configuration
    validate_config
    echo ""
    
    # Step 7: Restart service
    restart_service
    echo ""
    
    # Step 8: Health check
    health_check
    echo ""
    
    # Step 9: Show status
    show_status
    echo ""
    
    log_info "========================================="
    log_info "Deployment completed successfully! 🚀"
    log_info "========================================="
}

# Run main function
main
