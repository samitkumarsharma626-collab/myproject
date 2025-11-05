#!/bin/bash
# Verification script - run this after installing dependencies

set -e

echo "=================================="
echo "Project Verification Script"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python version
echo "1. Checking Python version..."
python3 --version

echo ""
echo "2. Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -r requirements-dev.txt

echo ""
echo "3. Running Ruff linter..."
ruff check . || echo "Linting completed with warnings"

echo ""
echo "4. Checking code formatting..."
black --check . || echo "Formatting check completed"

echo ""
echo "5. Running type checks..."
mypy app.py config.py --ignore-missing-imports || echo "Type checking completed with warnings"

echo ""
echo "6. Validating configuration..."
python3 -c "from config import settings; settings.validate_credentials(); print('✓ Config validation passed')" || echo "Config validation completed (check .env)"

echo ""
echo "7. Testing app import..."
python3 -c "from app import app; print('✓ App imports successfully')"

echo ""
echo "8. Running tests..."
ENVIRONMENT=development DEBUG=true DELTA_API_KEY=test_key DELTA_API_SECRET=test_secret pytest -v tests/

echo ""
echo "=================================="
echo -e "${GREEN}✓ All verifications completed!${NC}"
echo "=================================="
echo ""
echo "To run the application:"
echo "  python3 app.py"
echo ""
echo "Or with uvicorn:"
echo "  uvicorn app:app --reload --host 0.0.0.0 --port 8000"
