#!/bin/bash
# Unified Deployment Validation Script
# This script validates that the unified deployment is correctly configured

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  Unified Deployment Validation"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        return 1
    fi
}

check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1/"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1/"
        return 1
    fi
}

ERRORS=0

echo "1. Checking required files..."
echo "-----------------------------------"
check_file "Dockerfile.unified" || ERRORS=$((ERRORS+1))
check_file ".dockerignore" || ERRORS=$((ERRORS+1))
check_file "backend/app/main.py" || ERRORS=$((ERRORS+1))
check_file "frontend/package.json" || ERRORS=$((ERRORS+1))
check_file "backend/requirements.txt" || ERRORS=$((ERRORS+1))
check_file "UNIFIED_DEPLOYMENT.md" || ERRORS=$((ERRORS+1))
echo ""

echo "2. Checking directory structure..."
echo "-----------------------------------"
check_directory "backend/app" || ERRORS=$((ERRORS+1))
check_directory "frontend/src" || ERRORS=$((ERRORS+1))
echo ""

echo "3. Validating backend/app/main.py..."
echo "-----------------------------------"
if grep -q "StaticFiles" backend/app/main.py; then
    echo -e "${GREEN}✓${NC} StaticFiles import found"
else
    echo -e "${RED}✗${NC} StaticFiles import missing"
    ERRORS=$((ERRORS+1))
fi

if grep -q "FRONTEND_DIST" backend/app/main.py; then
    echo -e "${GREEN}✓${NC} FRONTEND_DIST configuration found"
else
    echo -e "${RED}✗${NC} FRONTEND_DIST configuration missing"
    ERRORS=$((ERRORS+1))
fi

if grep -q "app.mount" backend/app/main.py; then
    echo -e "${GREEN}✓${NC} Frontend mount code found"
else
    echo -e "${RED}✗${NC} Frontend mount code missing"
    ERRORS=$((ERRORS+1))
fi
echo ""

echo "4. Validating Dockerfile.unified..."
echo "-----------------------------------"
if grep -q "FROM node:18-alpine AS frontend-builder" Dockerfile.unified; then
    echo -e "${GREEN}✓${NC} Frontend build stage defined"
else
    echo -e "${RED}✗${NC} Frontend build stage missing"
    ERRORS=$((ERRORS+1))
fi

if grep -q "FROM python:3.11-slim AS backend-builder" Dockerfile.unified; then
    echo -e "${GREEN}✓${NC} Backend build stage defined"
else
    echo -e "${RED}✗${NC} Backend build stage missing"
    ERRORS=$((ERRORS+1))
fi

if grep -q "npm run build" Dockerfile.unified; then
    echo -e "${GREEN}✓${NC} Frontend build command found"
else
    echo -e "${RED}✗${NC} Frontend build command missing"
    ERRORS=$((ERRORS+1))
fi

if grep -q "\${PORT:-8000}" Dockerfile.unified; then
    echo -e "${GREEN}✓${NC} PORT environment variable support found"
else
    echo -e "${RED}✗${NC} PORT environment variable support missing"
    ERRORS=$((ERRORS+1))
fi

if grep -q "COPY --from=frontend-builder /app/frontend/dist ./backend/frontend/dist" Dockerfile.unified; then
    echo -e "${GREEN}✓${NC} Frontend dist copy command found"
else
    echo -e "${RED}✗${NC} Frontend dist copy command missing"
    ERRORS=$((ERRORS+1))
fi
echo ""

echo "5. Checking .dockerignore..."
echo "-----------------------------------"
if grep -q "node_modules" .dockerignore; then
    echo -e "${GREEN}✓${NC} node_modules excluded"
else
    echo -e "${YELLOW}⚠${NC} node_modules not explicitly excluded"
fi

if grep -q "__pycache__" .dockerignore; then
    echo -e "${GREEN}✓${NC} __pycache__ excluded"
else
    echo -e "${YELLOW}⚠${NC} __pycache__ not explicitly excluded"
fi

if grep -q "dist" .dockerignore; then
    echo -e "${GREEN}✓${NC} dist excluded from build context"
else
    echo -e "${YELLOW}⚠${NC} dist not excluded from build context"
fi
echo ""

echo "6. Optional: Testing frontend build..."
echo "-----------------------------------"
if [ -d "frontend/node_modules" ]; then
    echo -e "${GREEN}✓${NC} Frontend dependencies installed"
    if command -v npm &> /dev/null; then
        echo -e "${YELLOW}ℹ${NC} npm found, you can test build with: cd frontend && npm run build"
    fi
else
    echo -e "${YELLOW}ℹ${NC} Frontend dependencies not installed. Run: cd frontend && npm install"
fi
echo ""

echo "7. Optional: Testing Docker availability..."
echo "-----------------------------------"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is available"
    DOCKER_VERSION=$(docker --version)
    echo "  $DOCKER_VERSION"
    echo -e "${YELLOW}ℹ${NC} You can build the image with:"
    echo "  docker build -f Dockerfile.unified -t blank-app:unified ."
else
    echo -e "${YELLOW}ℹ${NC} Docker not found. Install Docker to build the unified image."
fi
echo ""

echo "================================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Validation Passed!${NC}"
    echo ""
    echo "Your unified deployment is correctly configured."
    echo ""
    echo "Next steps:"
    echo "  1. Test locally: docker build -f Dockerfile.unified -t blank-app:unified ."
    echo "  2. Run locally: docker run -p 8000:8000 blank-app:unified"
    echo "  3. Deploy to Render using Dockerfile.unified"
    echo ""
    echo "See UNIFIED_DEPLOYMENT.md for detailed instructions."
    exit 0
else
    echo -e "${RED}✗ Validation Failed${NC}"
    echo ""
    echo "Found $ERRORS error(s). Please review the output above."
    echo "See UNIFIED_DEPLOYMENT.md for configuration details."
    exit 1
fi
