#!/bin/bash
# Test script for unified deployment
# This script verifies the deployment works correctly

set -e  # Exit on error

echo "=================================="
echo "Unified Deployment Test Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check files exist
echo "Step 1: Checking required files..."
files=("Dockerfile" ".dockerignore" "backend/app/main.py" "frontend/package.json")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
        exit 1
    fi
done
echo ""

# Step 2: Build frontend
echo "Step 2: Building frontend..."
cd frontend
npm ci --silent 2>&1 > /dev/null || echo "npm install had warnings"
npm run build
if [ -f "dist/index.html" ]; then
    echo -e "${GREEN}✓${NC} Frontend built successfully"
else
    echo -e "${RED}✗${NC} Frontend build failed"
    exit 1
fi
cd ..
echo ""

# Step 3: Copy frontend to backend
echo "Step 3: Copying frontend to backend..."
mkdir -p backend/frontend
cp -r frontend/dist backend/frontend/dist
if [ -f "backend/frontend/dist/index.html" ]; then
    echo -e "${GREEN}✓${NC} Frontend copied to backend/frontend/dist"
else
    echo -e "${RED}✗${NC} Frontend copy failed"
    exit 1
fi
echo ""

# Step 4: Verify backend code
echo "Step 4: Verifying backend code..."
if grep -q "StaticFiles" backend/app/main.py && grep -q "FRONTEND_DIST" backend/app/main.py; then
    echo -e "${GREEN}✓${NC} Backend has frontend mounting code"
else
    echo -e "${RED}✗${NC} Backend missing frontend mounting code"
    exit 1
fi
echo ""

# Step 5: Run tests
echo "Step 5: Running tests..."
cd backend
python -m pytest tests/test_frontend_mounting.py -v --tb=short
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All tests passed"
else
    echo -e "${RED}✗${NC} Tests failed"
    exit 1
fi
cd ..
echo ""

# Step 6: Verify Dockerfile structure
echo "Step 6: Verifying Dockerfile..."
required_strings=("FROM node:18-alpine AS frontend-builder" "FROM python:3.11-slim AS backend-builder" "npm run build" "python -m venv" "COPY --from=frontend-builder" "frontend/dist")
for str in "${required_strings[@]}"; do
    if grep -q "$str" Dockerfile; then
        echo -e "${GREEN}✓${NC} Found: $str"
    else
        echo -e "${RED}✗${NC} Missing: $str"
        exit 1
    fi
done
echo ""

echo "=================================="
echo -e "${GREEN}All checks passed!${NC}"
echo "=================================="
echo ""
echo "The unified deployment is ready!"
echo ""
echo "To deploy on Render:"
echo "1. Push this code to your repository"
echo "2. Create a Web Service on Render"
echo "3. Point it to this repository"
echo "4. Render will automatically use the Dockerfile"
echo "5. Set environment variables (DATABASE_URL, SECRET_KEY, etc.)"
echo ""
echo "To test locally with Docker:"
echo "  docker build -t blank-unified ."
echo "  docker run -p 8000:8000 -e PORT=8000 blank-unified"
echo "  curl http://localhost:8000/"
echo "  curl http://localhost:8000/health"
echo ""
