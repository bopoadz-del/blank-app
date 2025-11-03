#!/bin/bash

# Deployment Script for Staging VPS
# This script deploys the Formula Execution API to a staging environment

set -e

# Configuration
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-staging.example.com}"
REMOTE_DIR="/opt/formula-api"
APP_NAME="formula-api"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Formula API - Staging Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Pre-deployment checks
echo -e "${YELLOW}Step 1: Pre-deployment checks${NC}"
echo "Checking local Docker setup..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Local environment OK${NC}"
echo ""

# Step 2: Run tests locally before deployment
echo -e "${YELLOW}Step 2: Running tests before deployment${NC}"
echo "Running unit tests..."
if pytest tests/ -v --tb=short; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed. Aborting deployment.${NC}"
    exit 1
fi
echo ""

# Step 3: Build Docker images
echo -e "${YELLOW}Step 3: Building Docker images${NC}"
docker-compose build
echo -e "${GREEN}✓ Docker images built${NC}"
echo ""

# Step 4: Check SSH connection
echo -e "${YELLOW}Step 4: Checking SSH connection to staging server${NC}"
if ssh -o ConnectTimeout=10 -o BatchMode=yes ${REMOTE_USER}@${REMOTE_HOST} exit 2>/dev/null; then
    echo -e "${GREEN}✓ SSH connection successful${NC}"
else
    echo -e "${RED}✗ Cannot connect to ${REMOTE_HOST}${NC}"
    echo "Please configure SSH access or set REMOTE_HOST environment variable"
    exit 1
fi
echo ""

# Step 5: Create remote directory structure
echo -e "${YELLOW}Step 5: Setting up remote directory structure${NC}"
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}/{app,backup,logs}"
echo -e "${GREEN}✓ Remote directories created${NC}"
echo ""

# Step 6: Copy files to staging server
echo -e "${YELLOW}Step 6: Copying files to staging server${NC}"
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'venv' \
    --exclude '.pytest_cache' \
    --exclude 'htmlcov' \
    --exclude 'test.db' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Step 7: Copy environment configuration
echo -e "${YELLOW}Step 7: Setting up environment configuration${NC}"
if [ -f .env.staging ]; then
    scp .env.staging ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env
    echo -e "${GREEN}✓ Staging environment file copied${NC}"
else
    echo -e "${YELLOW}⚠ No .env.staging file found, using .env.example${NC}"
    scp .env.example ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env
    echo -e "${YELLOW}⚠ Please update ${REMOTE_DIR}/.env with production values${NC}"
fi
echo ""

# Step 8: Deploy with Docker Compose
echo -e "${YELLOW}Step 8: Deploying with Docker Compose${NC}"
ssh ${REMOTE_USER}@${REMOTE_HOST} << 'ENDSSH'
cd /opt/formula-api
docker-compose down
docker-compose pull || true
docker-compose up -d
ENDSSH
echo -e "${GREEN}✓ Services deployed${NC}"
echo ""

# Step 9: Wait for services to be ready
echo -e "${YELLOW}Step 9: Waiting for services to be ready${NC}"
echo "Waiting 30 seconds for services to start..."
sleep 30
echo -e "${GREEN}✓ Services should be ready${NC}"
echo ""

# Step 10: Run verification script
echo -e "${YELLOW}Step 10: Running deployment verification${NC}"
if ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && bash scripts/verify-deployment.sh"; then
    echo -e "${GREEN}✓ Deployment verification passed${NC}"
else
    echo -e "${RED}✗ Deployment verification failed${NC}"
    echo "Check logs with: ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && docker-compose logs'"
    exit 1
fi
echo ""

# Step 11: Display deployment information
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "API URL: http://${REMOTE_HOST}:8000"
echo "API Docs: http://${REMOTE_HOST}:8000/docs"
echo "MLflow UI: http://${REMOTE_HOST}:5000"
echo ""
echo "Useful commands:"
echo "  View logs: ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && docker-compose logs -f backend'"
echo "  Check status: ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && docker-compose ps'"
echo "  Restart: ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && docker-compose restart'"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run verify-deployment.sh to verify the deployment"
echo "2. Monitor logs for 24 hours"
echo "3. Test backup/restore procedures"
echo ""
