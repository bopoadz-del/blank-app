#!/bin/bash
# Production Deployment Script for The Reasoner AI Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Reasoner AI Platform Deployment${NC}"
echo -e "${GREEN}Environment: $ENVIRONMENT${NC}"
echo -e "${GREEN}=====================================${NC}"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required but not installed.${NC}" >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}Docker Compose is required but not installed.${NC}" >&2; exit 1; }

echo -e "${GREEN}✓ Prerequisites check passed${NC}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
echo -e "\n${YELLOW}Backing up database...${NC}"
docker-compose exec -T db pg_dump -U postgres formulas > "$BACKUP_DIR/database_backup.sql" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Could not backup database (container might not be running)${NC}"
}

# Backup environment files
cp .env "$BACKUP_DIR/.env.backup" 2>/dev/null || true

echo -e "${GREEN}✓ Backup completed: $BACKUP_DIR${NC}"

# Pull latest changes
echo -e "\n${YELLOW}Pulling latest changes...${NC}"
git pull origin main

# Build new images
echo -e "\n${YELLOW}Building Docker images...${NC}"
docker-compose build --no-cache --parallel

# Stop old containers
echo -e "\n${YELLOW}Stopping old containers...${NC}"
docker-compose down

# Start new containers
echo -e "\n${YELLOW}Starting new containers...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo -e "\n${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check service health
echo -e "\n${YELLOW}Checking service health...${NC}"
BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
FRONTEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)

if [ "$BACKEND_HEALTH" = "200" ] && [ "$FRONTEND_HEALTH" = "200" ]; then
    echo -e "${GREEN}✓ All services are healthy${NC}"
else
    echo -e "${RED}✗ Service health check failed${NC}"
    echo -e "${RED}Backend status: $BACKEND_HEALTH${NC}"
    echo -e "${RED}Frontend status: $FRONTEND_HEALTH${NC}"

    # Rollback on failure
    echo -e "\n${YELLOW}Rolling back to previous version...${NC}"
    docker-compose down
    git reset --hard HEAD~1
    docker-compose up -d
    exit 1
fi

# Run database migrations
echo -e "\n${YELLOW}Running database migrations...${NC}"
docker-compose exec -T backend alembic upgrade head

# Clean up old Docker images
echo -e "\n${YELLOW}Cleaning up old Docker images...${NC}"
docker image prune -f

# Display running containers
echo -e "\n${YELLOW}Running containers:${NC}"
docker-compose ps

# Show logs
echo -e "\n${YELLOW}Recent logs:${NC}"
docker-compose logs --tail=20

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "\nApplication URLs:"
echo -e "  Backend:  ${GREEN}http://localhost:8000${NC}"
echo -e "  Frontend: ${GREEN}http://localhost:8080${NC}"
echo -e "  Grafana:  ${GREEN}http://localhost:3000${NC}"
echo -e "  Prometheus: ${GREEN}http://localhost:9090${NC}"
echo -e "\nBackup location: ${GREEN}$BACKUP_DIR${NC}"
