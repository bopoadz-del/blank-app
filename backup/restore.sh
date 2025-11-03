#!/bin/bash

# Restore Script for Formula Execution API
# Restores database, volumes, and configuration from backup

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if backup file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    echo ""
    echo "Available backups:"
    ls -lh ./backup/backups/*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Error: Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Formula API - Restore${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Backup file: $BACKUP_FILE"
echo "Timestamp: $(date)"
echo ""

# Warning
echo -e "${YELLOW}WARNING: This will replace existing data!${NC}"
echo -e "${YELLOW}Current containers will be stopped.${NC}"
echo ""
read -p "Do you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

echo ""

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Step 1: Extract backup
echo -e "${YELLOW}[1/7] Extracting backup...${NC}"
tar xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)
EXTRACT_DIR="$TEMP_DIR/$BACKUP_NAME"

if [ ! -d "$EXTRACT_DIR" ]; then
    echo -e "${RED}✗ Failed to extract backup${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo -e "${GREEN}✓ Backup extracted${NC}"

# Step 2: Display backup info
echo -e "${YELLOW}[2/7] Backup information:${NC}"
cat "$EXTRACT_DIR/metadata.txt"
echo ""

# Step 3: Stop existing containers
echo -e "${YELLOW}[3/7] Stopping containers...${NC}"
docker-compose down
echo -e "${GREEN}✓ Containers stopped${NC}"

# Step 4: Restore database
echo -e "${YELLOW}[4/7] Restoring database...${NC}"
docker-compose up -d db
echo "Waiting for database to be ready..."
sleep 10

if [ -f "$EXTRACT_DIR/database.sql" ]; then
    docker-compose exec -T db psql -U postgres < "$EXTRACT_DIR/database.sql"
    echo -e "${GREEN}✓ Database restored${NC}"
else
    echo -e "${RED}✗ Database backup file not found${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Step 5: Restore Redis
echo -e "${YELLOW}[5/7] Restoring Redis...${NC}"
if [ -f "$EXTRACT_DIR/redis.rdb" ]; then
    docker-compose up -d redis
    sleep 5
    docker cp "$EXTRACT_DIR/redis.rdb" formula-api-redis:/data/dump.rdb
    docker-compose restart redis
    echo -e "${GREEN}✓ Redis restored${NC}"
else
    echo -e "${YELLOW}⚠ No Redis backup found, skipping${NC}"
fi

# Step 6: Restore MLflow
echo -e "${YELLOW}[6/7] Restoring MLflow artifacts...${NC}"
if [ -f "$EXTRACT_DIR/mlflow_data.tar.gz" ]; then
    docker-compose up -d mlflow
    sleep 5
    docker run --rm \
        --volumes-from formula-api-mlflow \
        -v "$EXTRACT_DIR":/backup \
        alpine sh -c "cd / && tar xzf /backup/mlflow_data.tar.gz"
    echo -e "${GREEN}✓ MLflow restored${NC}"
else
    echo -e "${YELLOW}⚠ No MLflow backup found, skipping${NC}"
fi

# Step 7: Restore configuration
echo -e "${YELLOW}[7/7] Restoring configuration...${NC}"
if [ -d "$EXTRACT_DIR/config" ]; then
    if [ -f "$EXTRACT_DIR/config/.env" ]; then
        echo "Found .env file. Do you want to restore it? (yes/no)"
        read -p "> " restore_env
        if [ "$restore_env" = "yes" ]; then
            cp "$EXTRACT_DIR/config/.env" .env
            echo -e "${GREEN}✓ .env restored${NC}"
        else
            echo -e "${YELLOW}⚠ .env not restored${NC}"
        fi
    fi
fi

echo ""

# Start all services
echo -e "${YELLOW}Starting all services...${NC}"
docker-compose up -d
echo "Waiting for services to be ready..."
sleep 15

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Restore Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Verify restoration
echo -e "${YELLOW}Verifying restoration...${NC}"
if docker-compose exec -T db psql -U postgres -d formulas -c "SELECT COUNT(*) FROM formula_executions;" > /dev/null 2>&1; then
    record_count=$(docker-compose exec -T db psql -U postgres -d formulas -t -c "SELECT COUNT(*) FROM formula_executions;" | tr -d ' ')
    echo -e "${GREEN}✓ Database accessible (${record_count} execution records)${NC}"
else
    echo -e "${RED}✗ Database verification failed${NC}"
fi

if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis accessible${NC}"
else
    echo -e "${RED}✗ Redis verification failed${NC}"
fi

# Cleanup
rm -rf "$TEMP_DIR"
echo ""
echo "Restoration complete!"
echo "Run verification: bash scripts/verify-deployment.sh"
