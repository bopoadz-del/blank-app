#!/bin/bash

# Backup Script for Formula Execution API
# Creates backups of database, volumes, and configuration

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-./backup/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="formula-api-backup-${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Formula API - Backup${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Timestamp: $(date)"
echo "Backup name: $BACKUP_NAME"
echo "Backup directory: $BACKUP_DIR"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Step 1: Backup PostgreSQL database
echo -e "${YELLOW}[1/6] Backing up PostgreSQL database...${NC}"
docker-compose exec -T db pg_dumpall -U postgres > "$BACKUP_DIR/$BACKUP_NAME/database.sql"
if [ $? -eq 0 ]; then
    db_size=$(du -h "$BACKUP_DIR/$BACKUP_NAME/database.sql" | cut -f1)
    echo -e "${GREEN}✓ Database backup complete (${db_size})${NC}"
else
    echo -e "${RED}✗ Database backup failed${NC}"
    exit 1
fi

# Step 2: Backup Redis data
echo -e "${YELLOW}[2/6] Backing up Redis data...${NC}"
docker-compose exec -T redis redis-cli SAVE > /dev/null 2>&1
docker cp formula-api-redis:/data/dump.rdb "$BACKUP_DIR/$BACKUP_NAME/redis.rdb" 2>/dev/null || true
if [ -f "$BACKUP_DIR/$BACKUP_NAME/redis.rdb" ]; then
    echo -e "${GREEN}✓ Redis backup complete${NC}"
else
    echo -e "${YELLOW}⚠ Redis backup skipped (no data)${NC}"
fi

# Step 3: Backup MLflow artifacts
echo -e "${YELLOW}[3/6] Backing up MLflow artifacts...${NC}"
docker run --rm \
    --volumes-from formula-api-mlflow \
    -v "$BACKUP_DIR/$BACKUP_NAME":/backup \
    alpine tar czf /backup/mlflow_data.tar.gz /mlflow 2>/dev/null || true
if [ -f "$BACKUP_DIR/$BACKUP_NAME/mlflow_data.tar.gz" ]; then
    mlflow_size=$(du -h "$BACKUP_DIR/$BACKUP_NAME/mlflow_data.tar.gz" | cut -f1)
    echo -e "${GREEN}✓ MLflow backup complete (${mlflow_size})${NC}"
else
    echo -e "${YELLOW}⚠ MLflow backup skipped${NC}"
fi

# Step 4: Backup configuration files
echo -e "${YELLOW}[4/6] Backing up configuration files...${NC}"
mkdir -p "$BACKUP_DIR/$BACKUP_NAME/config"
cp .env "$BACKUP_DIR/$BACKUP_NAME/config/.env" 2>/dev/null || cp .env.example "$BACKUP_DIR/$BACKUP_NAME/config/.env.example"
cp docker-compose.yml "$BACKUP_DIR/$BACKUP_NAME/config/"
cp -r .streamlit "$BACKUP_DIR/$BACKUP_NAME/config/" 2>/dev/null || true
echo -e "${GREEN}✓ Configuration backup complete${NC}"

# Step 5: Create metadata file
echo -e "${YELLOW}[5/6] Creating backup metadata...${NC}"
cat > "$BACKUP_DIR/$BACKUP_NAME/metadata.txt" << EOF
Backup Information
==================
Backup Name: $BACKUP_NAME
Created: $(date)
Hostname: $(hostname)
Docker Compose Version: $(docker-compose --version)

Database Size: $(du -h "$BACKUP_DIR/$BACKUP_NAME/database.sql" 2>/dev/null | cut -f1 || echo "N/A")
Redis Backup: $([ -f "$BACKUP_DIR/$BACKUP_NAME/redis.rdb" ] && echo "Yes" || echo "No")
MLflow Backup: $([ -f "$BACKUP_DIR/$BACKUP_NAME/mlflow_data.tar.gz" ] && echo "Yes" || echo "No")

Container Status:
$(docker-compose ps)

Database Info:
$(docker-compose exec -T db psql -U postgres -d formulas -c "SELECT COUNT(*) as execution_count FROM formula_executions;" 2>/dev/null || echo "N/A")
EOF
echo -e "${GREEN}✓ Metadata created${NC}"

# Step 6: Compress backup
echo -e "${YELLOW}[6/6] Compressing backup...${NC}"
cd "$BACKUP_DIR"
tar czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"
cd - > /dev/null

backup_size=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)
echo -e "${GREEN}✓ Backup compressed (${backup_size})${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Backup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Backup file: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo "Size: $backup_size"
echo ""

# Cleanup old backups
echo -e "${YELLOW}Cleaning up old backups (keeping last ${RETENTION_DAYS} days)...${NC}"
find "$BACKUP_DIR" -name "formula-api-backup-*.tar.gz" -mtime +${RETENTION_DAYS} -delete
remaining=$(ls -1 "$BACKUP_DIR"/formula-api-backup-*.tar.gz 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Cleanup complete (${remaining} backups remaining)${NC}"
echo ""

echo "To restore this backup:"
echo "  bash backup/restore.sh $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo ""
