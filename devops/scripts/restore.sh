#!/bin/bash
#
# ==============================================================================
# Database Restore Script
# ==============================================================================
# Restores database from a backup file
#
# Usage: ./restore.sh <backup_file>
# Example: ./restore.sh reasoner_backup_20251103_120000.sql.gz
#

set -e

if [ -z "$1" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh "$(dirname "$(dirname "$SCRIPT_DIR")")/data/backups"/*.sql.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_DIR="$PROJECT_ROOT/data/backups"

# Check if backup file exists
if [ ! -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "❌ Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    BACKUP_PATH="$BACKUP_FILE"
else
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"
fi

# Load environment
set -a
source "$PROJECT_ROOT/.env"
set +a

echo "=========================================="
echo "  Database Restore"
echo "=========================================="
echo "Backup file: $BACKUP_PATH"
echo ""

# Confirm
read -p "⚠️  This will REPLACE the current database. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    exit 0
fi

# Stop backend to prevent connections
echo "🛑 Stopping backend service..."
docker-compose stop backend

# Drop and recreate database
echo "🗑️  Dropping existing database..."
docker-compose exec -T postgres psql -U "$POSTGRES_USER" -c "DROP DATABASE IF EXISTS $POSTGRES_DB;"
docker-compose exec -T postgres psql -U "$POSTGRES_USER" -c "CREATE DATABASE $POSTGRES_DB;"

# Restore backup
echo "📥 Restoring database..."
gunzip -c "$BACKUP_PATH" | docker-compose exec -T postgres psql -U "$POSTGRES_USER" "$POSTGRES_DB"

echo "✅ Database restored"
echo ""

# Restart backend
echo "🚀 Restarting backend service..."
docker-compose start backend

# Wait for service
echo "⏳ Waiting for service to be ready..."
sleep 5

# Health check
HEALTH_CHECK=$(curl -s http://localhost:${PORT:-8000}/health || echo "failed")
if echo "$HEALTH_CHECK" | grep -q "healthy"; then
    echo "✅ Service is healthy"
else
    echo "⚠️  Service health check: $HEALTH_CHECK"
fi
echo ""

echo "=========================================="
echo "  ✅ Restore Complete"
echo "=========================================="
