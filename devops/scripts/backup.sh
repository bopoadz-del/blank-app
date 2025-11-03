#!/bin/bash
#
# ==============================================================================
# Database Backup Script
# ==============================================================================
# Creates a timestamped backup of the PostgreSQL database
#
# Usage: ./backup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_DIR="$PROJECT_ROOT/data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="reasoner_backup_$TIMESTAMP.sql.gz"

# Load environment
set -a
source "$PROJECT_ROOT/.env"
set +a

echo "=========================================="
echo "  Database Backup"
echo "=========================================="
echo "Backup file: $BACKUP_FILE"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
echo "📦 Creating database backup..."
docker-compose exec -T postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_DIR/$BACKUP_FILE"

BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_FILE" | cut -f1)
echo "✅ Backup created: $BACKUP_FILE ($BACKUP_SIZE)"
echo ""

# Backup formula data
echo "📚 Backing up formula files..."
FORMULA_BACKUP="$BACKUP_DIR/formulas_$TIMESTAMP.tar.gz"
tar -czf "$FORMULA_BACKUP" -C "$PROJECT_ROOT" data/formulas data/datasets data/bounds 2>/dev/null || true

if [ -f "$FORMULA_BACKUP" ]; then
    FORMULA_SIZE=$(du -h "$FORMULA_BACKUP" | cut -f1)
    echo "✅ Formula backup: formulas_$TIMESTAMP.tar.gz ($FORMULA_SIZE)"
else
    echo "⚠️  No formula files to backup"
fi
echo ""

# Clean old backups (keep last 30 days)
echo "🧹 Cleaning old backups (keeping 30 days)..."
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
echo "✅ Cleanup complete"
echo ""

# List recent backups
echo "📋 Recent backups:"
ls -lh "$BACKUP_DIR" | grep "backup_" | tail -5
echo ""

echo "=========================================="
echo "  ✅ Backup Complete"
echo "=========================================="
echo "Location: $BACKUP_DIR"
echo ""
echo "To restore:"
echo "  ./restore.sh $BACKUP_FILE"
echo ""
