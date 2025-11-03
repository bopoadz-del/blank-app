#!/bin/bash

# Quick Rollback Script
# For emergency use when GitHub Actions is not available

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Environment to rollback (staging|production)"
    echo "  -b, --backup TIMESTAMP   Backup timestamp (YYYYMMDD_HHMMSS) or 'previous'"
    echo "  -r, --reason REASON      Reason for rollback"
    echo "  -y, --yes                Skip confirmation prompt"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e staging -b previous -r \"Critical bug\""
    echo "  $0 -e production -b 20240101_120000 -r \"Performance issue\" -y"
}

# Parse arguments
ENVIRONMENT=""
BACKUP_TIMESTAMP=""
REASON=""
AUTO_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--backup)
            BACKUP_TIMESTAMP="$2"
            shift 2
            ;;
        -r|--reason)
            REASON="$2"
            shift 2
            ;;
        -y|--yes)
            AUTO_CONFIRM=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ENVIRONMENT" ] || [ -z "$BACKUP_TIMESTAMP" ] || [ -z "$REASON" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    show_usage
    exit 1
fi

if [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    echo -e "${RED}Error: Environment must be 'staging' or 'production'${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Emergency Rollback${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Backup: $BACKUP_TIMESTAMP"
echo "Reason: $REASON"
echo ""

# Confirmation
if [ "$AUTO_CONFIRM" = false ]; then
    echo -e "${YELLOW}WARNING: This will rollback the $ENVIRONMENT environment!${NC}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Rollback cancelled"
        exit 0
    fi
fi

echo ""
echo -e "${YELLOW}Starting rollback process...${NC}"
echo ""

# Determine backup file
if [ "$BACKUP_TIMESTAMP" = "previous" ]; then
    BACKUP_FILE=$(ls -t backup/backups/*.tar.gz 2>/dev/null | head -1)
    if [ "$ENVIRONMENT" = "production" ]; then
        # For production, get second-to-last (skip the emergency backup)
        BACKUP_FILE=$(ls -t backup/backups/*.tar.gz 2>/dev/null | head -2 | tail -1)
    fi
else
    BACKUP_FILE="backup/backups/formula-api-backup-${BACKUP_TIMESTAMP}.tar.gz"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Error: Backup file not found: $BACKUP_FILE${NC}"
    echo ""
    echo "Available backups:"
    ls -lh backup/backups/*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

echo -e "${GREEN}Using backup: $BACKUP_FILE${NC}"
echo ""

# Step 1: Create emergency backup (production only)
if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${YELLOW}[1/6] Creating emergency backup...${NC}"
    if bash backup/backup.sh; then
        echo -e "${GREEN}✓ Emergency backup created${NC}"
    else
        echo -e "${RED}✗ Emergency backup failed${NC}"
        exit 1
    fi
    echo ""
fi

# Step 2: Stop services
STEP=$((ENVIRONMENT == "production" ? 2 : 1))
echo -e "${YELLOW}[$STEP/6] Stopping services...${NC}"
docker-compose down
echo -e "${GREEN}✓ Services stopped${NC}"
echo ""

# Step 3: Restore from backup
STEP=$((STEP + 1))
echo -e "${YELLOW}[$STEP/6] Restoring from backup...${NC}"
if echo "yes" | bash backup/restore.sh "$BACKUP_FILE"; then
    echo -e "${GREEN}✓ Backup restored${NC}"
else
    echo -e "${RED}✗ Restore failed${NC}"
    exit 1
fi
echo ""

# Step 4: Start services
STEP=$((STEP + 1))
echo -e "${YELLOW}[$STEP/6] Starting services...${NC}"
docker-compose up -d
echo "Waiting for services to initialize..."
sleep 15
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Step 5: Verify rollback
STEP=$((STEP + 1))
echo -e "${YELLOW}[$STEP/6] Verifying rollback...${NC}"
if bash scripts/verify-deployment.sh; then
    echo -e "${GREEN}✓ Verification passed${NC}"
else
    echo -e "${RED}✗ Verification failed${NC}"
    echo "Services may need manual intervention"
fi
echo ""

# Step 6: Generate report
STEP=$((STEP + 1))
echo -e "${YELLOW}[$STEP/6] Generating rollback report...${NC}"

REPORT_FILE="backup/rollback-report-$(date +%Y%m%d_%H%M%S).txt"
cat > "$REPORT_FILE" << EOF
Rollback Report
===============

Environment: $ENVIRONMENT
Backup Used: $BACKUP_FILE
Reason: $REASON
Executed By: $(whoami)
Executed At: $(date)
Hostname: $(hostname)

Services Status:
$(docker-compose ps)

Database Status:
$(docker-compose exec -T db psql -U postgres -d formulas -c "SELECT COUNT(*) FROM formula_executions;" 2>/dev/null || echo "N/A")

Recent Logs:
$(docker-compose logs backend --tail 20)
EOF

echo -e "${GREEN}✓ Report saved: $REPORT_FILE${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Rollback Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Monitor application logs: docker-compose logs -f backend"
echo "2. Check metrics and health: curl http://localhost:8000/health"
echo "3. Review rollback report: $REPORT_FILE"
echo "4. Notify team about the rollback"
echo ""
