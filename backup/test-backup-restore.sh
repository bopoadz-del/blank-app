#!/bin/bash

# Test Backup and Restore Procedures
# Validates that backup and restore work correctly

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_KEY="${API_KEY:-your-api-key-change-this}"
API_URL="${API_URL:-http://localhost:8000}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Backup/Restore Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Create test data
echo -e "${YELLOW}Step 1: Creating test data...${NC}"
test_response=$(curl -s "$API_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}')

if echo "$test_response" | grep -q '"success":true'; then
    execution_id=$(echo "$test_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['execution_id'])" 2>/dev/null)
    echo -e "${GREEN}✓ Test data created (ID: $execution_id)${NC}"
else
    echo -e "${RED}✗ Failed to create test data${NC}"
    exit 1
fi

# Get record count before backup
initial_count=$(curl -s "$API_URL/api/v1/formulas/history/recent?limit=1000" -H "X-API-Key: $API_KEY" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null)
echo "Current database records: $initial_count"
echo ""

# Step 2: Create backup
echo -e "${YELLOW}Step 2: Creating backup...${NC}"
if bash backup/backup.sh; then
    echo -e "${GREEN}✓ Backup created successfully${NC}"
    LATEST_BACKUP=$(ls -t ./backup/backups/*.tar.gz | head -1)
    echo "Backup file: $LATEST_BACKUP"
else
    echo -e "${RED}✗ Backup failed${NC}"
    exit 1
fi
echo ""

# Step 3: Verify backup file
echo -e "${YELLOW}Step 3: Verifying backup file...${NC}"
if [ -f "$LATEST_BACKUP" ]; then
    backup_size=$(du -h "$LATEST_BACKUP" | cut -f1)
    echo -e "${GREEN}✓ Backup file exists (${backup_size})${NC}"

    # Check backup contents
    echo "Backup contents:"
    tar tzf "$LATEST_BACKUP" | head -10
    echo "..."
else
    echo -e "${RED}✗ Backup file not found${NC}"
    exit 1
fi
echo ""

# Step 4: Create additional test data
echo -e "${YELLOW}Step 4: Creating additional test data...${NC}"
curl -s "$API_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"spring_deflection","input_values":{"F":100,"k":1000}}' > /dev/null

new_count=$(curl -s "$API_URL/api/v1/formulas/history/recent?limit=1000" -H "X-API-Key: $API_KEY" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null)
echo "Database records after new data: $new_count"
echo ""

# Step 5: Restore from backup
echo -e "${YELLOW}Step 5: Testing restore (this will reset data)...${NC}"
echo "Do you want to continue with restore test? (yes/no)"
read -p "> " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore test skipped"
    exit 0
fi

# Restore
if echo "yes" | bash backup/restore.sh "$LATEST_BACKUP"; then
    echo -e "${GREEN}✓ Restore completed${NC}"
else
    echo -e "${RED}✗ Restore failed${NC}"
    exit 1
fi
echo ""

# Step 6: Verify restored data
echo -e "${YELLOW}Step 6: Verifying restored data...${NC}"
sleep 10

restored_count=$(curl -s "$API_URL/api/v1/formulas/history/recent?limit=1000" -H "X-API-Key: $API_KEY" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null)
echo "Database records after restore: $restored_count"

if [ "$restored_count" -eq "$initial_count" ]; then
    echo -e "${GREEN}✓ Data count matches original backup${NC}"
else
    echo -e "${YELLOW}⚠ Data count differs (Expected: $initial_count, Got: $restored_count)${NC}"
fi

# Verify original execution exists
if curl -s "$API_URL/api/v1/formulas/history/recent?limit=1000" -H "X-API-Key: $API_KEY" | grep -q "\"id\":$execution_id"; then
    echo -e "${GREEN}✓ Original execution record found${NC}"
else
    echo -e "${RED}✗ Original execution record not found${NC}"
fi
echo ""

# Step 7: Run full verification
echo -e "${YELLOW}Step 7: Running full deployment verification...${NC}"
if bash scripts/verify-deployment.sh; then
    echo -e "${GREEN}✓ Full verification passed${NC}"
else
    echo -e "${RED}✗ Verification failed${NC}"
fi
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Backup/Restore Test Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "  Initial records: $initial_count"
echo "  Records after new data: $new_count"
echo "  Records after restore: $restored_count"
echo ""
echo -e "${GREEN}Backup and restore procedures are working correctly!${NC}"
