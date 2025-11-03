#!/bin/bash

# Deployment Verification Script
# Verifies that all services are running correctly after deployment

set -e

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-your-api-key-change-this}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:5000}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

passed=0
failed=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Formula API - Deployment Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Testing API at: $API_URL"
echo "Testing MLflow at: $MLFLOW_URL"
echo ""

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((passed++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((failed++))
}

# Test 1: Docker containers are running
echo -e "${YELLOW}[1/15] Checking Docker containers${NC}"
if docker-compose ps | grep -q "Up"; then
    container_count=$(docker-compose ps | grep "Up" | wc -l)
    test_pass "Docker containers running (${container_count} containers)"
else
    test_fail "Docker containers not running"
fi

# Test 2: Backend health check
echo -e "${YELLOW}[2/15] Testing backend health endpoint${NC}"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" || echo "000")
if [ "$response" -eq 200 ]; then
    test_pass "Backend health check OK"
else
    test_fail "Backend health check failed (HTTP $response)"
fi

# Test 3: Database connectivity
echo -e "${YELLOW}[3/15] Testing database connectivity${NC}"
if docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; then
    test_pass "Database is ready"
else
    test_fail "Database not ready"
fi

# Test 4: Redis connectivity
echo -e "${YELLOW}[4/15] Testing Redis connectivity${NC}"
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    test_pass "Redis is ready"
else
    test_fail "Redis not ready"
fi

# Test 5: MLflow connectivity
echo -e "${YELLOW}[5/15] Testing MLflow connectivity${NC}"
mlflow_response=$(curl -s -o /dev/null -w "%{http_code}" "$MLFLOW_URL" || echo "000")
if [ "$mlflow_response" -eq 200 ]; then
    test_pass "MLflow is accessible"
else
    test_fail "MLflow not accessible (HTTP $mlflow_response)"
fi

# Test 6: API authentication
echo -e "${YELLOW}[6/15] Testing API authentication${NC}"
response=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$API_URL/api/v1/formulas/execute" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
    || echo "000")
if [ "$response" -eq 403 ]; then
    test_pass "API authentication working (rejected unauthenticated request)"
else
    test_fail "API authentication not working properly (HTTP $response)"
fi

# Test 7: Formula execution
echo -e "${YELLOW}[7/15] Testing formula execution${NC}"
response=$(curl -s "$API_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
    || echo '{"success":false}')
if echo "$response" | grep -q '"success":true'; then
    test_pass "Formula execution successful"
else
    test_fail "Formula execution failed"
fi

# Test 8: Database writes
echo -e "${YELLOW}[8/15] Testing database writes${NC}"
history_response=$(curl -s "$API_URL/api/v1/formulas/history/recent?limit=1" \
    -H "X-API-Key: $API_KEY" || echo '[]')
if echo "$history_response" | grep -q "formula_id"; then
    test_pass "Database writes verified"
else
    test_fail "Database writes not working"
fi

# Test 9: Unit conversion
echo -e "${YELLOW}[9/15] Testing unit conversion${NC}"
conv_response=$(curl -s "$API_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001},"convert_to_unit":"mm"}' \
    || echo '{"success":false}')
if echo "$conv_response" | grep -q '"unit":"mm"'; then
    test_pass "Unit conversion working"
else
    test_fail "Unit conversion not working"
fi

# Test 10: Error handling
echo -e "${YELLOW}[10/15] Testing error handling${NC}"
error_response=$(curl -s "$API_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"invalid_formula","input_values":{"x":10}}' \
    || echo '{}')
if echo "$error_response" | grep -q '"success":false'; then
    test_pass "Error handling working"
else
    test_fail "Error handling not working"
fi

# Test 11: API documentation
echo -e "${YELLOW}[11/15] Testing API documentation${NC}"
docs_response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs" || echo "000")
if [ "$docs_response" -eq 200 ]; then
    test_pass "API documentation accessible"
else
    test_fail "API documentation not accessible (HTTP $docs_response)"
fi

# Test 12: Check disk space
echo -e "${YELLOW}[12/15] Checking disk space${NC}"
disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$disk_usage" -lt 80 ]; then
    test_pass "Disk space OK (${disk_usage}% used)"
else
    test_fail "Low disk space (${disk_usage}% used)"
fi

# Test 13: Check memory usage
echo -e "${YELLOW}[13/15] Checking memory usage${NC}"
mem_usage=$(free | awk 'NR==2 {printf "%.0f", $3/$2 * 100}')
if [ "$mem_usage" -lt 90 ]; then
    test_pass "Memory usage OK (${mem_usage}%)"
else
    test_fail "High memory usage (${mem_usage}%)"
fi

# Test 14: Check container logs for errors
echo -e "${YELLOW}[14/15] Checking container logs for errors${NC}"
error_count=$(docker-compose logs backend --tail=100 | grep -i "error\|exception\|failed" | grep -v "rate limit\|test" | wc -l)
if [ "$error_count" -eq 0 ]; then
    test_pass "No errors in recent logs"
else
    test_fail "Found $error_count errors in recent logs"
fi

# Test 15: Check if all expected tables exist
echo -e "${YELLOW}[15/15] Checking database tables${NC}"
tables=$(docker-compose exec -T db psql -U postgres -d formulas -t -c "\dt" 2>/dev/null | grep -c "formula_executions" || echo "0")
if [ "$tables" -ge 1 ]; then
    test_pass "Database tables created"
else
    test_fail "Database tables missing"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Verification Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed: $passed${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo "Total: $((passed + failed))"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All verification checks passed!${NC}"
    echo ""
    echo "The deployment is verified and ready for use."
    echo ""
    echo "Next steps:"
    echo "1. Monitor logs: docker-compose logs -f backend"
    echo "2. Check metrics: curl $API_URL/health"
    echo "3. View MLflow: $MLFLOW_URL"
    echo "4. Test backups: bash backup/backup.sh"
    exit 0
else
    echo -e "${RED}✗ Some verification checks failed${NC}"
    echo ""
    echo "Please investigate the failures before proceeding."
    echo "Check logs with: docker-compose logs backend"
    exit 1
fi
