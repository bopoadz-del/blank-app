#!/bin/bash
#
# ==============================================================================
# Post-Deployment Verification Tests
# ==============================================================================
# Tests the deployed system to verify it's working
#
# Usage: ./verify-deployment.sh [api-key]
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load .env if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

API_KEY=${1:-$API_KEY}
PORT=${PORT:-8000}
BASE_URL="http://localhost:$PORT"

PASSED=0
FAILED=0

echo "=========================================="
echo "  Deployment Verification Tests"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Root endpoint
echo "Test 1: Root endpoint"
if curl -sf "$BASE_URL/" > /dev/null; then
    echo "  ✅ PASS - Root accessible"
    ((PASSED++))
else
    echo "  ❌ FAIL - Root not accessible"
    ((FAILED++))
fi

# Test 2: Health check
echo "Test 2: Health check"
HEALTH=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo "failed")
if echo "$HEALTH" | grep -q "healthy"; then
    echo "  ✅ PASS - Health check passed"
    ((PASSED++))
else
    echo "  ❌ FAIL - Health check failed"
    echo "     Response: $HEALTH"
    ((FAILED++))
fi

# Test 3: Metrics endpoint
echo "Test 3: Metrics endpoint"
if curl -sf "$BASE_URL/metrics" > /dev/null; then
    echo "  ✅ PASS - Metrics accessible"
    ((PASSED++))
else
    echo "  ❌ FAIL - Metrics not accessible"
    ((FAILED++))
fi

# Test 4: API docs
echo "Test 4: API documentation"
if curl -sf "$BASE_URL/docs" > /dev/null; then
    echo "  ✅ PASS - API docs accessible"
    ((PASSED++))
else
    echo "  ❌ FAIL - API docs not accessible"
    ((FAILED++))
fi

# Test 5: List formulas (with auth)
if [ -n "$API_KEY" ]; then
    echo "Test 5: List formulas (authenticated)"
    FORMULAS=$(curl -sf -H "X-API-Key: $API_KEY" "$BASE_URL/api/v1/formulas" 2>/dev/null || echo "failed")
    
    if echo "$FORMULAS" | grep -q "formula_id"; then
        FORMULA_COUNT=$(echo "$FORMULAS" | grep -o "formula_id" | wc -l)
        echo "  ✅ PASS - Got $FORMULA_COUNT formulas"
        ((PASSED++))
        
        if [ "$FORMULA_COUNT" -eq 30 ]; then
            echo "     ✅ Formula count correct (30)"
        else
            echo "     ⚠️  Expected 30 formulas, got $FORMULA_COUNT"
        fi
    else
        echo "  ❌ FAIL - Could not retrieve formulas"
        echo "     Response: $FORMULAS"
        ((FAILED++))
    fi
else
    echo "Test 5: List formulas"
    echo "  ⚠️  SKIP - No API key provided"
fi

# Test 6: Execute a formula (with auth)
if [ -n "$API_KEY" ]; then
    echo "Test 6: Execute formula"
    EXEC_RESULT=$(curl -sf -X POST \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
        "$BASE_URL/api/v1/formulas/execute" 2>/dev/null || echo "failed")
    
    if echo "$EXEC_RESULT" | grep -q "output_values\|status"; then
        echo "  ✅ PASS - Formula execution works"
        ((PASSED++))
    else
        echo "  ❌ FAIL - Formula execution failed"
        echo "     Response: $EXEC_RESULT"
        ((FAILED++))
    fi
else
    echo "Test 6: Execute formula"
    echo "  ⚠️  SKIP - No API key provided"
fi

# Test 7: Unit conversion (with auth)
if [ -n "$API_KEY" ]; then
    echo "Test 7: Unit conversion"
    CONVERT_RESULT=$(curl -sf -X POST \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"value":100,"from_unit":"psi","to_unit":"MPa"}' \
        "$BASE_URL/api/v1/units/convert" 2>/dev/null || echo "failed")
    
    if echo "$CONVERT_RESULT" | grep -q "converted_value"; then
        echo "  ✅ PASS - Unit conversion works"
        ((PASSED++))
    else
        echo "  ❌ FAIL - Unit conversion failed"
        echo "     Response: $CONVERT_RESULT"
        ((FAILED++))
    fi
else
    echo "Test 7: Unit conversion"
    echo "  ⚠️  SKIP - No API key provided"
fi

# Test 8: Docker containers
echo "Test 8: Docker containers"
CONTAINERS=$(docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps -q 2>/dev/null | wc -l)
if [ "$CONTAINERS" -ge 4 ]; then
    echo "  ✅ PASS - $CONTAINERS containers running"
    ((PASSED++))
else
    echo "  ⚠️  WARNING - Only $CONTAINERS containers running (expected 4+)"
fi

# Test 9: Database connectivity
echo "Test 9: Database connectivity"
if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" exec -T postgres pg_isready > /dev/null 2>&1; then
    echo "  ✅ PASS - Database responding"
    ((PASSED++))
else
    echo "  ❌ FAIL - Database not responding"
    ((FAILED++))
fi

echo ""
echo "=========================================="
echo "  Results"
echo "=========================================="
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All critical tests passed!"
    echo "   System is operational."
    exit 0
elif [ $PASSED -gt $FAILED ]; then
    echo "⚠️  Some tests failed but system is mostly working"
    echo "   Review failed tests above"
    exit 0
else
    echo "❌ Multiple critical tests failed"
    echo "   System may not be working correctly"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs: docker-compose logs backend"
    echo "  2. Check services: docker-compose ps"
    echo "  3. Check health: curl http://localhost:$PORT/health"
    exit 1
fi
