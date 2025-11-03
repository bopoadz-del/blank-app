#!/bin/bash

# Comprehensive Test Script for Formula Execution API
# Tests: Formula execution, database writes, MLflow tracking, unit conversions, error scenarios

set -e

API_KEY="${API_KEY:-your-api-key-change-this}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "========================================="
echo "Formula Execution API - Comprehensive Tests"
echo "========================================="
echo "API Key: ${API_KEY:0:10}..."
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

test_passed() {
    echo -e "${GREEN}✓ PASSED${NC}: $1"
    ((pass_count++))
}

test_failed() {
    echo -e "${RED}✗ FAILED${NC}: $1"
    ((fail_count++))
}

# Test 1: Health Check
echo -e "${BLUE}Test 1: Health Check${NC}"
response=$(curl -s "$BASE_URL/health")
if echo "$response" | grep -q "healthy"; then
    test_passed "Health check endpoint"
else
    test_failed "Health check endpoint"
fi
echo ""

# Test 2: Formula Execution (End-to-End)
echo -e "${BLUE}Test 2: Formula Execution (Beam Deflection)${NC}"
response=$(curl -s -X POST "$BASE_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "formula_id": "beam_deflection_simply_supported",
        "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001}
    }')

if echo "$response" | grep -q '"success":true'; then
    test_passed "Formula execution succeeded"
    execution_id=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('execution_id', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "  Execution ID: $execution_id"
else
    test_failed "Formula execution"
fi
echo ""

# Test 3: Unit Conversion
echo -e "${BLUE}Test 3: Unit Conversion (m to mm)${NC}"
response=$(curl -s -X POST "$BASE_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "formula_id": "beam_deflection_simply_supported",
        "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001},
        "convert_to_unit": "mm"
    }')

if echo "$response" | grep -q '"unit":"mm"' && echo "$response" | grep -q '"original_unit":"m"'; then
    test_passed "Unit conversion (m to mm)"
else
    test_failed "Unit conversion"
fi
echo ""

# Test 4: Database Persistence - Check History
echo -e "${BLUE}Test 4: Database Persistence (History Endpoint)${NC}"
response=$(curl -s "$BASE_URL/api/v1/formulas/history/recent?limit=5" \
    -H "X-API-Key: $API_KEY")

if echo "$response" | grep -q "formula_id"; then
    test_passed "Database persistence verified via history endpoint"
    count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    echo "  Recent executions: $count"
else
    test_failed "Database persistence"
fi
echo ""

# Test 5: Error Scenario - Invalid Formula
echo -e "${BLUE}Test 5: Error Handling (Invalid Formula)${NC}"
response=$(curl -s -X POST "$BASE_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "formula_id": "nonexistent_formula",
        "input_values": {"x": 10}
    }')

if echo "$response" | grep -q '"success":false' && echo "$response" | grep -q "not found"; then
    test_passed "Error handling for invalid formula"
else
    test_failed "Error handling for invalid formula"
fi
echo ""

# Test 6: Error Scenario - Missing Parameters
echo -e "${BLUE}Test 6: Error Handling (Missing Parameters)${NC}"
response=$(curl -s -X POST "$BASE_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "formula_id": "beam_deflection_simply_supported",
        "input_values": {"w": 10, "L": 5}
    }')

if echo "$response" | grep -q '"success":false' && echo "$response" | grep -qi "missing"; then
    test_passed "Error handling for missing parameters"
else
    test_failed "Error handling for missing parameters"
fi
echo ""

# Test 7: Error Scenario - Incompatible Unit Conversion
echo -e "${BLUE}Test 7: Error Handling (Incompatible Unit Conversion)${NC}"
response=$(curl -s -X POST "$BASE_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "formula_id": "beam_deflection_simply_supported",
        "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001},
        "convert_to_unit": "Pa"
    }')

if echo "$response" | grep -q '"success":false'; then
    test_passed "Error handling for incompatible unit conversion"
else
    test_failed "Error handling for incompatible unit conversion"
fi
echo ""

# Test 8: Multiple Formulas
echo -e "${BLUE}Test 8: Multiple Formula Types${NC}"
formulas=("spring_deflection:F=100,k=1000" "reynolds_number:rho=1000,v=2,L=0.5,mu=0.001" "beam_stress:M=1000,c=0.05,I=0.0001")
for formula_spec in "${formulas[@]}"; do
    IFS=':' read -r formula_id params <<< "$formula_spec"

    # Build JSON input
    json_params=$(echo "$params" | awk -F',' '{
        printf "{"
        for(i=1; i<=NF; i++) {
            split($i, a, "=")
            if (i > 1) printf ","
            printf "\"%s\":%s", a[1], a[2]
        }
        printf "}"
    }')

    response=$(curl -s -X POST "$BASE_URL/api/v1/formulas/execute" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"formula_id\":\"$formula_id\",\"input_values\":$json_params}")

    if echo "$response" | grep -q '"success":true'; then
        test_passed "Formula execution: $formula_id"
    else
        test_failed "Formula execution: $formula_id"
    fi
done
echo ""

# Test 9: List Formulas
echo -e "${BLUE}Test 9: List Available Formulas${NC}"
response=$(curl -s "$BASE_URL/api/v1/formulas/list" \
    -H "X-API-Key: $API_KEY")

if echo "$response" | grep -q "beam_deflection_simply_supported"; then
    test_passed "List formulas endpoint"
    formula_count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    echo "  Available formulas: $formula_count"
else
    test_failed "List formulas endpoint"
fi
echo ""

# Test 10: Formula Info
echo -e "${BLUE}Test 10: Get Formula Information${NC}"
response=$(curl -s "$BASE_URL/api/v1/formulas/beam_deflection_simply_supported" \
    -H "X-API-Key: $API_KEY")

if echo "$response" | grep -q "parameters"; then
    test_passed "Get formula info endpoint"
else
    test_failed "Get formula info endpoint"
fi
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"
echo "Total: $((pass_count + fail_count))"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the logs.${NC}"
    exit 1
fi
