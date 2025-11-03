#!/bin/bash

################################################################################
# Apache Bench (ab) Load Testing Script
# Tests API performance under various load conditions
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_HOST="${API_HOST:-http://localhost:8000}"
API_KEY="${API_KEY:-test-key-1}"
REPORT_DIR="performance-testing/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/ab_load_test_${TIMESTAMP}.txt"

# Create report directory
mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Apache Bench (ab) Load Testing Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "API Host: ${API_HOST}"
echo "Report: ${REPORT_FILE}"
echo ""

# Check if ab is installed
if ! command -v ab &> /dev/null; then
    echo -e "${RED}Error: Apache Bench (ab) is not installed${NC}"
    echo "Install with: sudo apt-get install apache2-utils (Ubuntu/Debian)"
    echo "           or: brew install apache2 (macOS)"
    exit 1
fi

# Health check
echo -e "${YELLOW}Running health check...${NC}"
if ! curl -s "${API_HOST}/health" > /dev/null; then
    echo -e "${RED}Error: API is not responding at ${API_HOST}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API is healthy${NC}"
echo ""

# Function to run load test
run_ab_test() {
    local test_name=$1
    local requests=$2
    local concurrency=$3
    local endpoint=$4
    local method=$5
    local data_file=$6

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test: ${test_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Requests: ${requests}"
    echo "Concurrency: ${concurrency}"
    echo "Endpoint: ${endpoint}"
    echo ""

    {
        echo "═══════════════════════════════════════════════════════════"
        echo "Test: ${test_name}"
        echo "Timestamp: $(date)"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
    } >> "${REPORT_FILE}"

    if [ "${method}" = "POST" ] && [ -n "${data_file}" ]; then
        ab -n "${requests}" -c "${concurrency}" \
           -T "application/json" \
           -H "X-API-Key: ${API_KEY}" \
           -p "${data_file}" \
           "${API_HOST}${endpoint}" 2>&1 | tee -a "${REPORT_FILE}"
    else
        ab -n "${requests}" -c "${concurrency}" \
           -H "X-API-Key: ${API_KEY}" \
           "${API_HOST}${endpoint}" 2>&1 | tee -a "${REPORT_FILE}"
    fi

    echo "" >> "${REPORT_FILE}"
    echo ""
    sleep 2
}

# Create test payload files
echo -e "${YELLOW}Creating test payload files...${NC}"

cat > /tmp/ab_formula_payload.json <<EOF
{
  "formula_id": "beam_deflection_simply_supported",
  "input_values": {
    "w": 10,
    "L": 5,
    "E": 200,
    "I": 0.0001
  }
}
EOF

cat > /tmp/ab_unit_conversion_payload.json <<EOF
{
  "formula_id": "beam_deflection_simply_supported",
  "input_values": {
    "w": 10,
    "L": 5,
    "E": 200,
    "I": 0.0001
  },
  "convert_to": "mm"
}
EOF

echo -e "${GREEN}✓ Payload files created${NC}"
echo ""

# Test Suite
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Load Tests${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Test 1: Health Endpoint - Baseline (Light Load)
run_ab_test \
    "Health Check - Light Load" \
    100 \
    10 \
    "/health" \
    "GET"

# Test 2: Health Endpoint - Moderate Load
run_ab_test \
    "Health Check - Moderate Load" \
    1000 \
    50 \
    "/health" \
    "GET"

# Test 3: Health Endpoint - Heavy Load
run_ab_test \
    "Health Check - Heavy Load" \
    5000 \
    100 \
    "/health" \
    "GET"

# Test 4: List Formulas - Light Load
run_ab_test \
    "List Formulas - Light Load" \
    100 \
    10 \
    "/api/v1/formulas/list" \
    "GET"

# Test 5: List Formulas - Moderate Load
run_ab_test \
    "List Formulas - Moderate Load" \
    1000 \
    50 \
    "/api/v1/formulas/list" \
    "GET"

# Test 6: Formula Execution - Light Load
run_ab_test \
    "Formula Execution - Light Load" \
    50 \
    5 \
    "/api/v1/formulas/execute" \
    "POST" \
    "/tmp/ab_formula_payload.json"

# Test 7: Formula Execution - Moderate Load
run_ab_test \
    "Formula Execution - Moderate Load" \
    200 \
    20 \
    "/api/v1/formulas/execute" \
    "POST" \
    "/tmp/ab_formula_payload.json"

# Test 8: Formula Execution - Heavy Load
run_ab_test \
    "Formula Execution - Heavy Load" \
    500 \
    50 \
    "/api/v1/formulas/execute" \
    "POST" \
    "/tmp/ab_formula_payload.json"

# Test 9: Formula with Unit Conversion - Moderate Load
run_ab_test \
    "Formula with Unit Conversion - Moderate Load" \
    200 \
    20 \
    "/api/v1/formulas/execute" \
    "POST" \
    "/tmp/ab_unit_conversion_payload.json"

# Test 10: History Endpoint - Moderate Load
run_ab_test \
    "History Endpoint - Moderate Load" \
    500 \
    25 \
    "/api/v1/formulas/history?limit=10" \
    "GET"

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Load Testing Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Full report saved to: ${REPORT_FILE}"
echo ""
echo -e "${YELLOW}Summary Statistics:${NC}"
echo ""

# Extract key metrics from report
if [ -f "${REPORT_FILE}" ]; then
    echo "Requests per second (mean):"
    grep "Requests per second:" "${REPORT_FILE}" | awk '{print "  " $4 " req/s"}' | nl
    echo ""

    echo "Time per request (mean):"
    grep "Time per request:" "${REPORT_FILE}" | head -n 10 | awk '{print "  " $4 " " $5}' | nl
    echo ""
fi

echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review detailed report: cat ${REPORT_FILE}"
echo "2. Run wrk tests: bash performance-testing/load-tests/load_test_wrk.sh"
echo "3. Run stress tests: bash performance-testing/stress-tests/stress_test.sh"
echo ""

# Cleanup
rm -f /tmp/ab_formula_payload.json /tmp/ab_unit_conversion_payload.json

exit 0
