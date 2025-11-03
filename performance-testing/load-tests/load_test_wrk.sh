#!/bin/bash

################################################################################
# wrk Load Testing Script
# Modern HTTP benchmarking tool with Lua scripting support
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
REPORT_FILE="${REPORT_DIR}/wrk_load_test_${TIMESTAMP}.txt"

# Create report directory
mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  wrk Load Testing Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "API Host: ${API_HOST}"
echo "Report: ${REPORT_FILE}"
echo ""

# Check if wrk is installed
if ! command -v wrk &> /dev/null; then
    echo -e "${YELLOW}Warning: wrk is not installed${NC}"
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install wrk"
    echo "  macOS: brew install wrk"
    echo "  Or build from source: https://github.com/wg/wrk"
    echo ""
    echo "Attempting to continue with available tools..."

    # Try using hey as alternative
    if command -v hey &> /dev/null; then
        echo -e "${GREEN}Found 'hey' as alternative load testing tool${NC}"
        USE_HEY=true
    else
        echo -e "${RED}Error: No load testing tool available${NC}"
        exit 1
    fi
fi

# Health check
echo -e "${YELLOW}Running health check...${NC}"
if ! curl -s "${API_HOST}/health" > /dev/null; then
    echo -e "${RED}Error: API is not responding at ${API_HOST}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API is healthy${NC}"
echo ""

# Create Lua scripts for wrk
mkdir -p performance-testing/load-tests/lua

# Lua script for POST requests
cat > performance-testing/load-tests/lua/post.lua <<'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["X-API-Key"] = "test-key-1"
wrk.body = '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}'

function response(status, headers, body)
    if status ~= 200 and status ~= 201 then
        print("Error response: " .. status)
        print("Body: " .. body)
    end
end
EOF

# Lua script for POST with unit conversion
cat > performance-testing/load-tests/lua/post_with_conversion.lua <<'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["X-API-Key"] = "test-key-1"
wrk.body = '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001},"convert_to":"mm"}'

function response(status, headers, body)
    if status ~= 200 and status ~= 201 then
        print("Error response: " .. status)
    end
end
EOF

# Lua script for GET with API key
cat > performance-testing/load-tests/lua/get.lua <<'EOF'
wrk.method = "GET"
wrk.headers["X-API-Key"] = "test-key-1"

function response(status, headers, body)
    if status ~= 200 then
        print("Error response: " .. status)
    end
end
EOF

# Lua script for latency tracking
cat > performance-testing/load-tests/lua/latency.lua <<'EOF'
wrk.method = "GET"
wrk.headers["X-API-Key"] = "test-key-1"

latencies = {}

function response(status, headers, body)
    if status ~= 200 then
        print("Error response: " .. status)
    end
end

function done(summary, latency, requests)
    print("\n--- Detailed Latency Statistics ---")
    print(string.format("  Min: %.3f ms", latency.min))
    print(string.format("  Max: %.3f ms", latency.max))
    print(string.format("  Mean: %.3f ms", latency.mean))
    print(string.format("  Stdev: %.3f ms", latency.stdev))
    print(string.format("  50th percentile: %.3f ms", latency:percentile(50)))
    print(string.format("  75th percentile: %.3f ms", latency:percentile(75)))
    print(string.format("  90th percentile: %.3f ms", latency:percentile(90)))
    print(string.format("  95th percentile: %.3f ms", latency:percentile(95)))
    print(string.format("  99th percentile: %.3f ms", latency:percentile(99)))
    print(string.format("  99.9th percentile: %.3f ms", latency:percentile(99.9)))
end
EOF

echo -e "${GREEN}✓ Lua scripts created${NC}"
echo ""

# Function to run wrk test
run_wrk_test() {
    local test_name=$1
    local duration=$2
    local threads=$3
    local connections=$4
    local endpoint=$5
    local lua_script=$6

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test: ${test_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Duration: ${duration}"
    echo "Threads: ${threads}"
    echo "Connections: ${connections}"
    echo "Endpoint: ${endpoint}"
    echo ""

    {
        echo "═══════════════════════════════════════════════════════════"
        echo "Test: ${test_name}"
        echo "Timestamp: $(date)"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
    } >> "${REPORT_FILE}"

    if [ -n "${lua_script}" ]; then
        wrk -t"${threads}" -c"${connections}" -d"${duration}" \
            --latency \
            -s "${lua_script}" \
            "${API_HOST}${endpoint}" 2>&1 | tee -a "${REPORT_FILE}"
    else
        wrk -t"${threads}" -c"${connections}" -d"${duration}" \
            --latency \
            -H "X-API-Key: ${API_KEY}" \
            "${API_HOST}${endpoint}" 2>&1 | tee -a "${REPORT_FILE}"
    fi

    echo "" >> "${REPORT_FILE}"
    echo ""
    sleep 2
}

# Test Suite
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting wrk Load Tests${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

if [ "${USE_HEY}" = "true" ]; then
    echo -e "${YELLOW}Using 'hey' instead of 'wrk'${NC}"
    echo -e "${YELLOW}Results may differ from wrk output${NC}"
    echo ""

    # Run tests with hey
    # Test 1: Health endpoint
    echo "Test: Health Check - 10s, 50 connections"
    hey -z 10s -c 50 -H "X-API-Key: ${API_KEY}" "${API_HOST}/health" | tee -a "${REPORT_FILE}"

    # Test 2: List formulas
    echo "Test: List Formulas - 10s, 50 connections"
    hey -z 10s -c 50 -H "X-API-Key: ${API_KEY}" "${API_HOST}/api/v1/formulas/list" | tee -a "${REPORT_FILE}"

else
    # Test 1: Health Endpoint - Baseline
    run_wrk_test \
        "Health Check - Baseline (30s)" \
        "30s" \
        4 \
        50 \
        "/health"

    # Test 2: Health Endpoint - High Concurrency
    run_wrk_test \
        "Health Check - High Concurrency (30s)" \
        "30s" \
        8 \
        200 \
        "/health"

    # Test 3: List Formulas - Baseline
    run_wrk_test \
        "List Formulas - Baseline (30s)" \
        "30s" \
        4 \
        50 \
        "/api/v1/formulas/list" \
        "performance-testing/load-tests/lua/get.lua"

    # Test 4: List Formulas - High Concurrency
    run_wrk_test \
        "List Formulas - High Concurrency (30s)" \
        "30s" \
        8 \
        200 \
        "/api/v1/formulas/list" \
        "performance-testing/load-tests/lua/get.lua"

    # Test 5: Formula Execution - Baseline
    run_wrk_test \
        "Formula Execution - Baseline (60s)" \
        "60s" \
        4 \
        50 \
        "/api/v1/formulas/execute" \
        "performance-testing/load-tests/lua/post.lua"

    # Test 6: Formula Execution - High Concurrency
    run_wrk_test \
        "Formula Execution - High Concurrency (60s)" \
        "60s" \
        8 \
        200 \
        "/api/v1/formulas/execute" \
        "performance-testing/load-tests/lua/post.lua"

    # Test 7: Formula with Unit Conversion
    run_wrk_test \
        "Formula with Unit Conversion (60s)" \
        "60s" \
        4 \
        100 \
        "/api/v1/formulas/execute" \
        "performance-testing/load-tests/lua/post_with_conversion.lua"

    # Test 8: History Endpoint - Baseline
    run_wrk_test \
        "History Endpoint - Baseline (30s)" \
        "30s" \
        4 \
        50 \
        "/api/v1/formulas/history?limit=10" \
        "performance-testing/load-tests/lua/get.lua"

    # Test 9: Detailed Latency Analysis
    run_wrk_test \
        "Detailed Latency Analysis - Formula Execution (60s)" \
        "60s" \
        4 \
        100 \
        "/api/v1/formulas/execute" \
        "performance-testing/load-tests/lua/latency.lua"

    # Test 10: Sustained Load Test (5 minutes)
    echo -e "${YELLOW}Running sustained load test (5 minutes)...${NC}"
    run_wrk_test \
        "Sustained Load Test - Formula Execution (5min)" \
        "5m" \
        4 \
        50 \
        "/api/v1/formulas/execute" \
        "performance-testing/load-tests/lua/post.lua"
fi

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Load Testing Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Full report saved to: ${REPORT_FILE}"
echo ""

# Extract key metrics
if [ -f "${REPORT_FILE}" ]; then
    echo -e "${YELLOW}Summary Statistics:${NC}"
    echo ""

    echo "Requests per second:"
    grep "Requests/sec:" "${REPORT_FILE}" | awk '{print "  " $2 " req/s"}' | nl
    echo ""

    echo "Transfer rate:"
    grep "Transfer/sec:" "${REPORT_FILE}" | awk '{print "  " $2 $3}' | nl
    echo ""

    echo "Latency (99th percentile):"
    grep "99%" "${REPORT_FILE}" | awk '{print "  " $2}' | nl
    echo ""
fi

echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review detailed report: cat ${REPORT_FILE}"
echo "2. Compare with Apache Bench results"
echo "3. Run stress tests: bash performance-testing/stress-tests/stress_test.sh"
echo "4. Analyze with profiling: bash performance-testing/profiling/memory_profiler.sh"
echo ""

exit 0
