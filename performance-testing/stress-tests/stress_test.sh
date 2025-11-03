#!/bin/bash

################################################################################
# Stress Testing Script
# Tests system behavior under extreme conditions
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
API_HOST="${API_HOST:-http://localhost:8000}"
API_KEY="${API_KEY:-test-key-1}"
REPORT_DIR="performance-testing/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/stress_test_${TIMESTAMP}.txt"
STRESS_DURATION="${STRESS_DURATION:-300}" # 5 minutes default

# Create report directory
mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Stress Testing Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "API Host: ${API_HOST}"
echo "Report: ${REPORT_FILE}"
echo "Stress Duration: ${STRESS_DURATION}s"
echo ""

# Initialize report
{
    echo "═══════════════════════════════════════════════════════════"
    echo "Stress Testing Report"
    echo "Started: $(date)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
} > "${REPORT_FILE}"

# Check dependencies
check_dependencies() {
    local missing=0

    echo -e "${YELLOW}Checking dependencies...${NC}"

    if ! command -v curl &> /dev/null; then
        echo -e "${RED}✗ curl not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ curl${NC}"
    fi

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ docker not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ docker${NC}"
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}✗ docker-compose not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ docker-compose${NC}"
    fi

    if [ $missing -eq 1 ]; then
        echo -e "${RED}Error: Missing required dependencies${NC}"
        exit 1
    fi

    echo ""
}

# Health check
health_check() {
    echo -e "${YELLOW}Running health check...${NC}"
    if ! curl -s "${API_HOST}/health" > /dev/null; then
        echo -e "${RED}Error: API is not responding at ${API_HOST}${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ API is healthy${NC}"
    echo ""
}

# Get initial metrics
get_initial_metrics() {
    echo -e "${YELLOW}Collecting initial metrics...${NC}"

    {
        echo "Initial System State"
        echo "===================="
        echo ""
        echo "Docker Stats:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}"
        echo ""
        echo "Container Logs (Last 20 lines):"
        docker-compose logs --tail 20
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Initial metrics collected${NC}"
    echo ""
}

# Stress Test 1: Gradual Load Increase (Ramp-up Test)
stress_test_ramp_up() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test 1: Gradual Load Increase (Ramp-up)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Testing how the system handles gradually increasing load"
    echo ""

    {
        echo "Test 1: Gradual Load Increase"
        echo "=============================="
        echo "Timestamp: $(date)"
        echo ""
    } >> "${REPORT_FILE}"

    local max_concurrent=500
    local step=50
    local duration_per_step=30

    for concurrent in $(seq $step $step $max_concurrent); do
        echo -e "${CYAN}Concurrent requests: ${concurrent}${NC}"

        # Run load test
        for i in $(seq 1 $concurrent); do
            curl -s -X POST "${API_HOST}/api/v1/formulas/execute" \
                -H "X-API-Key: ${API_KEY}" \
                -H "Content-Type: application/json" \
                -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
                > /dev/null 2>&1 &
        done

        # Wait for requests to complete
        sleep $duration_per_step

        # Collect metrics
        {
            echo "Concurrent: ${concurrent}"
            docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
            echo ""
        } >> "${REPORT_FILE}"

        # Check if system is still responsive
        if ! curl -s "${API_HOST}/health" > /dev/null; then
            echo -e "${RED}✗ System became unresponsive at ${concurrent} concurrent requests${NC}"
            {
                echo "FAILURE: System became unresponsive at ${concurrent} concurrent requests"
                echo ""
            } >> "${REPORT_FILE}"
            break
        fi

        echo -e "${GREEN}✓ System still responsive${NC}"
    done

    echo ""
}

# Stress Test 2: Spike Test (Sudden Load)
stress_test_spike() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test 2: Spike Test (Sudden Load)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Testing system recovery from sudden traffic spikes"
    echo ""

    {
        echo "Test 2: Spike Test"
        echo "=================="
        echo "Timestamp: $(date)"
        echo ""
    } >> "${REPORT_FILE}"

    local spike_requests=1000

    echo -e "${YELLOW}Baseline (10 seconds)...${NC}"
    sleep 10

    echo -e "${RED}SPIKE: Sending ${spike_requests} concurrent requests${NC}"
    {
        echo "Sending spike: ${spike_requests} requests"
        echo "Before spike:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
        echo ""
    } >> "${REPORT_FILE}"

    # Send spike
    for i in $(seq 1 $spike_requests); do
        curl -s -X POST "${API_HOST}/api/v1/formulas/execute" \
            -H "X-API-Key: ${API_KEY}" \
            -H "Content-Type: application/json" \
            -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
            > /dev/null 2>&1 &
    done

    echo -e "${YELLOW}Monitoring system during spike...${NC}"
    sleep 10

    {
        echo "During spike:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${YELLOW}Waiting for recovery...${NC}"
    sleep 30

    {
        echo "After spike (30s recovery):"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
        echo ""
    } >> "${REPORT_FILE}"

    # Check recovery
    if curl -s "${API_HOST}/health" > /dev/null; then
        echo -e "${GREEN}✓ System recovered from spike${NC}"
        echo "SUCCESS: System recovered" >> "${REPORT_FILE}"
    else
        echo -e "${RED}✗ System did not recover from spike${NC}"
        echo "FAILURE: System did not recover" >> "${REPORT_FILE}"
    fi

    echo ""
}

# Stress Test 3: Sustained High Load
stress_test_sustained() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test 3: Sustained High Load${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Testing system stability under sustained high load"
    echo "Duration: ${STRESS_DURATION} seconds"
    echo ""

    {
        echo "Test 3: Sustained High Load"
        echo "============================"
        echo "Timestamp: $(date)"
        echo "Duration: ${STRESS_DURATION}s"
        echo ""
    } >> "${REPORT_FILE}"

    local concurrent=100
    local end_time=$(($(date +%s) + STRESS_DURATION))

    echo -e "${YELLOW}Starting sustained load (${concurrent} concurrent)...${NC}"

    # Monitor in background
    (
        while [ $(date +%s) -lt $end_time ]; do
            {
                echo "$(date): Metrics"
                docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
                echo ""
            } >> "${REPORT_FILE}"
            sleep 30
        done
    ) &
    MONITOR_PID=$!

    # Generate sustained load
    while [ $(date +%s) -lt $end_time ]; do
        for i in $(seq 1 $concurrent); do
            curl -s -X POST "${API_HOST}/api/v1/formulas/execute" \
                -H "X-API-Key: ${API_KEY}" \
                -H "Content-Type: application/json" \
                -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
                > /dev/null 2>&1 &
        done
        sleep 2
    done

    # Wait for monitor to finish
    wait $MONITOR_PID

    echo -e "${GREEN}✓ Sustained load test complete${NC}"
    echo ""
}

# Stress Test 4: Rate Limit Breach Test
stress_test_rate_limit() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test 4: Rate Limit Breach Test${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Testing rate limiting behavior under excessive requests"
    echo ""

    {
        echo "Test 4: Rate Limit Breach"
        echo "========================="
        echo "Timestamp: $(date)"
        echo ""
    } >> "${REPORT_FILE}"

    local total_requests=100
    local success=0
    local rate_limited=0

    echo -e "${YELLOW}Sending ${total_requests} rapid requests...${NC}"

    for i in $(seq 1 $total_requests); do
        response=$(curl -s -w "\n%{http_code}" -X POST "${API_HOST}/api/v1/formulas/execute" \
            -H "X-API-Key: ${API_KEY}" \
            -H "Content-Type: application/json" \
            -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}')

        http_code=$(echo "$response" | tail -n1)

        if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
            ((success++))
        elif [ "$http_code" = "429" ]; then
            ((rate_limited++))
        fi

        # Small delay to avoid overwhelming the system
        sleep 0.1
    done

    {
        echo "Results:"
        echo "  Total requests: ${total_requests}"
        echo "  Successful: ${success}"
        echo "  Rate limited (429): ${rate_limited}"
        echo "  Other errors: $((total_requests - success - rate_limited))"
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Rate limit test complete${NC}"
    echo "  Successful: ${success}"
    echo "  Rate limited: ${rate_limited}"
    echo ""
}

# Stress Test 5: Database Connection Pool Exhaustion
stress_test_db_pool() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test 5: Database Connection Pool Test${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Testing behavior when database connections are exhausted"
    echo ""

    {
        echo "Test 5: Database Connection Pool"
        echo "================================="
        echo "Timestamp: $(date)"
        echo ""
    } >> "${REPORT_FILE}"

    local concurrent=200
    local duration=30

    echo -e "${YELLOW}Sending ${concurrent} concurrent requests for ${duration}s...${NC}"

    local end_time=$(($(date +%s) + duration))

    while [ $(date +%s) -lt $end_time ]; do
        for i in $(seq 1 $concurrent); do
            curl -s -X POST "${API_HOST}/api/v1/formulas/execute" \
                -H "X-API-Key: ${API_KEY}" \
                -H "Content-Type: application/json" \
                -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
                > /dev/null 2>&1 &
        done
        sleep 1
    done

    echo -e "${YELLOW}Waiting for requests to complete...${NC}"
    sleep 10

    {
        echo "Database connections after test:"
        docker-compose exec -T db psql -U postgres -d formulas -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE datname = 'formulas';" 2>&1
        echo ""
        echo "Recent errors from logs:"
        docker-compose logs --tail 50 backend | grep -i "error\|connection\|pool" || echo "No errors found"
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Database pool test complete${NC}"
    echo ""
}

# Stress Test 6: Memory Leak Detection
stress_test_memory_leak() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test 6: Memory Leak Detection${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Monitoring memory usage over time"
    echo ""

    {
        echo "Test 6: Memory Leak Detection"
        echo "=============================="
        echo "Timestamp: $(date)"
        echo ""
    } >> "${REPORT_FILE}"

    local duration=120
    local interval=10
    local iterations=$((duration / interval))

    echo -e "${YELLOW}Monitoring for ${duration} seconds...${NC}"

    for i in $(seq 1 $iterations); do
        # Send batch of requests
        for j in $(seq 1 50); do
            curl -s -X POST "${API_HOST}/api/v1/formulas/execute" \
                -H "X-API-Key: ${API_KEY}" \
                -H "Content-Type: application/json" \
                -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
                > /dev/null 2>&1 &
        done

        # Collect memory stats
        {
            echo "Iteration ${i}/${iterations} ($(date)):"
            docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"
            echo ""
        } >> "${REPORT_FILE}"

        sleep $interval
    done

    echo -e "${GREEN}✓ Memory leak detection complete${NC}"
    echo -e "${YELLOW}Review the report to check for memory growth trends${NC}"
    echo ""
}

# Get final metrics
get_final_metrics() {
    echo -e "${YELLOW}Collecting final metrics...${NC}"

    {
        echo "Final System State"
        echo "=================="
        echo ""
        echo "Docker Stats:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}"
        echo ""
        echo "Container Logs (Last 50 lines - errors only):"
        docker-compose logs --tail 50 | grep -i "error\|exception\|fail" || echo "No errors found"
        echo ""
        echo "Database Statistics:"
        docker-compose exec -T db psql -U postgres -d formulas -c "SELECT COUNT(*) as total_executions FROM formula_executions;" 2>&1
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Final metrics collected${NC}"
    echo ""
}

# Generate summary
generate_summary() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Stress Testing Complete!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    {
        echo "═══════════════════════════════════════════════════════════"
        echo "Test Summary"
        echo "Completed: $(date)"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "All stress tests completed successfully."
        echo ""
        echo "Review the detailed report for:"
        echo "  - System behavior under gradual load increase"
        echo "  - Recovery from traffic spikes"
        echo "  - Stability under sustained high load"
        echo "  - Rate limiting effectiveness"
        echo "  - Database connection pool behavior"
        echo "  - Potential memory leaks"
        echo ""
    } >> "${REPORT_FILE}"

    echo "Full report saved to: ${REPORT_FILE}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review the detailed report: cat ${REPORT_FILE}"
    echo "2. Check for memory leaks in the memory section"
    echo "3. Analyze error logs: docker-compose logs backend | grep -i error"
    echo "4. Run memory profiler: bash performance-testing/profiling/memory_profiler.sh"
    echo "5. Optimize database queries: bash performance-testing/profiling/db_query_analyzer.sh"
    echo ""
}

# Main execution
main() {
    check_dependencies
    health_check
    get_initial_metrics

    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Running Stress Tests${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    stress_test_ramp_up
    stress_test_spike
    stress_test_sustained
    stress_test_rate_limit
    stress_test_db_pool
    stress_test_memory_leak

    get_final_metrics
    generate_summary
}

# Run main function
main

exit 0
