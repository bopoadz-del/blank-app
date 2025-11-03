#!/bin/bash

################################################################################
# Memory Profiling Script
# Profiles memory usage and detects potential leaks
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
REPORT_DIR="performance-testing/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/memory_profile_${TIMESTAMP}.txt"
PROFILE_DURATION="${PROFILE_DURATION:-300}" # 5 minutes default
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}" # 5 seconds

mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Memory Profiling Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Duration: ${PROFILE_DURATION}s"
echo "Sample Interval: ${SAMPLE_INTERVAL}s"
echo "Report: ${REPORT_FILE}"
echo ""

# Initialize report
{
    echo "═══════════════════════════════════════════════════════════"
    echo "Memory Profiling Report"
    echo "Started: $(date)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
} > "${REPORT_FILE}"

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

echo -e "${YELLOW}Collecting initial memory baseline...${NC}"

# Get initial memory stats
{
    echo "Initial Memory Baseline"
    echo "======================="
    echo ""
    docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}"
    echo ""
} >> "${REPORT_FILE}"

echo -e "${GREEN}✓ Baseline collected${NC}"
echo ""

# Profile 1: Container Memory Usage Over Time
profile_container_memory() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Profile 1: Container Memory Usage Over Time${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Profile 1: Container Memory Usage"
        echo "=================================="
        echo ""
    } >> "${REPORT_FILE}"

    local samples=$((PROFILE_DURATION / SAMPLE_INTERVAL))

    echo -e "${CYAN}Collecting ${samples} samples (${PROFILE_DURATION}s total)${NC}"

    for i in $(seq 1 $samples); do
        echo -ne "\rSample ${i}/${samples}..."

        {
            echo "Sample ${i} ($(date)):"
            docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}"
            echo ""
        } >> "${REPORT_FILE}"

        sleep ${SAMPLE_INTERVAL}
    done

    echo ""
    echo -e "${GREEN}✓ Container memory profiling complete${NC}"
    echo ""
}

# Profile 2: Python Memory Usage (memory_profiler)
profile_python_memory() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Profile 2: Python Memory Usage${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Profile 2: Python Memory Usage"
        echo "==============================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Installing memory_profiler in backend container...${NC}"

    # Install memory_profiler
    docker-compose exec -T backend pip install memory-profiler psutil > /dev/null 2>&1 || true

    # Create a test script that profiles a formula execution
    cat > /tmp/profile_api_memory.py <<'EOF'
#!/usr/bin/env python3
from memory_profiler import profile
import requests
import time

@profile
def make_request():
    """Profile a single API request"""
    response = requests.post(
        'http://localhost:8000/api/v1/formulas/execute',
        headers={'X-API-Key': 'test-key-1'},
        json={
            'formula_id': 'beam_deflection_simply_supported',
            'input_values': {'w': 10, 'L': 5, 'E': 200, 'I': 0.0001}
        }
    )
    return response.json()

@profile
def make_multiple_requests(count=10):
    """Profile multiple API requests"""
    results = []
    for i in range(count):
        result = make_request()
        results.append(result)
        time.sleep(0.1)
    return results

if __name__ == '__main__':
    print("Profiling single request:")
    make_request()
    print("\nProfiling 10 requests:")
    make_multiple_requests(10)
EOF

    # Copy script to container
    docker cp /tmp/profile_api_memory.py formula-api-backend-1:/tmp/profile_api_memory.py 2>/dev/null || \
    docker cp /tmp/profile_api_memory.py $(docker-compose ps -q backend):/tmp/profile_api_memory.py

    echo -e "${CYAN}Running memory profiler...${NC}"

    # Run profiler
    {
        echo "Python Memory Profile:"
        docker-compose exec -T backend python /tmp/profile_api_memory.py 2>&1 || echo "Could not run memory profiler"
        echo ""
    } >> "${REPORT_FILE}"

    # Cleanup
    rm -f /tmp/profile_api_memory.py

    echo -e "${GREEN}✓ Python memory profiling complete${NC}"
    echo ""
}

# Profile 3: Database Memory Usage
profile_database_memory() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Profile 3: Database Memory Usage${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Profile 3: Database Memory Usage"
        echo "================================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Collecting database memory stats...${NC}"

    {
        echo "Database Size:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                pg_database.datname,
                pg_size_pretty(pg_database_size(pg_database.datname)) AS size
            FROM pg_database
            WHERE datname = 'formulas';
        " 2>&1

        echo ""
        echo "Table Sizes:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        " 2>&1

        echo ""
        echo "Cache Hit Ratio:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                sum(heap_blks_read) as heap_read,
                sum(heap_blks_hit) as heap_hit,
                sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
            FROM pg_statio_user_tables;
        " 2>&1

        echo ""
        echo "Connection Pool Stats:"
        docker-compose exec -T db psql -U postgres -c "
            SELECT
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections
            FROM pg_stat_activity;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Database memory profiling complete${NC}"
    echo ""
}

# Profile 4: Redis Memory Usage
profile_redis_memory() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Profile 4: Redis Memory Usage${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Profile 4: Redis Memory Usage"
        echo "=============================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Collecting Redis memory stats...${NC}"

    {
        echo "Redis INFO Memory:"
        docker-compose exec -T redis redis-cli INFO MEMORY 2>&1

        echo ""
        echo "Redis Key Statistics:"
        docker-compose exec -T redis redis-cli INFO KEYSPACE 2>&1

        echo ""
        echo "Redis Stats:"
        docker-compose exec -T redis redis-cli INFO STATS 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Redis memory profiling complete${NC}"
    echo ""
}

# Profile 5: Memory Leak Detection
profile_memory_leaks() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Profile 5: Memory Leak Detection${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Profile 5: Memory Leak Detection"
        echo "================================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Monitoring memory growth under load...${NC}"

    # Get initial memory
    local initial_mem=$(docker stats --no-stream --format "{{.MemUsage}}" $(docker-compose ps -q backend) | cut -d'/' -f1)

    echo "Initial memory: ${initial_mem}"
    {
        echo "Memory Growth Analysis:"
        echo "Initial: ${initial_mem}"
        echo ""
    } >> "${REPORT_FILE}"

    # Generate load and monitor
    echo -e "${CYAN}Generating load (60 seconds)...${NC}"

    local API_HOST="${API_HOST:-http://localhost:8000}"
    local API_KEY="${API_KEY:-test-key-1}"

    # Background load generation
    (
        for iteration in {1..60}; do
            for i in {1..10}; do
                curl -s -X POST "${API_HOST}/api/v1/formulas/execute" \
                    -H "X-API-Key: ${API_KEY}" \
                    -H "Content-Type: application/json" \
                    -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
                    > /dev/null 2>&1 &
            done
            sleep 1
        done
    ) &

    LOAD_PID=$!

    # Monitor memory during load
    for i in {1..12}; do
        sleep 5
        local current_mem=$(docker stats --no-stream --format "{{.MemUsage}}" $(docker-compose ps -q backend) | cut -d'/' -f1)
        echo "Sample ${i}: ${current_mem}"
        echo "  ${i} ($(date)): ${current_mem}" >> "${REPORT_FILE}"
    done

    # Wait for load to finish
    wait $LOAD_PID

    # Get final memory
    local final_mem=$(docker stats --no-stream --format "{{.MemUsage}}" $(docker-compose ps -q backend) | cut -d'/' -f1)

    echo ""
    echo "Final memory: ${final_mem}"

    {
        echo ""
        echo "Final: ${final_mem}"
        echo ""
        echo "Analysis:"
        echo "  If memory increased significantly and did not return to baseline,"
        echo "  there may be a memory leak."
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Memory leak detection complete${NC}"
    echo ""
}

# Profile 6: Python Object Tracking
profile_python_objects() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Profile 6: Python Object Tracking${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Profile 6: Python Object Tracking"
        echo "=================================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Creating object tracking script...${NC}"

    # Create object tracker
    cat > /tmp/track_objects.py <<'EOF'
#!/usr/bin/env python3
import gc
import sys
from collections import Counter

def track_objects():
    """Track Python objects in memory"""
    gc.collect()

    # Count objects by type
    objects = gc.get_objects()
    type_counts = Counter(type(obj).__name__ for obj in objects)

    print(f"\nTotal objects in memory: {len(objects)}")
    print(f"\nTop 20 object types:")
    print("-" * 60)
    print(f"{'Type':<40} {'Count':>10}")
    print("-" * 60)

    for obj_type, count in type_counts.most_common(20):
        print(f"{obj_type:<40} {count:>10}")

    # Check for potential leaks (large lists, dicts)
    print(f"\nLarge collections:")
    print("-" * 60)

    large_lists = [obj for obj in objects if isinstance(obj, list) and len(obj) > 1000]
    large_dicts = [obj for obj in objects if isinstance(obj, dict) and len(obj) > 100]

    print(f"Lists with > 1000 items: {len(large_lists)}")
    print(f"Dicts with > 100 items: {len(large_dicts)}")

if __name__ == '__main__':
    track_objects()
EOF

    # Copy and run
    docker cp /tmp/track_objects.py $(docker-compose ps -q backend):/tmp/track_objects.py 2>/dev/null || true

    {
        echo "Python Object Tracking:"
        docker-compose exec -T backend python /tmp/track_objects.py 2>&1 || echo "Could not track objects"
        echo ""
    } >> "${REPORT_FILE}"

    rm -f /tmp/track_objects.py

    echo -e "${GREEN}✓ Object tracking complete${NC}"
    echo ""
}

# Generate summary
generate_summary() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Memory Profiling Complete!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    {
        echo "═══════════════════════════════════════════════════════════"
        echo "Profiling Summary"
        echo "Completed: $(date)"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "Final Memory Stats:"
        docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"
        echo ""
    } >> "${REPORT_FILE}"

    echo "Full report saved to: ${REPORT_FILE}"
    echo ""
    echo -e "${YELLOW}Key Findings:${NC}"

    # Extract some key metrics
    if [ -f "${REPORT_FILE}" ]; then
        echo ""
        echo "Container Memory Usage:"
        tail -n 20 "${REPORT_FILE}" | grep -A 5 "Final Memory Stats" || echo "  See report for details"
    fi

    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review full report: cat ${REPORT_FILE}"
    echo "2. Check for memory growth patterns"
    echo "3. Look for large object allocations"
    echo "4. Run database query optimization: bash performance-testing/profiling/db_query_analyzer.sh"
    echo "5. If leaks detected, use py-spy for deeper analysis:"
    echo "   docker-compose exec backend pip install py-spy"
    echo "   docker-compose exec backend py-spy record -o profile.svg -- python -m uvicorn app.main:app"
    echo ""
}

# Main execution
main() {
    echo -e "${BLUE}Starting memory profiling...${NC}"
    echo ""

    profile_container_memory
    profile_python_memory
    profile_database_memory
    profile_redis_memory
    profile_memory_leaks
    profile_python_objects

    generate_summary
}

# Run main
main

exit 0
