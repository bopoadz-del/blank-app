#!/bin/bash

# 24-Hour Monitoring Script
# Continuously monitors the Formula Execution API and logs metrics

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-your-api-key-change-this}"
CHECK_INTERVAL=300  # 5 minutes
LOG_DIR="./monitoring/logs"
ALERT_EMAIL="${ALERT_EMAIL:-}"

mkdir -p "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Formula API - 24-Hour Monitoring${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Starting at: $(date)"
echo "Check interval: ${CHECK_INTERVAL}s ($(($CHECK_INTERVAL / 60)) minutes)"
echo "Logs directory: $LOG_DIR"
echo ""

# Function to send alert
send_alert() {
    local message="$1"
    echo -e "${RED}ALERT: $message${NC}"

    # Log alert
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $message" >> "$LOG_DIR/alerts.log"

    # Send email if configured
    if [ -n "$ALERT_EMAIL" ]; then
        echo "$message" | mail -s "Formula API Alert" "$ALERT_EMAIL" 2>/dev/null || true
    fi
}

# Function to check health
check_health() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" 2>&1)

    if [ "$response" = "200" ]; then
        echo "[$timestamp] ✓ Health check OK"
        echo "[$timestamp] OK" >> "$LOG_DIR/health.log"
        return 0
    else
        echo "[$timestamp] ✗ Health check FAILED (HTTP $response)"
        echo "[$timestamp] FAILED - HTTP $response" >> "$LOG_DIR/health.log"
        send_alert "Health check failed - HTTP $response"
        return 1
    fi
}

# Function to check database
check_database() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; then
        echo "[$timestamp] ✓ Database OK"
        echo "[$timestamp] OK" >> "$LOG_DIR/database.log"
        return 0
    else
        echo "[$timestamp] ✗ Database FAILED"
        echo "[$timestamp] FAILED" >> "$LOG_DIR/database.log"
        send_alert "Database connection failed"
        return 1
    fi
}

# Function to check Redis
check_redis() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "[$timestamp] ✓ Redis OK"
        echo "[$timestamp] OK" >> "$LOG_DIR/redis.log"
        return 0
    else
        echo "[$timestamp] ✗ Redis FAILED"
        echo "[$timestamp] FAILED" >> "$LOG_DIR/redis.log"
        send_alert "Redis connection failed"
        return 1
    fi
}

# Function to collect metrics
collect_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

    # Memory usage
    mem_usage=$(free | awk 'NR==2 {printf "%.2f", $3/$2 * 100}')

    # Disk usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

    # Docker container status
    container_count=$(docker-compose ps | grep "Up" | wc -l)

    # Log metrics
    echo "[$timestamp] CPU: ${cpu_usage}%, MEM: ${mem_usage}%, DISK: ${disk_usage}%, CONTAINERS: ${container_count}" >> "$LOG_DIR/metrics.log"

    # Check thresholds
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        send_alert "High CPU usage: ${cpu_usage}%"
    fi

    if (( $(echo "$mem_usage > 85" | bc -l) )); then
        send_alert "High memory usage: ${mem_usage}%"
    fi

    if [ "$disk_usage" -gt 80 ]; then
        send_alert "Low disk space: ${disk_usage}% used"
    fi

    if [ "$container_count" -lt 4 ]; then
        send_alert "Some containers are down (running: $container_count)"
    fi

    echo "[$timestamp] Metrics: CPU ${cpu_usage}%, MEM ${mem_usage}%, DISK ${disk_usage}%"
}

# Function to test API endpoint
test_api() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local start_time=$(date +%s%3N)

    local response=$(curl -s "$API_URL/api/v1/formulas/execute" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
        2>&1)

    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))

    if echo "$response" | grep -q '"success":true'; then
        echo "[$timestamp] ✓ API test OK (${response_time}ms)"
        echo "[$timestamp] OK - ${response_time}ms" >> "$LOG_DIR/api_tests.log"

        # Check response time threshold
        if [ "$response_time" -gt 5000 ]; then
            send_alert "Slow API response: ${response_time}ms"
        fi

        return 0
    else
        echo "[$timestamp] ✗ API test FAILED"
        echo "[$timestamp] FAILED" >> "$LOG_DIR/api_tests.log"
        send_alert "API test execution failed"
        return 1
    fi
}

# Function to check error logs
check_error_logs() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local error_count=$(docker-compose logs backend --tail=100 --since=5m | grep -i "error\|exception" | grep -v "test\|rate limit" | wc -l)

    echo "[$timestamp] Error count (last 5min): $error_count" >> "$LOG_DIR/errors.log"

    if [ "$error_count" -gt 10 ]; then
        send_alert "High error rate: $error_count errors in last 5 minutes"
    fi

    echo "[$timestamp] Errors in last 5min: $error_count"
}

# Function to generate report
generate_report() {
    local report_file="$LOG_DIR/report_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "==================================="
        echo "Monitoring Report"
        echo "Generated: $(date)"
        echo "==================================="
        echo ""

        echo "Health Checks:"
        tail -20 "$LOG_DIR/health.log" 2>/dev/null || echo "No data"
        echo ""

        echo "Metrics (last 10):"
        tail -10 "$LOG_DIR/metrics.log" 2>/dev/null || echo "No data"
        echo ""

        echo "API Tests (last 10):"
        tail -10 "$LOG_DIR/api_tests.log" 2>/dev/null || echo "No data"
        echo ""

        echo "Alerts:"
        tail -20 "$LOG_DIR/alerts.log" 2>/dev/null || echo "No alerts"
        echo ""

        echo "Error Summary:"
        tail -10 "$LOG_DIR/errors.log" 2>/dev/null || echo "No data"
        echo ""
    } > "$report_file"

    echo -e "${GREEN}Report saved: $report_file${NC}"
}

# Main monitoring loop
echo "Starting monitoring loop..."
echo "Press Ctrl+C to stop"
echo ""

iteration=0
start_time=$(date +%s)

while true; do
    iteration=$((iteration + 1))
    elapsed=$(($(date +%s) - start_time))
    elapsed_hours=$((elapsed / 3600))
    elapsed_minutes=$(((elapsed % 3600) / 60))

    echo ""
    echo -e "${BLUE}========== Check #$iteration (Elapsed: ${elapsed_hours}h ${elapsed_minutes}m) ==========${NC}"

    check_health
    check_database
    check_redis
    collect_metrics
    test_api
    check_error_logs

    # Generate report every 4 hours
    if [ $((iteration % 48)) -eq 0 ]; then
        generate_report
    fi

    # Stop after 24 hours
    if [ "$elapsed" -ge 86400 ]; then
        echo ""
        echo -e "${GREEN}24-hour monitoring period complete!${NC}"
        generate_report
        break
    fi

    echo "Next check in ${CHECK_INTERVAL}s..."
    sleep "$CHECK_INTERVAL"
done

echo ""
echo "Monitoring stopped at: $(date)"
echo "Logs saved in: $LOG_DIR"
