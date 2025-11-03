#!/bin/bash

# Log Checking Utility
# Analyzes application logs for errors, warnings, and issues

# Configuration
BACKEND_CONTAINER="formula-api-backend"
DB_CONTAINER="formula-api-db"
REDIS_CONTAINER="formula-api-redis"
MLFLOW_CONTAINER="formula-api-mlflow"
LINES="${LINES:-1000}"

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
    echo "  -s, --service SERVICE   Check logs for specific service (backend|db|redis|mlflow|all)"
    echo "  -n, --lines N          Number of lines to check (default: 1000)"
    echo "  -e, --errors           Show only errors"
    echo "  -w, --warnings         Show only warnings"
    echo "  -f, --follow           Follow logs in real-time"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --service backend --errors"
    echo "  $0 --service all --lines 500"
    echo "  $0 --follow"
}

# Parse command line arguments
SERVICE="all"
SHOW_ERRORS_ONLY=false
SHOW_WARNINGS_ONLY=false
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -e|--errors)
            SHOW_ERRORS_ONLY=true
            shift
            ;;
        -w|--warnings)
            SHOW_WARNINGS_ONLY=true
            shift
            ;;
        -f|--follow)
            FOLLOW=true
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Formula API - Log Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Service: $SERVICE"
echo "Lines: $LINES"
echo ""

# Function to analyze logs
analyze_logs() {
    local container=$1
    local service_name=$2

    echo -e "${YELLOW}Analyzing $service_name logs...${NC}"

    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo -e "${RED}✗ Container $container is not running${NC}"
        return 1
    fi

    local logs=$(docker logs "$container" --tail "$LINES" 2>&1)

    # Count errors
    local error_count=$(echo "$logs" | grep -i "error\|exception\|failed" | grep -v "rate limit\|test" | wc -l)
    local warning_count=$(echo "$logs" | grep -i "warning\|warn" | wc -l)
    local info_count=$(echo "$logs" | grep -i "info" | wc -l)

    echo "Summary:"
    echo -e "  ${RED}Errors: $error_count${NC}"
    echo -e "  ${YELLOW}Warnings: $warning_count${NC}"
    echo -e "  ${GREEN}Info: $info_count${NC}"
    echo ""

    # Show errors if requested or if errors exist
    if [ "$SHOW_ERRORS_ONLY" = true ] || [ "$error_count" -gt 0 ]; then
        echo -e "${RED}Errors:${NC}"
        echo "$logs" | grep -i "error\|exception\|failed" | grep -v "rate limit\|test" | tail -20
        echo ""
    fi

    # Show warnings if requested or if warnings exist
    if [ "$SHOW_WARNINGS_ONLY" = true ] || [ "$warning_count" -gt 0 ]; then
        echo -e "${YELLOW}Warnings:${NC}"
        echo "$logs" | grep -i "warning\|warn" | tail -20
        echo ""
    fi

    # Common issues detection
    echo -e "${BLUE}Common Issues:${NC}"

    # Check for database connection issues
    if echo "$logs" | grep -qi "database.*error\|connection.*refused.*postgres"; then
        echo -e "${RED}  ✗ Database connection issues detected${NC}"
    else
        echo -e "${GREEN}  ✓ No database connection issues${NC}"
    fi

    # Check for Redis connection issues
    if echo "$logs" | grep -qi "redis.*error\|connection.*refused.*6379"; then
        echo -e "${RED}  ✗ Redis connection issues detected${NC}"
    else
        echo -e "${GREEN}  ✓ No Redis connection issues${NC}"
    fi

    # Check for rate limiting
    local rate_limit_count=$(echo "$logs" | grep -i "rate limit exceeded" | wc -l)
    if [ "$rate_limit_count" -gt 0 ]; then
        echo -e "${YELLOW}  ⚠ Rate limit exceeded $rate_limit_count times${NC}"
    fi

    # Check for out of memory
    if echo "$logs" | grep -qi "out of memory\|oom"; then
        echo -e "${RED}  ✗ Out of memory issues detected${NC}"
    else
        echo -e "${GREEN}  ✓ No memory issues${NC}"
    fi

    echo ""
}

# Function to follow logs
follow_logs() {
    if [ "$SERVICE" = "all" ]; then
        docker-compose logs -f
    else
        case $SERVICE in
            backend)
                docker logs -f "$BACKEND_CONTAINER"
                ;;
            db)
                docker logs -f "$DB_CONTAINER"
                ;;
            redis)
                docker logs -f "$REDIS_CONTAINER"
                ;;
            mlflow)
                docker logs -f "$MLFLOW_CONTAINER"
                ;;
            *)
                echo "Unknown service: $SERVICE"
                exit 1
                ;;
        esac
    fi
}

# Main logic
if [ "$FOLLOW" = true ]; then
    echo "Following logs (Ctrl+C to stop)..."
    echo ""
    follow_logs
    exit 0
fi

# Analyze logs based on service selection
case $SERVICE in
    backend)
        analyze_logs "$BACKEND_CONTAINER" "Backend"
        ;;
    db)
        analyze_logs "$DB_CONTAINER" "Database"
        ;;
    redis)
        analyze_logs "$REDIS_CONTAINER" "Redis"
        ;;
    mlflow)
        analyze_logs "$MLFLOW_CONTAINER" "MLflow"
        ;;
    all)
        analyze_logs "$BACKEND_CONTAINER" "Backend"
        echo ""
        analyze_logs "$DB_CONTAINER" "Database"
        echo ""
        analyze_logs "$REDIS_CONTAINER" "Redis"
        echo ""
        analyze_logs "$MLFLOW_CONTAINER" "MLflow"
        ;;
    *)
        echo "Unknown service: $SERVICE"
        show_usage
        exit 1
        ;;
esac

echo -e "${BLUE}========================================${NC}"
echo "Log analysis complete"
echo ""
echo "For real-time monitoring:"
echo "  $0 --follow"
echo ""
echo "For more details:"
echo "  docker logs $BACKEND_CONTAINER --tail 1000"
