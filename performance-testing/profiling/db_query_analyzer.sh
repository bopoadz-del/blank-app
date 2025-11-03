#!/bin/bash

################################################################################
# Database Query Analyzer and Optimizer
# Analyzes slow queries and provides optimization recommendations
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
REPORT_DIR="performance-testing/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/db_query_analysis_${TIMESTAMP}.txt"

mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Database Query Analyzer & Optimizer${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Report: ${REPORT_FILE}"
echo ""

# Initialize report
{
    echo "═══════════════════════════════════════════════════════════"
    echo "Database Query Analysis Report"
    echo "Generated: $(date)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
} > "${REPORT_FILE}"

# Check if database is accessible
if ! docker-compose exec -T db psql -U postgres -c "\l" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to database${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Database connection established${NC}"
echo ""

# Analysis 1: Database Statistics
analyze_database_stats() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 1: Database Statistics${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 1: Database Statistics"
        echo "================================"
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Collecting database statistics...${NC}"

    {
        echo "Database Size:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                pg_size_pretty(pg_database_size('formulas')) as size;
        " 2>&1

        echo ""
        echo "Table Statistics:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
                pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS indexes_size,
                n_live_tup AS row_count,
                n_dead_tup AS dead_rows
            FROM pg_tables
            LEFT JOIN pg_stat_user_tables USING (schemaname, tablename)
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Database statistics collected${NC}"
    echo ""
}

# Analysis 2: Index Analysis
analyze_indexes() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 2: Index Analysis${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 2: Index Analysis"
        echo "=========================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Analyzing indexes...${NC}"

    {
        echo "Current Indexes:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                schemaname,
                tablename,
                indexname,
                pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
                idx_scan AS times_used,
                idx_tup_read AS tuples_read,
                idx_tup_fetch AS tuples_fetched
            FROM pg_indexes
            LEFT JOIN pg_stat_user_indexes USING (schemaname, tablename, indexname)
            WHERE schemaname = 'public'
            ORDER BY pg_relation_size(indexrelid) DESC;
        " 2>&1

        echo ""
        echo "Unused Indexes (never scanned):"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                schemaname,
                tablename,
                indexname,
                pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
            FROM pg_indexes
            LEFT JOIN pg_stat_user_indexes USING (schemaname, tablename, indexname)
            WHERE schemaname = 'public'
              AND idx_scan = 0
            ORDER BY pg_relation_size(indexrelid) DESC;
        " 2>&1

        echo ""
        echo "Missing Indexes (tables with sequential scans):"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                schemaname,
                tablename,
                seq_scan AS sequential_scans,
                seq_tup_read AS rows_read,
                idx_scan AS index_scans,
                n_live_tup AS table_rows,
                CASE
                    WHEN seq_scan = 0 THEN 0
                    ELSE ROUND((seq_tup_read::numeric / seq_scan), 2)
                END AS avg_rows_per_seq_scan
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
              AND seq_scan > 0
            ORDER BY seq_scan DESC;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Index analysis complete${NC}"
    echo ""
}

# Analysis 3: Query Performance
analyze_query_performance() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 3: Query Performance${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 3: Query Performance"
        echo "============================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Enabling pg_stat_statements extension...${NC}"

    # Try to enable pg_stat_statements (may already be enabled or unavailable)
    docker-compose exec -T db psql -U postgres -d formulas -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" 2>&1 | grep -v "already exists" || true

    echo -e "${CYAN}Analyzing query performance...${NC}"

    {
        echo "Query Statistics (if pg_stat_statements is available):"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
            ) as extension_enabled;
        " 2>&1

        # If extension is available, show query stats
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                query,
                calls,
                ROUND(total_exec_time::numeric, 2) as total_time_ms,
                ROUND(mean_exec_time::numeric, 2) as mean_time_ms,
                ROUND(max_exec_time::numeric, 2) as max_time_ms,
                rows
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname = 'formulas')
            ORDER BY total_exec_time DESC
            LIMIT 10;
        " 2>&1 || echo "pg_stat_statements not available (requires configuration)"

        echo ""
        echo "Active Queries:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                pid,
                usename,
                application_name,
                client_addr,
                state,
                query,
                EXTRACT(EPOCH FROM (now() - query_start)) as query_duration_sec
            FROM pg_stat_activity
            WHERE datname = 'formulas'
              AND state != 'idle'
              AND query NOT LIKE '%pg_stat_activity%'
            ORDER BY query_start;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Query performance analysis complete${NC}"
    echo ""
}

# Analysis 4: Table Bloat Analysis
analyze_table_bloat() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 4: Table Bloat${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 4: Table Bloat"
        echo "======================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Analyzing table bloat...${NC}"

    {
        echo "Dead Tuples and Bloat:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                schemaname,
                tablename,
                n_live_tup AS live_tuples,
                n_dead_tup AS dead_tuples,
                CASE
                    WHEN n_live_tup > 0
                    THEN ROUND((n_dead_tup::numeric / n_live_tup * 100), 2)
                    ELSE 0
                END AS bloat_percentage,
                last_vacuum,
                last_autovacuum
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY n_dead_tup DESC;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Table bloat analysis complete${NC}"
    echo ""
}

# Analysis 5: Connection Pool Analysis
analyze_connections() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 5: Connection Pool${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 5: Connection Pool"
        echo "==========================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Analyzing database connections...${NC}"

    {
        echo "Connection Statistics:"
        docker-compose exec -T db psql -U postgres -c "
            SELECT
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active,
                count(*) FILTER (WHERE state = 'idle') as idle,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
                count(*) FILTER (WHERE wait_event IS NOT NULL) as waiting
            FROM pg_stat_activity;
        " 2>&1

        echo ""
        echo "Connections by Database:"
        docker-compose exec -T db psql -U postgres -c "
            SELECT
                datname,
                count(*) as connections,
                max(EXTRACT(EPOCH FROM (now() - query_start))) as longest_query_sec
            FROM pg_stat_activity
            WHERE datname IS NOT NULL
            GROUP BY datname
            ORDER BY connections DESC;
        " 2>&1

        echo ""
        echo "Long Running Connections:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                pid,
                usename,
                application_name,
                state,
                EXTRACT(EPOCH FROM (now() - query_start)) as duration_sec,
                query
            FROM pg_stat_activity
            WHERE datname = 'formulas'
              AND query_start < now() - interval '30 seconds'
              AND state != 'idle'
            ORDER BY query_start;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Connection analysis complete${NC}"
    echo ""
}

# Analysis 6: Cache Hit Ratio
analyze_cache_performance() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 6: Cache Performance${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 6: Cache Performance"
        echo "============================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Analyzing cache hit ratios...${NC}"

    {
        echo "Buffer Cache Hit Ratio (should be > 99%):"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                sum(heap_blks_read) as heap_read,
                sum(heap_blks_hit) as heap_hit,
                CASE
                    WHEN sum(heap_blks_hit) + sum(heap_blks_read) = 0 THEN 0
                    ELSE ROUND(
                        sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100,
                        2
                    )
                END as cache_hit_ratio_percent
            FROM pg_statio_user_tables;
        " 2>&1

        echo ""
        echo "Index Cache Hit Ratio:"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            SELECT
                sum(idx_blks_read) as idx_read,
                sum(idx_blks_hit) as idx_hit,
                CASE
                    WHEN sum(idx_blks_hit) + sum(idx_blks_read) = 0 THEN 0
                    ELSE ROUND(
                        sum(idx_blks_hit) / (sum(idx_blks_hit) + sum(idx_blks_read)) * 100,
                        2
                    )
                END as index_cache_hit_ratio_percent
            FROM pg_statio_user_indexes;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Cache performance analysis complete${NC}"
    echo ""
}

# Analysis 7: Explain Query Plans
analyze_query_plans() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Analysis 7: Query Execution Plans${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "Analysis 7: Query Execution Plans"
        echo "=================================="
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${CYAN}Analyzing query execution plans...${NC}"

    {
        echo "Query Plan: SELECT recent executions"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            EXPLAIN ANALYZE
            SELECT *
            FROM formula_executions
            ORDER BY created_at DESC
            LIMIT 10;
        " 2>&1

        echo ""
        echo "Query Plan: SELECT by formula_id"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            EXPLAIN ANALYZE
            SELECT *
            FROM formula_executions
            WHERE formula_id = 'beam_deflection_simply_supported'
            ORDER BY created_at DESC
            LIMIT 10;
        " 2>&1

        echo ""
        echo "Query Plan: COUNT executions"
        docker-compose exec -T db psql -U postgres -d formulas -c "
            EXPLAIN ANALYZE
            SELECT COUNT(*)
            FROM formula_executions;
        " 2>&1

        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Query plan analysis complete${NC}"
    echo ""
}

# Generate Optimization Recommendations
generate_recommendations() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Generating Optimization Recommendations${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    {
        echo "═══════════════════════════════════════════════════════════"
        echo "Optimization Recommendations"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "1. Add Indexes"
        echo "--------------"
        echo "Consider adding these indexes to improve query performance:"
        echo ""
        echo "-- Index for filtering by formula_id"
        echo "CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_id"
        echo "ON formula_executions(formula_id);"
        echo ""
        echo "-- Index for sorting by created_at (most recent first)"
        echo "CREATE INDEX IF NOT EXISTS idx_formula_executions_created_at"
        echo "ON formula_executions(created_at DESC);"
        echo ""
        echo "-- Composite index for formula_id + created_at (common query pattern)"
        echo "CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_created"
        echo "ON formula_executions(formula_id, created_at DESC);"
        echo ""
        echo "-- Index for filtering successful executions"
        echo "CREATE INDEX IF NOT EXISTS idx_formula_executions_success"
        echo "ON formula_executions(success);"
        echo ""
        echo "2. Query Optimization"
        echo "---------------------"
        echo "Optimize your queries with these patterns:"
        echo ""
        echo "-- Use LIMIT for pagination"
        echo "SELECT * FROM formula_executions"
        echo "WHERE formula_id = 'your_formula'"
        echo "ORDER BY created_at DESC"
        echo "LIMIT 100 OFFSET 0;"
        echo ""
        echo "-- Use WHERE clauses to filter early"
        echo "SELECT * FROM formula_executions"
        echo "WHERE success = true"
        echo "  AND created_at > NOW() - INTERVAL '7 days'"
        echo "ORDER BY created_at DESC;"
        echo ""
        echo "3. Maintenance Tasks"
        echo "--------------------"
        echo "Regular maintenance improves performance:"
        echo ""
        echo "-- Manually run VACUUM to clean up dead tuples"
        echo "VACUUM ANALYZE formula_executions;"
        echo ""
        echo "-- Rebuild indexes if needed"
        echo "REINDEX TABLE formula_executions;"
        echo ""
        echo "-- Update table statistics"
        echo "ANALYZE formula_executions;"
        echo ""
        echo "4. Connection Pool Settings"
        echo "---------------------------"
        echo "Update docker-compose.yml database settings:"
        echo ""
        echo "environment:"
        echo "  POSTGRES_MAX_CONNECTIONS: 100"
        echo "  POSTGRES_SHARED_BUFFERS: 256MB"
        echo "  POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB"
        echo ""
        echo "5. Application Level Optimization"
        echo "----------------------------------"
        echo "Optimize your application code:"
        echo ""
        echo "-- Use connection pooling in SQLAlchemy"
        echo "engine = create_engine("
        echo "    DATABASE_URL,"
        echo "    pool_size=20,"
        echo "    max_overflow=0,"
        echo "    pool_pre_ping=True"
        echo ")"
        echo ""
        echo "-- Add caching for frequently accessed data"
        echo "from functools import lru_cache"
        echo ""
        echo "@lru_cache(maxsize=128)"
        echo "def get_formula_info(formula_id):"
        echo "    # cached lookup"
        echo "    pass"
        echo ""
        echo "6. Monitoring"
        echo "-------------"
        echo "Set up continuous monitoring:"
        echo ""
        echo "-- Enable pg_stat_statements"
        echo "# Add to postgresql.conf:"
        echo "shared_preload_libraries = 'pg_stat_statements'"
        echo "pg_stat_statements.track = all"
        echo ""
        echo "-- Monitor slow queries"
        echo "log_min_duration_statement = 1000  # Log queries > 1s"
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Recommendations generated${NC}"
    echo ""
}

# Generate SQL optimization script
generate_optimization_script() {
    echo -e "${YELLOW}Generating optimization SQL script...${NC}"

    local OPT_SCRIPT="${REPORT_DIR}/optimize_database_${TIMESTAMP}.sql"

    cat > "${OPT_SCRIPT}" <<'EOF'
-- Database Optimization Script
-- Generated by db_query_analyzer.sh
-- Run this script to apply recommended optimizations

-- 1. Create recommended indexes
CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_id
ON formula_executions(formula_id);

CREATE INDEX IF NOT EXISTS idx_formula_executions_created_at
ON formula_executions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_created
ON formula_executions(formula_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_formula_executions_success
ON formula_executions(success);

-- 2. Update table statistics
ANALYZE formula_executions;

-- 3. Clean up dead tuples
VACUUM ANALYZE formula_executions;

-- 4. Show index usage after creation
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as times_used
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Done!
EOF

    echo "CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_id ON formula_executions(formula_id);" >> "${OPT_SCRIPT}"

    {
        echo ""
        echo "Optimization Script Created: ${OPT_SCRIPT}"
        echo ""
        echo "To apply optimizations, run:"
        echo "docker-compose exec -T db psql -U postgres -d formulas < ${OPT_SCRIPT}"
        echo ""
    } >> "${REPORT_FILE}"

    echo -e "${GREEN}✓ Optimization script created: ${OPT_SCRIPT}${NC}"
    echo ""
}

# Generate summary
generate_summary() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Database Analysis Complete!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    {
        echo "═══════════════════════════════════════════════════════════"
        echo "Analysis Complete"
        echo "Timestamp: $(date)"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
    } >> "${REPORT_FILE}"

    echo "Full report saved to: ${REPORT_FILE}"
    echo ""
    echo -e "${YELLOW}Quick Summary:${NC}"

    # Extract key metrics
    if [ -f "${REPORT_FILE}" ]; then
        echo ""
        echo "Database Size:"
        grep -A 3 "Database Size:" "${REPORT_FILE}" | tail -n 2
        echo ""
        echo "Cache Hit Ratio:"
        grep -A 5 "Buffer Cache Hit Ratio" "${REPORT_FILE}" | tail -n 4
        echo ""
    fi

    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review full report: cat ${REPORT_FILE}"
    echo "2. Apply optimizations: docker-compose exec -T db psql -U postgres -d formulas < ${REPORT_DIR}/optimize_database_${TIMESTAMP}.sql"
    echo "3. Monitor query performance after optimizations"
    echo "4. Run performance tests to measure improvements"
    echo ""
}

# Main execution
main() {
    analyze_database_stats
    analyze_indexes
    analyze_query_performance
    analyze_table_bloat
    analyze_connections
    analyze_cache_performance
    analyze_query_plans
    generate_recommendations
    generate_optimization_script
    generate_summary
}

# Run main
main

exit 0
