# Performance Testing Guide

Comprehensive guide for load testing, stress testing, memory profiling, and database query optimization.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Load Testing](#load-testing)
3. [Stress Testing](#stress-testing)
4. [Memory Profiling](#memory-profiling)
5. [Database Query Optimization](#database-query-optimization)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Interpreting Results](#interpreting-results)
8. [Optimization Strategies](#optimization-strategies)
9. [Continuous Monitoring](#continuous-monitoring)

---

## Overview

Performance testing ensures the Formula API can handle production load and identifies bottlenecks before they impact users.

### Testing Types

| Test Type | Purpose | Tools | Duration |
|-----------|---------|-------|----------|
| **Load Testing** | Measure performance under expected load | Apache Bench, wrk | 5-30 min |
| **Stress Testing** | Find breaking points and recovery | Custom scripts | 5-60 min |
| **Memory Profiling** | Detect memory leaks and optimize usage | memory_profiler, Docker stats | 5-30 min |
| **Query Optimization** | Improve database performance | PostgreSQL tools | 5-15 min |

### Quick Start

```bash
# 1. Ensure services are running
docker-compose up -d

# 2. Run load tests
bash performance-testing/load-tests/load_test_ab.sh
bash performance-testing/load-tests/load_test_wrk.sh

# 3. Run stress tests
bash performance-testing/stress-tests/stress_test.sh

# 4. Profile memory
bash performance-testing/profiling/memory_profiler.sh

# 5. Optimize database
bash performance-testing/profiling/db_query_analyzer.sh

# 6. Review reports
ls -lh performance-testing/reports/
```

---

## Load Testing

Load testing measures how the system performs under expected and peak traffic conditions.

### Apache Bench (ab)

**Best for:** Quick, simple load tests with clear metrics

#### Basic Usage

```bash
# Run all load tests
bash performance-testing/load-tests/load_test_ab.sh

# Custom configuration
API_HOST=http://localhost:8000 \
API_KEY=your-api-key \
bash performance-testing/load-tests/load_test_ab.sh
```

#### Test Scenarios

The script runs 10 different test scenarios:

1. **Health Check - Light Load**: 100 requests, 10 concurrent
2. **Health Check - Moderate Load**: 1,000 requests, 50 concurrent
3. **Health Check - Heavy Load**: 5,000 requests, 100 concurrent
4. **List Formulas - Light Load**: 100 requests, 10 concurrent
5. **List Formulas - Moderate Load**: 1,000 requests, 50 concurrent
6. **Formula Execution - Light Load**: 50 requests, 5 concurrent
7. **Formula Execution - Moderate Load**: 200 requests, 20 concurrent
8. **Formula Execution - Heavy Load**: 500 requests, 50 concurrent
9. **Formula with Unit Conversion**: 200 requests, 20 concurrent
10. **History Endpoint**: 500 requests, 25 concurrent

#### Key Metrics

```bash
# View summary
cat performance-testing/reports/ab_load_test_*.txt | grep "Requests per second"

# Expected values (adjust based on your hardware):
# - Health endpoint: > 1000 req/s
# - List formulas: > 500 req/s
# - Formula execution: > 100 req/s
# - With unit conversion: > 80 req/s
```

### wrk (Modern Load Testing)

**Best for:** Advanced load testing with Lua scripting

#### Basic Usage

```bash
# Run all wrk tests
bash performance-testing/load-tests/load_test_wrk.sh

# Custom duration
PROFILE_DURATION=600 bash performance-testing/load-tests/load_test_wrk.sh
```

#### Test Scenarios

The script runs 10 comprehensive tests:

1. **Health Check - Baseline**: 30s, 4 threads, 50 connections
2. **Health Check - High Concurrency**: 30s, 8 threads, 200 connections
3. **List Formulas - Baseline**: 30s, 4 threads, 50 connections
4. **List Formulas - High Concurrency**: 30s, 8 threads, 200 connections
5. **Formula Execution - Baseline**: 60s, 4 threads, 50 connections
6. **Formula Execution - High Concurrency**: 60s, 8 threads, 200 connections
7. **Formula with Unit Conversion**: 60s, 4 threads, 100 connections
8. **History Endpoint**: 30s, 4 threads, 50 connections
9. **Detailed Latency Analysis**: 60s, 4 threads, 100 connections
10. **Sustained Load Test**: 5 minutes, 4 threads, 50 connections

#### Key Metrics

```bash
# View latency percentiles
cat performance-testing/reports/wrk_load_test_*.txt | grep -A 10 "Latency"

# Expected values:
# - 50th percentile: < 50ms
# - 90th percentile: < 200ms
# - 99th percentile: < 500ms
# - 99.9th percentile: < 1s
```

### Manual Load Testing

For custom scenarios:

```bash
# Simple GET request load test
ab -n 1000 -c 50 \
   -H "X-API-Key: test-key-1" \
   http://localhost:8000/health

# POST request with JSON
ab -n 500 -c 25 \
   -T "application/json" \
   -H "X-API-Key: test-key-1" \
   -p payload.json \
   http://localhost:8000/api/v1/formulas/execute

# wrk with custom Lua script
wrk -t4 -c100 -d30s \
    -s performance-testing/load-tests/lua/post.lua \
    http://localhost:8000/api/v1/formulas/execute
```

---

## Stress Testing

Stress testing pushes the system beyond normal operating conditions to find breaking points.

### Running Stress Tests

```bash
# Run full stress test suite
bash performance-testing/stress-tests/stress_test.sh

# Custom duration (in seconds)
STRESS_DURATION=600 bash performance-testing/stress-tests/stress_test.sh
```

### Test Scenarios

#### Test 1: Gradual Load Increase (Ramp-up)
Tests how the system handles gradually increasing load from 50 to 500 concurrent requests.

**What to Look For:**
- At what concurrency level does the system start slowing down?
- Does response time increase linearly or exponentially?
- Does the system remain stable throughout?

#### Test 2: Spike Test (Sudden Load)
Sends 1,000 simultaneous requests to test recovery from traffic spikes.

**What to Look For:**
- Does the system handle the spike without crashing?
- How long does it take to recover?
- Are there any error responses?

#### Test 3: Sustained High Load
Maintains 100 concurrent requests for the specified duration (default 5 minutes).

**What to Look For:**
- Does memory usage remain stable?
- Are there any memory leaks?
- Does performance degrade over time?

#### Test 4: Rate Limit Breach
Tests rate limiting by sending 100 rapid requests.

**What to Look For:**
- Are 429 (Too Many Requests) errors returned correctly?
- Does rate limiting protect the system?
- How many requests succeed before rate limiting kicks in?

#### Test 5: Database Connection Pool
Tests database connection pool exhaustion with 200 concurrent requests.

**What to Look For:**
- Does the system handle connection pool exhaustion gracefully?
- Are there "too many connections" errors?
- Does the pool recover after load decreases?

#### Test 6: Memory Leak Detection
Monitors memory usage over time while generating load.

**What to Look For:**
- Does memory usage increase steadily?
- Does memory return to baseline after load stops?
- Are there signs of memory leaks?

### Interpreting Stress Test Results

```bash
# View full report
cat performance-testing/reports/stress_test_*.txt

# Check for failures
grep -i "FAILURE\|error\|exception" performance-testing/reports/stress_test_*.txt

# Analyze memory growth
grep -A 5 "Memory Growth" performance-testing/reports/stress_test_*.txt
```

**Success Criteria:**
- ✅ System remains responsive throughout all tests
- ✅ No crashes or unrecoverable errors
- ✅ Memory returns to baseline after load
- ✅ Rate limiting works correctly
- ✅ Recovery time from spikes < 30 seconds

---

## Memory Profiling

Memory profiling identifies memory leaks and optimization opportunities.

### Running Memory Profiler

```bash
# Run full memory profile
bash performance-testing/profiling/memory_profiler.sh

# Custom duration and sampling
PROFILE_DURATION=600 \
SAMPLE_INTERVAL=10 \
bash performance-testing/profiling/memory_profiler.sh
```

### Profiling Components

#### Profile 1: Container Memory Usage
Tracks Docker container memory over time.

```bash
# Check for memory growth
grep "Sample" performance-testing/reports/memory_profile_*.txt
```

#### Profile 2: Python Memory Usage
Uses `memory_profiler` to track Python function memory.

```bash
# View detailed Python memory usage
grep -A 20 "Python Memory Profile" performance-testing/reports/memory_profile_*.txt
```

#### Profile 3: Database Memory Usage
Analyzes PostgreSQL memory and cache usage.

```bash
# Check database cache hit ratio (should be > 99%)
grep -A 5 "Cache Hit Ratio" performance-testing/reports/memory_profile_*.txt
```

#### Profile 4: Redis Memory Usage
Monitors Redis memory and key statistics.

```bash
# View Redis memory info
grep -A 10 "Redis INFO Memory" performance-testing/reports/memory_profile_*.txt
```

#### Profile 5: Memory Leak Detection
Generates load and monitors for memory growth.

```bash
# Check for leaks
grep -A 20 "Memory Growth Analysis" performance-testing/reports/memory_profile_*.txt
```

#### Profile 6: Python Object Tracking
Tracks Python objects in memory.

```bash
# View top object types
grep -A 30 "Python Object Tracking" performance-testing/reports/memory_profile_*.txt
```

### Memory Optimization Tips

**If memory usage is high:**

1. **Check for connection leaks:**
   ```python
   # Ensure database sessions are closed
   try:
       # use session
       pass
   finally:
       session.close()
   ```

2. **Limit result set sizes:**
   ```python
   # Always use LIMIT
   query = session.query(FormulaExecution).limit(1000)
   ```

3. **Clear caches periodically:**
   ```python
   # For Redis
   redis_client.flushdb()
   ```

4. **Use generators for large datasets:**
   ```python
   # Instead of loading all results
   for result in query.yield_per(100):
       process(result)
   ```

### Advanced Memory Profiling

For deeper analysis, use py-spy:

```bash
# Install py-spy in container
docker-compose exec backend pip install py-spy

# Record CPU and memory profile
docker-compose exec backend py-spy record \
    -o profile.svg \
    -- python -m uvicorn app.main:app

# Download and view the SVG
docker cp $(docker-compose ps -q backend):/app/profile.svg ./
```

---

## Database Query Optimization

Database optimization ensures fast query performance and efficient resource usage.

### Running Query Analyzer

```bash
# Run full database analysis
bash performance-testing/profiling/db_query_analyzer.sh

# Results include:
# - Database statistics
# - Index analysis
# - Query performance
# - Table bloat
# - Connection pool stats
# - Cache hit ratios
# - Query execution plans
# - Optimization recommendations
```

### Analysis Components

#### Analysis 1: Database Statistics
Shows database size, table sizes, and row counts.

```bash
# View database size
grep -A 10 "Database Size" performance-testing/reports/db_query_analysis_*.txt
```

#### Analysis 2: Index Analysis
Identifies missing or unused indexes.

```bash
# Check for unused indexes
grep -A 10 "Unused Indexes" performance-testing/reports/db_query_analysis_*.txt

# Check for missing indexes
grep -A 10 "Missing Indexes" performance-testing/reports/db_query_analysis_*.txt
```

**Recommended Indexes:**

```sql
-- Index for filtering by formula_id
CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_id
ON formula_executions(formula_id);

-- Index for sorting by created_at
CREATE INDEX IF NOT EXISTS idx_formula_executions_created_at
ON formula_executions(created_at DESC);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_formula_executions_formula_created
ON formula_executions(formula_id, created_at DESC);

-- Index for filtering successful executions
CREATE INDEX IF NOT EXISTS idx_formula_executions_success
ON formula_executions(success);
```

#### Analysis 3: Query Performance
Tracks slow queries and execution times.

```bash
# View query statistics
grep -A 20 "Query Statistics" performance-testing/reports/db_query_analysis_*.txt
```

#### Analysis 4: Table Bloat
Identifies dead tuples and bloat.

```bash
# Check bloat percentage
grep -A 10 "Dead Tuples and Bloat" performance-testing/reports/db_query_analysis_*.txt
```

**If bloat > 20%:**

```sql
-- Run VACUUM
VACUUM ANALYZE formula_executions;

-- For immediate results
VACUUM FULL formula_executions;  -- Requires table lock
```

#### Analysis 5: Connection Pool
Monitors database connections.

```bash
# View connection stats
grep -A 10 "Connection Statistics" performance-testing/reports/db_query_analysis_*.txt
```

#### Analysis 6: Cache Performance
Measures buffer cache hit ratio.

```bash
# Check cache hit ratio (should be > 99%)
grep -A 5 "Buffer Cache Hit Ratio" performance-testing/reports/db_query_analysis_*.txt
```

**If cache hit ratio < 90%:**

Increase shared_buffers in PostgreSQL:

```yaml
# docker-compose.yml
services:
  db:
    environment:
      - POSTGRES_SHARED_BUFFERS=256MB
      - POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
```

#### Analysis 7: Query Execution Plans
Shows how PostgreSQL executes queries.

```bash
# View execution plans
grep -A 30 "Query Plan" performance-testing/reports/db_query_analysis_*.txt
```

### Applying Optimizations

The analyzer generates an optimization script:

```bash
# Apply recommended optimizations
docker-compose exec -T db psql -U postgres -d formulas \
    < performance-testing/reports/optimize_database_*.sql

# Verify indexes were created
docker-compose exec db psql -U postgres -d formulas -c "\di"
```

### Manual Query Optimization

```sql
-- Analyze a specific query
EXPLAIN ANALYZE
SELECT *
FROM formula_executions
WHERE formula_id = 'beam_deflection_simply_supported'
ORDER BY created_at DESC
LIMIT 10;

-- Look for:
-- - Seq Scan (bad) vs Index Scan (good)
-- - High cost values
-- - Large row counts
```

---

## Performance Benchmarks

Target performance metrics based on typical production workloads.

### Response Time Targets

| Endpoint | 50th %ile | 90th %ile | 99th %ile | Max |
|----------|-----------|-----------|-----------|-----|
| /health | < 10ms | < 20ms | < 50ms | < 100ms |
| /api/v1/formulas/list | < 50ms | < 100ms | < 200ms | < 500ms |
| /api/v1/formulas/execute | < 100ms | < 300ms | < 1s | < 2s |
| /api/v1/formulas/history | < 100ms | < 200ms | < 500ms | < 1s |

### Throughput Targets

| Endpoint | Target req/s | Minimum |
|----------|--------------|---------|
| /health | > 1000 | > 500 |
| /api/v1/formulas/list | > 500 | > 200 |
| /api/v1/formulas/execute | > 100 | > 50 |
| /api/v1/formulas/history | > 200 | > 100 |

### Resource Usage Targets

| Resource | Normal | Warning | Critical |
|----------|--------|---------|----------|
| CPU (backend) | < 50% | > 70% | > 90% |
| Memory (backend) | < 512MB | > 1GB | > 2GB |
| Database connections | < 50 | > 80 | > 95 |
| Cache hit ratio | > 99% | < 95% | < 90% |

### Concurrency Targets

The system should handle:
- **Normal load**: 50 concurrent users
- **Peak load**: 200 concurrent users
- **Burst capacity**: 500 concurrent requests (temporary)

---

## Interpreting Results

### Good Performance Indicators

✅ **Response times are consistent**
- Low standard deviation
- Predictable latency percentiles
- No sudden spikes

✅ **Linear scaling**
- 2x load = ~2x response time
- No exponential degradation

✅ **High cache hit ratio**
- Database: > 99%
- Redis: > 95%

✅ **Low resource usage**
- CPU < 50% under normal load
- Memory stable over time
- Disk I/O < 50%

✅ **No errors**
- All requests successful
- No 500 errors
- Rate limiting works correctly

### Warning Signs

⚠️ **Increasing response times**
- May indicate memory leak
- Or database performance degradation

⚠️ **High CPU usage**
- May need more workers
- Or code optimization

⚠️ **Memory growth**
- Potential memory leak
- Or connection leak

⚠️ **Low cache hit ratio**
- Database needs tuning
- Or queries not optimized

⚠️ **Connection pool exhaustion**
- Too many concurrent requests
- Or connections not being released

### Critical Issues

🔴 **System unresponsive**
- Crashes or hangs
- Out of memory errors
- Database connection failures

🔴 **Data corruption**
- Failed transactions
- Inconsistent state

🔴 **Memory leaks**
- Steady memory growth
- Eventually leads to OOM

---

## Optimization Strategies

### Strategy 1: Index Optimization

**Problem:** Slow database queries

**Solution:**
```sql
-- Add indexes for common query patterns
CREATE INDEX idx_formula_executions_formula_created
ON formula_executions(formula_id, created_at DESC);
```

**Expected Improvement:** 10-100x faster queries

### Strategy 2: Connection Pooling

**Problem:** Too many database connections

**Solution:**
```python
# app/database.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,        # Base connections
    max_overflow=10,      # Additional connections
    pool_pre_ping=True,  # Verify connections
    pool_recycle=3600    # Recycle after 1 hour
)
```

**Expected Improvement:** Better resource usage, fewer connection errors

### Strategy 3: Caching

**Problem:** Repeated expensive computations

**Solution:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_formula_metadata(formula_id):
    # Cached for repeated calls
    return FORMULAS[formula_id]
```

**Expected Improvement:** 100-1000x faster for cached results

### Strategy 4: Async Processing

**Problem:** Slow synchronous operations

**Solution:**
```python
# Use background tasks for non-critical operations
from fastapi import BackgroundTasks

@router.post("/execute")
async def execute_formula(background_tasks: BackgroundTasks):
    result = await execute_async()
    background_tasks.add_task(log_to_mlflow, result)
    return result
```

**Expected Improvement:** 2-10x faster response times

### Strategy 5: Rate Limiting

**Problem:** System overwhelmed by requests

**Solution:**
```python
# Already implemented in app/middleware/rate_limit.py
# Adjust limits based on capacity

RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds
```

**Expected Improvement:** Protects system from abuse

### Strategy 6: Horizontal Scaling

**Problem:** Single instance can't handle load

**Solution:**
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3  # Run 3 instances
    # Add load balancer (nginx, traefik, etc.)
```

**Expected Improvement:** 3x throughput

---

## Continuous Monitoring

Set up continuous monitoring to catch issues early.

### Automated Performance Testing

Add to CI/CD pipeline:

```yaml
# .github/workflows/performance-test.yml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run load tests
        run: bash performance-testing/load-tests/load_test_ab.sh
      - name: Check thresholds
        run: |
          # Fail if p99 latency > 1s
          # Fail if error rate > 1%
```

### Production Monitoring

Use monitoring stack:

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### Key Metrics to Monitor

1. **Response time (p50, p95, p99)**
2. **Request rate (req/s)**
3. **Error rate (%)**
4. **CPU usage (%)**
5. **Memory usage (MB)**
6. **Database connections**
7. **Cache hit ratio**

### Alerting

Set up alerts for:
- Response time p99 > 1s
- Error rate > 1%
- CPU > 80%
- Memory > 1.5GB
- Cache hit ratio < 90%
- Database connections > 80

---

## Quick Reference

```bash
# Run all performance tests
bash performance-testing/load-tests/load_test_ab.sh
bash performance-testing/load-tests/load_test_wrk.sh
bash performance-testing/stress-tests/stress_test.sh
bash performance-testing/profiling/memory_profiler.sh
bash performance-testing/profiling/db_query_analyzer.sh

# View all reports
ls -lh performance-testing/reports/

# Apply database optimizations
docker-compose exec -T db psql -U postgres -d formulas \
    < performance-testing/reports/optimize_database_*.sql

# Monitor resources
docker stats

# Check logs for errors
docker-compose logs backend | grep -i error
```

---

## Troubleshooting

### Issue: Load tests fail immediately

```bash
# Check if services are running
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Check logs
docker-compose logs backend
```

### Issue: Out of memory during tests

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory

# Or reduce test load
API_HOST=http://localhost:8000 \
bash performance-testing/load-tests/load_test_ab.sh
```

### Issue: Database connection errors

```bash
# Check database
docker-compose exec db psql -U postgres -c "\l"

# Increase connection pool
# Edit docker-compose.yml:
environment:
  POSTGRES_MAX_CONNECTIONS: 200
```

### Issue: Rate limiting blocking tests

```bash
# Temporarily increase rate limits
# Edit app/middleware/rate_limit.py:
RATE_LIMIT_REQUESTS = 1000  # Increase for testing
```

---

## Additional Resources

- **Apache Bench Manual**: `man ab`
- **wrk Documentation**: https://github.com/wg/wrk
- **PostgreSQL Performance**: https://www.postgresql.org/docs/current/performance-tips.html
- **FastAPI Performance**: https://fastapi.tiangolo.com/advanced/performance/
- **Python Memory Profiler**: https://pypi.org/project/memory-profiler/

---

For more details, see:
- `QUICK_REFERENCE.md` - Fast command reference
- `CI_CD_TESTING.md` - CI/CD testing procedures
- `PRODUCTION_DEPLOYMENT.md` - Production deployment guide
