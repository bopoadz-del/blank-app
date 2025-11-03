# Performance Testing Tools

Comprehensive performance testing suite for the Formula API project.

## 📁 Directory Structure

```
performance-testing/
├── load-tests/           # Load testing scripts
│   ├── load_test_ab.sh   # Apache Bench load tests
│   ├── load_test_wrk.sh  # wrk load tests
│   └── lua/              # wrk Lua scripts
│       ├── post.lua
│       ├── post_with_conversion.lua
│       ├── get.lua
│       └── latency.lua
├── stress-tests/         # Stress testing scripts
│   └── stress_test.sh    # Comprehensive stress tests
├── profiling/            # Profiling and optimization tools
│   ├── memory_profiler.sh     # Memory profiling
│   └── db_query_analyzer.sh   # Database query optimization
└── reports/              # Generated test reports
    ├── ab_load_test_*.txt
    ├── wrk_load_test_*.txt
    ├── stress_test_*.txt
    ├── memory_profile_*.txt
    └── db_query_analysis_*.txt
```

## 🚀 Quick Start

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

# 6. View reports
ls -lh performance-testing/reports/
```

## 📊 Testing Tools

### Load Testing

**Apache Bench (ab)**
- Simple, fast load testing
- Clear metrics
- Good for quick tests

```bash
bash performance-testing/load-tests/load_test_ab.sh
```

**wrk**
- Advanced load testing
- Lua scripting support
- Detailed latency percentiles

```bash
bash performance-testing/load-tests/load_test_wrk.sh
```

### Stress Testing

Tests system behavior under extreme conditions:
- Gradual load increase (ramp-up)
- Sudden load spikes
- Sustained high load
- Rate limit breach
- Database connection pool exhaustion
- Memory leak detection

```bash
bash performance-testing/stress-tests/stress_test.sh
```

### Memory Profiling

Identifies memory leaks and optimization opportunities:
- Container memory tracking
- Python memory usage
- Database memory
- Redis memory
- Memory leak detection
- Python object tracking

```bash
bash performance-testing/profiling/memory_profiler.sh
```

### Database Query Optimization

Analyzes and optimizes database performance:
- Database statistics
- Index analysis
- Query performance
- Table bloat
- Connection pool stats
- Cache hit ratios
- Query execution plans

```bash
bash performance-testing/profiling/db_query_analyzer.sh
```

## 📈 Performance Benchmarks

### Target Metrics

**Response Times (ms):**
- Health endpoint: < 10ms (p50), < 50ms (p99)
- List formulas: < 50ms (p50), < 200ms (p99)
- Execute formula: < 100ms (p50), < 1000ms (p99)

**Throughput (req/s):**
- Health endpoint: > 1000 req/s
- List formulas: > 500 req/s
- Execute formula: > 100 req/s

**Resource Usage:**
- CPU: < 50% (normal), < 90% (peak)
- Memory: < 512MB (normal), < 2GB (peak)
- Cache hit ratio: > 99%

## 🔧 Configuration

All scripts support environment variables:

```bash
# API configuration
export API_HOST="http://localhost:8000"
export API_KEY="test-key-1"

# Test duration
export STRESS_DURATION=300  # 5 minutes
export PROFILE_DURATION=600 # 10 minutes

# Sampling
export SAMPLE_INTERVAL=5    # 5 seconds

# Run tests
bash performance-testing/load-tests/load_test_ab.sh
```

## 📝 Reports

All tests generate detailed reports in `performance-testing/reports/`:

```bash
# View latest reports
ls -lt performance-testing/reports/ | head -10

# Search for errors
grep -i "error\|fail" performance-testing/reports/stress_test_*.txt

# View summary
tail -n 50 performance-testing/reports/ab_load_test_*.txt
```

## 🎯 Common Workflows

### Before Deployment

```bash
# Run full test suite
bash performance-testing/load-tests/load_test_ab.sh
bash performance-testing/stress-tests/stress_test.sh
bash performance-testing/profiling/memory_profiler.sh
bash performance-testing/profiling/db_query_analyzer.sh

# Verify no critical issues
grep -i "fail\|error\|critical" performance-testing/reports/*.txt
```

### After Code Changes

```bash
# Quick load test
bash performance-testing/load-tests/load_test_ab.sh

# Compare with previous results
diff performance-testing/reports/ab_load_test_PREV.txt \
     performance-testing/reports/ab_load_test_NEW.txt
```

### Investigating Performance Issues

```bash
# 1. Profile memory
bash performance-testing/profiling/memory_profiler.sh

# 2. Check for leaks
grep -A 20 "Memory Growth" performance-testing/reports/memory_profile_*.txt

# 3. Analyze database
bash performance-testing/profiling/db_query_analyzer.sh

# 4. Check slow queries
grep -A 10 "Query Plan" performance-testing/reports/db_query_analysis_*.txt

# 5. Apply optimizations
docker-compose exec -T db psql -U postgres -d formulas \
    < performance-testing/reports/optimize_database_*.sql
```

## 🐛 Troubleshooting

### Tests fail to start

```bash
# Check services
docker-compose ps

# Check API health
curl http://localhost:8000/health

# View logs
docker-compose logs backend
```

### Out of memory

```bash
# Increase Docker memory
# Docker Desktop → Settings → Resources

# Or reduce test load
STRESS_DURATION=60 bash performance-testing/stress-tests/stress_test.sh
```

### Database connection errors

```bash
# Check database
docker-compose exec db psql -U postgres -c "\l"

# Check connections
docker-compose exec db psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
```

## 📚 Additional Documentation

- **`../PERFORMANCE_TESTING.md`** - Complete performance testing guide
- **`../QUICK_REFERENCE.md`** - Fast command reference
- **`../CI_CD_TESTING.md`** - CI/CD testing procedures

## 🔗 External Resources

- Apache Bench: https://httpd.apache.org/docs/2.4/programs/ab.html
- wrk: https://github.com/wg/wrk
- PostgreSQL Performance: https://www.postgresql.org/docs/current/performance-tips.html
- Python memory_profiler: https://pypi.org/project/memory-profiler/

---

For questions or issues, see the main project README or open an issue on GitHub.
