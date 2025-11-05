# Load Testing Suite

This directory contains load testing scripts for the application using Locust.

## Prerequisites

```bash
pip install locust
```

## Running Load Tests

### Quick Start

```bash
# Run with default settings (100 users, 5 minutes)
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Run headless with 50 users for 2 minutes
locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless --users 50 --spawn-rate 5 --run-time 2m
```

### Predefined Scenarios

```bash
# Smoke test (10 users, 2 minutes)
python tests/load/load_test_config.py smoke

# Baseline load test (50 users, 10 minutes)
python tests/load/load_test_config.py baseline

# Stress test (200 users, 15 minutes)
python tests/load/load_test_config.py stress

# Spike test (500 users, 5 minutes)
python tests/load/load_test_config.py spike

# Endurance test (100 users, 60 minutes)
python tests/load/load_test_config.py endurance
```

### Running with Tags

```bash
# Test only read operations
locust -f tests/load/locustfile.py --tags read

# Test only write operations
locust -f tests/load/locustfile.py --tags write

# Test admin endpoints
locust -f tests/load/locustfile.py --tags admin
```

## Test Results

Results are saved to:
- HTML report: `reports/load_test_<scenario>.html`
- CSV data: `reports/load_test_<scenario>_stats.csv`

## User Classes

### ReasonerUser
Simulates regular users performing typical operations:
- Authentication
- Project management
- Conversations
- Messages
- Notifications
- Reports

### AdminUser
Simulates admin users:
- User management
- System metrics
- Admin operations

### StressTestUser
High-frequency requests for stress testing:
- Rapid health checks
- Fast API calls

## Performance Thresholds

Target metrics:
- P50 Response Time: < 200ms
- P95 Response Time: < 500ms
- P99 Response Time: < 1000ms
- Error Rate: < 1%
- Requests/Second: > 100

## Monitoring During Load Tests

Monitor the application using:

```bash
# Watch API logs
tail -f logs/app.log

# Monitor system resources
htop

# Check database connections
# PostgreSQL: SELECT count(*) FROM pg_stat_activity;
```

## Analyzing Results

1. Check the HTML report for overall statistics
2. Review CSV files for detailed metrics
3. Look for:
   - High response times
   - Error rates
   - Failed requests
   - Resource bottlenecks

## Tips for Load Testing

1. **Start Small**: Begin with smoke tests before large-scale tests
2. **Ramp Gradually**: Use appropriate spawn rates
3. **Monitor Resources**: Watch CPU, memory, and database
4. **Test Production-Like**: Use similar data volumes and configurations
5. **Clean Up**: Remove test data after testing
