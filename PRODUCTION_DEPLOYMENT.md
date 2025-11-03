# Formula Execution API - Production Deployment Checklist

Complete checklist for deploying to staging VPS, monitoring, and maintenance.

---

## 📋 Deployment Checklist

### Step 1: Deploy to Staging VPS ✅

**Preparation:**

```bash
# 1. Configure staging environment
cp .env.example .env.staging
nano .env.staging  # Update API_KEY, SECRET_KEY, and other settings

# 2. Set remote server details
export REMOTE_HOST="your-staging-server.example.com"
export REMOTE_USER="root"

# 3. Ensure SSH access works
ssh $REMOTE_USER@$REMOTE_HOST "echo 'SSH connection OK'"
```

**Deploy:**

```bash
# Run deployment script
bash deployment/deploy-staging.sh
```

**What it does:**
- ✅ Runs pre-deployment checks
- ✅ Executes local tests
- ✅ Builds Docker images
- ✅ Copies files to staging server via rsync
- ✅ Deploys with Docker Compose
- ✅ Automatically runs verification

**Expected Output:**
```
========================================
Deployment Complete!
========================================

API URL: http://your-server:8000
API Docs: http://your-server:8000/docs
MLflow UI: http://your-server:5000
```

---

### Step 2: Run verify-deployment.sh ✅

**On Staging Server:**

```bash
# SSH into staging
ssh $REMOTE_USER@$REMOTE_HOST

# Navigate to app directory
cd /opt/formula-api

# Run verification
bash scripts/verify-deployment.sh
```

**Verification Checks (15 Total):**

1. ✓ Docker containers running
2. ✓ Backend health endpoint
3. ✓ Database connectivity
4. ✓ Redis connectivity
5. ✓ MLflow connectivity
6. ✓ API authentication working
7. ✓ Formula execution successful
8. ✓ Database writes verified
9. ✓ Unit conversion working
10. ✓ Error handling working
11. ✓ API documentation accessible
12. ✓ Disk space OK (<80%)
13. ✓ Memory usage OK (<90%)
14. ✓ No errors in recent logs
15. ✓ Database tables created

**Expected Output:**
```
========================================
Verification Summary
========================================
Passed: 15
Failed: 0
Total: 15

✓ All verification checks passed!
```

**If Verification Fails:**

```bash
# Check specific service logs
bash scripts/check-logs.sh --service backend --errors

# Check Docker status
docker-compose ps

# Restart if needed
docker-compose restart
```

---

### Step 3: Monitor for 24 Hours ✅

**Start Monitoring:**

```bash
# On staging server or from local machine
bash monitoring/monitor.sh
```

**Monitoring Configuration:**

```bash
# Optional: Set alert email
export ALERT_EMAIL="your-email@example.com"

# Optional: Change check interval (default: 5 minutes)
export CHECK_INTERVAL=300

# Start monitoring
bash monitoring/monitor.sh
```

**What Gets Monitored:**

| Check | Frequency | Alert Threshold |
|-------|-----------|-----------------|
| Health endpoint | 5 min | HTTP != 200 |
| Database connection | 5 min | Connection fails |
| Redis connection | 5 min | Ping fails |
| API functionality | 5 min | Execution fails |
| CPU usage | 5 min | > 80% |
| Memory usage | 5 min | > 85% |
| Disk space | 5 min | > 80% |
| Error count | 5 min | > 10 errors/5min |
| Response time | 5 min | > 5000ms |

**Monitoring Output:**

```
========================================
Formula API - 24-Hour Monitoring
========================================
Starting at: 2024-01-01 10:00:00
Check interval: 300s (5 minutes)

========== Check #1 (Elapsed: 0h 0m) ==========
[2024-01-01 10:00:00] ✓ Health check OK
[2024-01-01 10:00:00] ✓ Database OK
[2024-01-01 10:00:00] ✓ Redis OK
[2024-01-01 10:00:00] Metrics: CPU 15%, MEM 45%, DISK 35%
[2024-01-01 10:00:00] ✓ API test OK (234ms)
[2024-01-01 10:00:00] Errors in last 5min: 0
```

**Monitoring Logs Location:**

```
monitoring/logs/
├── health.log          # Health check history
├── database.log        # DB connection history
├── redis.log           # Redis connection history
├── metrics.log         # System metrics over time
├── api_tests.log       # API test results
├── errors.log          # Error count tracking
├── alerts.log          # All alerts triggered
└── report_*.txt        # Generated reports (every 4h)
```

**Real-Time Monitoring (Alternative):**

```bash
# Terminal 1: Backend logs
docker-compose logs -f backend

# Terminal 2: All service logs
docker-compose logs -f

# Terminal 3: System stats
watch -n 5 'docker stats --no-stream'
```

---

### Step 4: Check Logs for Errors ✅

**Automated Log Checking:**

```bash
# Check all services
bash scripts/check-logs.sh --service all

# Check specific service
bash scripts/check-logs.sh --service backend

# Show only errors
bash scripts/check-logs.sh --service backend --errors

# Show only warnings
bash scripts/check-logs.sh --service backend --warnings

# Check last 500 lines
bash scripts/check-logs.sh --service backend --lines 500

# Follow logs in real-time
bash scripts/check-logs.sh --follow
```

**Manual Log Checking:**

```bash
# Backend logs
docker logs formula-api-backend --tail 100

# Database logs
docker logs formula-api-db --tail 100

# Redis logs
docker logs formula-api-redis --tail 100

# MLflow logs
docker logs formula-api-mlflow --tail 100

# Search for errors
docker-compose logs backend | grep -i "error\|exception\|failed"

# Search for warnings
docker-compose logs backend | grep -i "warning\|warn"
```

**Log Analysis Output:**

```
========================================
Formula API - Log Analysis
========================================
Service: backend
Lines: 1000

Analyzing Backend logs...
Summary:
  Errors: 0
  Warnings: 2
  Info: 145

Common Issues:
  ✓ No database connection issues
  ✓ No Redis connection issues
  ⚠ Rate limit exceeded 5 times
  ✓ No memory issues
```

**Common Issues to Look For:**

- Database connection errors
- Redis connection failures
- Rate limit exceeded (high traffic)
- Out of memory errors
- Slow API responses (>5s)
- Unhandled exceptions
- Failed formula executions

---

### Step 5: Test Backup/Restore ✅

#### A. Create Backup

```bash
# Create manual backup
bash backup/backup.sh
```

**What Gets Backed Up:**
- ✅ PostgreSQL database (all tables)
- ✅ Redis data snapshots
- ✅ MLflow artifacts and experiments
- ✅ Configuration files (.env, docker-compose.yml)
- ✅ Backup metadata

**Output:**
```
========================================
Formula API - Backup
========================================
Timestamp: 2024-01-01 12:00:00
Backup name: formula-api-backup-20240101_120000

[1/6] Backing up PostgreSQL database...
✓ Database backup complete (12MB)

[2/6] Backing up Redis data...
✓ Redis backup complete

[3/6] Backing up MLflow artifacts...
✓ MLflow backup complete (5.2MB)

[4/6] Backing up configuration files...
✓ Configuration backup complete

[5/6] Creating backup metadata...
✓ Metadata created

[6/6] Compressing backup...
✓ Backup compressed (8.5MB)

========================================
Backup Complete!
========================================
Backup file: ./backup/backups/formula-api-backup-20240101_120000.tar.gz
Size: 8.5MB
```

**Backup Location:**
```
backup/backups/formula-api-backup-20240101_120000.tar.gz
```

#### B. Verify Backup Contents

```bash
# List backup contents
tar tzf backup/backups/formula-api-backup-20240101_120000.tar.gz
```

#### C. Test Restore

```bash
# Run automated backup/restore test
bash backup/test-backup-restore.sh
```

**What the Test Does:**
1. Creates test data in database
2. Records current data count
3. Creates a backup
4. Adds additional test data
5. Restores from backup
6. Verifies original data is restored
7. Runs full deployment verification

**Test Output:**
```
========================================
Backup/Restore Test
========================================

Step 1: Creating test data...
✓ Test data created (ID: 42)
Current database records: 15

Step 2: Creating backup...
✓ Backup created successfully

Step 3: Verifying backup file...
✓ Backup file exists (8.5M)

Step 4: Creating additional test data...
Database records after new data: 16

Step 5: Testing restore...
✓ Restore completed

Step 6: Verifying restored data...
Database records after restore: 15
✓ Data count matches original backup
✓ Original execution record found

Step 7: Running full deployment verification...
✓ Full verification passed

========================================
Backup/Restore Test Complete!
========================================

Summary:
  Initial records: 15
  Records after new data: 16
  Records after restore: 15

✓ Backup and restore procedures are working correctly!
```

#### D. Manual Restore (If Needed)

```bash
# List available backups
ls -lh backup/backups/

# Restore specific backup
bash backup/restore.sh backup/backups/formula-api-backup-20240101_120000.tar.gz
```

**Restore Process:**
- ⚠️ Warning: Displays confirmation prompt
- Stops all containers
- Extracts backup
- Restores database
- Restores Redis
- Restores MLflow
- Restarts services
- Verifies restoration

#### E. Automated Backup Schedule

**Set up daily backups:**

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * cd /opt/formula-api && bash backup/backup.sh >> backup/backup.log 2>&1

# Add weekly backup to remote location
0 3 * * 0 cd /opt/formula-api && bash backup/backup.sh && rsync -avz backup/backups/ backup-server:/backups/
```

---

## 🎯 Quick Command Reference

### Deployment
```bash
bash deployment/deploy-staging.sh        # Deploy to staging
bash scripts/verify-deployment.sh        # Verify deployment
```

### Monitoring
```bash
bash monitoring/monitor.sh               # Start 24h monitoring
bash scripts/check-logs.sh --service all # Check all logs
docker-compose logs -f backend           # Follow backend logs
```

### Backup & Restore
```bash
bash backup/backup.sh                    # Create backup
bash backup/restore.sh <backup-file>     # Restore backup
bash backup/test-backup-restore.sh       # Test backup/restore
```

### Troubleshooting
```bash
docker-compose ps                        # Check container status
docker-compose restart                   # Restart all services
docker-compose logs backend --tail 100   # View recent logs
bash scripts/check-logs.sh --errors      # Find errors
```

---

## 📊 Success Criteria

### Deployment Success
- ✅ All 15 verification checks pass
- ✅ All 4 Docker containers running
- ✅ API responds to health check
- ✅ Formula execution successful
- ✅ Database writes confirmed

### Monitoring Success (24 Hours)
- ✅ No critical alerts triggered
- ✅ Error count < 10 per 5 minutes
- ✅ CPU usage < 80%
- ✅ Memory usage < 85%
- ✅ Disk usage < 80%
- ✅ API response time < 5 seconds
- ✅ All services remain healthy

### Backup/Restore Success
- ✅ Backup completes without errors
- ✅ Backup file contains all components
- ✅ Restore completes successfully
- ✅ Data integrity verified after restore
- ✅ All services functional after restore

---

## 🚨 Alert Response

### If Monitoring Alerts Trigger

**Database Connection Failed:**
```bash
docker-compose restart db
bash scripts/verify-deployment.sh
```

**Redis Connection Failed:**
```bash
docker-compose restart redis
bash scripts/verify-deployment.sh
```

**High CPU/Memory:**
```bash
docker stats
docker-compose restart backend
```

**High Error Rate:**
```bash
bash scripts/check-logs.sh --service backend --errors
# Investigate and fix root cause
docker-compose restart backend
```

---

## 📚 Additional Resources

- **Deployment Guide:** `deployment/DEPLOYMENT.md`
- **API Documentation:** http://your-server:8000/docs
- **MLflow UI:** http://your-server:5000
- **Test Scripts:** `test_comprehensive.sh`, `test_rate_limit.sh`

---

## ✅ Completion Checklist

After completing all steps, you should have:

- [x] Successfully deployed to staging VPS
- [x] All 15 verification checks passing
- [x] 24-hour monitoring logs collected
- [x] Log analysis completed with no critical errors
- [x] Backup created and verified
- [x] Restore tested and validated
- [x] Automated backup schedule configured
- [x] Alert system configured (if email provided)
- [x] All services running stably for 24 hours
- [x] Documentation reviewed

---

**System is now ready for production deployment! 🚀**
