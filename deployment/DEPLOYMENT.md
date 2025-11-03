# Formula Execution API - Deployment Guide

This guide covers deploying the Formula Execution API to a staging VPS, monitoring, and maintenance procedures.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Verification](#verification)
- [Monitoring](#monitoring)
- [Log Management](#log-management)
- [Backup and Restore](#backup-and-restore)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Local Machine
- Docker and Docker Compose
- SSH access to staging server
- Python 3.11+ with pytest
- Git

### Staging Server
- Ubuntu 20.04+ or similar Linux distribution
- Docker and Docker Compose installed
- At least 2GB RAM
- At least 20GB free disk space
- Open ports: 8000 (API), 5000 (MLflow), 5432 (PostgreSQL), 6379 (Redis)

## Deployment Steps

### 1. Configure Environment

Create `.env.staging` file with production values:

```bash
# Copy example
cp .env.example .env.staging

# Edit with production values
nano .env.staging
```

**Important settings to change:**
- `API_KEY` - Generate a strong API key
- `SECRET_KEY` - Generate a strong secret key
- `ENVIRONMENT=production`
- `DATABASE_URL` - Verify PostgreSQL connection
- `REDIS_HOST` - Verify Redis connection

### 2. Set Remote Server Details

Export your staging server details:

```bash
export REMOTE_HOST="your-staging-server.com"
export REMOTE_USER="root"  # or your user with sudo
```

### 3. Run Deployment Script

```bash
bash deployment/deploy-staging.sh
```

The script will:
1. Run pre-deployment checks
2. Run tests locally
3. Build Docker images
4. Copy files to staging server
5. Deploy with Docker Compose
6. Run verification checks

### 4. Verify Deployment

```bash
# On staging server
ssh $REMOTE_USER@$REMOTE_HOST
cd /opt/formula-api
bash scripts/verify-deployment.sh
```

Expected output: All 15 checks should pass.

## Verification

### Manual Verification Steps

#### 1. Check Services Status

```bash
docker-compose ps
```

All services should show "Up" status.

#### 2. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"healthy","version":"1.0.0",...}`

#### 3. Test Formula Execution

```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001}
  }'
```

Expected: `{"success":true,"result":0.00065104,...}`

#### 4. Verify Database Persistence

```bash
curl http://localhost:8000/api/v1/formulas/history/recent?limit=5 \
  -H "X-API-Key: YOUR_API_KEY"
```

Should return recent executions.

#### 5. Check MLflow UI

```bash
open http://YOUR_STAGING_SERVER:5000
```

Should show MLflow tracking interface.

## Monitoring

### 24-Hour Monitoring

Start continuous monitoring for 24 hours:

```bash
bash monitoring/monitor.sh
```

This script will:
- Check health every 5 minutes
- Test API endpoints
- Monitor database and Redis
- Collect system metrics (CPU, memory, disk)
- Log errors and warnings
- Send alerts for issues
- Generate reports every 4 hours

**Monitoring Logs Location:**
```
monitoring/logs/
├── health.log          # Health check results
├── database.log        # Database connectivity
├── redis.log           # Redis connectivity
├── metrics.log         # System metrics
├── api_tests.log       # API test results
├── errors.log          # Error counts
└── alerts.log          # Alert history
```

### Real-time Monitoring

For real-time monitoring, use multiple terminal windows:

**Terminal 1 - Backend Logs:**
```bash
docker-compose logs -f backend
```

**Terminal 2 - All Services:**
```bash
docker-compose logs -f
```

**Terminal 3 - System Metrics:**
```bash
watch -n 5 'docker stats --no-stream'
```

### Alert Configuration

To receive email alerts, set the `ALERT_EMAIL` environment variable:

```bash
export ALERT_EMAIL="your-email@example.com"
bash monitoring/monitor.sh
```

## Log Management

### Check Logs Script

The `check-logs.sh` script provides comprehensive log analysis:

```bash
# Check all services
bash scripts/check-logs.sh --service all

# Check specific service
bash scripts/check-logs.sh --service backend

# Show only errors
bash scripts/check-logs.sh --service backend --errors

# Follow logs in real-time
bash scripts/check-logs.sh --follow

# Check last 500 lines
bash scripts/check-logs.sh --lines 500
```

### Common Log Locations

**Docker Logs:**
```bash
docker logs formula-api-backend
docker logs formula-api-db
docker logs formula-api-redis
docker logs formula-api-mlflow
```

**Application Logs:**
```bash
# View last 100 lines
docker-compose logs backend --tail 100

# Follow logs
docker-compose logs -f backend

# Search for errors
docker-compose logs backend | grep -i error
```

### Log Rotation

Docker automatically rotates logs. Configure in `docker-compose.yml`:

```yaml
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Backup and Restore

### Create Backup

```bash
bash backup/backup.sh
```

This creates a compressed backup containing:
- PostgreSQL database dump
- Redis data
- MLflow artifacts
- Configuration files
- Metadata

**Backup Location:** `backup/backups/formula-api-backup-TIMESTAMP.tar.gz`

**Retention:** Backups older than 7 days are automatically deleted.

### Restore from Backup

```bash
bash backup/restore.sh backup/backups/formula-api-backup-20240101_120000.tar.gz
```

**Warning:** This will replace existing data!

### Test Backup/Restore

Verify backup and restore procedures work correctly:

```bash
bash backup/test-backup-restore.sh
```

This script:
1. Creates test data
2. Creates a backup
3. Adds more data
4. Restores from backup
5. Verifies original data is restored

### Automated Backup Schedule

Set up cron job for automatic backups:

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * cd /opt/formula-api && bash backup/backup.sh >> backup/backup.log 2>&1
```

### Off-site Backup

For disaster recovery, copy backups off-site:

```bash
# Sync to remote server
rsync -avz backup/backups/ backup-server:/backups/formula-api/

# Or upload to S3
aws s3 sync backup/backups/ s3://your-bucket/formula-api-backups/
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs backend

# Verify configuration
cat .env

# Restart services
docker-compose restart
```

### Database Connection Failed

```bash
# Check database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Test connection
docker-compose exec db psql -U postgres -d formulas -c "SELECT 1;"

# Restart database
docker-compose restart db
```

### Redis Connection Failed

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

### High Memory Usage

```bash
# Check container stats
docker stats

# Restart specific service
docker-compose restart backend

# Or restart all services
docker-compose restart
```

### Disk Space Issues

```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a

# Remove old logs
docker-compose logs --tail=0
```

### API Slow Response

```bash
# Check system resources
top
free -h
df -h

# Check database performance
docker-compose exec db psql -U postgres -d formulas -c "
  SELECT * FROM pg_stat_activity WHERE datname = 'formulas';
"

# Restart services
docker-compose restart
```

## Security Checklist

- [ ] Change default API_KEY and SECRET_KEY
- [ ] Set ENVIRONMENT=production
- [ ] Configure firewall rules
- [ ] Enable HTTPS/SSL
- [ ] Regularly update Docker images
- [ ] Monitor logs for suspicious activity
- [ ] Implement rate limiting
- [ ] Regular security audits
- [ ] Keep backups encrypted
- [ ] Restrict SSH access

## Performance Optimization

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_formula_executions_created_at ON formula_executions(created_at);
CREATE INDEX idx_formula_executions_formula_id ON formula_executions(formula_id);

-- Vacuum database
VACUUM ANALYZE;
```

### Redis Optimization

Configure Redis persistence in `docker-compose.yml`:

```yaml
redis:
  command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Docker Optimization

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
```

## Maintenance Schedule

### Daily
- Check monitoring logs
- Verify services are running
- Review error logs

### Weekly
- Run full verification
- Test backup restore
- Review disk space
- Update dependencies

### Monthly
- Security audit
- Performance review
- Update documentation
- Disaster recovery drill

## Support

For issues or questions:
- Check logs: `bash scripts/check-logs.sh`
- Run verification: `bash scripts/verify-deployment.sh`
- Review documentation: `deployment/DEPLOYMENT.md`
- Contact: support@example.com
