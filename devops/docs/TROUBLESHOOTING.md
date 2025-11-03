# 🔧 Troubleshooting Quick Reference

**Quick solutions for common issues**

---

## 🚨 Common Issues

### 1. "Connection refused" on port 8000

**Cause:** Backend service not running

**Solution:**
```bash
# Check if running
docker-compose ps backend

# View logs
docker-compose logs backend

# Restart
docker-compose restart backend
```

---

### 2. "Database connection failed"

**Cause:** PostgreSQL not ready or wrong credentials

**Solution:**
```bash
# Check database status
docker-compose ps postgres

# Test connection
docker-compose exec postgres psql -U reasoner_user -d reasoner_db -c "SELECT 1"

# Verify .env settings
grep DATABASE_URL .env
grep POSTGRES_ .env

# Restart database
docker-compose restart postgres
```

---

### 3. "401 Unauthorized" on API calls

**Cause:** Missing or invalid API key

**Solution:**
```bash
# Check API key in .env
grep API_KEY .env

# Verify it's enabled
grep API_KEY_ENABLED .env

# Test with correct header
curl -H "X-API-Key: YOUR_KEY_HERE" http://localhost:8000/health

# Temporarily disable for testing
# Edit .env: API_KEY_ENABLED=false
# Then: docker-compose restart backend
```

---

### 4. Service crashes immediately

**Cause:** Configuration error or missing .env

**Solution:**
```bash
# Check logs for error
docker-compose logs backend | tail -50

# Verify .env exists
ls -la .env

# Check for syntax errors in .env
cat .env | grep -v "^#" | grep "="

# Start with defaults
mv .env .env.backup
cp .env.example .env
# Edit .env with correct values
docker-compose up -d
```

---

### 5. "Out of memory" errors

**Cause:** Insufficient resources

**Solution:**
```bash
# Check memory usage
docker stats

# Reduce workers in .env
WORKERS=2

# Add memory limit to docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G

# Restart
docker-compose restart backend
```

---

### 6. Slow response times

**Cause:** Too many concurrent requests or slow queries

**Solution:**
```bash
# Check CPU/memory
docker stats

# Increase workers (if CPU available)
# Edit .env: WORKERS=8
# Restart: docker-compose restart backend

# Check slow queries
docker-compose logs backend | grep "duration"

# Enable query logging
# docker-compose.yml postgres:
command: ["-c", "log_min_duration_statement=1000"]

# Add indexes if needed
docker-compose exec backend alembic revision --autogenerate -m "add_indexes"
```

---

### 7. "Formula not found" errors

**Cause:** Database not initialized

**Solution:**
```bash
# Check formula count
docker-compose exec backend python -c "
from app.core.database import SessionLocal
from app.models.database import Formula
db = SessionLocal()
print(f'Formulas in DB: {db.query(Formula).count()}')
"

# Load formulas
docker-compose exec backend python -m app.core.init_db

# Verify
curl http://localhost:8000/api/v1/formulas | jq '.[] | .formula_id'
```

---

### 8. "Redis connection failed"

**Cause:** Redis not running or wrong password

**Solution:**
```bash
# Check Redis status
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli -a YOUR_REDIS_PASSWORD ping

# Verify password in .env
grep REDIS_PASSWORD .env

# Restart Redis
docker-compose restart redis
```

---

### 9. Migrations fail

**Cause:** Schema conflicts or migration errors

**Solution:**
```bash
# Check current version
docker-compose exec backend alembic current

# View migration history
docker-compose exec backend alembic history

# If stuck, drop and recreate
docker-compose down -v
docker-compose up -d postgres
sleep 10
docker-compose exec postgres psql -U reasoner_user -c "DROP DATABASE reasoner_db;"
docker-compose exec postgres psql -U reasoner_user -c "CREATE DATABASE reasoner_db;"
docker-compose up -d backend
docker-compose exec backend alembic upgrade head
docker-compose exec backend python -m app.core.init_db
```

---

### 10. High disk usage

**Cause:** Large logs or backups

**Solution:**
```bash
# Check disk space
df -h
du -sh data/ logs/

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Clean old backups (keep 30 days)
find data/backups/ -name "*.sql.gz" -mtime +30 -delete

# Limit log size in docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## 🔍 Diagnostic Commands

### Check Everything
```bash
# Service status
docker-compose ps

# Resource usage
docker stats --no-stream

# Recent logs
docker-compose logs --tail=100

# Health check
curl http://localhost:8000/health | jq

# Formula count
curl http://localhost:8000/api/v1/formulas | jq 'length'
```

### View Logs
```bash
# All logs
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Last 50 lines
docker-compose logs --tail=50 backend

# Errors only
docker-compose logs backend 2>&1 | grep -i error

# With timestamps
docker-compose logs -f -t backend
```

### Database Checks
```bash
# Connection test
docker-compose exec postgres pg_isready

# Table list
docker-compose exec postgres psql -U reasoner_user -d reasoner_db -c "\dt"

# Row counts
docker-compose exec postgres psql -U reasoner_user -d reasoner_db -c "
SELECT 'formulas' as table, COUNT(*) FROM formulas
UNION ALL
SELECT 'executions', COUNT(*) FROM formula_executions;"

# Active connections
docker-compose exec postgres psql -U reasoner_user -d reasoner_db -c "
SELECT count(*) as connections FROM pg_stat_activity;"
```

### Performance Checks
```bash
# Response time test
time curl -s http://localhost:8000/health > /dev/null

# Load test (simple)
ab -n 100 -c 10 http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics | grep reasoner
```

---

## 🛠️ Quick Fixes

### Full Reset
```bash
# Nuclear option - resets everything
docker-compose down -v
rm -rf data/postgres data/redis
docker-compose up -d
sleep 15
docker-compose exec backend alembic upgrade head
docker-compose exec backend python -m app.core.init_db
```

### Config Reload
```bash
# After changing .env
docker-compose restart backend

# After changing docker-compose.yml
docker-compose down
docker-compose up -d
```

### Emergency Stop
```bash
# Stop all services
docker-compose stop

# Force stop
docker-compose kill

# Remove containers
docker-compose down
```

---

## 📞 When to Escalate

Contact backend developer if:
- ❌ Data corruption suspected
- ❌ Security breach detected  
- ❌ Persistent crashes after restart
- ❌ Unknown error messages
- ❌ Need to modify code
- ❌ Database schema changes needed

---

## 🔗 Quick Links

- Health: `http://localhost:8000/health`
- Docs: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`
- Deployment Guide: `devops/docs/DEPLOYMENT.md`

---

**Updated:** 2025-11-03  
**Version:** 1.0.0
