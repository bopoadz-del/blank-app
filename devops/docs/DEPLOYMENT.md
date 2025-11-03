# 🚀 DevOps Deployment Guide

**The Reasoner AI Platform - Production Deployment on VPS**

---

## 📋 Quick Start (5 Minutes)

```bash
# 1. Clone/Extract package
cd reasoner_complete/

# 2. Run deployment script
chmod +x devops/scripts/*.sh
./devops/scripts/deploy.sh production

# 3. Test
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/health
```

**That's it!** The script handles everything.

---

## 🎯 What You Get

- ✅ PostgreSQL database with migrations
- ✅ Redis for caching/rate limiting
- ✅ MLflow for experiment tracking
- ✅ 30 mathematical formulas loaded
- ✅ API authentication enabled
- ✅ Structured logging
- ✅ Prometheus metrics
- ✅ Health checks
- ✅ Auto-backup capability

---

## 🔧 Manual Deployment (If Needed)

### Prerequisites

**Required:**
- Ubuntu 20.04+ or Debian 11+ VPS
- Docker 20.10+
- Docker Compose 2.0+
- 2GB RAM minimum (4GB recommended)
- 20GB disk space

**Install Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in
```

**Install Docker Compose:**
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

---

### Step-by-Step Deployment

#### 1. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit configuration
nano .env
```

**Critical settings to change:**
```bash
# Security (MUST CHANGE)
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 32)

# Database
POSTGRES_PASSWORD=your_strong_password_here

# Redis
REDIS_PASSWORD=your_redis_password_here

# CORS (your domain)
CORS_ORIGINS=https://yourdomain.com

# Environment
ENVIRONMENT=production
DEBUG=false
```

#### 2. Deploy Services

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

#### 3. Initialize Database

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Load formulas (30 formulas)
docker-compose exec backend python -m app.core.init_db
```

#### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test API (use your API key from .env)
curl -H "X-API-Key: YOUR_API_KEY" \
     http://localhost:8000/api/v1/formulas

# Check metrics
curl http://localhost:8000/metrics
```

---

## 🔐 Security Setup

### 1. API Key Authentication

**Generate API key:**
```bash
openssl rand -hex 32
```

**Add to .env:**
```bash
API_KEY_ENABLED=true
API_KEY=your_generated_key_here
```

**Use in requests:**
```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     http://localhost:8000/api/v1/formulas
```

### 2. CORS Configuration

**Update .env:**
```bash
CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

### 3. Rate Limiting

**Configure in .env:**
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60  # Adjust based on needs
```

### 4. SSL/TLS Setup

Use nginx or Traefik as reverse proxy:

**Nginx example:**
```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 📊 Monitoring

### Health Checks

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-03T06:00:00",
  "version": "1.0.0",
  "components": {
    "database": {"status": "up"},
    "redis": {"status": "up"},
    "formulas": {"status": "up", "count": 30}
  }
}
```

**Set up monitoring:**
```bash
# Add to crontab for alerts
*/5 * * * * curl -f http://localhost:8000/health || echo "API down" | mail -s "Alert" admin@example.com
```

### Prometheus Metrics

**Endpoint:** `GET /metrics`

**Available metrics:**
- `reasoner_http_requests_total` - Total requests
- `reasoner_http_request_duration_seconds` - Request latency
- `reasoner_formula_executions_total` - Formula executions
- `reasoner_formula_execution_seconds` - Execution time

**Grafana setup:**
```yaml
# Add to docker-compose.yml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=your_password
  volumes:
    - grafana_data:/var/lib/grafana
```

### Logging

**View logs:**
```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Filter errors
docker-compose logs backend | grep ERROR
```

**Log levels:**
- `DEBUG` - Development only
- `INFO` - Production default
- `WARNING` - Important events
- `ERROR` - Errors only

---

## 💾 Backup & Recovery

### Automated Backups

**Create backup:**
```bash
./devops/scripts/backup.sh
```

**What's backed up:**
- PostgreSQL database (compressed)
- Formula library files
- Configuration data

**Schedule automatic backups:**
```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/reasoner_complete/devops/scripts/backup.sh
```

### Restore from Backup

```bash
./devops/scripts/restore.sh reasoner_backup_20251103_120000.sql.gz
```

### Backup to S3 (Optional)

**Install AWS CLI:**
```bash
apt-get install awscli
aws configure
```

**Update backup script:**
```bash
# Add to backup.sh
aws s3 cp "$BACKUP_DIR/$BACKUP_FILE" s3://your-bucket/backups/
```

---

## 🔄 Updates & Maintenance

### Updating the Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d

# Run migrations
docker-compose exec backend alembic upgrade head
```

### Database Migrations

**Create new migration:**
```bash
docker-compose exec backend alembic revision --autogenerate -m "description"
```

**Apply migrations:**
```bash
docker-compose exec backend alembic upgrade head
```

**Rollback migration:**
```bash
docker-compose exec backend alembic downgrade -1
```

### Scaling

**Increase workers:**
```bash
# Edit .env
WORKERS=8  # (2 x CPU cores) + 1

# Restart
docker-compose restart backend
```

**Horizontal scaling:**
```bash
# docker-compose.yml
backend:
  deploy:
    replicas: 3
  
# Add load balancer (nginx/traefik)
```

---

## 🚨 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs backend

# Check ports
netstat -tulpn | grep 8000

# Verify .env
cat .env | grep -v "^#" | grep -v "^$"

# Restart fresh
docker-compose down -v
docker-compose up -d
```

### Database Connection Failed

```bash
# Check database is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres psql -U reasoner_user -d reasoner_db -c "SELECT 1"

# Check credentials in .env
# Ensure DATABASE_URL matches POSTGRES_* variables
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Reduce workers
# Edit .env: WORKERS=2

# Limit container memory
# docker-compose.yml:
backend:
  mem_limit: 1g
```

### Slow Queries

```bash
# Enable query logging
# In docker-compose.yml postgres:
command: 
  - "postgres"
  - "-c"
  - "log_statement=all"

# Check slow queries
docker-compose exec postgres tail -f /var/lib/postgresql/data/log/postgresql.log
```

### API Returns 401 Unauthorized

```bash
# Check API key in .env
grep API_KEY .env

# Test without auth (temporarily disable)
# Edit .env: API_KEY_ENABLED=false
docker-compose restart backend

# Verify key format
echo -n "X-API-Key: YOUR_KEY" | base64
```

---

## 📈 Performance Tuning

### Database Optimization

**PostgreSQL settings:**
```yaml
# docker-compose.yml
postgres:
  command:
    - "postgres"
    - "-c"
    - "max_connections=200"
    - "-c"
    - "shared_buffers=256MB"
    - "-c"
    - "effective_cache_size=1GB"
```

### Redis Caching

**Enable caching:**
```python
# Results are cached automatically
# Configure in .env:
REDIS_MAX_CONNECTIONS=50
```

### Request Optimization

**Enable compression:**
```python
# Add to main.py
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## 🔒 Security Checklist

- [ ] Change all default passwords
- [ ] Generate new SECRET_KEY and API_KEY
- [ ] Configure CORS with actual domains
- [ ] Enable rate limiting
- [ ] Set up SSL/TLS (reverse proxy)
- [ ] Restrict database to internal network only
- [ ] Enable firewall (UFW)
- [ ] Set up fail2ban for SSH
- [ ] Regular security updates
- [ ] Enable audit logging
- [ ] Backup encryption (if sensitive data)

---

## 📞 Support & Maintenance

### Regular Tasks

**Daily:**
- Check health endpoint
- Monitor error logs
- Review metrics

**Weekly:**
- Review backup success
- Check disk space
- Security updates

**Monthly:**
- Database vacuum/analyze
- Review and rotate logs
- Performance review

### Monitoring Commands

```bash
# System resources
docker stats

# Disk space
df -h
du -sh data/

# Service status
docker-compose ps

# Recent errors
docker-compose logs --since 1h backend | grep ERROR
```

---

## 📚 Additional Resources

- API Documentation: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`
- Health: `http://localhost:8000/health`
- Project README: `../README.md`
- Integration Guide: `../INTEGRATION_GUIDE.md`

---

## ✅ Deployment Checklist

**Before deployment:**
- [ ] Review .env configuration
- [ ] Generate secure keys
- [ ] Configure CORS
- [ ] Set up SSL certificate
- [ ] Plan backup strategy
- [ ] Test restore procedure
- [ ] Configure monitoring alerts

**After deployment:**
- [ ] Verify health check
- [ ] Test API endpoints
- [ ] Check metrics endpoint
- [ ] Review logs for errors
- [ ] Test backup script
- [ ] Document API key
- [ ] Set up monitoring
- [ ] Schedule backups

**Ongoing:**
- [ ] Monitor health daily
- [ ] Review logs weekly
- [ ] Test backups monthly
- [ ] Update dependencies quarterly
- [ ] Security audit annually

---

**Questions?** Check troubleshooting section or contact backend developer.

**Status:** ✅ Production Ready
