# Production Ready Documentation

This document outlines all production-ready features implemented in The Reasoner AI Platform.

## Table of Contents
1. [Load Testing](#load-testing)
2. [Security Hardening](#security-hardening)
3. [Documentation](#documentation)
4. [Deployment Automation](#deployment-automation)
5. [Monitoring](#monitoring)
6. [Quick Start](#quick-start)

---

## Load Testing

### Overview
Comprehensive load testing suite using Locust for performance validation.

### Location
- Load test files: `backend/tests/load/`
- Configuration: `backend/tests/load/load_test_config.py`
- Documentation: `backend/tests/load/README.md`

### Test Scenarios
1. **Smoke Test** (10 users, 2 minutes)
2. **Baseline** (50 users, 10 minutes)
3. **Stress Test** (200 users, 15 minutes)
4. **Spike Test** (500 users, 5 minutes)
5. **Endurance Test** (100 users, 60 minutes)

### Running Load Tests

```bash
# Install locust
pip install locust

# Run with UI
locust -f backend/tests/load/locustfile.py --host=http://localhost:8000

# Run headless (baseline scenario)
locust -f backend/tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --headless \
  --users 50 \
  --spawn-rate 5 \
  --run-time 10m \
  --html reports/load_test.html
```

### Performance Targets
- P50 Response Time: < 200ms
- P95 Response Time: < 500ms
- P99 Response Time: < 1000ms
- Error Rate: < 1%
- Throughput: > 100 req/s

---

## Security Hardening

### Security Middleware

**Location**: `backend/app/core/security_middleware.py`

#### Features Implemented:

1. **Security Headers**
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - X-XSS-Protection
   - Strict-Transport-Security (HSTS)
   - Content-Security-Policy (CSP)
   - Referrer-Policy

2. **Rate Limiting**
   - Configurable per endpoint
   - Default: 100 requests per 60 seconds
   - Redis-backed storage support
   - Per-IP tracking

3. **Request Validation**
   - SQL injection detection
   - XSS attack prevention
   - Path traversal protection
   - Request size limits (10MB default)

4. **IP Filtering**
   - Whitelist support
   - Blacklist support
   - Configurable per environment

5. **Audit Logging**
   - All API requests logged
   - Sensitive paths tracked
   - Failed authentication attempts logged

### Input Validation

**Location**: `backend/app/core/input_validation.py`

#### Features:
- Email validation
- Username validation
- Password strength checking
- HTML sanitization
- SQL injection prevention
- XSS prevention
- File upload validation

### Security Configuration

**Location**: `backend/app/core/security_config.py`

#### Environment-specific Settings:

**Production**:
- Strict CORS origins
- HTTPS required
- Strong password policy (12+ chars)
- 2FA enabled
- SSL/TLS for database
- Limited login attempts (3)
- 1-hour lockout

**Development**:
- Relaxed CORS (for testing)
- No HTTPS requirement
- Relaxed password policy
- Higher rate limits

### Applying Security Middleware

```python
from app.core.security_middleware import setup_security_middleware
from app.core.security_config import security_settings

# In main.py
setup_security_middleware(app, config={
    "allowed_hosts": security_settings.allowed_hosts,
    "rate_limit_calls": security_settings.rate_limit_calls,
    "rate_limit_period": security_settings.rate_limit_period,
    "ip_whitelist": security_settings.ip_whitelist,
    "ip_blacklist": security_settings.ip_blacklist,
})
```

---

## Documentation

### MkDocs Site

**Location**: `docs/`

#### Structure:
- Getting Started guides
- User documentation
- API reference
- Architecture docs
- Deployment guides
- Development guides
- Security best practices

#### Building Documentation

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Serve locally
cd docs
mkdocs serve

# Build static site
mkdocs build
```

The documentation will be available at `http://localhost:8000`.

#### Deployment
Documentation can be deployed to:
- GitHub Pages
- Netlify
- Vercel
- Custom hosting

---

## Deployment Automation

### Docker Containerization

#### Backend Dockerfile
**Location**: `backend/Dockerfile`

**Features**:
- Multi-stage build (smaller images)
- Non-root user
- Health checks
- Security hardening
- Production-optimized

#### Frontend Dockerfile
**Location**: `frontend/Dockerfile`

**Features**:
- Multi-stage build with Node.js and Nginx
- Optimized asset caching
- Security headers in Nginx
- Gzip compression
- Non-root user

### Docker Compose

**Location**: `docker-compose.yml`

**Services**:
1. PostgreSQL (with health checks)
2. Redis (with persistence)
3. MLflow (experiment tracking)
4. Backend API
5. Frontend
6. Prometheus (monitoring)
7. Grafana (visualization)

**Running with Docker Compose**:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Deployment Script

**Location**: `scripts/deploy.sh`

**Features**:
- Automated deployment
- Database backup before deployment
- Health checks
- Automatic rollback on failure
- Migration execution
- Old image cleanup

**Usage**:

```bash
# Make executable
chmod +x scripts/deploy.sh

# Run deployment
./scripts/deploy.sh

# With custom environment
ENVIRONMENT=production ./scripts/deploy.sh
```

### CI/CD Pipeline

**Location**: `.github/workflows/ci-cd.yml`

**Pipeline Stages**:

1. **Test**
   - Backend unit tests
   - Frontend unit tests
   - Integration tests
   - Code coverage

2. **Security Scan**
   - Trivy vulnerability scanning
   - SAST (Static Application Security Testing)
   - Dependency scanning

3. **Build**
   - Build Docker images
   - Tag with version
   - Push to container registry

4. **Deploy**
   - Deploy to staging
   - Run smoke tests
   - Deploy to production (on main branch)
   - Database migrations

**Triggering**:
- Automatic on push to `main` or `develop`
- Manual workflow dispatch
- Pull requests (test only)

---

## Monitoring

### Prometheus

**Location**: `monitoring/prometheus.yml`

**Metrics Collected**:
- HTTP request rates
- Response times
- Error rates
- System resources (CPU, memory)
- Database connections
- Cache hit rates

**Access**: `http://localhost:9090`

### Grafana

**Dashboards** (pre-configured):
- API Performance
- System Resources
- Database Metrics
- Error Tracking
- User Activity

**Access**: `http://localhost:3000`
**Default Credentials**: admin/admin (change on first login)

### Health Checks

All services expose `/health` endpoints:
- **Backend**: `http://localhost:8000/health`
- **Frontend**: `http://localhost:8080/health`

Health check format:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "1.0.0",
  "components": {
    "database": {"status": "up"},
    "redis": {"status": "up"},
    "mlflow": {"status": "assumed_up"}
  }
}
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd blank-app

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Run migrations
docker-compose exec backend alembic upgrade head

# Create admin user
docker-compose exec backend python -m app.create_admin
```

### Production Deployment

```bash
# 1. Configure environment
cp .env.example .env.production
nano .env.production  # Update with production values

# 2. Run deployment script
ENVIRONMENT=production ./scripts/deploy.sh

# 3. Verify deployment
curl http://localhost:8000/health
curl http://localhost:8080/health

# 4. Access services
# - Application: http://your-domain.com
# - Grafana: http://your-domain.com:3000
# - Prometheus: http://your-domain.com:9090
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployments
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/backend

# Scale services
kubectl scale deployment/backend --replicas=3
```

---

## Security Checklist

Before deploying to production:

- [ ] Change all default passwords
- [ ] Set strong `SECRET_KEY`
- [ ] Configure CORS origins
- [ ] Enable HTTPS/SSL
- [ ] Set up firewall rules
- [ ] Configure backup strategy
- [ ] Enable rate limiting
- [ ] Review and set security headers
- [ ] Configure logging and monitoring
- [ ] Set up alerts
- [ ] Enable 2FA for admin accounts
- [ ] Review environment variables
- [ ] Set up database SSL
- [ ] Configure IP whitelist (if needed)
- [ ] Review and audit dependencies

---

## Performance Optimization

### Backend
- Enable caching with Redis
- Use connection pooling (configured in `config.py`)
- Enable query optimization
- Use async endpoints where possible
- Configure worker processes (4+ for production)

### Frontend
- Static asset caching (configured in Nginx)
- Gzip compression enabled
- Code splitting
- Lazy loading
- Image optimization

### Database
- Enable query logging for slow queries
- Set up read replicas for scaling
- Configure connection pool size
- Enable prepared statements
- Regular VACUUM and ANALYZE

---

## Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check logs
docker-compose logs

# Check disk space
df -h

# Check Docker status
docker ps -a
```

#### Database Connection Errors
```bash
# Check database is running
docker-compose ps db

# Test connection
docker-compose exec backend python -c "from app.core.database import engine; engine.connect()"
```

#### High Memory Usage
```bash
# Check container stats
docker stats

# Adjust worker count in docker-compose.yml
# Reduce: --workers 2
```

### Getting Help
- Documentation: `docs/`
- Issues: GitHub Issues
- Security: security@reasoner.ai

---

## Maintenance

### Regular Tasks

**Daily**:
- Monitor error logs
- Check service health
- Review security alerts

**Weekly**:
- Review performance metrics
- Check disk usage
- Update dependencies (dev environment)
- Run load tests

**Monthly**:
- Security updates
- Database optimization
- Backup verification
- Review access logs
- Rotate API keys

### Backup Strategy

**Automated Backups**:
- Database: Daily at 2 AM
- Configuration: With each deployment
- Logs: Retained for 90 days

**Manual Backup**:
```bash
# Database
docker-compose exec db pg_dump -U postgres formulas > backup.sql

# Full system
./scripts/backup.sh
```

---

## Version History

- **v1.0.0** - Initial production release
  - Load testing suite
  - Security hardening
  - Documentation site
  - Deployment automation
  - Monitoring setup

---

For detailed information on any topic, refer to the relevant section in the `docs/` directory.
