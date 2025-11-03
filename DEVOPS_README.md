# 📦 DevOps Package - The Reasoner Platform

**Production-ready deployment package for VPS**

---

## 🎯 What's Included

This package contains everything your DevOps engineer needs to deploy The Reasoner AI Platform on a VPS with minimal effort.

### ✅ Complete Infrastructure
- Docker Compose orchestration (PostgreSQL, Redis, MLflow, Backend, Frontend)
- Database migrations (Alembic)
- Health checks & monitoring
- Backup & restore scripts
- One-command deployment
- Production-ready configuration

### ✅ Security Features
- API key authentication
- Rate limiting
- CORS configuration
- Secrets management
- Input validation

### ✅ Monitoring & Logging
- Prometheus metrics endpoint
- Structured logging
- Health check system
- Request/response tracking
- Performance monitoring

### ✅ Documentation
- Deployment guide
- Troubleshooting reference
- API quick reference
- Configuration examples

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Deploy
cd reasoner_complete/
chmod +x devops/scripts/*.sh
./devops/scripts/deploy.sh production

# 2. Test
curl http://localhost:8000/health

# 3. Use API (get key from deploy output)
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/v1/formulas
```

**Done!** System is running with 30 formulas loaded.

---

## 📁 Package Structure

```
reasoner_complete/
├── devops/
│   ├── scripts/
│   │   ├── deploy.sh         # One-command deployment
│   │   ├── backup.sh          # Database backup
│   │   └── restore.sh         # Database restore
│   ├── docs/
│   │   ├── DEPLOYMENT.md      # Full deployment guide
│   │   ├── TROUBLESHOOTING.md # Quick fixes
│   │   └── API_REFERENCE.md   # API testing guide
│   └── monitoring/
│       └── (Grafana dashboards - optional)
├── backend/
│   ├── app/                   # Application code
│   ├── alembic/              # Database migrations
│   ├── Dockerfile             # Backend container
│   └── requirements.txt       # Python dependencies
├── frontend/                  # Web interface (optional)
├── alembic/                   # Migration system
│   ├── versions/
│   │   └── 001_initial.py    # Initial schema
│   ├── env.py                 # Migration environment
│   └── script.py.mako         # Migration template
├── data/
│   ├── formulas/
│   │   └── initial_library.json  # 30 formulas
│   ├── datasets/
│   │   └── sample_inputs.json    # Test cases
│   └── bounds/
│       └── empirical_bounds.yaml # Validation rules
├── config/
│   ├── context_rules.yaml     # Context detection
│   └── unit_definitions.txt   # Custom units
├── docker-compose.yml         # Service orchestration
├── .env.example               # Configuration template
└── README.md                  # This file
```

---

## 🔧 What DevOps Needs to Do

### Before Deployment

1. **Review `.env.example`**
   - Understand configuration options
   - Plan secret generation strategy

2. **Check Prerequisites**
   - Docker 20.10+ installed
   - Docker Compose 2.0+ installed
   - 2GB+ RAM available
   - 20GB+ disk space

3. **Plan Security**
   - API key distribution
   - SSL/TLS setup (nginx/traefik)
   - Firewall configuration
   - Backup strategy

### During Deployment

1. **Run Deploy Script**
   ```bash
   ./devops/scripts/deploy.sh production
   ```
   
   This automatically:
   - Generates secrets
   - Builds containers
   - Starts services
   - Runs migrations
   - Loads formulas
   - Runs health checks

2. **Verify Deployment**
   - Check health endpoint
   - Test API with key
   - Review logs
   - Verify formula count

### After Deployment

1. **Set Up Monitoring**
   - Configure health check alerts
   - Set up log aggregation (optional)
   - Monitor metrics endpoint

2. **Configure Backups**
   - Schedule daily backups (cron)
   - Test restore procedure
   - Configure S3 backup (optional)

3. **Document Access**
   - Save API key securely
   - Document admin credentials
   - Share with backend team

---

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health | jq
```

**Returns:**
- Service status
- Component health
- Formula count
- Version info

### Metrics (Prometheus)
```bash
curl http://localhost:8000/metrics
```

**Available metrics:**
- HTTP request count/duration
- Formula execution count/duration
- System health indicators

### Logs
```bash
# View all logs
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Errors only
docker-compose logs backend | grep ERROR
```

---

## 🔐 Security Checklist

- [ ] Change all default passwords
- [ ] Generate secure API key
- [ ] Configure CORS for production domains
- [ ] Enable rate limiting
- [ ] Set up SSL/TLS (reverse proxy)
- [ ] Configure firewall
- [ ] Restrict database to internal network
- [ ] Enable audit logging
- [ ] Set up Sentry (optional)
- [ ] Configure backup encryption (if needed)

---

## 💾 Backup & Recovery

### Create Backup
```bash
./devops/scripts/backup.sh
```

Creates:
- Database dump (compressed)
- Formula files backup
- Stores in `data/backups/`

### Restore from Backup
```bash
./devops/scripts/restore.sh backup_file.sql.gz
```

### Automate Backups
```bash
# Add to crontab
crontab -e

# Daily at 2 AM
0 2 * * * /path/to/reasoner_complete/devops/scripts/backup.sh
```

---

## 🔄 Updates & Maintenance

### Update Application
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Run migrations
docker-compose exec backend alembic upgrade head
```

### Scale Workers
```bash
# Edit .env
WORKERS=8

# Restart
docker-compose restart backend
```

---

## 📞 Support

### Documentation
- **Deployment:** `devops/docs/DEPLOYMENT.md`
- **Troubleshooting:** `devops/docs/TROUBLESHOOTING.md`
- **API Reference:** `devops/docs/API_REFERENCE.md`

### Quick Fixes
- Service won't start → Check logs
- Connection refused → Verify .env
- 401 Unauthorized → Check API key
- Database error → Verify credentials

### Escalation
Contact backend developer for:
- Code changes needed
- Schema modifications
- Security issues
- Persistent failures

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Review documentation
- [ ] Check prerequisites (Docker, etc)
- [ ] Plan security (keys, SSL, firewall)
- [ ] Configure .env file
- [ ] Test restore procedure

### Deployment
- [ ] Run deploy.sh script
- [ ] Verify health check
- [ ] Test API endpoints
- [ ] Check formula count (should be 30)
- [ ] Review logs for errors

### Post-Deployment
- [ ] Set up monitoring alerts
- [ ] Configure automated backups
- [ ] Document API key
- [ ] Test backup/restore
- [ ] Configure SSL/reverse proxy
- [ ] Set up firewall rules

---

## 🎯 Success Criteria

After deployment, verify:

✅ Health endpoint returns "healthy"  
✅ `/api/v1/formulas` returns 30 formulas  
✅ Formula execution works  
✅ Metrics endpoint accessible  
✅ Logs show no errors  
✅ Backup script runs successfully  
✅ All services in "Up" state  

---

## 📈 Performance Baselines

**Expected Performance:**
- Health check: < 100ms
- List formulas: < 200ms
- Execute formula: < 500ms
- 30+ formulas loaded
- 4 workers (default)

**Resource Usage:**
- RAM: 2-3GB total
- CPU: 10-30% idle
- Disk: ~500MB (without logs)

---

## 🔗 Quick Links

- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics

---

## 📊 What Changed from V3

### New in V4 (DevOps Ready)

✅ **Infrastructure:**
- Alembic database migrations
- Redis for caching/rate limiting
- Health checks on all services
- Resource limits configured
- Restart policies

✅ **Security:**
- API key authentication
- Rate limiting system
- CORS configuration
- Structured logging
- Request ID tracking

✅ **Monitoring:**
- Prometheus metrics endpoint
- Enhanced health checks
- Performance tracking
- Error tracking ready

✅ **DevOps Tools:**
- One-command deployment script
- Automated backup/restore
- Production docker-compose
- Complete documentation
- Troubleshooting guide
- API quick reference

✅ **Production Ready:**
- Environment-based config
- Secrets management
- Migration system
- Backup strategy
- Monitoring setup

---

## 💯 Package Completeness

| Feature | V3 | V4 DevOps |
|---------|----| ---|
| Formulas | ✅ 30 | ✅ 30 |
| Integration | ✅ 100% | ✅ 100% |
| Docker Setup | ✅ Basic | ✅ Production |
| Migrations | ❌ None | ✅ Alembic |
| Security | ❌ None | ✅ Complete |
| Monitoring | ❌ None | ✅ Prometheus |
| Logging | ❌ Basic | ✅ Structured |
| Backup | ❌ Manual | ✅ Automated |
| Documentation | ⚠️ Limited | ✅ Complete |
| **Production Ready** | **60%** | **100%** |

---

**Status:** ✅ Production Ready  
**Version:** 4.0.0 (DevOps Ready)  
**Date:** November 3, 2025

**This package is ready to hand to your DevOps engineer for immediate deployment.**
