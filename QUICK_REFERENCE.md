# Quick Reference Guide

Fast access to common commands and workflows for the Formula API project.

---

## 🚀 Quick Start

```bash
# Local development
docker-compose up -d
curl http://localhost:8000/health
docker-compose logs -f backend

# Run tests
docker-compose exec backend pytest -v
bash test_comprehensive.sh

# Deploy to staging
bash deployment/deploy-staging.sh

# Monitor services
bash monitoring/monitor.sh 24

# Create backup
bash backup/backup.sh

# Emergency rollback
bash scripts/rollback.sh -e staging -b previous -r "Critical bug"
```

---

## 📋 GitHub Actions Workflows

### Trigger Workflows

```bash
# CI/CD Pipeline (test + deploy)
gh workflow run ci-cd.yml -f environment=staging

# Security Scan
gh workflow run security-scan.yml

# Rollback
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=previous \
  -f reason="Deployment issue"
```

### Monitor Workflows

```bash
# List workflows
gh workflow list

# Watch active run
gh run watch

# View recent runs
gh run list --limit 10

# View specific run
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

---

## 🔒 GitHub Secrets

```bash
# Set secrets
gh secret set STAGING_HOST -b "staging.example.com"
gh secret set STAGING_USER -b "deploy"
gh secret set STAGING_SSH_KEY < ~/.ssh/staging_key
gh secret set STAGING_API_KEY -b "sk_staging_test123"

# List secrets
gh secret list

# View secret (value hidden)
gh secret get STAGING_HOST
```

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest --cov=app --cov-report=term tests/

# Specific test file
pytest tests/test_app.py -v

# Integration tests
pytest tests/integration/ -v
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Execute formula
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "X-API-Key: test-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001}
  }'

# List formulas
curl http://localhost:8000/api/v1/formulas/list \
  -H "X-API-Key: test-key-1"

# Get execution history
curl http://localhost:8000/api/v1/formulas/history?limit=10 \
  -H "X-API-Key: test-key-1"
```

### Rate Limiting Test

```bash
# Test rate limit (should block after 10 requests)
for i in {1..15}; do
  echo "Request $i:"
  curl -X POST http://localhost:8000/api/v1/formulas/execute \
    -H "X-API-Key: test-key-1" \
    -H "Content-Type: application/json" \
    -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
    -w "\nStatus: %{http_code}\n\n"
  sleep 1
done
```

---

## 🚢 Deployment

### Staging Deployment

```bash
# Option 1: Automated (push to develop)
git checkout develop
git merge feature-branch
git push origin develop
# Workflow automatically deploys

# Option 2: Manual deployment script
bash deployment/deploy-staging.sh

# Option 3: Manual workflow trigger
gh workflow run ci-cd.yml -f environment=staging
```

### Production Deployment

```bash
# Merge to main (requires approval)
git checkout main
git merge develop
git push origin main
# Workflow automatically creates backup and deploys

# Or manual trigger
gh workflow run ci-cd.yml -f environment=production
```

### Verify Deployment

```bash
# Run verification script
bash scripts/verify-deployment.sh

# Or on remote server
ssh deploy@staging.example.com
cd /opt/formula-api
bash scripts/verify-deployment.sh
```

---

## 🔄 Rollback

### Quick Rollback

```bash
# Via GitHub Actions (recommended)
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=previous \
  -f reason="Critical bug"

# Via script (emergency)
ssh deploy@staging.example.com
cd /opt/formula-api
bash scripts/rollback.sh -e staging -b previous -r "Emergency" -y
```

### List Available Backups

```bash
# On server
ssh deploy@staging.example.com
ls -lh /opt/formula-api/backup/backups/

# Find specific backup
ssh deploy@staging.example.com \
  "ls -lh /opt/formula-api/backup/backups/ | grep '20250103'"
```

### Rollback to Specific Backup

```bash
# With exact timestamp
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=20250103_120000 \
  -f reason="Revert to known good state"
```

---

## 📊 Monitoring

### Check Services

```bash
# Docker status
docker-compose ps

# Service logs
docker-compose logs backend --tail 50 -f
docker-compose logs db --tail 50
docker-compose logs redis --tail 50
docker-compose logs mlflow --tail 50

# All logs
docker-compose logs --tail 100

# Check specific errors
bash scripts/check-logs.sh --errors
bash scripts/check-logs.sh --service backend --lines 100
```

### Continuous Monitoring

```bash
# Monitor for 24 hours
bash monitoring/monitor.sh 24

# Monitor for 1 hour
bash monitoring/monitor.sh 1

# View monitoring logs
tail -f monitoring/logs/health_checks.log
tail -f monitoring/logs/database_checks.log
tail -f monitoring/logs/api_tests.log
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Database health
docker-compose exec db psql -U postgres -c "\l"

# Redis health
docker-compose exec redis redis-cli ping

# MLflow health
curl http://localhost:5000
```

---

## 💾 Backup & Restore

### Create Backup

```bash
# Manual backup
bash backup/backup.sh

# Verify backup created
ls -lh backup/backups/

# View backup metadata
tar -xzf backup/backups/formula-api-backup-*.tar.gz -O backup/metadata.json | jq
```

### Restore from Backup

```bash
# Interactive restore
bash backup/restore.sh backup/backups/formula-api-backup-20250103_120000.tar.gz

# Automated restore (skip confirmation)
echo "yes" | bash backup/restore.sh backup/backups/formula-api-backup-20250103_120000.tar.gz
```

### Test Backup System

```bash
# Run comprehensive backup/restore test
bash backup/test-backup-restore.sh
```

---

## 🔍 Security Scanning

### Run Security Scans

```bash
# All scans via GitHub Actions
gh workflow run security-scan.yml

# Local dependency scan
pip install pip-audit
pip-audit -r requirements.txt

# Local code scan
pip install bandit
bandit -r app/ -f json -o bandit-report.json
bandit -r app/

# Docker image scan
docker run --rm aquasec/trivy image formula-api:latest
```

### View Security Reports

```bash
# Download from GitHub Actions
SECURITY_RUN=$(gh run list --workflow=security-scan.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download $SECURITY_RUN

# View reports
cat pip-audit-report/pip-audit-report.json | jq
cat security-reports/bandit-report.json | jq
cat trivy-report/trivy-report.json | jq
cat security-summary/security-summary.md
```

### Check GitHub Security Alerts

```bash
# Code scanning alerts
gh api /repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/code-scanning/alerts

# Dependabot alerts
gh api /repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/dependabot/alerts

# View in browser
gh browse --settings
```

---

## 🐳 Docker Commands

### Service Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart backend

# View running containers
docker-compose ps

# Rebuild and restart
docker-compose up -d --build
```

### Container Access

```bash
# Execute command in backend
docker-compose exec backend pytest -v

# Access backend shell
docker-compose exec backend bash

# Access database
docker-compose exec db psql -U postgres -d formulas

# Access Redis CLI
docker-compose exec redis redis-cli
```

### Logs and Debugging

```bash
# Follow logs
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail 100 backend

# Search logs
docker-compose logs backend | grep ERROR

# Container stats
docker stats
```

---

## 🗄️ Database

### Access Database

```bash
# Via docker-compose
docker-compose exec db psql -U postgres -d formulas

# Direct connection
psql -h localhost -U postgres -d formulas
```

### Common Queries

```sql
-- View all formula executions
SELECT * FROM formula_executions ORDER BY created_at DESC LIMIT 10;

-- Count executions by formula
SELECT formula_id, COUNT(*) as count
FROM formula_executions
GROUP BY formula_id;

-- Recent successful executions
SELECT formula_id, result, unit, created_at
FROM formula_executions
WHERE success = true
ORDER BY created_at DESC
LIMIT 20;

-- Failed executions
SELECT formula_id, error_message, created_at
FROM formula_executions
WHERE success = false
ORDER BY created_at DESC;

-- Average execution time
SELECT formula_id, AVG(execution_time_ms) as avg_time_ms
FROM formula_executions
WHERE success = true
GROUP BY formula_id;
```

### Database Backup/Restore

```bash
# Manual database backup
docker-compose exec db pg_dumpall -U postgres > backup/manual-db-backup.sql

# Restore database
docker-compose exec -T db psql -U postgres < backup/manual-db-backup.sql
```

---

## 📈 MLflow

### Access MLflow UI

```bash
# Open in browser
open http://localhost:5000

# Or
curl http://localhost:5000
```

### MLflow CLI

```bash
# List experiments
docker-compose exec mlflow mlflow experiments list

# Search runs
docker-compose exec mlflow mlflow runs list --experiment-id 0

# Get run details
docker-compose exec mlflow mlflow runs describe --run-id <run-id>
```

---

## 🔧 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port
export BACKEND_PORT=8001
docker-compose up -d
```

**Database connection error:**
```bash
# Restart database
docker-compose restart db

# Check database logs
docker-compose logs db

# Reset database
docker-compose down -v
docker-compose up -d
```

**Redis connection error:**
```bash
# Restart Redis
docker-compose restart redis

# Check Redis
docker-compose exec redis redis-cli ping

# View Redis logs
docker-compose logs redis
```

**Out of disk space:**
```bash
# Check disk usage
df -h

# Clean Docker
docker system prune -a --volumes

# Remove old backups
find backup/backups/ -mtime +7 -delete
```

---

## 📚 Documentation

- `README.md` - Project overview
- `CI_CD_TESTING.md` - Complete CI/CD testing guide
- `.github/WORKFLOWS.md` - GitHub Actions documentation
- `PRODUCTION_DEPLOYMENT.md` - Production deployment guide
- `backup/README.md` - Backup/restore procedures
- `tests/README.md` - Testing documentation

---

## 🔗 Useful Links

### Local Services
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Database: localhost:5432

### GitHub Actions
```bash
# View workflows in browser
gh browse --actions

# View specific workflow
gh workflow view ci-cd.yml --web
```

### Remote Servers
```bash
# SSH to staging
ssh deploy@staging.example.com

# SSH to production
ssh deploy@production.example.com
```

---

## 📞 Support

### View Workflow Status
```bash
# Check all runs
gh run list

# Check specific workflow
gh run list --workflow=ci-cd.yml --limit 5

# View failed runs only
gh run list --status=failure
```

### Re-run Failed Workflows
```bash
# Re-run failed jobs only
gh run rerun <run-id> --failed

# Re-run entire workflow
gh run rerun <run-id>
```

### Get Help
```bash
# Workflow help
gh workflow --help

# Run help
gh run --help

# View workflow file
cat .github/workflows/ci-cd.yml
```

---

## 🎯 Common Workflows

### New Feature Development
```bash
# 1. Create feature branch
git checkout -b feature/new-formula

# 2. Make changes and test locally
docker-compose up -d
pytest tests/ -v

# 3. Commit and push
git add .
git commit -m "Add new formula"
git push origin feature/new-formula

# 4. Create PR (triggers CI/CD tests)
gh pr create --title "Add new formula" --body "Description"

# 5. After approval, merge to develop
gh pr merge --merge

# 6. Verify staging deployment
gh run watch
```

### Emergency Hotfix
```bash
# 1. Create hotfix branch from main
git checkout main
git checkout -b hotfix/critical-bug

# 2. Fix and test
# ... make changes ...
pytest tests/ -v

# 3. Deploy to production immediately
git checkout main
git merge hotfix/critical-bug
git push origin main

# 4. If deployment fails, rollback
gh workflow run rollback.yml \
  -f environment=production \
  -f backup_timestamp=previous \
  -f reason="Hotfix deployment failed"
```

### Weekly Security Check
```bash
# Run security scan
gh workflow run security-scan.yml

# Wait for completion
sleep 300

# Download and review reports
SCAN_RUN=$(gh run list --workflow=security-scan.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download $SCAN_RUN

# Update dependencies if needed
pip install --upgrade pip
pip-audit -r requirements.txt --fix
```

---

## ⚡ Performance Tips

### Speed up local development
```bash
# Use volume mounts instead of rebuilding
docker-compose up -d

# Run tests without coverage (faster)
pytest tests/ -v --no-cov

# Use pytest markers for selective testing
pytest tests/ -v -m "not slow"
```

### Speed up CI/CD
```bash
# Use Docker layer caching
# Already configured in .github/workflows/ci-cd.yml

# Skip tests for documentation changes
# Add [skip ci] to commit message for docs-only changes
git commit -m "[skip ci] Update README"
```

---

This quick reference covers the most common operations. For detailed documentation, see the individual guides listed in the Documentation section.
