# GitHub Actions CI/CD - Complete Guide

This guide covers the automated CI/CD pipeline, security scanning, deployments, and rollback procedures.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Workflows](#workflows)
- [Setup](#setup)
- [Testing](#testing)
- [Security Scanning](#security-scanning)
- [Deployments](#deployments)
- [Rollback](#rollback)
- [Secrets Configuration](#secrets-configuration)

---

## Overview

The CI/CD pipeline includes:

### ✅ Automated Testing
- Unit tests on Python 3.9, 3.10, 3.11
- Integration tests
- Code coverage reporting
- Linting and formatting checks

### 🔒 Security Scanning
- Dependency vulnerability scanning
- Code security analysis (Bandit)
- Docker image scanning (Trivy)
- Secret scanning (Gitleaks)
- CodeQL analysis

### 🚀 Automated Deployment
- Staging deployment (develop branch)
- Production deployment (main branch)
- Automated backups before deployment
- Post-deployment verification
- Smoke tests

### 🔄 Rollback Capability
- One-click rollback to previous version
- Automated backup restoration
- Emergency rollback for production
- Rollback verification

---

## Workflows

### 1. CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

**Triggers:**
- Push to `main`, `master`, `develop`
- Pull requests
- Manual dispatch

**Jobs:**
1. **Test** - Run all tests with coverage
2. **Lint** - Code quality checks
3. **Build** - Build and push Docker image
4. **Deploy Staging** - Deploy to staging environment
5. **Deploy Production** - Deploy to production environment
6. **Smoke Test** - Post-deployment validation

### 2. Security Scan (`.github/workflows/security-scan.yml`)

**Triggers:**
- Push to main branches
- Pull requests
- Weekly schedule (Sunday)
- Manual dispatch

**Jobs:**
1. **Dependency Scan** - Check for vulnerable dependencies
2. **Code Scan** - Static analysis for security issues
3. **CodeQL Analysis** - GitHub's semantic code analysis
4. **Docker Scan** - Container image vulnerability scan
5. **Secret Scan** - Detect exposed secrets
6. **Security Summary** - Aggregate results

### 3. Rollback (`.github/workflows/rollback.yml`)

**Triggers:**
- Manual dispatch only (intentional safety measure)

**Jobs:**
1. **Validate** - Validate rollback request
2. **Rollback** - Restore from backup
3. **Verify** - Verify rollback success
4. **Post-Rollback** - Generate reports and notifications

---

## Setup

### 1. Required GitHub Secrets

Navigate to: **Settings → Secrets and variables → Actions**

#### Staging Secrets
```
STAGING_HOST          # staging.example.com
STAGING_USER          # root or deploy user
STAGING_SSH_KEY       # Private SSH key for staging
STAGING_API_KEY       # API key for staging environment
```

#### Production Secrets
```
PRODUCTION_HOST       # api.example.com
PRODUCTION_USER       # root or deploy user
PRODUCTION_SSH_KEY    # Private SSH key for production
PRODUCTION_API_KEY    # API key for production environment
```

### 2. Generate SSH Keys

```bash
# Generate deployment key
ssh-keygen -t ed25519 -C "github-actions-deploy" -f deploy_key

# Add public key to server
ssh-copy-id -i deploy_key.pub user@staging.example.com

# Add private key to GitHub Secrets
cat deploy_key  # Copy this to STAGING_SSH_KEY secret
```

### 3. Configure Environments

Navigate to: **Settings → Environments**

Create two environments:
- **staging** - Auto-deploy from develop branch
- **production** - Require manual approval for deployment

**Protection Rules:**
- [x] Required reviewers (for production)
- [x] Wait timer: 5 minutes (for production)
- [ ] Deployment branches: `main`, `master`

---

## Testing

### Running Tests Locally

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest --cov=app --cov-report=term tests/

# Integration tests only
pytest tests/integration/ -v
```

### CI Test Matrix

Tests run on multiple Python versions:
- Python 3.9
- Python 3.10
- Python 3.11 (with coverage report)

### Test Workflow

```yaml
# Automatic on push/PR
git push origin feature-branch

# Manual trigger
gh workflow run ci-cd.yml
```

---

## Security Scanning

### Automated Scans

Security scans run:
- On every push to main branches
- On every pull request
- Weekly (Sunday at midnight)
- Manually via workflow dispatch

### Scan Types

#### 1. Dependency Scan (`pip-audit`)
Checks Python dependencies for known vulnerabilities.

#### 2. Code Security Scan (`bandit`)
Static analysis for common security issues:
- SQL injection
- Command injection
- Hard-coded credentials
- Insecure random
- And more...

#### 3. CodeQL Analysis
GitHub's semantic code analysis engine:
- Security vulnerabilities
- Code quality issues
- Best practice violations

#### 4. Docker Image Scan (`trivy`)
Scans Docker images for:
- OS vulnerabilities
- Library vulnerabilities
- Misconfigurations

#### 5. Secret Scanning (`gitleaks`)
Detects accidentally committed secrets:
- API keys
- Passwords
- Private keys
- Tokens

### Viewing Scan Results

**Option 1: Security Tab**
```
GitHub Repo → Security → Code scanning alerts
```

**Option 2: Workflow Artifacts**
```
Actions → Select workflow run → Artifacts
Download: security-reports, trivy-report, etc.
```

**Option 3: Pull Request Comments**
Security summary automatically posted on PRs.

### Manual Security Scan

```bash
# Run all security scans
gh workflow run security-scan.yml

# Check status
gh workflow view security-scan.yml

# Download reports
gh run download <run-id>
```

---

## Deployments

### Automated Deployments

#### Staging Deployment
**Trigger:** Push to `develop` branch

```bash
git checkout develop
git merge feature-branch
git push origin develop

# Workflow automatically:
# 1. Runs all tests
# 2. Builds Docker image
# 3. Deploys to staging
# 4. Runs verification
# 5. Executes smoke tests
```

#### Production Deployment
**Trigger:** Push to `main` branch

```bash
git checkout main
git merge develop
git push origin main

# Workflow automatically:
# 1. Runs all tests
# 2. Builds Docker image
# 3. Creates backup
# 4. Deploys to production
# 5. Runs verification
# 6. Creates version tag
```

### Manual Deployment

```bash
# Deploy to staging
gh workflow run ci-cd.yml \
  -f environment=staging

# Deploy to production
gh workflow run ci-cd.yml \
  -f environment=production
```

### Deployment Process

1. **Pre-Deployment**
   - Run all tests
   - Build Docker image
   - Create backup (production only)

2. **Deployment**
   - Pull latest images
   - Update containers
   - Wait for services to start

3. **Post-Deployment**
   - Run verification script
   - Execute smoke tests
   - Create deployment tag
   - Send notifications

### Monitoring Deployment

```bash
# View workflow run
gh run list --workflow=ci-cd.yml

# Watch specific run
gh run watch <run-id>

# View logs
gh run view <run-id> --log
```

---

## Rollback

### When to Rollback

Perform a rollback if:
- Deployment introduces critical bugs
- Services are failing verification
- Performance degradation detected
- Security vulnerabilities discovered

### Rollback Methods

#### Method 1: GitHub Actions (Recommended)

```bash
# Rollback staging to previous backup
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=previous \
  -f reason="Critical bug in formula execution"

# Rollback production to specific backup
gh workflow run rollback.yml \
  -f environment=production \
  -f backup_timestamp=20240101_120000 \
  -f reason="Performance regression"
```

#### Method 2: Manual Rollback

```bash
# SSH to server
ssh user@production-server

# Navigate to app
cd /opt/formula-api

# List backups
ls -lh backup/backups/

# Restore
bash backup/restore.sh backup/backups/formula-api-backup-20240101_120000.tar.gz
```

### Rollback Process

1. **Validation**
   - Validate rollback request
   - Create tracking issue

2. **Emergency Backup** (Production only)
   - Create backup of current state
   - Store as emergency restore point

3. **Service Stop**
   - Gracefully stop containers
   - Ensure clean shutdown

4. **Restoration**
   - Extract backup
   - Restore database
   - Restore Redis
   - Restore MLflow
   - Restore configuration

5. **Restart**
   - Start all services
   - Wait for initialization

6. **Verification**
   - Run full verification
   - Execute smoke tests
   - Check logs for errors

7. **Post-Rollback**
   - Create rollback tag
   - Generate report
   - Send notifications

### Rollback Verification

After rollback, verify:
```bash
# On the server
bash scripts/verify-deployment.sh

# Check specific services
docker-compose ps
docker-compose logs backend --tail 50

# Test API
curl https://api.example.com/health
```

---

## Secrets Configuration

### Required Secrets

| Secret | Description | Example |
|--------|-------------|---------|
| `STAGING_HOST` | Staging server hostname | staging.example.com |
| `STAGING_USER` | SSH user for staging | deploy |
| `STAGING_SSH_KEY` | Private SSH key | -----BEGIN OPENSSH PRIVATE KEY----- |
| `STAGING_API_KEY` | API key for staging | sk_staging_abc123 |
| `PRODUCTION_HOST` | Production server hostname | api.example.com |
| `PRODUCTION_USER` | SSH user for production | deploy |
| `PRODUCTION_SSH_KEY` | Private SSH key | -----BEGIN OPENSSH PRIVATE KEY----- |
| `PRODUCTION_API_KEY` | API key for production | sk_prod_xyz789 |

### Adding Secrets via CLI

```bash
# Add staging secrets
gh secret set STAGING_HOST -b "staging.example.com"
gh secret set STAGING_USER -b "deploy"
gh secret set STAGING_SSH_KEY < deploy_key
gh secret set STAGING_API_KEY -b "your-api-key"

# Add production secrets
gh secret set PRODUCTION_HOST -b "api.example.com"
gh secret set PRODUCTION_USER -b "deploy"
gh secret set PRODUCTION_SSH_KEY < production_key
gh secret set PRODUCTION_API_KEY -b "your-api-key"
```

### Viewing Secrets

```bash
# List all secrets
gh secret list

# View secret (value hidden)
gh secret get STAGING_HOST
```

---

## Workflow Status Badges

Add to your README.md:

```markdown
[![CI/CD](https://github.com/username/repo/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/username/repo/actions/workflows/ci-cd.yml)
[![Security Scan](https://github.com/username/repo/workflows/Security%20Scan/badge.svg)](https://github.com/username/repo/actions/workflows/security-scan.yml)
```

---

## Troubleshooting

### Workflow Fails

```bash
# View failed run
gh run view <run-id> --log

# Re-run failed jobs
gh run rerun <run-id> --failed

# Re-run entire workflow
gh run rerun <run-id>
```

### Deployment Fails

**Check logs:**
```bash
# View deployment logs
gh run view <run-id> --job=deploy-staging --log

# SSH to server and check
ssh user@server
cd /opt/formula-api
docker-compose logs backend
```

**Common issues:**
- SSH key authentication failure → Check `STAGING_SSH_KEY`
- Service not starting → Check Docker logs
- Verification fails → Run manual verification

### Security Scan Failures

**View detailed reports:**
```bash
gh run download <run-id>
cd security-reports/
cat bandit-report.json | jq
```

**Common issues:**
- False positives → Add to exclude list
- Known vulnerabilities → Update dependencies
- Secret detected → Remove and rotate

---

## Best Practices

### Branching Strategy

```
main/master     → Production deployments
develop         → Staging deployments
feature/*       → Development
hotfix/*        → Emergency fixes
```

### Deployment Flow

```
Feature → Develop → Staging → Main → Production
   ↓         ↓         ↓        ↓         ↓
  Test    Test+    Deploy   Test++   Deploy
           Lint     Test    Security  Verify
```

### Rollback Strategy

- Always have automated backups
- Test rollback procedures regularly
- Document rollback reasons
- Monitor post-rollback metrics

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Security Scanning Tools](https://github.com/analysis-tools-dev/static-analysis)

---

## Support

For issues with workflows:
1. Check workflow logs
2. Review this documentation
3. Check GitHub Actions status
4. Contact DevOps team
