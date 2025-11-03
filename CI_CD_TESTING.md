# CI/CD Testing Guide

Complete guide for testing the GitHub Actions CI/CD pipeline.

---

## ✅ Testing Checklist

This guide covers all 5 testing requirements:

1. ✓ Push to GitHub
2. Verify workflow runs
3. Test automated deployment
4. Verify security scans
5. Test rollback

---

## 1. ✓ Push to GitHub (COMPLETED)

**Status:** Successfully pushed to branch `claude/overwrite-repo-011CUkgR4MVFZiaCLhmPrvLw`

**Workflows Added:**
- `.github/workflows/ci-cd.yml` - Main CI/CD pipeline
- `.github/workflows/security-scan.yml` - Security scanning
- `.github/workflows/rollback.yml` - Emergency rollback
- `.github/WORKFLOWS.md` - Complete documentation
- `scripts/rollback.sh` - Manual rollback script

**Commit:** `325ba7f`

---

## 2. Verify Workflow Runs

### Expected Behavior

After pushing, the following workflows should automatically trigger:

**CI/CD Pipeline** (`ci-cd.yml`)
- Triggers on: Push to main/master/develop branches
- Current branch: Feature branch (won't auto-trigger)
- Status: Will trigger on next push to develop/main

**Security Scan** (`security-scan.yml`)
- Triggers on: Push to main/master/develop branches
- Current branch: Feature branch (won't auto-trigger)
- Status: Will trigger on next push to develop/main

### How to Verify

#### Option 1: Via GitHub Web Interface

```bash
# Open your repository in browser
# Navigate to: Actions tab
# You should see:
# - CI/CD Pipeline workflow
# - Security Scan workflow
# - Rollback Deployment workflow (manual only)
```

**Check:**
- All workflows are listed under "All workflows"
- No errors in workflow file syntax
- Workflows show as "waiting to run" or "queued" if triggered

#### Option 2: Via GitHub CLI

```bash
# List all workflows
gh workflow list

# Expected output:
# CI/CD Pipeline         active  12345
# Security Scan          active  12346
# Rollback Deployment    active  12347

# View workflow details
gh workflow view "CI/CD Pipeline"
gh workflow view "Security Scan"
gh workflow view "Rollback Deployment"

# Check recent runs
gh run list --workflow=ci-cd.yml
gh run list --workflow=security-scan.yml
```

### Trigger Test Runs

Since we're on a feature branch, manually trigger the workflows:

```bash
# Trigger CI/CD pipeline for staging
gh workflow run ci-cd.yml -f environment=staging

# Trigger security scan
gh workflow run security-scan.yml

# Watch the runs
gh run list --limit 5

# View real-time logs
gh run watch
```

### Verify Workflow Execution

```bash
# Get the latest run ID
RUN_ID=$(gh run list --workflow=ci-cd.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# View the run
gh run view $RUN_ID

# View logs
gh run view $RUN_ID --log

# Check job status
gh run view $RUN_ID --json jobs --jq '.jobs[] | {name: .name, status: .status, conclusion: .conclusion}'
```

**Expected Jobs:**
- ✓ test - Runs on Python 3.9, 3.10, 3.11
- ✓ lint - Code quality checks
- ✓ build - Docker image build (only on push/workflow_dispatch)
- ⏸ deploy-staging - Only on develop branch
- ⏸ deploy-production - Only on main branch

---

## 3. Test Automated Deployment

### Prerequisites

Before testing deployment, you need to configure secrets:

```bash
# Set up staging secrets
gh secret set STAGING_HOST -b "staging.example.com"
gh secret set STAGING_USER -b "deploy"
gh secret set STAGING_SSH_KEY < ~/.ssh/staging_key
gh secret set STAGING_API_KEY -b "sk_staging_test123"

# Set up production secrets
gh secret set PRODUCTION_HOST -b "api.example.com"
gh secret set PRODUCTION_USER -b "deploy"
gh secret set PRODUCTION_SSH_KEY < ~/.ssh/production_key
gh secret set PRODUCTION_API_KEY -b "sk_prod_test456"

# Verify secrets
gh secret list
```

### Test Staging Deployment

**Method 1: Push to develop branch**

```bash
# Merge feature branch to develop
git checkout develop
git merge claude/overwrite-repo-011CUkgR4MVFZiaCLhmPrvLw
git push origin develop

# This automatically triggers:
# 1. test job
# 2. lint job
# 3. build job
# 4. deploy-staging job
# 5. smoke-test job
```

**Method 2: Manual trigger**

```bash
# Trigger deployment manually
gh workflow run ci-cd.yml -f environment=staging

# Watch deployment
gh run watch
```

### Test Production Deployment

**WARNING:** Only do this when ready for production!

```bash
# Merge develop to main
git checkout main
git merge develop
git push origin main

# This automatically triggers production deployment
# with required manual approval (if configured)
```

### Verify Deployment

**Check deployment logs:**

```bash
# View latest deployment run
gh run list --workflow=ci-cd.yml --limit 1

# View deployment job logs
gh run view <run-id> --job=deploy-staging --log
# or
gh run view <run-id> --job=deploy-production --log
```

**Check deployed services:**

```bash
# If you have SSH access to the staging/production server
ssh deploy@staging.example.com

# Check services
cd /opt/formula-api
docker-compose ps

# Check verification results
cat /tmp/verify-deployment-*.log

# Test API
curl https://staging.example.com/health
```

**Expected Deployment Steps:**
1. ✓ SSH connection established
2. ✓ Docker images pulled
3. ✓ Services restarted
4. ✓ Health check passed
5. ✓ Verification script succeeded
6. ✓ Smoke tests passed (staging only)
7. ✓ Deployment tag created

---

## 4. Verify Security Scans

### Trigger Security Scan

```bash
# Manual trigger
gh workflow run security-scan.yml

# Wait for completion
gh run watch

# Get latest run
RUN_ID=$(gh run list --workflow=security-scan.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

### Check Scan Results

**View scan summary:**

```bash
# View overall status
gh run view $RUN_ID

# Check individual scan jobs
gh run view $RUN_ID --json jobs --jq '.jobs[] | {name: .name, conclusion: .conclusion}'
```

**Expected Jobs:**
- ✓ dependency-scan - Scans Python dependencies
- ✓ code-scan - Bandit & Safety checks
- ✓ codeql-analysis - GitHub CodeQL
- ✓ docker-scan - Trivy image scanning
- ✓ secret-scan - Gitleaks secret detection
- ✓ security-summary - Aggregated results

### Download Security Reports

```bash
# Download all artifacts
gh run download $RUN_ID

# View reports
cd pip-audit-report/
cat pip-audit-report.json | jq

cd ../security-reports/
cat bandit-report.json | jq
cat safety-report.json | jq

cd ../trivy-report/
cat trivy-report.json | jq

cd ../security-summary/
cat security-summary.md
```

### Check GitHub Security Tab

```bash
# Open in browser
gh browse --settings

# Navigate to:
# Security → Code scanning alerts
# Security → Secret scanning alerts

# Or via CLI
gh api /repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/code-scanning/alerts
```

### Review Security Issues

**Check for vulnerabilities:**

```bash
# View dependency alerts
gh api /repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/dependabot/alerts

# View CodeQL findings
gh api /repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/code-scanning/alerts | jq '.[] | {number, rule_id, severity}'
```

**Expected Results:**
- No critical vulnerabilities
- No exposed secrets
- Minor warnings acceptable
- All scans complete successfully

---

## 5. Test Rollback

### Prerequisites

For rollback testing, you need:
1. A deployed environment (staging or production)
2. At least one backup available
3. SSH access to the server

### Verify Backups Exist

```bash
# SSH to server
ssh deploy@staging.example.com

# Check backups
cd /opt/formula-api/backup/backups
ls -lh

# Expected: At least one backup file
# Format: formula-api-backup-YYYYMMDD_HHMMSS.tar.gz
```

### Test Staging Rollback

**Method 1: Via GitHub Actions (Recommended)**

```bash
# Trigger rollback workflow
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=previous \
  -f reason="Testing rollback procedure"

# Watch rollback
gh run watch

# Get rollback run ID
ROLLBACK_RUN_ID=$(gh run list --workflow=rollback.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# View detailed logs
gh run view $ROLLBACK_RUN_ID --log
```

**Method 2: Manual Script (Emergency)**

```bash
# SSH to server
ssh deploy@staging.example.com
cd /opt/formula-api

# Run rollback script
bash scripts/rollback.sh \
  -e staging \
  -b previous \
  -r "Manual rollback test" \
  -y

# Check rollback report
cat backup/rollback-report-*.txt
```

### Test Production Rollback

**⚠️ WARNING:** Production rollback should only be tested with proper planning!

```bash
# Trigger production rollback (requires approval)
gh workflow run rollback.yml \
  -f environment=production \
  -f backup_timestamp=previous \
  -f reason="Critical bug in new deployment"

# This will:
# 1. Create tracking issue
# 2. Wait for environment approval (if configured)
# 3. Create emergency backup
# 4. Stop services
# 5. Restore from backup
# 6. Verify rollback
# 7. Generate report
```

### Verify Rollback Success

**Check rollback status:**

```bash
# View rollback run
gh run view $ROLLBACK_RUN_ID

# Check rollback jobs
gh run view $ROLLBACK_RUN_ID --json jobs --jq '.jobs[] | {name: .name, conclusion: .conclusion}'
```

**Expected Jobs:**
- ✓ validate-rollback - Validates request & creates issue
- ✓ rollback-staging/production - Performs rollback
- ✓ post-rollback - Generates report

**Verify on server:**

```bash
# SSH to server
ssh deploy@staging.example.com
cd /opt/formula-api

# Check services
docker-compose ps

# Run verification
bash scripts/verify-deployment.sh

# Check rollback report
cat backup/rollback-report-*.txt
```

**Download rollback report:**

```bash
# Download artifacts
gh run download $ROLLBACK_RUN_ID

# View report
cat rollback-report/rollback-report.md
```

### Check Rollback Issue

```bash
# List recent issues with rollback label
gh issue list --label rollback

# View specific issue
gh issue view <issue-number>
```

**Expected Issue Content:**
- Environment name
- Backup timestamp used
- Reason for rollback
- Workflow run link
- Status updates

---

## Complete Testing Workflow

Here's a complete end-to-end testing sequence:

```bash
# 1. ✓ Already pushed to GitHub

# 2. Verify workflows
gh workflow list
gh workflow view "CI/CD Pipeline"

# 3. Trigger test runs
gh workflow run ci-cd.yml -f environment=staging
gh workflow run security-scan.yml

# 4. Watch execution
gh run watch

# 5. Check staging deployment
ssh deploy@staging.example.com "cd /opt/formula-api && docker-compose ps"

# 6. Download security reports
SECURITY_RUN=$(gh run list --workflow=security-scan.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download $SECURITY_RUN

# 7. Test rollback
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=previous \
  -f reason="Testing rollback functionality"

# 8. Verify rollback
ssh deploy@staging.example.com "cd /opt/formula-api && bash scripts/verify-deployment.sh"
```

---

## Troubleshooting

### Workflow Not Triggering

**Issue:** Workflow doesn't run after push

**Solutions:**
```bash
# Check workflow syntax
cat .github/workflows/ci-cd.yml | grep "^on:" -A 10

# Verify branch name
git branch --show-current

# Manually trigger
gh workflow run ci-cd.yml
```

### Deployment Fails

**Issue:** Deploy job fails with SSH error

**Solutions:**
```bash
# Verify secrets exist
gh secret list

# Test SSH key locally
ssh -i ~/.ssh/staging_key deploy@staging.example.com "echo 'SSH works'"

# Check workflow logs
gh run view <run-id> --log | grep -i "ssh\|error\|fail"
```

### Security Scan Failures

**Issue:** Security scan finds vulnerabilities

**Solutions:**
```bash
# Download and review reports
gh run download <run-id>

# Update vulnerable dependencies
pip-audit -r requirements.txt
pip install --upgrade <vulnerable-package>

# Re-run security scan
gh workflow run security-scan.yml
```

### Rollback Fails

**Issue:** Rollback cannot find backup

**Solutions:**
```bash
# SSH to server and check backups
ssh deploy@staging.example.com "ls -lh /opt/formula-api/backup/backups/"

# Create manual backup
ssh deploy@staging.example.com "cd /opt/formula-api && bash backup/backup.sh"

# Specify exact backup timestamp
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=20250103_120000 \
  -f reason="Manual rollback with specific backup"
```

---

## Success Criteria

All 5 testing requirements should be met:

### ✓ 1. Push to GitHub
- [x] All workflow files committed
- [x] Pushed to feature branch
- [x] No merge conflicts

### 2. Verify Workflow Runs
- [ ] Workflows visible in Actions tab
- [ ] No syntax errors
- [ ] Test run completes successfully
- [ ] All jobs execute as expected

### 3. Test Automated Deployment
- [ ] Staging deployment succeeds
- [ ] Services start correctly
- [ ] Verification passes
- [ ] Smoke tests pass (staging)
- [ ] Deployment tags created

### 4. Verify Security Scans
- [ ] All 5 scan jobs complete
- [ ] Reports generated
- [ ] No critical vulnerabilities
- [ ] Results visible in Security tab
- [ ] PR comments working (if applicable)

### 5. Test Rollback
- [ ] Rollback workflow triggers
- [ ] Backup restoration succeeds
- [ ] Services restart correctly
- [ ] Verification passes after rollback
- [ ] Rollback report generated
- [ ] Tracking issue created

---

## Next Steps

After completing all testing:

1. **Review Results** - Check all workflow runs succeeded
2. **Merge to Develop** - Merge feature branch to develop for staging deployment
3. **Monitor Staging** - Watch staging for 24 hours using `monitoring/monitor.sh`
4. **Merge to Main** - Merge develop to main for production deployment
5. **Production Monitoring** - Continuous monitoring in production

---

## Quick Reference

```bash
# View all workflows
gh workflow list

# Trigger CI/CD
gh workflow run ci-cd.yml -f environment=staging

# Trigger security scan
gh workflow run security-scan.yml

# Trigger rollback
gh workflow run rollback.yml \
  -f environment=staging \
  -f backup_timestamp=previous \
  -f reason="Your reason here"

# Watch runs
gh run watch

# Download artifacts
gh run download <run-id>

# View logs
gh run view <run-id> --log
```

---

## Documentation

For more details, see:
- `.github/WORKFLOWS.md` - Complete workflow documentation
- `PRODUCTION_DEPLOYMENT.md` - Deployment guide
- `backup/README.md` - Backup/restore procedures
- `scripts/rollback.sh --help` - Rollback script usage
