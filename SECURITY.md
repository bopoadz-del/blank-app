# Security Configuration Guide

## ⚠️ IMPORTANT: Security Requirements for Production

### Critical Security Items

#### 1. Secret Key (CRITICAL)
**Default**: `dev-secret-key`

**Must Change Before Production!**

Generate a secure secret key:
```bash
openssl rand -hex 32
```

Set in environment:
```bash
export SECRET_KEY="your-generated-secure-secret-key"
```

Or in `.env`:
```
SECRET_KEY=your-generated-secure-secret-key
```

**Risk if not changed**: 
- JWT tokens can be forged
- User sessions can be hijacked
- Authentication bypass possible

#### 2. API Key (CRITICAL)
**Default**: `test-api-key-12345`

**Must Change Before Production!**

Generate a secure API key:
```bash
openssl rand -hex 24
```

Set in environment:
```bash
export API_KEY="your-generated-secure-api-key"
```

**Risk if not changed**:
- Unauthorized API access
- Rate limiting bypass
- Data exposure

#### 3. Database Credentials (CRITICAL)
**Default**: `postgresql://postgres:postgres@localhost:5432/formulas`

**Must Change Before Production!**

Use environment variable:
```bash
export DATABASE_URL="postgresql://secure_user:secure_pass@db-host:5432/dbname"
```

**Never use default postgres/postgres credentials in production!**

**Risk if not changed**:
- Complete database compromise
- Data theft
- Data manipulation

### Medium Priority Security Items

#### 4. CORS Origins
**Default**: `["http://localhost:3000", "http://localhost:8000"]`

**Update for Production:**
```bash
export CORS_ORIGINS="https://yourdomain.com,https://api.yourdomain.com"
```

Or for development, use `*` carefully:
```bash
export CORS_ORIGINS="*"
```

**Risk**:
- Cross-origin attacks
- Unauthorized frontend access

#### 5. Host Binding
**Default**: `0.0.0.0` (binds to all interfaces)

**For Production**: This is intentional for container deployments (Docker, Render, etc.)

**For Local Development**: Consider using `127.0.0.1` for localhost-only access

```bash
export HOST="127.0.0.1"  # localhost only
# OR
export HOST="0.0.0.0"    # all interfaces (Docker/production)
```

## Security Checklist for Production

### Before Deploying:

- [ ] Change `SECRET_KEY` to a randomly generated value
- [ ] Change `API_KEY` to a randomly generated value
- [ ] Update `DATABASE_URL` with secure credentials
- [ ] Set `CORS_ORIGINS` to your actual domain(s)
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=False`
- [ ] Review all environment variables
- [ ] Never commit `.env` file to git
- [ ] Use secrets management (Render secrets, AWS Secrets Manager, etc.)

### Security Headers

The application automatically includes:
- CORS headers (configurable)
- Rate limiting (10 req/min default)
- JWT authentication for protected routes

### Additional Recommendations

#### 1. Enable HTTPS
Always use HTTPS in production (automatic on Render.com)

#### 2. Rate Limiting
Adjust based on your needs:
```bash
export RATE_LIMIT_PER_MINUTE=100  # Increase for production
```

#### 3. Database SSL
Use SSL for database connections in production:
```bash
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require"
```

#### 4. Redis Password
If using Redis, set a password:
```bash
export REDIS_PASSWORD="your-secure-redis-password"
```

#### 5. Log Level
Set appropriate log level:
```bash
export LOG_LEVEL=WARNING  # or ERROR for production
```

## Security Best Practices

### 1. Secrets Management

**DO**:
- ✅ Use environment variables for all secrets
- ✅ Use secrets management services (AWS Secrets Manager, Render secrets)
- ✅ Rotate secrets regularly
- ✅ Use different secrets for different environments

**DON'T**:
- ❌ Hard-code secrets in code
- ❌ Commit secrets to git
- ❌ Share secrets in plain text
- ❌ Use same secrets across environments

### 2. Database Security

**DO**:
- ✅ Use strong passwords
- ✅ Limit database user permissions
- ✅ Use SSL/TLS for connections
- ✅ Regular backups
- ✅ Monitor for unusual activity

**DON'T**:
- ❌ Use default credentials
- ❌ Grant superuser permissions to app
- ❌ Expose database publicly
- ❌ Store sensitive data unencrypted

### 3. API Security

**DO**:
- ✅ Use API key authentication
- ✅ Implement rate limiting
- ✅ Validate all inputs
- ✅ Use HTTPS only
- ✅ Log access attempts

**DON'T**:
- ❌ Accept unauthenticated requests to protected endpoints
- ❌ Trust client-side validation only
- ❌ Expose internal errors to clients
- ❌ Allow unlimited API calls

### 4. Frontend Security

**DO**:
- ✅ Keep dependencies updated
- ✅ Use Content Security Policy
- ✅ Sanitize user inputs
- ✅ Use HTTPS only

**DON'T**:
- ❌ Store secrets in frontend code
- ❌ Trust user input
- ❌ Use outdated dependencies with known vulnerabilities

## Vulnerability Scanning

### Python Backend
```bash
# Install bandit
pip install bandit

# Scan for security issues
bandit -r app/ -f txt
```

### Frontend
```bash
# Audit npm packages
cd frontend
npm audit

# Fix automatically (when safe)
npm audit fix

# Fix with breaking changes (review carefully)
npm audit fix --force
```

### Dependencies
```bash
# Check for known vulnerabilities
pip install safety
safety check --file requirements.txt
```

## Known Issues and Mitigations

### 1. Development Dependencies (Frontend)
**Issue**: esbuild vulnerability (GHSA-67mh-4wv8-2f99)
**Severity**: Moderate
**Impact**: Development only, does not affect production builds
**Mitigation**: Not required for production (build output is safe)

### 2. Hardcoded Defaults
**Issue**: Default credentials in config.py
**Severity**: Critical if not changed
**Mitigation**: MUST change via environment variables before production
**Status**: Documented and warnings added

## Monitoring and Auditing

### Enable Audit Logging
The platform includes audit logging. Ensure it's enabled:
```bash
export ENABLE_AUDIT_LOG=true
```

### Monitor for Security Events
- Failed authentication attempts
- Rate limit violations
- Unusual API access patterns
- Database connection failures

### Regular Security Reviews
- Weekly: Review access logs
- Monthly: Update dependencies
- Quarterly: Security audit
- Annually: Penetration testing

## Incident Response

### If Secret Key Compromised:
1. Generate new secret key immediately
2. Update production environment
3. Invalidate all existing JWT tokens
4. Force all users to re-authenticate
5. Review access logs
6. Notify affected users if necessary

### If Database Compromised:
1. Isolate database immediately
2. Change all database credentials
3. Review database logs
4. Assess data exposure
5. Notify affected parties per regulations
6. Restore from backup if necessary

### If API Key Compromised:
1. Rotate API key immediately
2. Update all legitimate clients
3. Review API access logs
4. Block suspicious IPs if identified

## Compliance

### GDPR Considerations
- Users can request data deletion
- Audit logs track data access
- Data encryption at rest and in transit

### HIPAA Considerations (if applicable)
- Enable all audit logging
- Use encrypted database
- Implement access controls
- Regular security reviews

## Contact

For security issues, please report to:
- Security team (if applicable)
- GitHub Security Advisories
- Direct communication (not public issues)

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Render Security Best Practices](https://render.com/docs/security)

---

**Last Updated**: November 12, 2025
**Security Version**: 1.0
**Status**: Active
