# Render Network Configuration

## Outbound IP Addresses (Oregon Region)

These are the IP addresses that Render uses for outbound network requests from our services.

### IP List:
- `44.229.227.142`
- `54.188.71.94`
- `52.13.128.108`
- `74.220.48.0/24` (range: 74.220.48.1 - 74.220.48.254)
- `74.220.56.0/24` (range: 74.220.56.1 - 74.220.56.254)

### Region:
**Oregon** (us-west-2)

### Important Notes:
- These IPs are **shared** across all Render services in Oregon region
- They are **not unique** to our service
- Requests from our backend will originate from one of these IPs
- IPs may change if service is migrated to different region

---

## When to Use These IPs

### ✅ Current Requirements: **NONE**
Our current integrations don't require IP whitelisting:
- Google Drive API → Uses OAuth 2.0
- OpenAI API → Uses API key authentication
- Render PostgreSQL → Internal network (no external IP needed)

### 🔮 Future Use Cases:

#### 1. Jetson Edge Device Firewall Configuration
If Jetson AGX Orin devices need to accept connections from backend:

**UFW Configuration:**
```bash
# On Jetson device
sudo ufw allow from 44.229.227.142 comment "Render backend"
sudo ufw allow from 54.188.71.94 comment "Render backend"
sudo ufw allow from 52.13.128.108 comment "Render backend"
sudo ufw allow from 74.220.48.0/24 comment "Render backend range"
sudo ufw allow from 74.220.56.0/24 comment "Render backend range"
sudo ufw reload
```

**iptables Configuration:**
```bash
# On Jetson device
iptables -A INPUT -s 44.229.227.142 -j ACCEPT -m comment --comment "Render backend"
iptables -A INPUT -s 54.188.71.94 -j ACCEPT -m comment --comment "Render backend"
iptables -A INPUT -s 52.13.128.108 -j ACCEPT -m comment --comment "Render backend"
iptables -A INPUT -s 74.220.48.0/24 -j ACCEPT -m comment --comment "Render backend range"
iptables -A INPUT -s 74.220.56.0/24 -j ACCEPT -m comment --comment "Render backend range"
iptables-save > /etc/iptables/rules.v4
```

#### 2. External Database IP Whitelisting

**MongoDB Atlas:**
1. Network Access → Add IP Address
2. Add all 5 IPs/ranges above
3. Description: "Render ml-platform-backend"

**AWS RDS Security Group:**
1. Add inbound rule for PostgreSQL (port 5432)
2. Source: Custom IP
3. Add all 5 IPs/ranges

**Google Cloud SQL:**
```bash
gcloud sql instances patch INSTANCE_NAME \
  --authorized-networks=44.229.227.142,54.188.71.94,52.13.128.108,74.220.48.0/24,74.220.56.0/24
```

#### 3. Enterprise API Integration

If integrating with enterprise systems (SAP, Oracle, banking APIs):
- Provide this IP list to vendor
- Request whitelisting for API endpoints
- Test connectivity after whitelisting

#### 4. Webhook Source Verification

If setting up webhooks to external systems that need to verify source:
- Configure external system to only accept webhooks from these IPs
- Add IP validation in external firewall

---

## Terraform Example (if using IaC)

```hcl
variable "render_outbound_ips" {
  type = list(string)
  default = [
    "44.229.227.142/32",
    "54.188.71.94/32",
    "52.13.128.108/32",
    "74.220.48.0/24",
    "74.220.56.0/24"
  ]
  description = "Render Oregon region outbound IPs"
}

# AWS Security Group example
resource "aws_security_group_rule" "allow_render" {
  count             = length(var.render_outbound_ips)
  type              = "ingress"
  from_port         = 5432
  to_port           = 5432
  protocol          = "tcp"
  cidr_blocks       = [var.render_outbound_ips[count.index]]
  security_group_id = aws_security_group.database.id
  description       = "Allow Render backend access"
}
```

---

## Testing IP Configuration

### Verify Outbound IP:

**From Render backend** (test in production):
```bash
# Add to backend for testing
curl https://api.ipify.org
# Should return one of the IPs above
```

### Test Firewall Rules:

**On Jetson device:**
```bash
# Check UFW status
sudo ufw status numbered

# Test connectivity from backend
# (run from Render shell)
curl http://JETSON_DEVICE_IP:PORT/health
```

---

## Security Best Practices

### ✅ DO:
- Use these IPs in combination with authentication (API keys, OAuth)
- Document which external services have these IPs whitelisted
- Review and update when migrating regions
- Use narrowest IP range needed (prefer /32 over /24 when possible)

### ❌ DON'T:
- Rely solely on IP whitelisting for security
- Assume these IPs are unique to your service
- Hardcode these in application code (use environment variables)
- Forget to update if migrating regions

---

## Region-Specific IPs

If you deploy to other Render regions in the future, you'll need different IPs:

| Region | Get IPs From |
|--------|--------------|
| Oregon (us-west-2) | ✅ Listed above |
| Ohio (us-east-2) | Render Dashboard → Network |
| Frankfurt (eu-central-1) | Render Dashboard → Network |
| Singapore (ap-southeast-1) | Render Dashboard → Network |

**How to find**: Render Dashboard → Your Service → Network tab

---

## Support

For questions about IP addresses:
- Render Docs: https://render.com/docs/static-outbound-ip-addresses
- Render Support: https://render.com/support

Last Updated: 2025-11-07
