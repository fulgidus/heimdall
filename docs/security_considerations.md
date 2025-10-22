# Security Considerations

## Overview

This document outlines security considerations and best practices for Heimdall deployment and usage.

## Data Security

### Encryption at Rest

**PostgreSQL**:
```bash
# Enable encryption with pgcrypto extension
CREATE EXTENSION pgcrypto;

# Encrypt sensitive columns
ALTER TABLE signal_measurements 
ADD COLUMN encrypted_coordinates bytea;

UPDATE signal_measurements SET encrypted_coordinates = 
  pgp_sym_encrypt(coordinates::text, 'encryption_key');
```

**MinIO (S3-Compatible)**:
```bash
# Enable server-side encryption
mc admin config set minio server_side_encryption/sse:kms
```

### Encryption in Transit

```bash
# Enable HTTPS with SSL certificates
# nginx/ingress configuration
ssl_certificate /path/to/cert.pem;
ssl_certificate_key /path/to/key.pem;

# Force HTTPS redirect
server {
    listen 80;
    return 301 https://$server_name$request_uri;
}
```

## API Security

### Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthCredential
from fastapi import Depends, HTTPException

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthCredential = Depends(security)):
    """Verify API key."""
    if credentials.credentials != os.getenv('API_KEY'):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/v1/tasks")
@limiter.limit("100/minute")
async def get_tasks():
    """Get tasks (rate limited to 100 per minute)."""
    pass
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heimdall.example.com"],  # Whitelist domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization"],
)
```

## Infrastructure Security

### Network Segmentation

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-gateway-policy
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

### Container Security

```dockerfile
# Use non-root user
FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
USER appuser

# No privileged mode
# Read-only filesystem where possible
# Resource limits
```

### Secrets Management

```bash
# Kubernetes Secrets
kubectl create secret generic heimdall-secrets \
  --from-literal=db_password=... \
  --from-literal=api_key=... \
  -n heimdall

# In deployment
apiVersion: v1
kind: Pod
metadata:
  name: api-gateway
spec:
  containers:
    - name: api-gateway
      env:
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: heimdall-secrets
              key: db_password
```

### Audit Logging

```python
import logging
from datetime import datetime

audit_logger = logging.getLogger("audit")

def log_api_access(request, response):
    """Log API access for audit trail."""
    audit_logger.info({
        "timestamp": datetime.utcnow().isoformat(),
        "user": request.headers.get("Authorization"),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "remote_addr": request.client.host,
    })
```

## Data Privacy

### GDPR Compliance

- **Consent**: Obtain explicit consent for data collection
- **Right to Access**: Implement data export functionality
- **Right to Deletion**: Implement data deletion procedures
- **Data Minimization**: Collect only necessary data

### Data Retention

```sql
-- Automated data deletion after retention period
CREATE OR REPLACE FUNCTION delete_old_data()
RETURNS void AS $$
BEGIN
  DELETE FROM signal_measurements 
  WHERE created_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly
SELECT cron.schedule('delete_old_data', '0 2 1 * *', 'SELECT delete_old_data()');
```

### Anonymization

```python
def anonymize_measurements(measurements):
    """Remove identifying information from measurements."""
    for m in measurements:
        # Round coordinates to reduce precision
        m['latitude'] = round(m['latitude'], 2)
        m['longitude'] = round(m['longitude'], 2)
        # Remove station names
        m['station_name'] = 'anonymous'
    return measurements
```

## Vulnerability Management

### Dependency Scanning

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Or use GitHub Dependabot
# Enable in repository settings

# SBOM generation
pip install cyclonedx-bom
cyclonedx-bom -o sbom.json
```

### Code Scanning

```bash
# Static analysis
pip install bandit
bandit -r services/

# Security linting
pip install pylint
pylint --load-plugins=pylint_security services/
```

### Penetration Testing

```bash
# Security testing framework
pip install requests
pip install hypothesis

# Example: Test SQL injection
def test_sql_injection():
    """Test API endpoint for SQL injection vulnerability."""
    payloads = ["' OR '1'='1", "'; DROP TABLE tasks; --"]
    for payload in payloads:
        response = client.get(f"/tasks?name={payload}")
        assert response.status_code != 500  # Should handle gracefully
```

## WebSDR Security

### Station Vetting

```python
# Verify WebSDR station authenticity
TRUSTED_STATIONS = {
    "giaveno": {
        "url": "http://sdr1.ik1jns.it:8076/",
        "fingerprint": "...",  # Certificate fingerprint
        "location": {"lat": 45.02, "lon": 7.29},
        "trusted": True
    }
}

def validate_station(station_name):
    """Verify station is in trusted list."""
    if station_name not in TRUSTED_STATIONS:
        raise SecurityError(f"Untrusted station: {station_name}")
    return TRUSTED_STATIONS[station_name]
```

## Monitoring & Detection

### Intrusion Detection

```bash
# Enable audit logging in Kubernetes
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: RequestResponse
    omitStages:
      - RequestReceived
    resources:
      - group: ""
        resources: ["secrets", "configmaps"]
```

### Anomaly Detection

```python
from scipy import stats

def detect_anomaly(request_data):
    """Detect unusual API requests."""
    # Check request size
    if len(request_data) > 1000000:  # 1MB
        alert("Large request detected")
    
    # Check request rate per IP
    if request_count_per_ip > 1000:
        alert("Possible DDoS attack")
    
    # Check request patterns
    if z_score > 3:  # Statistical anomaly
        alert("Unusual pattern detected")
```

## Incident Response

### Security Incident Procedure

1. **Detection**: Monitor logs and alerts
2. **Containment**: Isolate affected systems
3. **Investigation**: Analyze incident
4. **Remediation**: Fix root cause
5. **Recovery**: Restore services
6. **Post-Mortem**: Document lessons learned

### Emergency Contacts

```yaml
Security Team:
  - Email: security@heimdall.example.com
  - Phone: +1-XXX-XXX-XXXX
  - On-call: https://incidents.heimdall.example.com

Emergency Procedures:
  - Incident: Create urgent issue on GitHub
  - Security Issue: Email security@heimdall.example.com
  - Escalation: Contact project owner (fulgidus)
```

## Compliance

### Standards Compliance

- **ISO 27001**: Information Security Management
- **OWASP Top 10**: Web application security
- **CWE/SANS Top 25**: Common weaknesses
- **NIST Cybersecurity Framework**: Risk management

### Regular Audits

```bash
# Monthly security checks
- Dependency scanning
- Code scanning
- Vulnerability assessment
- Access review
- Log analysis

# Quarterly
- Penetration testing
- Security audit
- Compliance check

# Annually
- Full security assessment
- Policy review
- Training
```

## Best Practices

### For Developers

1. Never commit secrets to Git
2. Use parameterized queries (prevent SQL injection)
3. Validate all input
4. Handle errors safely (don't leak info)
5. Keep dependencies updated
6. Use strong cryptography

### For Operations

1. Enable HTTPS everywhere
2. Use strong authentication
3. Implement rate limiting
4. Monitor for anomalies
5. Regular backups
6. Incident response plan

### For Users

1. Use strong API keys
2. Rotate credentials regularly
3. Monitor API usage
4. Report security issues
5. Keep software updated

---

**Related**: [Deployment Guide](./deployment_instructions.md) | [FAQ](./faqs.md)

**Security Issues**: Email security@heimdall.example.com (do not use public issues)

**Last Updated**: October 2025
