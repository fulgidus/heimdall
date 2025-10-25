# Development Default Credentials

> ⚠️ **SECURITY WARNING - DEVELOPMENT ONLY**
> 
> **These credentials are ONLY for local development and testing environments.**
> 
> **DO NOT USE THESE CREDENTIALS IN PRODUCTION.**
> 
> These default values are publicly documented and must be changed before any production deployment. For production environments, use proper secret management tools (Kubernetes Secrets, HashiCorp Vault, AWS Secrets Manager, etc.).

---

## Overview

This document provides all default usernames and passwords for Heimdall services when running in local development mode using `docker-compose`. These credentials allow you to quickly access various management UIs and services after starting the development environment.

**Quick Start:**
```bash
# Copy environment template
cp .env.example .env

# Start all services
docker-compose up -d

# Wait for services to be healthy (2-3 minutes)
make health-check
```

---

## 📊 Default Credentials Table

| Service | Default User | Default Password | URL / Port | Notes |
|---------|--------------|------------------|------------|-------|
| **PostgreSQL** | `heimdall_user` | `changeme` | `localhost:5432` | Main database with TimescaleDB extension |
| **pgAdmin** | `admin@pg.com` | `admin` | http://localhost:5050 | PostgreSQL web management UI |
| **RabbitMQ Management** | `guest` | `guest` | http://localhost:15672 | Message queue management UI |
| **Redis** | *(no username)* | `changeme` | `localhost:6379` | Cache and Celery result backend |
| **Redis Commander** | *(no auth)* | *(no auth)* | http://localhost:8081 | Redis web management UI |
| **MinIO Console** | `minioadmin` | `minioadmin` | http://localhost:9001 | S3-compatible object storage UI |
| **Grafana** | `admin` | `admin` | http://localhost:3000 | Metrics visualization and dashboards |
| **Prometheus** | *(no auth)* | *(no auth)* | http://localhost:9090 | Metrics collection and queries |
| **Keycloak Admin Console** | `admin` | `admin` | http://localhost:8080 | Authentication & Authorization Provider |
| **API Gateway** | *JWT Bearer Token* | *(via Keycloak)* | http://localhost:8000 | Main REST API entry point - requires authentication |
| **RF Acquisition Service** | *JWT Bearer Token* | *(via Keycloak)* | http://localhost:8001 | WebSDR data collection service - requires authentication |
| **Training Service** | *JWT Bearer Token* | *(via Keycloak)* | http://localhost:8002 | ML model training service - requires authentication |
| **Inference Service** | *JWT Bearer Token* | *(via Keycloak)* | http://localhost:8003 | Real-time inference service - requires authentication |
| **Data Ingestion Web** | *JWT Bearer Token* | *(via Keycloak)* | http://localhost:8004 | Data collection UI and API - requires authentication |
| **Frontend (Dev)** | *SSO via Keycloak* | *(login required)* | http://localhost:5173 | React development server with SSO authentication |

---

## 🚀 Getting Started

### 1. Initial Setup

```bash
# Navigate to project directory
cd heimdall

# Copy the environment template
cp .env.example .env

# (Optional) Edit .env if you want to change default values
# For development, the defaults work fine

# Start all infrastructure and services
docker-compose up -d

# Monitor startup logs (optional)
docker-compose logs -f

# Wait for services to become healthy
# This usually takes 2-3 minutes
make health-check
```

### 2. Verify Services Are Running

```bash
# Check all container status
docker-compose ps

# All services should show "healthy" status
# Example output:
# NAME                        STATUS
# heimdall-postgres          Up (healthy)
# heimdall-rabbitmq          Up (healthy)
# heimdall-redis             Up (healthy)
# heimdall-minio             Up (healthy)
# heimdall-grafana           Up (healthy)
# ... and so on
```

---

## 🔐 Connection Examples

### PostgreSQL Database

**Using psql CLI:**
```bash
# Connect to main database
psql -h localhost -U heimdall_user -d heimdall -p 5432
# Password: changeme

# Or using connection string
psql "postgresql://heimdall_user:changeme@localhost:5432/heimdall"
```

**Connection String Format:**
```
postgresql://heimdall_user:changeme@localhost:5432/heimdall
```

**Using pgAdmin Web UI:**
1. Open http://localhost:5050
2. Login with `admin@pg.com` / `admin`
3. Add new server:
   - Name: `Heimdall Local`
   - Host: `postgres` (or `localhost` if connecting from host)
   - Port: `5432`
   - Username: `heimdall_user`
   - Password: `changeme`

### RabbitMQ

**Management UI:**
1. Open http://localhost:15672
2. Login with `guest` / `guest`
3. Navigate to:
   - **Queues** tab to see task queues
   - **Connections** tab to see active connections
   - **Exchanges** tab to see routing configuration

**Connection String:**
```
amqp://guest:guest@localhost:5672//
```

### Redis

**Using redis-cli:**
```bash
# Connect with password
redis-cli -h localhost -p 6379 -a changeme

# Test connection
redis-cli -h localhost -p 6379 -a changeme PING
# Expected output: PONG
```

**Using Redis Commander Web UI:**
1. Open http://localhost:8081
2. No authentication required (dev only!)
3. Browse keys, run commands, monitor memory

**Connection String:**
```
redis://:changeme@localhost:6379/0
```

### MinIO (S3-compatible Object Storage)

**MinIO Console (Web UI):**
1. Open http://localhost:9001
2. Login with `minioadmin` / `minioadmin`
3. Browse buckets:
   - `heimdall-raw-iq` - RF signal data
   - `heimdall-models` - Trained ML models
   - `heimdall-mlflow` - MLflow artifacts
   - `heimdall-datasets` - Training datasets

**Using AWS CLI:**
```bash
# Configure AWS CLI for MinIO
aws configure --profile heimdall-local
# AWS Access Key ID: minioadmin
# AWS Secret Access Key: minioadmin
# Default region: us-east-1
# Default output format: json

# List buckets
aws --profile heimdall-local --endpoint-url http://localhost:9000 s3 ls

# Upload a file
aws --profile heimdall-local --endpoint-url http://localhost:9000 s3 cp file.txt s3://heimdall-raw-iq/
```

**Using Python boto3:**
```python
import boto3

s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
)

# List buckets
response = s3_client.list_buckets()
print(response['Buckets'])
```

### Grafana

**Access Dashboard:**
1. Open http://localhost:3000
2. Login with `admin` / `admin`
3. On first login, you'll be prompted to change the password
   - For dev, you can skip this or set to any value
4. Add Prometheus data source:
   - URL: `http://prometheus:9090`
   - No authentication required

### Prometheus

**Access Metrics:**
1. Open http://localhost:9090
2. No authentication required
3. Try sample queries:
   - `up` - Check which targets are up
   - `rate(http_requests_total[5m])` - HTTP request rate
   - `container_memory_usage_bytes` - Container memory usage

### Keycloak

**Admin Console:**
1. Open http://localhost:8080
2. Login with `admin` / `admin`
3. Select "Heimdall SDR" realm from dropdown (top left)
4. Navigate to:
   - **Users** - Manage users and passwords
   - **Clients** - Configure OAuth2/OIDC clients
   - **Realm Roles** - Manage user roles (admin, operator, viewer)
   - **Sessions** - Monitor active user sessions

**Default User Accounts:**
- **Admin**: `admin` / `admin` - Full system access
- **Operator**: `operator` / `operator` - Read/write access to signals and models
- **Viewer**: `viewer` / `viewer` - Read-only access

**API Testing with JWT:**
```bash
# Get access token for admin user
curl -X POST http://localhost:8080/realms/heimdall/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=heimdall-frontend" \
  -d "username=admin" \
  -d "password=admin" \
  | jq -r '.access_token'

# Use token to access protected endpoint
TOKEN="<your-token-here>"
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/health
```

---

## 🛠️ Service-Specific Login Instructions

### API Gateway (Main REST API)

```bash
# Health check
curl http://localhost:8000/health

# Get API documentation
curl http://localhost:8000/docs
# Or open in browser: http://localhost:8000/docs

# Example: Submit RF acquisition task
curl -X POST http://localhost:8000/api/v1/acquire \
  -H "Content-Type: application/json" \
  -d '{
    "frequency_mhz": 145.500,
    "duration_seconds": 10
  }'
```

### RF Acquisition Service

```bash
# Health check
curl http://localhost:8001/health

# API documentation
http://localhost:8001/docs
```

### Training Service

```bash
# Health check
curl http://localhost:8002/health

# API documentation
http://localhost:8002/docs
```

### Inference Service

```bash
# Health check
curl http://localhost:8003/health

# API documentation
http://localhost:8003/docs

# Example: Run inference
curl -X POST http://localhost:8003/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{ "iq_data": [...] }'
```

### Data Ingestion Web

```bash
# Health check
curl http://localhost:8004/health

# API documentation
http://localhost:8004/docs
```

### Frontend (Development Server)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Open in browser
# http://localhost:5173
```

---

## 📝 .env.example Reference

The `.env.example` file at the project root contains all these default values. Here's a quick reference:

```bash
# PostgreSQL
POSTGRES_DB=heimdall
POSTGRES_USER=heimdall_user
POSTGRES_PASSWORD=changeme

# RabbitMQ
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest

# Redis
REDIS_PASSWORD=changeme

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# pgAdmin
PGADMIN_EMAIL=admin@pg.com
PGADMIN_PASSWORD=admin
```

**To use:**
```bash
cp .env.example .env
# Edit .env if needed, or use as-is for development
```

---

## 🔄 Post-Setup: Changing Passwords

### Why Change Passwords?

Even in development, you may want to change passwords if:
- Working with sensitive or real data
- Exposing services to a network (not just localhost)
- Preparing for production deployment
- Following security best practices

### How to Change Passwords

#### Option 1: Update .env File (Recommended)

1. Edit the `.env` file:
```bash
nano .env
# or
vim .env
```

2. Change the password values:
```bash
# Example: Change PostgreSQL password
POSTGRES_PASSWORD=my_new_secure_password

# Example: Change Redis password
REDIS_PASSWORD=redis_new_password
```

3. Restart services for changes to take effect:
```bash
docker-compose down
docker-compose up -d
```

4. **Important:** If you changed database passwords, you may need to recreate volumes:
```bash
docker-compose down -v  # WARNING: This deletes all data!
docker-compose up -d
```

#### Option 2: Update docker-compose.yml

For permanent changes across environments, edit `docker-compose.yml` default values. However, for personal development setups, using `.env` is preferred.

#### Option 3: Use Secret Management (Production)

For production deployments:
- **Kubernetes:** Use Kubernetes Secrets
- **Docker Swarm:** Use Docker Secrets
- **Cloud:** Use cloud provider secret managers (AWS Secrets Manager, Azure Key Vault, Google Secret Manager)
- **Self-hosted:** Use HashiCorp Vault or similar

### Rotating Credentials

If credentials are compromised:

1. **Immediate action:**
```bash
# Stop all services
docker-compose down

# Remove all data (if sensitive data exists)
docker-compose down -v

# Update .env with new passwords
nano .env

# Restart with new credentials
docker-compose up -d
```

2. **Verify new credentials:**
```bash
# Test each service with new credentials
make health-check
```

---

## 🚨 Security Best Practices

### Development Environment

1. **Never commit .env file**
   - The `.env` file is in `.gitignore` by default
   - Always use `.env.example` as a template

2. **Use localhost only**
   - Don't expose ports to external networks in development
   - Use firewall rules if needed

3. **Regular updates**
   - Keep Docker images updated: `docker-compose pull`
   - Update service versions regularly

4. **Clean up**
   - Remove unused containers: `docker system prune`
   - Clear sensitive data when done: `docker-compose down -v`

### Production Environment

1. **Never use these default credentials**
   - Generate strong, random passwords
   - Use password managers or secret generators

2. **Use proper secret management**
   - Kubernetes Secrets for K8s deployments
   - Cloud provider secret managers
   - HashiCorp Vault for advanced scenarios

3. **Enable authentication everywhere**
   - Add authentication to all API endpoints
   - Use OAuth2/OIDC for frontend authentication
   - Implement role-based access control (RBAC)

4. **Network security**
   - Use TLS/SSL for all connections
   - Implement network policies in Kubernetes
   - Use VPNs or private networks

5. **Audit and monitoring**
   - Enable audit logging
   - Monitor for suspicious activity
   - Set up alerts for security events

---

## 🔒 What If Credentials Are Accidentally Committed?

If you accidentally commit credentials to Git:

### 1. Remove from Git History

```bash
# If commit not pushed yet
git reset --soft HEAD~1  # Undo last commit, keep changes
# Edit files to remove credentials
git add .
git commit -m "chore: update configuration"

# If commit already pushed (use with caution!)
# Consider the repository compromised and rotate all credentials
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: Rewrites history)
git push origin --force --all
```

### 2. Rotate All Compromised Credentials

```bash
# Generate new passwords
# Update .env with new values
# Restart all services
docker-compose down -v
docker-compose up -d
```

### 3. Consider Repository as Compromised

- If real production credentials were committed, consider them public
- Change them immediately in production
- Review security logs for unauthorized access
- Notify security team if applicable

### 4. Use Git Secrets Prevention (Optional)

```bash
# Install git-secrets
git secrets --install

# Add patterns to prevent committing
git secrets --register-aws
git secrets --add 'password.*=.*'
git secrets --add 'POSTGRES_PASSWORD=.*'
git secrets --add 'REDIS_PASSWORD=.*'
```

---

## 📚 Additional Resources

- [Main README](../README.md) - Project overview
- [Installation Guide](installation.md) - Detailed setup instructions
- [Architecture Documentation](ARCHITECTURE.md) - System architecture
- [API Documentation](api_documentation.md) - REST API reference
- [Troubleshooting Guide](troubleshooting_guide.md) - Common issues and solutions
- [Phase 1 Guide](agents/20251022_080000_phase1_guide.md) - Infrastructure setup details

---

## 🆘 Troubleshooting

### Can't connect to PostgreSQL

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Try resetting the database
docker-compose down
docker volume rm heimdall_postgres_data
docker-compose up -d postgres
```

### RabbitMQ Management UI not loading

```bash
# Check if RabbitMQ is healthy
docker-compose ps rabbitmq

# Restart RabbitMQ
docker-compose restart rabbitmq

# Check logs
docker-compose logs rabbitmq
```

### MinIO buckets not created

```bash
# Check minio-init logs
docker-compose logs minio-init

# Manually create buckets
docker-compose exec minio mc alias set heimdall http://localhost:9000 minioadmin minioadmin
docker-compose exec minio mc mb heimdall/heimdall-raw-iq
docker-compose exec minio mc mb heimdall/heimdall-models
docker-compose exec minio mc mb heimdall/heimdall-mlflow
docker-compose exec minio mc mb heimdall/heimdall-datasets
```

### Redis connection refused

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli -h localhost -p 6379 -a changeme PING

# Check logs
docker-compose logs redis
```

### Services won't start

```bash
# Check Docker resources
docker system df

# Clean up unused resources
docker system prune -a

# Restart Docker daemon (if needed)
sudo systemctl restart docker  # Linux
# or restart Docker Desktop on macOS/Windows

# Try starting services again
docker-compose down
docker-compose up -d
```

---

## ⚠️ Final Security Reminder

**THESE CREDENTIALS ARE FOR DEVELOPMENT ONLY**

Before deploying to any production or public-facing environment:

✅ Generate strong, unique passwords for all services  
✅ Use proper secret management tools  
✅ Enable authentication on all endpoints  
✅ Use TLS/SSL for all connections  
✅ Implement network security policies  
✅ Enable audit logging and monitoring  
✅ Follow security best practices for your deployment platform  

**Never use these default development credentials in production!**

---

*Last updated: 2025-10-24*
