# Deployment Instructions

## Overview

This guide covers deploying Heimdall to production environments using Docker and Kubernetes.

## Prerequisites

- Docker 24.0+ and Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- Helm 3.0+ (for Helm charts)
- Cloud provider account (AWS, GCP, Azure, etc.)

## Docker Deployment

### Building Production Images

```bash
# Build all images
docker-compose -f docker-compose.prod.yml build

# Build specific service
docker-compose -f docker-compose.prod.yml build api-gateway

# Push to registry
docker tag heimdall-api registry.example.com/heimdall/api:latest
docker push registry.example.com/heimdall/api:latest
```

### Environment Configuration

Create `.env.prod` with production settings:

```env
# Database
DATABASE_URL=postgresql://user:password@prod-db.example.com:5432/heimdall
POSTGRES_PASSWORD=secure-password-here

# Cache & Queue
REDIS_URL=redis://prod-redis.example.com:6379
RABBITMQ_URL=amqp://rabbitmq_user:password@prod-rabbitmq.example.com:5672

# Storage
MINIO_ACCESS_KEY=prod-access-key
MINIO_SECRET_KEY=prod-secret-key
MINIO_ENDPOINT=s3.example.com

# MLflow
MLFLOW_TRACKING_URI=http://prod-mlflow.example.com:5000

# Security
SECRET_KEY=your-production-secret-key
API_KEY=your-api-key

# Features
DEBUG=False
LOG_LEVEL=INFO
```

### Running Production Stack

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=5

# Stop environment
docker-compose -f docker-compose.prod.yml down
```

## Kubernetes Deployment

### Prerequisites

```bash
# Create namespace
kubectl create namespace heimdall

# Create secrets
kubectl create secret generic heimdall-secrets \
  --from-literal=db_password=your-password \
  --from-literal=api_key=your-api-key \
  -n heimdall
```

### Using Helm Charts

```bash
# Add Heimdall Helm repository
helm repo add heimdall https://charts.heimdall.example.com
helm repo update

# Install Heimdall
helm install heimdall heimdall/heimdall \
  -n heimdall \
  -f values-prod.yaml

# Check deployment
kubectl get pods -n heimdall
kubectl get svc -n heimdall

# Upgrade deployment
helm upgrade heimdall heimdall/heimdall \
  -n heimdall \
  -f values-prod.yaml

# Rollback if needed
helm rollback heimdall -n heimdall
```

### Custom Kubernetes Manifests

If not using Helm, deploy with kubectl:

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/database.yaml
kubectl apply -f k8s/cache.yaml
kubectl apply -f k8s/message-queue.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/api-gateway.yaml
kubectl apply -f k8s/workers.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/mlflow.yaml

# Check status
kubectl get pods -n heimdall -w
```

### Scaling in Kubernetes

```bash
# Scale API Gateway
kubectl scale deployment api-gateway --replicas=3 -n heimdall

# Scale Workers
kubectl scale deployment heimdall-worker --replicas=5 -n heimdall

# Auto-scaling (requires metrics-server)
kubectl autoscale deployment heimdall-worker \
  --min=2 --max=10 \
  --cpu-percent=70 \
  -n heimdall
```

## Database Setup

### PostgreSQL with TimescaleDB

```bash
# Using managed service (recommended for production)
# AWS RDS, Google Cloud SQL, Azure Database, etc.

# Connection string
postgresql://user:password@hostname:5432/heimdall

# Or self-hosted
docker run -d \
  --name timescaledb \
  -e POSTGRES_PASSWORD=password \
  -v timescaledb_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg14
```

### Initial Setup

```bash
# Connect to database
psql postgresql://user:password@hostname:5432/heimdall

# Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Create tables
psql -U user -d heimdall -f db/schema.sql

# Create hypertables for time-series data
SELECT create_hypertable('signal_measurements', 'created_at', 
  if_not_exists => TRUE);
```

## Storage Configuration

### MinIO (S3-Compatible)

```bash
# Deploy MinIO
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -v minio_data:/minio_data \
  minio/minio server /minio_data --console-address ":9001"

# Or use managed service
# AWS S3, Google Cloud Storage, Azure Blob Storage
```

### Create Buckets

```bash
# Using mc (MinIO Client)
mc alias set minio http://localhost:9000 minioadmin minioadmin

mc mb minio/heimdall-signals
mc mb minio/heimdall-models
mc mb minio/heimdall-results
```

## Monitoring & Logging

### Prometheus Setup

```yaml
# prometheus.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'heimdall'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8001']
```

### Grafana Dashboards

```bash
# Add Prometheus as data source
# Import Heimdall dashboards from grafana.com

# Or manually create dashboards:
# - API Response Times
# - Worker Queue Depth
# - Database Performance
# - Memory Usage
# - Error Rate
```

### Logging Stack

```bash
# Deploy ELK Stack
docker-compose up -d elasticsearch logstash kibana

# Configure log shipping
# Update docker-compose.prod.yml to ship logs to Logstash
```

## SSL/TLS Configuration

### Self-Signed Certificate

```bash
# Generate certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Copy to deployment
cp cert.pem /path/to/deployment/
cp key.pem /path/to/deployment/
```

### Let's Encrypt (Recommended)

```bash
# Using cert-manager in Kubernetes
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.10.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f k8s/cluster-issuer.yaml

# Ingress with automatic HTTPS
kubectl apply -f k8s/ingress-tls.yaml
```

## Health Checks

### Configure Health Check Endpoints

```bash
# API health
curl https://api.heimdall.example.com/health

# WebSDR health
curl https://api.heimdall.example.com/health/websdrs

# Database health
curl https://api.heimdall.example.com/health/database
```

### Kubernetes Liveness & Readiness

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Backup & Disaster Recovery

### Database Backups

```bash
# Daily backup to S3
PGPASSWORD=password pg_dump -h hostname -U user heimdall | \
  aws s3 cp - s3://backup-bucket/heimdall-$(date +%Y%m%d).sql

# Restore from backup
aws s3 cp s3://backup-bucket/heimdall-backup.sql - | \
  PGPASSWORD=password psql -h hostname -U user -d heimdall
```

### Model & Data Backups

```bash
# Backup MinIO data
mc mirror minio/heimdall-models s3://backup-bucket/models

# Backup MLflow registry
# (Configure MLflow to use cloud storage backend)
```

## Deployment Validation

### Post-Deployment Checks

```bash
# Check all pods running
kubectl get pods -n heimdall

# Check services
kubectl get svc -n heimdall

# Test API endpoint
curl https://api.heimdall.example.com/health

# Check logs for errors
kubectl logs -n heimdall -l app=heimdall

# Run smoke tests
pytest tests/smoke/ -v
```

## Troubleshooting

### Pod Crashloop

```bash
# Check pod logs
kubectl logs pod_name -n heimdall

# Check pod events
kubectl describe pod pod_name -n heimdall

# Check resource limits
kubectl top pods -n heimdall
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  sh -c 'nc -zv postgres-host 5432'

# Check secrets
kubectl get secret heimdall-secrets -n heimdall -o yaml
```

---

**Related**: [Installation Guide](./installation.md) | [Architecture](./ARCHITECTURE.md)
