# Heimdall SDR - Installation & Setup Guide

## Prerequisites

### Required Software

- **Docker** 24.0+ and **Docker Compose** 2.0+
- **Node.js** 18+ and **npm** 9+
- **Python** 3.11+ with **pip** and **venv**
- **Git** 2.30+
- **Make** (or compatible build tool)

### Recommended Tools

- **VS Code** with extensions:
  - Python Extension Pack
  - Docker Extension
  - React/TypeScript extensions
  - GitHub Copilot
- **pgAdmin** or **DBeaver** for database management
- **Postman** or **Insomnia** for API testing

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/fulgidus/heimdall.git
cd heimdall
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (adjust passwords and ports as needed)
nano .env  # or use your preferred editor
```

### 3. Start Development Environment

```bash
# Start all services
make dev-up

# Verify services are running
docker compose ps
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000

# Run tests
make test
```

## Detailed Setup

### Database Configuration

The development environment uses PostgreSQL with the following default configuration:

```yaml
Database: heimdall
Host: localhost
Port: 5432
Username: heimdall_user
Password: changeme (change in .env file)
```

#### Initial Database Setup

```bash
# Run database migrations
make db-migrate

# Seed with sample data (optional)
make db-seed
```

#### Database Management

```bash
# Connect to database
docker-compose exec postgres psql -U heimdall_user -d heimdall

# Backup database
make db-backup

# Restore database
make db-restore BACKUP_FILE=backup_file.sql
```

### Service Architecture

The development environment includes these services:

#### Core Services

1. **PostgreSQL** (port 5432)
   - Primary database for persistent data
   - Stores signal data, analysis results, and metadata

2. **Redis** (port 6379)
   - Caching layer for real-time data
   - Session storage and pub/sub messaging

3. **RabbitMQ** (port 5672, management UI: 15672)
   - Message queue for asynchronous processing
   - Task distribution and event notifications

4. **MinIO** (port 9000, console: 9001)
   - S3-compatible object storage
   - Signal files, ML models, and artifacts

#### Application Services

5. **API Gateway** (port 8000)
   - FastAPI-based REST API
   - WebSocket endpoints for real-time data

6. **WebSDR Collector** (port 8001)
   - Data collection from WebSDR stations
   - Signal ingestion and preprocessing

7. **Signal Processor** (port 8002)
   - DSP pipeline and feature extraction
   - Real-time signal analysis

8. **ML Detector** (port 8003)
   - Anomaly detection models
   - Prediction and classification services

9. **Frontend** (port 3000)
   - React/Next.js web application
   - Real-time dashboards and visualizations

10. **MLflow** (port 5000)
    - ML experiment tracking
    - Model registry and deployment

### Python Development Setup

#### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Code Quality Tools

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
make lint

# Format code
make format

# Type checking
make type-check
```

#### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/test_signal_processor.py -v

# Run integration tests
make test-integration
```

### Frontend Development Setup

#### Node.js Dependencies

```bash
cd frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run linting
npm run lint
```

#### Development Workflow

```bash
# Hot reload development
make frontend-dev

# Build and serve locally
make frontend-build
make frontend-serve

# Run Storybook (component development)
make frontend-storybook
```

## WebSDR Configuration

### Station Registration

Configure WebSDR stations in the database:

```sql
INSERT INTO websdr_stations (id, name, url, location, frequency_min, frequency_max, status) VALUES
('twente-nl', 'University of Twente', 'http://websdr.ewi.utwente.nl:8901/', 'Netherlands', 0, 29000000, 'active'),
('hackgreen-uk', 'Hack Green SDR', 'http://hackgreensdr.org:8073/', 'United Kingdom', 0, 30000000, 'active');
-- Add other stations...
```

### API Integration

Each WebSDR station requires specific API configuration:

```python
# Example configuration in services/websdr-collector/config.py
WEBSDR_STATIONS = {
    "twente-nl": {
        "url": "http://websdr.ewi.utwente.nl:8901/",
        "api_type": "http_streaming",
        "frequency_range": (0, 29_000_000),
        "poll_interval": 1.0,
        "authentication": None
    },
    "hackgreen-uk": {
        "url": "http://hackgreensdr.org:8073/",
        "api_type": "websocket",
        "frequency_range": (0, 30_000_000),
        "reconnect_interval": 5.0,
        "authentication": None
    }
}
```

## Machine Learning Setup

### MLflow Configuration

```bash
# Start MLflow server
make mlflow-start

# Access MLflow UI
open http://localhost:5000

# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Model Training

```bash
# Train initial models
make ml-train-initial

# Run training pipeline
python services/ml-detector/train_anomaly_model.py

# Evaluate models
make ml-evaluate
```

## Production Deployment

### Docker Production Build

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Kubernetes Deployment

```bash
# Install Helm charts
helm install heimdall ./helm/heimdall

# Check deployment status
kubectl get pods -n heimdall

# View logs
kubectl logs -n heimdall -l app=heimdall
```

## Troubleshooting

### Common Issues

#### Containers won't start

```bash
# Check Docker daemon
docker ps

# Rebuild all containers
docker-compose build --no-cache

# Start with verbose logging
docker-compose up --verbose
```

#### Database connection errors

```bash
# Check database logs
docker-compose logs postgres

# Verify database is running
docker-compose exec postgres psql -U heimdall_user -d heimdall -c "SELECT 1"
```

#### Memory issues

```bash
# Increase Docker memory limit in .env
DOCKER_MEMORY=4g

# Check current usage
docker stats
```

### Getting Help

- Check [troubleshooting guide](./troubleshooting_guide.md)
- Review [project documentation](./index.md)
- Open an [issue on GitHub](https://github.com/fulgidus/heimdall/issues)

---

**Last Updated**: October 2025  
**Next Steps**: [Architecture Guide](./ARCHITECTURE.md) | [Developer Guide](./developer_guide.md)
