# Heimdall SDR - Development Setup Guide

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
nano .env
```

### 3. Start Development Environment

```bash
# Start all services
make dev-up

# Verify services are running
docker-compose ps
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

### Model Deployment

```bash
# Deploy model to staging
make ml-deploy-staging MODEL_NAME=anomaly-detector VERSION=1

# Deploy to production
make ml-deploy-production MODEL_NAME=anomaly-detector VERSION=1
```

## Development Workflow

### Daily Development

```bash
# Start development environment
make dev-up

# Make code changes...

# Run tests continuously
make test-watch

# Check code quality
make lint format

# Commit changes
git add .
git commit -m "feat: implement signal processing pipeline"

# Push to feature branch
git push origin feature/signal-processing
```

### Pull Request Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Run full test suite: `make test-all`
4. Ensure code quality: `make lint format type-check`
5. Update documentation if needed
6. Push branch and create pull request
7. Address review feedback
8. Merge when approved

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add signal_detections table"

# Review generated migration
nano db/migrations/versions/xxx_add_signal_detections_table.py

# Apply migration
make db-migrate

# Downgrade if needed
alembic downgrade -1
```

## Monitoring and Debugging

### Application Logs

```bash
# View all service logs
make logs

# View specific service logs
docker-compose logs -f api-gateway

# View logs with timestamps
docker-compose logs -t websdr-collector
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Monitor database performance
make db-monitor

# Monitor Redis
redis-cli monitor
```

### Debugging

#### Python Services

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use IPython
import IPython; IPython.embed()

# Remote debugging (VS Code)
import debugpy
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()
```

#### Frontend Debugging

```bash
# Enable React DevTools
npm install -g react-devtools

# Debug in browser
# Open Chrome DevTools -> React tab
```

## Troubleshooting

### Common Issues

#### Docker Issues

```bash
# Problem: Port already in use
# Solution: Check and kill processes
lsof -i :5432
kill -9 <PID>

# Problem: Docker out of space
# Solution: Clean up Docker
docker system prune -a
make docker-clean
```

#### Database Issues

```bash
# Problem: Connection refused
# Solution: Check PostgreSQL is running
docker-compose ps postgres
docker-compose logs postgres

# Problem: Migration errors
# Solution: Reset database
make db-reset
make db-migrate
```

#### WebSDR Connection Issues

```bash
# Problem: Cannot connect to WebSDR
# Solution: Check station availability
curl -I http://websdr.ewi.utwente.nl:8901/

# Problem: Rate limiting
# Solution: Adjust collection intervals
# Edit services/websdr-collector/config.py
```

#### ML Pipeline Issues

```bash
# Problem: Model training fails
# Solution: Check data quality
make ml-validate-data

# Problem: MLflow server not starting
# Solution: Check database connection
docker-compose logs mlflow
```

### Performance Issues

#### High Memory Usage

```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Adjust memory limits in docker-compose.yml
mem_limit: 1g
```

#### High CPU Usage

```bash
# Profile Python services
pip install py-spy
py-spy top --pid <process_id>

# Optimize signal processing
# Check buffer sizes and processing intervals
```

### Getting Help

1. **Documentation**: Check `docs/` directory for detailed guides
2. **GitHub Issues**: Search existing issues or create new one
3. **Discussions**: Use GitHub Discussions for questions
4. **Logs**: Always include relevant logs when reporting issues

## Contributing

### Code Style

- Follow PEP 8 for Python code
- Use Black for code formatting
- Add type hints to all functions
- Write comprehensive docstrings
- Maintain test coverage >90%

### Commit Guidelines

```bash
# Use conventional commit format
feat: add new signal processing algorithm
fix: resolve WebSDR connection timeout
docs: update API documentation
test: add integration tests for ML pipeline
```

### Testing Requirements

- Unit tests for all new functions
- Integration tests for API endpoints
- End-to-end tests for critical workflows
- Performance tests for signal processing

---

**Next Steps**:
1. Complete setup following this guide
2. Read `docs/ARCHITECTURE.md` for system design
3. Check `docs/API.md` for API specifications
4. Start with Phase 1 tasks in `AGENTS.md`

For additional support, see the troubleshooting section or create an issue on GitHub.
