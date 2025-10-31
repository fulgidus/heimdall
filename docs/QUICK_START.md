# Quick Start Guide

Get Heimdall running in under 5 minutes!

## Prerequisites

- **Docker** (with Docker Compose)
- **Git**
- **8GB RAM** minimum (16GB recommended)
- **20GB disk space**

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/fulgidus/heimdall.git
cd heimdall
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Optional: Edit .env if you want to customize settings
# The defaults work fine for local development
```

### 3. Start All Services

```bash
# Start infrastructure (PostgreSQL, RabbitMQ, Redis, MinIO, etc.)
docker-compose up -d

# Wait 30 seconds for services to initialize
# Then verify all services are healthy
make health-check
```

### 4. Access the Application

**Frontend UI:**
- http://localhost:3000

**Service UIs:**
- RabbitMQ management: http://localhost:15672 - `guest` / `guest`
- MinIO console: http://localhost:9001 - `minioadmin` / `minioadmin`

## Verify Installation

```bash
# Check all containers are running
docker-compose ps

# Check health status
make health-check

# View logs
docker-compose logs -f
```

## Development Credentials

All services use default development credentials:

| Service | Username | Password | Port |
|---------|----------|----------|------|
| PostgreSQL | heimdall_user | changeme | 5432 |
| RabbitMQ | guest | guest | 5672 |
| MinIO | minioadmin | minioadmin | 9000 |
| Redis | (requires password) | changeme | 6379 |

⚠️ **Security Warning:** These credentials are for **development only**. Never use these in production!

## Next Steps

### Run Tests

```bash
# Backend tests
make test

# Frontend tests
cd frontend && pnpm test

# E2E tests (requires backend running)
cd frontend && pnpm test:e2e
```

### Explore the System

1. **Dashboard**: View WebSDR status and system health at http://localhost:3000
2. **WebSDRs**: Check receiver status at http://localhost:3000/websdrs
3. **Recording Sessions**: Create and manage sessions at http://localhost:3000/sessions
4. **Localization**: View real-time localization results at http://localhost:3000/localization

### Development Workflow

```bash
# Start development mode (hot-reload)
make dev

# Run linting
make lint

# Format code
make format

# View API documentation
open http://localhost:8000/docs  # FastAPI Swagger UI
```

## Troubleshooting

### Services Won't Start

```bash
# Stop all containers
docker-compose down -v

# Clean up and restart
docker-compose up -d --build
```

### Health Check Fails

```bash
# Check container logs
docker-compose logs <service-name>

# Restart specific service
docker-compose restart <service-name>
```

### Port Conflicts

If you get port conflict errors:

```bash
# Check what's using the port
lsof -i :5432  # Replace 5432 with conflicting port

# Stop conflicting service or change port in .env
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Cannot connect to Docker daemon" | Start Docker Desktop or Docker service |
| "Port already in use" | Stop conflicting service or change port in .env |
| "Health check timeout" | Wait longer or check logs with `docker-compose logs` |
| "Permission denied" | Run with `sudo` or add user to docker group |

## Getting Help

- **Documentation**: [Full Documentation](index.md)
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting_guide.md)
- **FAQ**: [Frequently Asked Questions](FAQ.md)
- **Issues**: [GitHub Issues](https://github.com/fulgidus/heimdall/issues)
- **Email**: alessio.corsi@gmail.com

## What's Next?

- **For Developers**: See [Development Guide](DEVELOPMENT.md) to contribute
- **Architecture**: See [Architecture Guide](ARCHITECTURE.md) to understand the system design
- **Training**: See [Training Guide](TRAINING.md) to train ML models
