# Frequently Asked Questions (FAQ)

Common questions and answers about Heimdall.

## General Questions

### What is Heimdall?

Heimdall is an AI-powered platform for real-time localization of radio transmissions using distributed WebSDR receivers. It uses machine learning to analyze radio signals from multiple receivers and triangulate the source location with uncertainty estimates.

### Why "Heimdall"?

In Norse mythology, Heimdall is the all-seeing guardian of Bifrost who can see and hear across all realms. Similarly, our system "sees" radio transmissions across a distributed network of receivers.

### Is Heimdall free to use?

Yes! Heimdall is open source under a Creative Commons Non-Commercial license. You can use it freely for personal, educational, and amateur radio purposes.

### Can I use Heimdall for commercial purposes?

No, the CC Non-Commercial license prohibits commercial use. For commercial licensing, please contact the project maintainer.

## Getting Started

### How do I install Heimdall?

See the [Quick Start Guide](QUICK_START.md) for a 5-minute installation process. You'll need Docker and 8GB of RAM.

### What are the system requirements?

**Minimum:**
- 8GB RAM
- 20GB disk space
- Docker 20.10+
- Internet connection

**Recommended:**
- 16GB RAM (for ML training)
- 50GB disk space
- GPU (for faster training)

### Do I need amateur radio experience?

Not required! The system is designed to be accessible. However, familiarity with radio concepts helps understand the results.

### Do I need machine learning expertise?

No! The ML models are pre-trained and ready to use. ML expertise is only needed if you want to retrain models or modify the architecture.

## Technical Questions

### How accurate is the localization?

Target accuracy is ±30m (68% confidence interval). Actual accuracy depends on:
- Number of active receivers
- Signal quality (SNR)
- Receiver geometry
- Radio propagation conditions

### How fast is the localization?

Real-time inference latency is <500ms from signal reception to location prediction.

### Which frequency bands are supported?

Currently configured for:
- **2m band** (144-148 MHz)
- **70cm band** (430-440 MHz)

The system can be adapted to other bands by retraining models with appropriate data.

### How many WebSDR receivers are needed?

**Minimum:** 3 receivers (for triangulation)
**Recommended:** 5-7 receivers (for better accuracy)
**Current setup:** 7 receivers in Northwestern Italy

### Can I add my own WebSDR receiver?

Yes! Edit `WEBSDRS.md` to add your receiver configuration. You'll need:
- WebSDR URL and API access
- Receiver coordinates (latitude, longitude, elevation)
- Antenna characteristics

## Development Questions

### How do I contribute?

See the [Contributing Guidelines](../CONTRIBUTING.md) for detailed instructions. Quick steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### What technologies does Heimdall use?

**Backend:**
- Python 3.11 (FastAPI, Celery)
- PostgreSQL + TimescaleDB
- RabbitMQ, Redis, MinIO

**ML Pipeline:**
- PyTorch Lightning
- MLflow tracking
- ONNX Runtime

**Frontend:**
- React + TypeScript
- Mapbox GL JS
- Bootstrap 5

**Deployment:**
- Docker + Docker Compose
- Kubernetes + Helm

### How do I run tests?

```bash
# Backend tests
make test

# Frontend tests
cd frontend && pnpm test

# E2E tests
cd frontend && pnpm test:e2e
```

See [Development Guide](DEVELOPMENT.md) for details.

### How do I report bugs?

Open an issue on [GitHub Issues](https://github.com/fulgidus/heimdall/issues) with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Docker version, etc.)

## Usage Questions

### How do I start a recording session?

1. Navigate to http://localhost:3000/sessions
2. Click "New Session"
3. Enter known source location (for training data)
4. Set frequency and duration
5. Click "Start Recording"

The system will fetch IQ data from all active WebSDR receivers simultaneously.

### What is "uncertainty" in localization results?

Uncertainty represents the confidence in the predicted location. It's displayed as an ellipse on the map:
- **Small ellipse**: High confidence
- **Large ellipse**: Low confidence

Factors affecting uncertainty:
- Signal strength (SNR)
- Receiver geometry
- Radio propagation conditions

### Can I use Heimdall without WebSDR access?

For development/testing, yes - use mock data. For production localization, you need access to real WebSDR receivers.

### How is training data collected?

**Supervised learning approach:**
1. Operator starts recording session
2. Operator enters known source location
3. System records IQ data from all receivers
4. IQ data + ground truth location = training sample

Accumulate many samples to train accurate models.

## Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker ps

# Restart all services
docker-compose down -v
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Health check fails

```bash
# Wait 30 seconds for services to initialize
sleep 30 && make health-check

# Check individual services
docker-compose ps
docker-compose logs <service-name>
```

### Port conflicts

```bash
# Find conflicting process
lsof -i :5432  # Replace with your port

# Stop conflicting service or change port in .env
```

### Database connection errors

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Test connection
psql -h localhost -U heimdall_user -d heimdall
```

### Frontend won't build

```bash
cd frontend

# Clear cache and reinstall
rm -rf node_modules pnpm-lock.yaml
pnpm install

# Rebuild
pnpm build
```

## Architecture Questions

### How does the ML model work?

1. **Input**: Mel-spectrograms from multiple WebSDR receivers
2. **Feature extraction**: CNN extracts spatial-temporal patterns
3. **Localization**: Fully connected layers predict coordinates
4. **Uncertainty**: Gaussian NLL loss quantifies prediction confidence

See [Architecture Guide](ARCHITECTURE.md) for details.

### Why use TimescaleDB instead of regular PostgreSQL?

TimescaleDB is optimized for time-series data:
- Efficient storage of measurement timestamps
- Fast time-range queries
- Automatic data compression
- Continuous aggregates

Perfect for radio signal measurements over time.

### Why use ONNX for inference?

**Benefits:**
- 1.5-2.5x faster than PyTorch
- Smaller model size
- Platform-independent
- Production-optimized

Models are trained in PyTorch, then exported to ONNX for inference.

### Can I deploy Heimdall without Kubernetes?

Yes! For development and small-scale deployments, use `docker-compose`:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

For production at scale, Kubernetes is recommended for auto-scaling, self-healing, and orchestration.

## Performance Questions

### Why is my first inference slow?

The first inference loads the ONNX model into memory. Subsequent inferences are much faster (<500ms) due to:
- Model caching
- Redis result caching
- Optimized inference session

### How can I improve accuracy?

1. **Add more receivers**: Better triangulation geometry
2. **Improve signal quality**: Better antennas, lower noise
3. **Collect more training data**: More diverse samples
4. **Retrain models**: Use local radio propagation conditions

### How much disk space do I need?

**Depends on usage:**
- Base system: ~5GB (Docker images)
- IQ data: ~10MB per recording session
- ML models: ~100-200MB per model version
- Logs and metrics: ~1GB per month

**Recommendation:** 50GB for active development/production.

## Data Privacy

### What data is collected?

**Local deployment:**
- Radio signal measurements (IQ data)
- Localization results
- System metrics and logs

No data is sent externally. Everything stays on your infrastructure.

### Can I delete collected data?

Yes, all data is stored locally:
- IQ files: MinIO buckets
- Measurements: PostgreSQL database
- Models: MinIO + MLflow

Use standard backup/deletion tools.

## Licensing

### Can I modify Heimdall?

Yes! The CC Non-Commercial license allows modifications for non-commercial purposes.

### Can I redistribute Heimdall?

Yes, under the same CC Non-Commercial license. Credit the original project.

### Can I train commercial models using Heimdall?

The license prohibits commercial use. For commercial licensing, contact the maintainer.

## Getting More Help

- **Documentation**: [Full Documentation](index.md)
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting_guide.md)
- **GitHub Issues**: [Report bugs](https://github.com/fulgidus/heimdall/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/fulgidus/heimdall/discussions)
- **Email**: alessio.corsi@gmail.com
