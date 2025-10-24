# Heimdall - Radio Source Localization

> *An intelligent platform for real-time radio signal localization*

[![License: CC Non-Commercial](https://img.shields.io/badge/License-CC%20Non--Commercial-orange.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](../AGENTS.md)
[![Community: Amateur Radio](https://img.shields.io/badge/Community-Amateur%20Radio-blue.svg)](https://www.iaru.org/)

An AI-powered platform that locates radio transmissions in real-time using machine learning and distributed WebSDR receivers.

## Overview

Heimdall analyzes radio signals from multiple WebSDR stations to triangulate transmission sources. The system uses neural networks trained on radio propagation data to predict location coordinates with uncertainty estimates.

**Key specifications:**
- Target accuracy: ±30m (68% confidence)
- Processing latency: <500ms
- Network: 7 distributed WebSDR receivers
- Frequency bands: 2m/70cm amateur radio

## Architecture

- **Backend**: Python microservices (FastAPI, Celery)
- **ML Pipeline**: PyTorch Lightning with MLflow tracking
- **Frontend**: React + TypeScript + Mapbox
- **Infrastructure**: PostgreSQL + TimescaleDB, Redis, RabbitMQ, MinIO
- **Deployment**: Kubernetes with Helm charts

## Applications

**Amateur Radio**
- DX station localization
- Interference source tracking
- Contest verification
- Emergency communication support

**Emergency Services**
- Search and rescue beacon location
- First responder coordination
- Unauthorized transmission monitoring

**Research**
- Radio propagation studies
- Spectrum management
- Educational demonstrations

## Technical Details

The system processes IQ data from WebSDR receivers, extracts mel-spectrograms for feature representation, and uses a CNN-based neural network to predict transmitter locations. A Gaussian negative log-likelihood loss function enables uncertainty quantification for each prediction.

### Performance Characteristics (Phase 4 Validated)

**API Performance**
- Task submission latency: **~52ms average** (well under 100ms SLA)
- P95 latency: **52.81ms** (consistent performance)
- P99 latency: **62.63ms** (stable under load)
- Success rate: **100%** on 50 concurrent submissions

**System Processing**
- RF Acquisition per WebSDR: **63-70 seconds** (network-bound, expected)
- Database operations: **<50ms** per measurement insertion
- Message queue latency: **<100ms** for task routing
- Container memory footprint: **100-300MB** per service (efficient)

**Infrastructure Throughput**
- Concurrent task handling: **50+ simultaneous RF acquisitions** verified
- RabbitMQ routing: **reliable under production load**
- Redis caching: **<50ms per operation**
- TimescaleDB: **stable high-velocity ingestion**

## Development Status

**Phase 4: Data Ingestion Validation** ✅ COMPLETE

- ✅ Phase 0: Repository Setup (Complete)
- ✅ Phase 1: Infrastructure & Database (Complete)
- ✅ Phase 2: Core Services Scaffolding (Complete)
- ✅ Phase 3: RF Acquisition Service (Complete)
- ✅ Phase 4: Data Ingestion & Validation (Complete - Infrastructure Verified)
  - E2E tests: 7/8 passing (87.5%)
  - Docker infrastructure: 13/13 containers healthy
  - Performance benchmarking: All SLAs met
  - Load testing: 50 concurrent tasks, 100% success rate
  - [Full Phase 4 Report →](../PHASE4_COMPLETION_FINAL.md)

**Phase 5: Training Pipeline** 🟡 READY TO START (All dependencies met)
- ML pipeline development with PyTorch Lightning
- Model training with MLflow tracking
- [Phase 5 Handoff →](../PHASE5_HANDOFF.md)

## Quick Start

```bash
# Clone repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Setup development environment
make dev-up

# Verify installation
curl http://localhost:8000/health
```

For detailed setup instructions, see [Installation Guide](./installation.md).

## Documentation

- [Installation & Setup](./installation.md)
- [Architecture Guide](./ARCHITECTURE.md)
- [API Reference](./api_reference.md)
- [Training Pipeline](./TRAINING.md)
- [WebSDR Configuration](./websdrs.md)
- [Developer Guide](./developer_guide.md)
- [Troubleshooting](./troubleshooting_guide.md)

## License

Heimdall is licensed under the [Creative Commons Non-Commercial License](../LICENSE).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](./contributing.md) for more information.

## Support

For questions or support, please contact:
- **Email**: alessio.corsi[AT]gmail.com
- **GitHub Issues**: [Heimdall Issues](https://github.com/fulgidus/heimdall/issues)
- **Discussions**: [Heimdall Discussions](https://github.com/fulgidus/heimdall/discussions)

---

**Last Updated**: October 2025  
**Status**: Phase 5 - Training Pipeline Ready  
**More Info**: [Project Overview →](../AGENTS.md)
