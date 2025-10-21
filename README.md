# Heimdall - Radio Source Localization

> *An intelligent platform for real-time radio signal localization*

[![License: CC Non-Commercial](https://img.shields.io/badge/License-CC%20Non--Commercial-orange.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](AGENTS.md)
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

## Development Status

**Phase 1: Infrastructure & Database** 🟡 IN PROGRESS

- ✅ Phase 0: Repository Setup (Complete)
- 🟡 Phase 1: Infrastructure & Database (In Progress)
  - Docker Compose infrastructure with PostgreSQL, RabbitMQ, Redis, MinIO
  - TimescaleDB for time-series optimization
  - Prometheus + Grafana monitoring stack
  - [Full Phase 1 Guide →](PHASE1_GUIDE.md)

### Quick Start

```bash
# Clone repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Setup environment
copy .env.example .env

# Start infrastructure (requires Docker)
docker-compose up -d

# Verify services
make health-check
```

See [PHASE1_GUIDE.md](PHASE1_GUIDE.md) for detailed setup instructions.

## License

Creative Commons Non-Commercial. Developed by fulgidus for the amateur radio community.

## 🎯 Mission Statement

**Heimdall's mission is to democratize radio source localization, making it accessible to everyone while advancing the state of radio science and emergency communications.**

We believe that **radio waves belong to everyone**, and everyone should have the tools to understand and explore them. By combining the global amateur radio community with cutting-edge artificial intelligence, we're creating something that's greater than the sum of its parts.

---

## 🌟 The Team

**Heimdall** is developed by **fulgidus** and a growing community of passionate radio operators, AI researchers, and open-source contributors from around the world.

---

## 🚀 Ready to See the Invisible?

**The radio spectrum has been hidden in plain sight for over a century.**  
**Today, we make it visible.**  
**Tomorrow, we make it yours.**

### [🌟 Start Your Journey →](https://fulgidus.github.io/heimdall)

---

*Heimdall - Where Radio Waves Meet Artificial Intelligence*