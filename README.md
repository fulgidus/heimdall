# 🎯 Heimdall - Real-Time Radio Source Localization

> AI-powered platform for locating radio transmissions using distributed WebSDR receivers.

![heimdall.png](heimdall.png)

[![License: CC Non-Commercial](https://img.shields.io/badge/License-CC%20Non--Commercial-orange.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](AGENTS.md)
[![Community: Amateur Radio](https://img.shields.io/badge/Community-Amateur%20Radio-blue.svg)](https://www.iaru.org/)

[![CI Tests](https://github.com/fulgidus/heimdall/workflows/CI%20Tests/badge.svg)](https://github.com/fulgidus/heimdall/actions/workflows/ci-test.yml)
[![Python Quality](https://github.com/fulgidus/heimdall/workflows/Python%20Code%20Quality/badge.svg)](https://github.com/fulgidus/heimdall/actions/workflows/python-quality.yml)
[![TypeScript Quality](https://github.com/fulgidus/heimdall/workflows/TypeScript%20Code%20Quality/badge.svg)](https://github.com/fulgidus/heimdall/actions/workflows/typescript-quality.yml)
[![Security Scan](https://github.com/fulgidus/heimdall/workflows/Security%20Scanning/badge.svg)](https://github.com/fulgidus/heimdall/actions/workflows/security-scan.yml)
[![E2E Tests](https://github.com/fulgidus/heimdall/workflows/E2E%20Tests/badge.svg)](https://github.com/fulgidus/heimdall/actions/workflows/e2e-tests.yml)
[![Integration Tests](https://github.com/fulgidus/heimdall/workflows/Integration%20Tests/badge.svg)](https://github.com/fulgidus/heimdall/actions/workflows/integration-tests.yml)

[![Coverage](https://raw.githubusercontent.com/fulgidus/heimdall/develop/docs/coverage/develop/badge.svg)](https://fulgidus.github.io/heimdall/coverage/)

---

## ⚡ Quick Facts

- **Accuracy**: ±30m (68% confidence)
- **Latency**: <500ms processing
- **Network**: 7 distributed WebSDR receivers (Italian 2m/70cm bands)
- **Status**: Phase 7 in progress (Frontend development)

---

## 🚀 Quick Start (5 minutes)

```bash
# Clone repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Configure environment
cp .env.example .env

# Start all services (Docker required)
docker-compose up -d

# Verify health
make health-check
```

**Done!** Open http://localhost:3000

→ **[Full Installation Guide](https://fulgidus.github.io/heimdall/QUICK_START.html)**

---

## 🏗️ Architecture

| Component          | Technology                  | Purpose              |
| ------------------ | --------------------------- | -------------------- |
| **Backend**        | Python (FastAPI + Celery)   | Microservices        |
| **ML Pipeline**    | PyTorch Lightning + MLflow  | Training & Inference |
| **Frontend**       | React + TypeScript + Mapbox | Web UI               |
| **Desktop App**    | Tauri + Rust                | Desktop wrapper      |
| **Storage**        | PostgreSQL + TimescaleDB    | Time-series data     |
| **Queue**          | RabbitMQ                    | Task orchestration   |
| **Object Storage** | MinIO (S3-compatible)       | IQ data & models     |
| **Deployment**     | Kubernetes + Helm           | Production           |

→ **[Architecture Deep-Dive](https://fulgidus.github.io/heimdall/ARCHITECTURE.html)**

### 🖥️ Desktop Application

Heimdall is also available as a **native desktop application** using Tauri:

```bash
# Development mode (with hot reload)
npm run tauri:dev

# Build production executable
npm run build:app
```

**Desktop Features**:
- Native GPU detection and monitoring
- Local settings persistence
- Direct backend process management (optional)
- Full web functionality + desktop integration

**Platform Support**: Windows 10/11, macOS 10.13+, Linux (AppImage)

---

## 📚 Documentation

**Getting Started**:
- **[Quick Start](https://fulgidus.github.io/heimdall/QUICK_START.html)** - Setup in 5 minutes
- **[Development Guide](https://fulgidus.github.io/heimdall/DEVELOPMENT.html)** - Contributing and local setup
- **[FAQ](https://fulgidus.github.io/heimdall/FAQ.html)** - Common questions

**Reference**:
- **[API Reference](https://fulgidus.github.io/heimdall/API.html)** - REST endpoints
- **[Architecture](https://fulgidus.github.io/heimdall/ARCHITECTURE.html)** - System design
- **[Training Guide](https://fulgidus.github.io/heimdall/TRAINING.html)** - ML model training

**Project**:
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Roadmap](AGENTS.md)** - Development phases
- **[Changelog](CHANGELOG.md)** - Version history

→ **[Full Documentation Index](https://fulgidus.github.io/heimdall/)**

---

## 📊 Development Status

**Phase Progress** (6/11 complete):

✅ **Phases 0-6**: Infrastructure, Services, ML Pipeline, Inference  
🟡 **Phase 7**: Frontend (In Progress)  
⏳ **Phases 8-10**: Kubernetes, QA, Release

→ **[Detailed Roadmap](AGENTS.md)**

**Performance Metrics** (validated in Phase 4):
- API latency: **52ms average** (P95: 52.81ms)
- Concurrent capacity: **50 simultaneous tasks**
- Success rate: **100%**
- Container memory: **100-300MB per service**

---

## 🌍 Use Cases

**Amateur Radio**:
- DX station localization
- Interference source tracking
- Contest verification

**Emergency Services**:
- Search and rescue beacon location
- First responder coordination

**Research**:
- Radio propagation studies
- Spectrum management
- Educational demonstrations

---

## 🧪 Testing

```bash
# Backend tests
make test

# Frontend tests
cd frontend && pnpm test

# E2E tests (real backend integration)
./scripts/run-e2e-tests.sh
```

**Coverage**: >80% across all services  
**E2E Tests**: 42 tests, 100% real HTTP calls

---

## 🤝 Contributing

We welcome contributions! See **[Contributing Guidelines](CONTRIBUTING.md)** for details.

**Quick Steps**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Submit a pull request

→ **[Development Setup](https://fulgidus.github.io/heimdall/DEVELOPMENT.html)**

---

## 📜 License

**CC Non-Commercial** - Developed by fulgidus for the amateur radio community.

See [LICENSE](LICENSE) for full details.

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/fulgidus/heimdall/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fulgidus/heimdall/discussions)
- **Email**: alessio.corsi@gmail.com

---

## 🌟 Mission

**Heimdall's mission is to democratize radio source localization, making it accessible to everyone while advancing radio science and emergency communications.**

We believe that **radio waves belong to everyone**, and everyone should have the tools to understand and explore them.

---

**Ready to see the invisible?** → **[Start Your Journey](https://fulgidus.github.io/heimdall)**
