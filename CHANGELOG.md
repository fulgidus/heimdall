# Changelog

All notable changes to the Heimdall SDR project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase 7: Frontend development in progress
- Comprehensive testing frameworks
- Improved orphan detection script (`scripts/find_orphan_docs.py`) with intelligent categorization and AI-assisted suggestions
- Phase index files for comprehensive navigation (Phase 4 and Phase 7)
- Comprehensive documentation portal organization in `docs/index.md`
- Project status and session tracking section in documentation
- Enhanced troubleshooting resources with specific diagnostic guides
- Technical guides section with WebSDR implementation documentation
- Frontend development documentation with bilingual support

### Changed
- **Documentation reorganization and standardization (100% complete)**
  - All 199 markdown files in docs/ now properly linked and discoverable (100% coverage)
  - Semantic organization with contextual introductions for each document
  - Hierarchical navigation structure through AGENTS.md and docs/index.md
  - Phase-specific documentation organized in dedicated index files
  - Task completion documentation organized by task (T5.6, T5.7, T5.8)
  - Session reports grouped in dedicated tracking section
  - Enhanced AGENTS.md with corrected tracking links for all 7 phases
  - Integrated Agent Handoff Protocol into Knowledge Base section
- Documentation structure now follows clear 3-level hierarchy (entry points → phase indices → individual documents)

### Removed
- Problematic `scripts/reorganize_docs.py` script (445 lines)
- Orphaned files dump report (`docs/agents/20251023_145400_orphaned_files_report.md`)

### Fixed
- 194 orphaned documentation files now properly linked with semantic placement
- Orphan finder script now correctly processes AGENTS.md links (root-level file handling)
- Broken links in all phase index files (Phase 1, 3, 4, 5, 6, 7)
- Broken resource links in docs/index.md
- Documentation discoverability through intentional navigation paths

---

## [0.0.1-phase6] - 2025-10-23

### Added
- ONNX model loader from MLflow registry
- Real-time prediction endpoint with preprocessing
- Redis caching for inference results
- Uncertainty ellipse calculation for visualization
- Batch prediction endpoint for multiple IQ samples
- Model versioning and A/B testing framework
- Performance monitoring (latency, throughput, cache hit rate)
- Load testing to verify <500ms latency requirement
- Endpoint for model metadata and performance metrics
- Graceful model reloading without downtime

### Changed
- Optimized inference latency to meet <500ms requirement
- Improved caching strategy with >80% hit rate target

### Fixed
- Model loading edge cases
- Memory optimization in ONNX runtime

---

## [0.0.1-phase5] - 2025-10-22

### Added
- Neural network architecture (LocalizationNet) for position prediction with uncertainty estimation
- Feature extraction utilities (IQ to mel-spectrogram, MFCC computation)
- HeimdallDataset for loading approved recordings from MinIO
- Gaussian negative log-likelihood loss for uncertainty-aware regression
- PyTorch Lightning module and trainer integration
- MLflow tracking (tracking URI via env/Postgres)
- ONNX export functionality and upload to MinIO
- Training entry point script
- Comprehensive test suite (800+ lines, 50+ test cases, >90% coverage)
- Complete training documentation (docs/TRAINING.md, 2,500+ lines)

### Changed
- Refactored training infrastructure for production readiness
- Improved model serialization and experiment tracking
- Enhanced PyTorch Lightning integration

### Fixed
- Memory efficiency in GPU training
- Data loading performance optimizations
- Model convergence issues

---

## [0.0.1-phase4] - 2025-10-22

### Added
- End-to-end test suite (7/8 tests passing, 87.5% pass rate)
- Docker infrastructure validation (13/13 containers healthy)
- Performance benchmarking tools and baselines
- Load testing framework (50 concurrent tasks tested successfully)
- API performance monitoring and metrics collection
- Task submission latency baseline (~52ms average)
- System health check automation
- Comprehensive performance reports (JSON and Markdown)

### Changed
- Optimized API response times (P95: 52.81ms, P99: 62.63ms)
- Improved container health checks (process-based instead of HTTP)
- Enhanced logging throughout all services
- Updated docker-compose.yml with better health check strategies

### Fixed
- WebSDR connection stability issues
- Task queue routing reliability
- Database ingestion performance bottlenecks
- Docker health check reliability (curl → process status check)
- HTTP status code handling (200 vs 202)
- Unicode encoding in logs (emoji → ASCII)
- Load test failure scenarios with proper error handling

---

## [0.0.1-phase3] - 2025-10-15

### Added
- RF Acquisition Service implementation
- WebSDR integration (7 receivers in Northwestern Italy)
- Simultaneous IQ data fetching from multiple WebSDR sources
- Signal preprocessing pipeline (SNR, PSD, frequency offset)
- Celery task queue integration for asynchronous processing
- MinIO storage backend for IQ data (HDF5 and NPY format)
- TimescaleDB hypertable for measurements
- Metadata persistence to PostgreSQL/TimescaleDB
- REST API endpoints for triggering acquisitions (`/acquire`, `/status/{task_id}`)
- Error handling and retry logic with exponential backoff
- WebSDR health checking and status monitoring
- Integration tests with mocked WebSDR endpoints

### Changed
- Refactored data flow architecture for better scalability
- Improved signal processing pipeline efficiency
- Enhanced WebSDR configuration management
- Updated to Italian WebSDR network (Piedmont & Liguria)

### Fixed
- WebSDR reliability issues and connection timeouts
- Task distribution performance in Celery
- IQ data format inconsistencies
- Partial failure handling when some receivers are offline

---

## [0.0.1-phase2] - 2025-10-08

### Added
- Core service scaffolding for all microservices
- FastAPI framework setup with structured logging
- Service scaffold generator script (`scripts/create_service.py`)
- Health check endpoints for all services (`/health`)
- Dockerfile templates (multi-stage builds with healthchecks)
- Common requirements.txt for shared dependencies
- docker-compose.services.yml for service orchestration
- Service containers: rf-acquisition, training, inference, data-ingestion-web, api-gateway
- Structured logging via structlog for all services

### Changed
- Service architecture refined for microservices pattern
- Communication patterns standardized across services
- Container networking configuration optimized

### Fixed
- Container networking issues
- Service dependency resolution
- Health check endpoint consistency

---

## [0.0.1-phase1] - 2025-10-01

### Added
- PostgreSQL 15 with TimescaleDB extension
- Redis 7 caching layer
- RabbitMQ 3.12 message queue (with management UI)
- MinIO S3-compatible object storage
- MLflow experiment tracker integration
- pgAdmin for database management
- Prometheus + Grafana for monitoring
- Development environment setup (docker-compose.yml)
- Production environment configuration (docker-compose.prod.yml)
- Database schema initialization (db/init-postgres.sql)
  - Tables: known_sources, measurements, training_datasets, models, websdr_stations
  - TimescaleDB hypertables for time-series data
  - PostGIS for geographic queries
- RabbitMQ configuration (db/rabbitmq.conf)
- Prometheus monitoring configuration (db/prometheus.yml)
- Grafana datasource provisioning
- Health check scripts (scripts/health-check.py)
- Makefile with 20+ lifecycle commands
- MinIO bucket auto-creation (heimdall-raw-iq, heimdall-models, heimdall-mlflow)

### Changed
- Infrastructure selection rationale documented
- Resource limits configured for production
- Logging configuration standardized

### Fixed
- Database connection pooling issues
- Service discovery in docker network

---

## [0.0.1-phase0] - 2025-09-24

### Added
- Repository initialization on GitHub
- Project structure and directory layout
  - `/services/` for microservices
  - `/frontend/` for React application
  - `/db/` for database scripts
  - `/docs/` for documentation
  - `/helm/` for Kubernetes deployment
- Documentation framework
  - README.md with project overview
  - AGENTS.md for phase management
  - WEBSDRS.md for receiver configuration
  - docs/ARCHITECTURE.md for system design
  - docs/API.md for API specifications
- CI/CD pipeline foundation
  - .github/workflows/ci-test.yml (pytest)
  - .github/workflows/build-docker.yml (Docker image build)
  - .github/workflows/deploy-k8s.yml (Kubernetes deployment)
- License (CC Non-Commercial)
- Development setup scripts and guidelines
- .copilot-instructions (350+ lines) with project conventions
- .env.example template with all required variables
- .gitignore for Python, Node.js, Docker
- Makefile with common development tasks
- Branch protection rules (main, develop)
- Gitflow workflow setup

### Changed
- Repository visibility set to Public
- Initial branch set to `develop`

---

## Version Format

This project follows [Semantic Versioning](https://semver.org/):
- **Major version** (X.0.0): Breaking changes, major milestones
- **Minor version** (0.X.0): New features, phase completions
- **Patch version** (0.0.X): Bug fixes, documentation updates
- **Phase suffix** (-phaseN): Development phase tracking

## Release Cadence

- **Major versions**: Upon completion of major milestones (v1.0.0 after all phases)
- **Minor versions**: Upon completion of development phases
- **Patch versions**: Bug fixes and documentation updates
- **Development tracking**: See [AGENTS.md](./AGENTS.md) for real-time progress

---

**Current Phase**: Phase 6 (Inference Service) - Complete  
**Current Status**: Phase 7 (Frontend) - In Progress  
**Last Updated**: October 2025

**Related Documentation**:
- [Project Status](./AGENTS.md) - Real-time phase tracking
- [Documentation Index](./docs/index.md) - Complete documentation portal
- [Architecture](./docs/ARCHITECTURE.md) - System design
- [Contributing](./docs/contributing.md) - Contribution guidelines
