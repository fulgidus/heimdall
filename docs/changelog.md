# Changelog

## [Unreleased]

### Added
- Phase 6: Inference Service planning
- Documentation migration to `/docs/`
- WebSDR configuration guide

### Changed
- Updated installation documentation
- Improved architecture guides

### Fixed
- Documentation structure and organization

## [Phase 5.10.0] - 2025-10-22

### Added
- ONNX model export functionality
- Model quantization support
- MLflow experiment tracking
- PyTorch Lightning training pipeline
- Comprehensive test coverage

### Changed
- Refactored training infrastructure
- Improved model serialization
- Enhanced experiment tracking

### Fixed
- Memory efficiency in GPU training
- Data loading performance

## [Phase 4.0.0] - 2025-10-22

### Added
- E2E test suite (7/8 tests passing)
- Docker infrastructure validation
- Performance benchmarking tools
- Load testing framework
- API performance baselines

### Changed
- Optimized API response times
- Improved container health checks
- Enhanced logging throughout

### Fixed
- WebSDR connection stability
- Task queue routing
- Database ingestion performance

## [Phase 3.0.0] - 2025-10-15

### Added
- RF Acquisition Service
- WebSDR integration
- Signal preprocessing pipeline
- Celery task queue integration
- MinIO storage backend

### Changed
- Refactored data flow architecture
- Improved signal processing pipeline

### Fixed
- WebSDR reliability issues
- Task distribution performance

## [Phase 2.0.0] - 2025-10-08

### Added
- Core service scaffolding
- FastAPI framework setup
- Database schema design
- Service containers
- Docker compose orchestration

### Changed
- Service architecture
- Communication patterns

### Fixed
- Container networking

## [Phase 1.0.0] - 2025-10-01

### Added
- PostgreSQL with TimescaleDB
- Redis caching layer
- RabbitMQ message queue
- MinIO object storage
- MLflow experiment tracker
- Development environment setup

### Changed
- Infrastructure selection rationale

## [Phase 0.0.0] - 2025-09-24

### Added
- Repository initialization
- Project structure
- Documentation framework
- License and guidelines
- Development setup scripts

---

## Version Format

This project follows [Semantic Versioning](https://semver.org/).

## Release Cadence

- Major versions: Upon completion of major phases
- Minor versions: Upon feature additions
- Patch versions: Upon bug fixes
- Development versions: Tracked in AGENTS.md

---

**Current Phase**: Phase 5 Complete, Phase 6 Planned  
**Last Updated**: October 2025

**Related**: [Project Status](../AGENTS.md) | [History](./index.md)
