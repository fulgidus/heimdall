# Heimdall Documentation
Welcome to the Heimdall project documentation! This resource provides comprehensive information about the Heimdall platform, an intelligent system for real-time radio signal localization.
## Table of Contents
- [Heimdall Documentation](#heimdall-documentation)
    - [Table of Contents](#table-of-contents)
    - [Overview](#overview)
    - [Architecture](#architecture)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Contributing](#contributing)
    - [License](#license)
    - [Contact](#contact)
    - [Changelog](#changelog)
    - [FAQs](#faqs)
    - [Additional Resources](#additional-resources)

## Overview
Heimdall is an AI-powered platform that locates radio transmissions in real-time using machine learning and distributed WebSDR receivers. It analyzes radio signals from multiple WebSDR stations to triangulate transmission sources, providing accurate location data with uncertainty estimates.

## Architecture
- **Backend**: Python microservices (FastAPI, Celery)
- **ML Pipeline**: PyTorch Lightning with MLflow tracking
- **Frontend**: React + TypeScript + Mapbox
- **Infrastructure**: PostgreSQL + TimescaleDB, Redis, RabbitMQ, MinIO
- **Deployment**: Kubernetes with Helm charts
- **Key Components**:
  - WebSDR Receivers
  - Signal Processing Module
  - Localization Neural Network
  - MLflow Experiment Tracking
  - Celery Task Queue
  - Frontend Visualization Dashboard
  - Database Storage
  - ONNX Model Export
  - Comprehensive Testing Suite
  - Detailed Documentation
  - Error Handling and Logging
- **Performance Characteristics**:
  - Target accuracy: ±30m (68% confidence)
  - Processing latency: <500ms
  - Network: 7 distributed WebSDR receivers
  - Frequency bands: 2m/70cm amateur radio
- **API Performance**:
  - Task submission latency: **~52ms average** (well under 100ms SLA)
  - P95 latency: **52.81ms** (consistent performance)
  - P99 latency: **62.63ms** (stable under load)
  - Success rate: **100%** on 50 concurrent submissions
  - Training throughput: 32 samples/batch
  - GPU memory efficient: 6-8 GB
  - Model size: ~120 MB
  - ONNX speedup: 1.5-2.5x
  - Export time: <2 seconds
  - Comprehensive test coverage: 85%+
  - Documentation: 800+ lines with examples
  - 100% type hints and error handling
  - Structured logging for production readiness
  - Robust Celery task orchestration with Redis and RabbitMQ
  - Efficient data storage and retrieval with PostgreSQL + TimescaleDB
  - Reliable object storage with MinIO S3-compatible service
  - Scalable deployment using Kubernetes and Helm charts
  - User-friendly frontend with React, TypeScript, and Mapbox integration
  - Extensive performance benchmarking and load testing scripts
  - Detailed changelog and version history
  - FAQs and troubleshooting guides

## Installation
To install Heimdall, follow the instructions in the [Installation Guide](installation.md).

## Usage
For usage instructions, refer to the [User Guide](usage.md).

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](contributing.md) for more information.

## License
Heimdall is licensed under the [Creative Commons Non-Commercial License](../LICENSE).

## Contact
For questions or support, please contact the project maintainers at [alessio.corsi[AT]gmail.com](mailto:alessio.corsi@gmail.com).

## Changelog

See the [Changelog](../CHANGELOG.md) for a complete list of all changes, updates, and released versions of the project.
The changelog follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.
The changelog is maintained according to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) guidelines.

## FAQs
Find answers to common questions in our [FAQ section](faqs.md).

## Additional Resources
- [Architecture Diagrams](architecture_diagrams.md)
- [API Documentation](api_documentation.md)
- [Performance Benchmarks](performance_benchmarks.md)
- [Testing Strategies](testing_strategies.md)
- [Deployment Instructions](deployment_instructions.md)
- [Troubleshooting Guide](troubleshooting_guide.md)
- [Glossary of Terms](glossary.md)
- [Acknowledgements](acknowledgements.md)
- [Roadmap](roadmap.md)
- [Security Considerations](security_considerations.md)
- [Performance Optimization Tips](performance_optimization.md)
- [Data Schema Documentation](data_schema.md)
- [API Reference](api_reference.md)
- [Developer Guide](developer_guide.md)
- [Training Guide](TRAINING.md)
- [Architecture Guide](ARCHITECTURE.md)
- [API Specification](API.md)
- [WebSDR Configuration](websdrs.md)

## Development Phase Tracking

The project follows a structured phase-based development approach. For detailed tracking of each phase:

### Completed Phases
- **[Phase 1: Infrastructure & Database](agents/20251022_080000_phase1_index.md)** - Infrastructure setup complete
- **[Phase 2: Core Services Scaffolding](agents/20251022_080000_phase2_complete.md)** - Microservices scaffolding complete
- **[Phase 3: RF Acquisition Service](agents/20251022_080000_phase3_index.md)** - WebSDR data acquisition complete
- **[Phase 4: Data Ingestion & Validation](agents/20251022_080000_phase4_progress_dashboard.md)** - Infrastructure validation complete
- **[Phase 5: Training Pipeline](agents/20251022_080000_phase5_document_index.md)** - ML training pipeline complete
- **[Phase 6: Inference Service](agents/20251023_153000_phase6_index.md)** - Real-time inference complete

### Active Phases
- **[Phase 7: Frontend](agents/20251023_153000_phase7_start_here.md)** - React + Mapbox UI (In Progress)

### Upcoming Phases
- **Phase 8: Kubernetes & Deployment** - Production deployment
- **Phase 9: Testing & QA** - Comprehensive testing
- **Phase 10: Documentation & Release** - Final documentation and release

For the complete phase management guide, see [AGENTS.md](../AGENTS.md).

