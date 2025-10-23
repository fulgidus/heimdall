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

Heimdall provides comprehensive setup options for different development scenarios. For a complete development environment with all prerequisites and tools, see the [Development Setup Guide](agents/20251022_080000_setup.md) which includes Docker configuration, Python environment setup, and recommended IDE extensions.

For quick installation, follow the instructions in the [Installation Guide](installation.md). Windows users can refer to the [Windows Quick Setup](agents/20251022_080000_setup_windows_quick.md) for platform-specific instructions.

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
- [E2E Test Failures Diagnostics](agents/20251022_080000_e2e_test_failures_diagnostics.md) - Detailed troubleshooting for end-to-end tests
- [Diagnostic 404 Issue](agents/20251023_153000_diagnostic_404_issue.md) - Troubleshooting 404 errors
- [Quick Answer 404](agents/20251023_153000_quick_answer_404.md) - Quick reference for 404 fixes
- [CI Debug Guide](agents/20251023_153000_ci_debug_guide.md) - CI/CD troubleshooting and debugging
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
- [Architecture Visual Guide](agents/20251023_153000_architecture_visual_guide.md) - Visual architecture diagrams and explanations
- [API Specification](API.md)
- [API Gateway Explanation](agents/20251023_153000_api_gateway_explanation.md) - Detailed API Gateway documentation
- [WebSDR Configuration](websdrs.md)
- [WebSDR API Integration Complete](agents/20251023_153000_websdr_api_integration_complete.md) - WebSDR integration implementation details
- [WebSDR Fix Guide](agents/20251023_153000_websdr_fix_guide.md) - WebSDR troubleshooting and fixes
- [WebSDR Fix Quickref](agents/20251023_153000_websdr_fix_quickref.md) - Quick reference for WebSDR fixes
- [WebSDR Test Instructions](agents/20251023_153000_websdr_test_instructions.md) - Testing WebSDR integration
- [README WebSDRs Implementation](agents/20251023_153000_readme_websdrs_implementation.md) - Implementation guide for WebSDR receivers
- [Testing WebSDRs Page](agents/20251023_153000_testing_websdrs_page.md) - WebSDR page testing procedures
- [WebSDRs Page Changes](agents/20251023_153000_websdrs_page_changes.md) - WebSDR UI changes documentation

### Frontend Development
- [Frontend Specification](agents/20251023_153000_frontend_specification.md) - Complete frontend requirements and design
- [Frontend Pages Complete](agents/20251023_153000_frontend_pages_complete.md) - Frontend implementation status
- [Debug Frontend Backend](agents/20251023_153000_debug_frontend_backend.md) - Frontend-backend integration troubleshooting
- [Visual Comparison](agents/20251023_153000_visual_comparison.md) - UI visual comparison and testing
- [Verifica Frontend Backend (IT)](agents/20251023_153000_verifica_frontend_backend_it.md) - Frontend-backend verification (Italian)

### Implementation Summaries & CI/CD
- [Implementation Summary](agents/20251023_153000_implementation_summary.md) - Overall implementation summary
- [Implementazione WebSDR Reale Summary (IT)](agents/20251023_153000_implementazione_websdr_reale_summary.md) - Real WebSDR implementation (Italian)
- [Final Summary](agents/20251023_153000_final_summary.md) - Comprehensive final summary
- [Fix Summary](agents/20251023_153000_fix_summary.md) - Summary of fixes and corrections
- [WebSDR Fix Summary](agents/20251023_153000_websdr_fix_summary.md) - WebSDR-specific fixes summary
- [WebSDR Fix Status](agents/20251023_153000_websdr_fix_status.md) - Current status of WebSDR fixes and integration
- [All Fixes Complete](agents/20251023_153000_all_fixes_complete.md) - All fixes completion report
- [CI Fixes Ready to Push](agents/20251023_153000_ci_fixes_ready_to_push.md) - CI/CD fixes ready for deployment
- [CI Test Fixes Complete](agents/20251023_153000_ci_test_fixes_complete.md) - CI testing fixes completion
- [Quick Verification Checklist](agents/20251023_153000_quick_verification_checklist.md) - Quick verification procedures
- [Answer to Your Question](agents/20251023_153000_answer_to_your_question.md) - FAQ and common questions
- [Why Mock Responses Answer](agents/20251023_153000_why_mock_responses_answer.md) - Explanation of mock responses usage

### Documentation & Maintenance
- [Documentation Index](agents/20251023_153000_documentation_index.md) - Complete documentation index
- [Documentation Reorganization Complete](agents/20251023_150000_documentation_reorganization_complete.md) - Documentation restructuring report
- [Orphan Resolution Summary](agents/20251023_182900_orphan_resolution_summary.md) - Orphaned files resolution process
- [Orphan Fix Plan](agents/ORPHAN_FIX_PLAN.md) - Plan for fixing orphaned documentation
- [docs/changelog.md](changelog.md) - Documentation-specific changelog
- [docs/handoff.md](handoff.md) - General handoff procedures
- [docs/README.md](README.md) - Documentation portal README

## Development Phase Tracking

The project follows a structured phase-based development approach. For detailed tracking of each phase:

### 📚 Navigation Hubs
- **[Agents Master Index](agents/MASTER_INDEX.md)** - Complete navigation hub for all agent documentation
- **[AGENTS.md](../AGENTS.md)** - Phase management guide (primary reference)

### Completed Phases
- **[Phase 1: Infrastructure & Database](agents/20251022_080000_phase1_index.md)** - Infrastructure setup complete
- **[Phase 2: Core Services Scaffolding](agents/20251022_080000_phase2_complete.md)** - Microservices scaffolding complete
- **[Phase 3: RF Acquisition Service](agents/20251022_080000_phase3_index.md)** - WebSDR data acquisition complete
- **[Phase 4: Data Ingestion & Validation](agents/20251022_080000_phase4_index.md)** - Infrastructure validation complete
- **[Phase 5: Training Pipeline](agents/20251022_080000_phase5_document_index.md)** - ML training pipeline complete
- **[Phase 6: Inference Service](agents/20251023_153000_phase6_index.md)** - Real-time inference complete

### Active Phases
- **[Phase 7: Frontend](agents/20251023_153000_phase7_index.md)** - React + Mapbox UI (In Progress)

### Upcoming Phases
- **Phase 8: Kubernetes & Deployment** - Production deployment
- **Phase 9: Testing & QA** - Comprehensive testing
- **Phase 10: Documentation & Release** - Final documentation and release

## Project Status & Session Tracking

For detailed project status updates and session-by-session progress tracking, see the following resources:

- [Project Status Guide](agents/20251022_080000_project_status_guide.md) - Comprehensive project status reference
- [Project Status Oct 2025](agents/20251023_153000_project_status_oct2025.md) - Current project status snapshot
- [Session Tracking](agents/20251022_080000_session_tracking.md) - Detailed session-by-session tracking
- [Session 2025-10-22 Complete](agents/20251022_080000_session_2025_10_22_complete.md) - Major session completion report
- [Session TimescaleDB Complete](agents/20251022_080000_session_timescaledb_complete.md) - TimescaleDB implementation session
- [README Session Complete](agents/20251022_080000_readme_session_complete.md) - Documentation update session

### Update Summaries
- [Update Summary 2025-10-22](agents/20251022_080000_update_summary_2025_10_22.md) - Daily progress update
- [Update Summary 2025-10-22 Metrics](agents/20251022_080000_update_summary_2025_10_22_metrics.md) - Performance metrics update

### Next Session Planning
- [Start Here Next Session](agents/20251022_080000_start_here_next_session.md) - Continuation guide for next work session

