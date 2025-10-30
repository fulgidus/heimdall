# Changelog

All notable changes to the Heimdall SDR project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet

### Fixed
- Nothing yet

---

## [0.2.0] - 2025-10-30

### Added
- **Centralized Dependency Management System** (2025-10-25, PR #3)
  - Created `services/requirements/` directory with modular requirement files
  - Added `base.txt`, `dev.txt`, `ml.txt`, `api.txt`, `data.txt` for shared dependencies
  - Implemented `scripts/lock_requirements.py` for generating version-pinned lock files
  - Implemented `scripts/audit_dependencies.py` for dependency analysis and conflict detection
  - Created `.github/workflows/dependency-updates.yml` for automated weekly dependency updates
  - Added comprehensive dependency management documentation in `docs/dependency_management.md`
  - Added Makefile targets: `lock-deps`, `audit-deps`, `deps-check`
  - Updated all service Dockerfiles to use centralized requirements with proper build contexts
  - Updated `docker-compose.yml` with consistent build contexts and PIP_NO_CACHE_DIR arg
  - Created `.github/ISSUE_TEMPLATE/dependency-update.md` for dependency issue reporting
  - Resolved version conflicts in boto3 and onnxruntime across services
  - Security: Automated vulnerability scanning with safety library
  - Security: Weekly dependency update workflow with security checks
  - Production stability: Version pinning strategy ensures reproducible and secure builds

- **WebSocket Real-Time Dashboard Updates** (2025-10-25)
  - Implemented WebSocket support for real-time updates to Dashboard without polling overhead
  - Frontend WebSocket manager with auto-reconnection and exponential backoff (1s → 30s max)
  - Connection state tracking (Connected, Connecting, Reconnecting, Disconnected)
  - Event subscription/unsubscription system for targeted updates
  - Heartbeat/ping-pong for connection keep-alive (30s interval)
  - Backend WebSocket endpoint at `/ws/updates` in API Gateway
  - Connection manager for broadcasting to multiple clients
  - Dashboard integration with connection status indicator (badge with color-coded states)
  - Reconnection button for manual reconnect attempts
  - Graceful fallback to 30s polling when WebSocket unavailable
  - Event types: `services:health`, `websdrs:status`, `signals:detected`, `localizations:updated`
  - Test coverage: 16/16 frontend WebSocket tests, 5/5 backend WebSocket tests, 11/11 Dashboard integration tests
  - Files: frontend/src/lib/websocket.ts, services/api-gateway/src/websocket_manager.py, services/api-gateway/src/main.py (WebSocket route)

### Fixed
- **Health Check Endpoints**: Fixed all service health check paths (2025-10-25)
  - Issue: Frontend dashboard calls `/api/v1/{service}/health` but backend services only had `/health` at root
  - Root cause: API Gateway's `proxy_request()` preserved full paths, but backends didn't have nested `/api/v1/*` paths
  - Solution:
    - API Gateway: Added path-stripping handlers for `/api/v1/api-gateway/health`, `/api/v1/rf-acquisition/health`, `/api/v1/inference/health`
    - Inference service: Added `/api/v1/inference/health` endpoint to match frontend expectations
    - All health endpoints now return proper HealthResponse JSON with status, service name, version, and timestamp
  - **Verification**: ✅ All 3 health check endpoints return HTTP 200
  - Changes: services/api-gateway/src/main.py (lines 175-230), services/inference/src/main.py (lines 38-41)

- **Frontend Tests**: Fixed all 283 frontend test failures (2025-10-25)
  - Added React to global scope in vitest test setup to resolve "React is not defined" errors
  - Fixed authStore test expectations to match API Gateway endpoint instead of Keycloak
  - All test files now properly support JSX syntax in React 19 with new JSX transform
  - 100% test success rate: 283 tests passing across 19 test files
  - Test execution time: ~14 seconds
  - Changes: frontend/src/test/setup.ts (added global React), frontend/src/store/authStore.test.ts (fixed endpoint)

### Added
- **Frontend Rebuild Phase 8**: Docker Integration (In Progress)
  - Updated Dockerfile to use Node 20 (required for rolldown-vite)
  - Switched from node+serve to nginx for production deployment
  - Created comprehensive nginx.conf with gzip, caching, API proxy, WebSocket support
  - Added security headers (X-Frame-Options, CSP, X-XSS-Protection)
  - Configured health check endpoint at /health
  - Added frontend service to docker compose.yml on port 3001
  - Environment variable support via build args (VITE_API_URL, VITE_ENV, etc.)
  - Multi-stage build for optimized image size
  - Logging configuration with rotation (10MB max, 3 files)
- **Frontend Rebuild Phase 7**: Testing & Validation (In Progress)
  - Created comprehensive responsive design tests for mobile/tablet/desktop viewports
  - Created real-time data update tests with timer mocking
  - Created interactive features validation tests (buttons, forms, modals, tables)
  - Added accessibility testing for all interactive elements
  - Test coverage for: responsive behavior, form submission, navigation, tab switching
  - Validation tests for: loading states, error handling, user interactions
- **Frontend Rebuild Phase 6**: API Integration Verification
  - Created comprehensive API integration test suite covering all services
  - Verified WebSDR service endpoints (list, health check, config)
  - Verified Acquisition service endpoints (trigger, status polling)
  - Verified Inference service endpoints (predict, recent localizations)
  - Verified Session service endpoints (list, create, update, delete)
  - Verified Analytics service endpoints (metrics, performance, distribution)
  - Verified System service endpoints (health check)
  - Added error handling tests for network, 404, and 500 errors
  - All API services properly connected and functional
- **Frontend Rebuild Phase 5**: Components Library - Reusable component creation
  - Created Table component with sortable columns, custom rendering, and multiple variants
  - Created StatCard component with 5 color variants, trend indicators, and icon support
  - Created ChartCard wrapper for Chart.js with loading states and error handling
  - Created Select component with validation, size variants, and full-width option
  - Created Textarea component with validation and auto-resize capabilities
  - Updated component exports with TypeScript type definitions
  - All components follow Datta Able Bootstrap design system
  - Components are fully responsive and support dark theme
- **Frontend Rebuild Phase 4**: Analytics page rebuild - real charts and API integration
  - Integrated Chart.js Line and Pie charts with real analytics data from analyticsStore
  - Added prediction trends line chart showing total/successful/failed predictions over time
  - Added accuracy distribution pie chart with real data from API endpoint
  - Connected Analytics page to analyticsStore for real-time data loading
  - Updated WebSDR performance table to use analytics data when available (uptime, SNR, acquisitions, success rate)
  - Added loading states and error handling for analytics data fetching
  - Fixed metric calculations to use time series data from predictionMetrics API

### Fixed
- **API Gateway Analytics Routing**: Fixed missing routing for `/api/v1/analytics/*` endpoints
  - Corrected INFERENCE_URL port from 8002 to 8003 in api-gateway service
  - Added analytics proxy route in api-gateway to forward requests to inference service
  - Fixed router imports in inference service __init__.py for proper module loading
  - Added test endpoint `/api/v1/analytics/test` for debugging routing issues
  - Temporarily disabled predict router to isolate analytics endpoint problems
  - Updated time range selector to reload analytics data for different periods (24h, 7d, 30d)
  - Added proper TypeScript interfaces and error handling throughout
  - Removed all mock data placeholders and replaced with real API calls
  - Analytics page now fully functional with real backend integration
- **Frontend Rebuild Phase 4**: Analytics page rebuild - API services and store setup
  - Installed Chart.js and react-chartjs-2 for data visualization
  - Created analytics API service with endpoints for prediction metrics, WebSDR performance, system performance, and accuracy distribution
  - Created analyticsStore with Zustand for managing analytics state and API calls
  - Added analytics store to main store exports
  - Ready to integrate real charts and analytics data into Analytics page
- **Frontend Rebuild Phase 4**: Localization page rebuild - API integration
  - Connected Localization page to real localizationStore instead of mock data
  - Updated page to use recentLocalizations from API instead of hardcoded data
  - Fixed TypeScript interfaces to match backend LocalizationResult structure
  - Updated field mappings: uncertainty_m, websdr_count, confidence as decimal
  - Added signal quality calculation based on snr_avg_db
  - Fixed CSS classes to use modern Tailwind (shrink-0, grow instead of flex-shrink-0, flex-grow-1)
  - Removed unused imports and variables
  - Page now ready to display real localization results from inference service
- **Frontend Rebuild Phase 4**: Localization API services and store
  - Added predictLocalization, predictLocalizationBatch, and getRecentLocalizations functions to inference API service
  - Created localizationStore with Zustand for managing localization state
  - Added PredictionRequest and PredictionResponse TypeScript interfaces
  - Integrated localization store into main store exports
  - Ready to connect Localization page to real API endpoints instead of mock data
- **Frontend Rebuild Phase 4**: Localization API services and store
  - Added predictLocalization, predictLocalizationBatch, and getRecentLocalizations functions to inference API service
  - Created localizationStore with Zustand for managing localization state
  - Added PredictionRequest and PredictionResponse TypeScript interfaces
  - Integrated localization store into main store exports
  - Ready to connect Localization page to real API endpoints instead of mock data
- **Frontend Rebuild Phase 4**: Localization API services and store
  - Added predictLocalization, predictLocalizationBatch, and getRecentLocalizations functions to inference API service
  - Created localizationStore with Zustand for managing localization state and API calls
  - Added PredictionRequest and PredictionResponse TypeScript interfaces
  - Integrated localization store into main store exports
  - Ready to connect Localization page to real API endpoints instead of mock data
- Frontend TypeScript compilation complete - all 31 errors resolved
- Unified type system across sessionStore, API layer, and all components
- Dev server running on http://localhost:3000/ with hot reload
- Production build successful (484.58 KB gzipped)
- Frontend testing integrated into CI pipeline with Node.js 20 and Vitest
- Parallel execution of backend and frontend tests for faster CI runs
- ESLint linting for frontend code (non-blocking)
- TypeScript build verification in CI (blocking on errors)
- Comprehensive frontend testing documentation in contributing guide
- Phase 7: Frontend UI complete rebuild with Datta Able template
- Analytics page rebuilt with real backend data integration
- Comprehensive testing frameworks
- Improved orphan detection script (`scripts/find_orphan_docs.py`) with intelligent categorization and AI-assisted suggestions
- Phase index files for comprehensive navigation (Phase 4 and Phase 7)
- Comprehensive documentation portal organization in `docs/index.md`
- Project status and session tracking section in documentation
- Enhanced troubleshooting resources with specific diagnostic guides
- Technical guides section with WebSDR implementation documentation
- Frontend development documentation with bilingual support

### Changed
- **Frontend TypeScript System**
  - Changed all sessionId parameters from `string` to `number` across store and API
  - Updated createSession call pattern from 3 parameters to object parameter
  - Added optional property handling for `source_frequency`, `started_at`, `session_start`
  - Modernized CSS classes: `flex-shrink-0` → `shrink-0`
  - Unified status enum to include `'in_progress'` variant

- **CI/CD Pipeline Enhancement**
  - Added `frontend-test` job to GitHub Actions workflow
  - Backend and frontend tests now run in parallel
  - Test summary job checks both backend and frontend results
  - npm caching for faster frontend dependency installation
  - Security: Added explicit permissions (contents: read) to frontend-test job
- **Documentation Updates**
  - Updated `docs/testing_strategies.md` with frontend testing examples
  - Updated `docs/contributing.md` with frontend test commands
  - Documented parallel execution strategy for CI
- **Frontend UI rebuild (Phase 4 COMPLETE - 8/8 pages 100%)**
  - Analytics page: Complete rebuild using Datta Able Bootstrap components
  - WebSDR Management page: Complete rebuild with real-time health monitoring
  - Data Ingestion page: Complete rebuild with Known Sources and Recording Sessions management
  - Localization page: Complete rebuild with map placeholder and results display
  - Settings page: Complete rebuild with tabbed configuration interface
  - Profile page: Complete rebuild with user information and security settings
  - Recording Session page: Complete rebuild with acquisition workflow and real-time status
  - Session History page: Complete rebuild with filtering, pagination, and detailed view
  - **ALL pages now use Datta Able Bootstrap components**
  - **ALL pages connect to real backend APIs via Zustand stores**
  - Replaced Lucide icons with Phosphor icons throughout
  - Implemented proper Bootstrap 5 grid system and card components
  - Consistent UI patterns across all pages (breadcrumbs, cards, tables, forms)
  - Real-time data updates with auto-refresh mechanisms
  - Complete CRUD operations for sessions, sources, and configuration
  - Status tracking with color-coded badges
  - Pagination and filtering for data tables
  - Loading states and error handling
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
- Updated docker compose.yml with better health check strategies

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
- docker compose.services.yml for service orchestration
- Service containers: rf-acquisition, training, inference, api-gateway
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
- Development environment setup (docker compose.yml)
- Production environment configuration (docker compose.prod.yml)
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
