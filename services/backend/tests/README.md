# Backend Service Tests

Test suite for the backend service handling RF acquisition and feature extraction.

## Structure

```
tests/
├── unit/                                    # Unit tests
│   ├── test_batch_feature_extraction.py    # Batch processing tests
│   ├── test_feature_extraction_task_extended.py  # Feature extraction tests
│   └── ...
├── integration/                             # Integration tests
│   ├── test_acquisition_endpoints.py
│   ├── test_minio_storage.py
│   └── ...
├── e2e/                                     # End-to-end tests
│   └── test_complete_workflow.py
├── conftest.py                              # Shared fixtures
└── README.md                                # This file
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires Docker)
pytest tests/integration/ -v -m integration

# End-to-end tests
pytest tests/e2e/ -v -m e2e
```

## Test Categories

### Unit Tests

#### Batch Feature Extraction
- `test_batch_feature_extraction.py` - Background batch processing
- Coverage: Finding recordings, task queuing, backfill operations

#### Feature Extraction Task
- `test_feature_extraction_task_extended.py` - Real recording feature extraction
- Coverage: IQ loading, feature extraction, error handling, metadata validation

### Integration Tests

Tests requiring Docker infrastructure:
- Database operations (PostgreSQL + TimescaleDB)
- MinIO storage operations
- RabbitMQ task queuing
- Full acquisition workflows

### E2E Tests

Complete workflows across multiple services:
- Recording session creation
- IQ data acquisition
- Feature extraction
- Storage and retrieval

## Coverage

Run with coverage:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

Target: ≥80% overall, ≥90% for critical paths

## See Also

- [Main Testing Guide](../../../docs/TESTING.md)
- [Backend Service README](../../README.md)
