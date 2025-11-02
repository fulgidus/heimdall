# Testing Guide

Comprehensive testing suite for the Heimdall RF source localization system.

## Overview

The testing suite covers:
- **Unit Tests**: Fast, isolated tests with no external dependencies
- **Integration Tests**: Tests requiring database, MinIO, or other services
- **Performance Tests**: Benchmarks validating performance targets
- **E2E Tests**: Full workflow tests across multiple services

## Test Structure

```
services/
├── training/
│   └── tests/
│       ├── unit/                      # Unit tests
│       ├── integration/               # Integration tests
│       │   └── test_synthetic_pipeline_integration.py
│       ├── test_feature_extractor_basic.py
│       ├── test_iq_generator.py
│       ├── test_performance.py        # Performance benchmarks
│       └── conftest.py
├── backend/
│   └── tests/
│       ├── unit/
│       │   ├── test_batch_feature_extraction.py
│       │   └── test_feature_extraction_task_extended.py
│       ├── integration/
│       └── e2e/
└── common/
    └── tests/
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run tests for specific service
cd services/training && pytest
cd services/backend && pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest services/training/tests/test_performance.py -v
```

### By Test Type

```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests (requires Docker services)
pytest -m integration

# Performance benchmarks
pytest -m performance

# Exclude slow tests
pytest -m "not slow"

# Performance tests excluding slow ones
pytest -m "performance and not slow"
```

### With Coverage

```bash
# Generate coverage report
pytest --cov=services --cov-report=html --cov-report=term-missing

# View HTML report
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html       # macOS
```

### Specific Service Tests

```bash
# Training service tests
cd services/training
pytest -v

# With coverage
pytest --cov=src --cov-report=html

# Only performance tests
pytest tests/test_performance.py -v -m performance

# Backend service tests
cd services/backend
pytest tests/unit/ -v
```

## Test Categories

### 1. Feature Extractor Tests

**File**: `services/training/tests/test_feature_extractor_basic.py`

Tests the RF feature extraction from IQ samples:
- ✅ Basic feature extraction
- ✅ Signal presence detection
- ✅ Chunked extraction with aggregation
- ✅ Frequency offset calculation
- ✅ Spectral features
- ✅ Confidence scoring

```bash
pytest services/training/tests/test_feature_extractor_basic.py -v
```

### 2. IQ Generator Tests

**File**: `services/training/tests/test_iq_generator.py`

Tests synthetic IQ sample generation:
- ✅ Generator initialization
- ✅ IQ sample shape and type validation
- ✅ SNR validation
- ✅ Multipath effects
- ✅ Rayleigh fading
- ✅ Reproducibility with seeds

```bash
pytest services/training/tests/test_iq_generator.py -v
```

### 3. Synthetic Pipeline Integration Tests

**File**: `services/training/tests/integration/test_synthetic_pipeline_integration.py`

Tests full synthetic data generation pipeline:
- ✅ Single sample generation
- ✅ Multiple receiver handling
- ✅ Reproducibility testing
- ✅ Feature structure validation
- ✅ Seed-based determinism

```bash
pytest services/training/tests/integration/ -v -m integration
```

### 4. Real Recording Feature Extraction Tests

**File**: `services/backend/tests/unit/test_feature_extraction_task_extended.py`

Tests feature extraction from real recordings:
- ✅ Task structure validation
- ✅ IQ sample loading
- ✅ Feature extraction from IQ
- ✅ Error handling
- ✅ Metadata structure validation
- ✅ Multi-receiver aggregation
- ✅ Beacon information handling

```bash
pytest services/backend/tests/unit/test_feature_extraction_task_extended.py -v
```

### 5. Batch Feature Extraction Tests

**File**: `services/backend/tests/unit/test_batch_feature_extraction.py`

Tests background batch processing:
- ✅ Finding recordings without features
- ✅ Batch task execution
- ✅ Task queueing
- ✅ Backfill operations
- ✅ Safety limits

```bash
pytest services/backend/tests/unit/test_batch_feature_extraction.py -v
```

### 6. Performance Benchmarks

**File**: `services/training/tests/test_performance.py`

Performance tests with strict targets:

| Test | Target | Expected | Command |
|------|--------|----------|---------|
| IQ Generation | <50ms | ~30ms | `pytest -k test_iq_generation_performance` |
| Feature Extraction | <100ms | ~60ms | `pytest -k test_feature_extraction_performance` |
| End-to-End | <150ms | ~90ms | `pytest -k test_end_to_end_performance` |
| Batch (3 RX) | <500ms | ~300ms | `pytest -k test_batch_generation_performance` |

```bash
# Run all performance tests
pytest services/training/tests/test_performance.py -v -m performance

# Run specific performance test
pytest -k test_iq_generation_performance -v

# Run without slow tests
pytest -m "performance and not slow" -v
```

Performance tests output detailed timing statistics:
```
IQ Generation Performance:
  Average: 28.45 ms
  StdDev:  3.12 ms
  Min:     24.10 ms
  Max:     35.20 ms
```

## Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit           # Fast unit test
@pytest.mark.integration    # Requires Docker services
@pytest.mark.performance    # Performance benchmark
@pytest.mark.slow           # Long-running test
@pytest.mark.asyncio        # Async test
@pytest.mark.e2e            # End-to-end test
```

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for [component name].
"""

import pytest
from src.module import Component


@pytest.mark.unit
def test_component_basic_functionality():
    """Test basic component operation."""
    component = Component()
    result = component.do_something()
    assert result is not None
```

### Integration Test Template

```python
"""
Integration tests for [feature name].
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_integration(test_db_pool):
    """Test database operations."""
    async with test_db_pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1
```

### Performance Test Template

```python
"""
Performance tests for [component name].
"""

import pytest
import time
from statistics import mean


@pytest.mark.performance
def test_operation_performance():
    """Benchmark operation speed."""
    times = []
    for _ in range(100):
        start = time.perf_counter()
        # Operation to benchmark
        result = expensive_operation()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = mean(times) * 1000  # ms
    print(f"\nAverage: {avg_time:.2f} ms")
    
    # Assert performance target
    assert avg_time < 50.0, f"Too slow: {avg_time:.2f}ms"
```

## Test Configuration

### Root Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = services
python_files = test_*.py
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance benchmarks
    slow: Slow tests
asyncio_mode = auto
```

### Training Service Configuration (`services/training/pytest.ini`)

```ini
[pytest]
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance benchmarks
    slow: Slow tests
```

## Coverage Targets

- **Overall**: ≥80%
- **Core modules**: ≥90%
- **Feature extraction**: ≥85%
- **Data generation**: ≥85%

Check coverage:
```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

## Continuous Integration

Tests run automatically on:
- Every push to main
- Every pull request
- Nightly builds

CI runs:
1. Unit tests (all services)
2. Integration tests (with Docker)
3. Performance tests (benchmarks only, not slow)
4. Coverage reporting

## Troubleshooting

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/heimdall/services/training/src:$PYTHONPATH

# Or use conftest.py (already configured)
```

### Docker Services Required

Some integration tests require Docker services:

```bash
# Start services
docker-compose up -d

# Run integration tests
pytest -m integration

# Stop services
docker-compose down
```

### Slow Performance Tests

Performance tests may timeout on slow machines:

```bash
# Skip slow tests
pytest -m "performance and not slow"

# Increase timeout
pytest --timeout=300
```

### Test Isolation

Each test should be independent:

```python
@pytest.fixture
def clean_database():
    """Ensure clean database state."""
    # Setup
    yield
    # Teardown
```

## Best Practices

1. **Test Naming**: Use descriptive names
   - ✅ `test_generate_single_sample_multiple_receivers`
   - ❌ `test_generation`

2. **Assertions**: Use specific assertions
   - ✅ `assert result == expected_value`
   - ❌ `assert result`

3. **Fixtures**: Reuse test data
   ```python
   @pytest.fixture
   def sample_config():
       return {"frequency_mhz": 144.0}
   ```

4. **Markers**: Categorize tests properly
   ```python
   @pytest.mark.unit
   @pytest.mark.performance
   def test_fast_operation():
       pass
   ```

5. **Documentation**: Add docstrings
   ```python
   def test_feature_extraction():
       """Test that features are extracted correctly from IQ samples."""
       pass
   ```

## Performance Targets

### Training Pipeline

| Operation | Target | Status |
|-----------|--------|--------|
| IQ Generation | <50ms | ✅ |
| Feature Extraction | <100ms | ✅ |
| End-to-End | <150ms | ✅ |
| 10k Samples (24 cores) | <3 min | ✅ |

### Backend Pipeline

| Operation | Target | Status |
|-----------|--------|--------|
| Recording Feature Extraction | <2s | ✅ |
| Batch Processing (50 recordings) | <30s | ✅ |

## Maintenance

- Run tests before every commit
- Update tests when changing functionality
- Keep performance benchmarks current
- Review coverage reports monthly
- Add regression tests for production bugs

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Project Architecture](./ARCHITECTURE.md)
- [Contributing Guide](../CONTRIBUTING.md)
