# Testing Strategies

## Overview

This document outlines the testing strategies employed throughout the Heimdall project to ensure code quality, reliability, and performance.

## Testing Pyramid

```
        ▲
       /│\
      / │ \
     /  │  \  E2E Tests (10%)
    /   │   \
   /    │    \
  /     │     \  Integration Tests (30%)
 /      │      \
/       │       \ Unit Tests (60%)
```

## Unit Tests

### Purpose
- Test individual components in isolation
- Verify business logic correctness
- Fast feedback loop (< 1 second per test)

### Coverage Target
- **Goal**: 80%+
- **Current**: 85%

### Example

```python
import pytest
import numpy as np
from heimdall.signal.processor import extract_features

@pytest.fixture
def sample_signal():
    """Generate synthetic IQ signal."""
    return np.random.randn(48000, 2)

def test_feature_extraction(sample_signal):
    """Test mel-spectrogram feature extraction."""
    features = extract_features(sample_signal)
    
    assert features is not None
    assert features.shape == (128, 431)  # Expected mel-spectrogram shape
    assert not np.isnan(features).any()
    assert features.min() >= -5  # Normalized dB scale

def test_invalid_input():
    """Test error handling for invalid input."""
    with pytest.raises(ValueError):
        extract_features(None)
```

### Running Unit Tests

```bash
# All unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=heimdall --cov-report=html

# Specific test
pytest tests/unit/test_signal_processor.py::test_feature_extraction -v
```

## Integration Tests

### Purpose
- Test component interactions
- Verify data flow between services
- Test external dependencies (database, Redis, etc.)

### Coverage Target
- **Goal**: 70%+
- **Current**: 75%

### Example

```python
import pytest
from fastapi.testclient import TestClient
from heimdall.api.main import app

@pytest.fixture
def client():
    """API test client."""
    return TestClient(app)

def test_rf_task_submission(client, db_session):
    """Test RF acquisition task submission."""
    response = client.post(
        "/api/v1/tasks/rf-acquisition",
        json={
            "frequencies": [145.500],
            "duration": 60
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["status"] == "submitted"

def test_task_status_retrieval(client, rf_task_in_db):
    """Test retrieving task status."""
    response = client.get(f"/api/v1/tasks/{rf_task_in_db.id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["pending", "processing", "completed"]
```

### Running Integration Tests

```bash
# All integration tests
pytest tests/integration/ -v

# With specific marker
pytest -m integration -v

# With database
pytest tests/integration/ -v --db-url postgresql://user:pass@localhost/test
```

## End-to-End (E2E) Tests

### Purpose
- Test complete user workflows
- Verify system as a whole
- Validate real data flows

### Coverage Target
- **Goal**: 50%+
- **Current**: 87.5% (7/8 passing)

### Example

```python
@pytest.mark.e2e
def test_complete_localization_workflow(client, api_client):
    """Test complete RF acquisition to localization workflow."""
    
    # Step 1: Submit RF acquisition
    task = api_client.submit_rf_acquisition(
        frequencies=[145.500],
        duration=60
    )
    assert task.id is not None
    
    # Step 2: Wait for completion
    result = api_client.wait_for_result(task.id, timeout=300)
    assert result.status == "completed"
    
    # Step 3: Verify results
    assert result.location.latitude is not None
    assert result.location.longitude is not None
    assert result.location.uncertainty_m > 0
    
    # Step 4: Retrieve from history
    history = api_client.get_results(
        start_time=result.created_at,
        end_time=result.created_at
    )
    assert len(history) >= 1
```

### Running E2E Tests

```bash
# All E2E tests
pytest tests/e2e/ -v

# With logging
pytest tests/e2e/ -v -s --log-cli-level=DEBUG

# Sequential (not parallel)
pytest tests/e2e/ -v -n 0
```

## Performance Tests

### Purpose
- Verify latency requirements
- Test under load
- Identify bottlenecks

### Example

```python
import pytest
from time import time

def test_api_latency_sla(client):
    """Verify API response latency meets SLA."""
    latencies = []
    
    for _ in range(100):
        start = time()
        response = client.post(
            "/api/v1/tasks/rf-acquisition",
            json={"frequencies": [145.500], "duration": 60}
        )
        latency = (time() - start) * 1000  # Convert to ms
        latencies.append(latency)
        
        assert response.status_code == 201
    
    # Verify SLA
    p95 = np.percentile(latencies, 95)
    assert p95 < 100, f"P95 latency {p95}ms exceeds 100ms SLA"
```

### Running Performance Tests

```bash
# With performance markers
pytest tests/performance/ -v -m performance

# With profiling
pytest tests/performance/ -v --profile

# Load testing
locust -f tests/load/locustfile.py -u 100 -r 10
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
      
      redis:
        image: redis:latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      
      - name: Run linting
        run: make lint
      
      - name: Run unit tests
        run: pytest tests/unit/ --cov --cov-report=xml
      
      - name: Run integration tests
        run: pytest tests/integration/
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Data & Fixtures

### Fixture Strategy

```python
@pytest.fixture(scope="session")
def db():
    """Session-scoped database for tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture
def rf_task(db_session):
    """Create a sample RF task."""
    task = RFTask(
        frequencies=[145.500],
        duration=60,
        status="completed"
    )
    db_session.add(task)
    db_session.commit()
    return task

@pytest.fixture
def sample_measurements(rf_task):
    """Create sample measurements for a task."""
    return [
        Measurement(
            task_id=rf_task.id,
            station_name="Giaveno",
            frequency=145.500,
            signal_strength=-85.5,
            bearing=120.5
        ),
        Measurement(
            task_id=rf_task.id,
            station_name="Torino",
            frequency=145.500,
            signal_strength=-82.3,
            bearing=245.2
        ),
    ]
```

## Debugging Tests

### Using pytest features

```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Show variable values on failure
pytest -l

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Using logging in tests

```python
import logging

logger = logging.getLogger(__name__)

def test_with_logging(caplog):
    """Test with logging capture."""
    caplog.set_level(logging.DEBUG)
    
    # Your test code
    logger.debug("This is logged")
    
    assert "This is logged" in caplog.text
```

## Test Organization

```
tests/
├── unit/
│   ├── test_signal_processor.py
│   ├── test_ml_model.py
│   └── test_localization.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_message_queue.py
├── e2e/
│   ├── test_acquisition_workflow.py
│   ├── test_inference_workflow.py
│   └── test_full_pipeline.py
├── performance/
│   ├── test_latency.py
│   └── test_throughput.py
├── conftest.py  # Shared fixtures
└── fixtures/    # Test data
```

## Best Practices

1. **Keep tests focused**: One assertion per test when possible
2. **Use descriptive names**: Test name should describe what it tests
3. **Mock external dependencies**: Use mocks for external APIs
4. **Clean up after tests**: Use fixtures with proper teardown
5. **Avoid hard-coded values**: Use fixtures and parameters
6. **Test error cases**: Test both happy path and failure cases
7. **Maintain fast feedback**: Unit tests should run in seconds

## Metrics

- **Test Count**: 250+ tests
- **Coverage**: 85% code coverage
- **Pass Rate**: 98%+ (excluding expected failures)
- **Average Runtime**: 2 minutes for full suite

---

**Related**: [Contributing Guidelines](./contributing.md) | [Performance Benchmarks](./performance_benchmarks.md)
