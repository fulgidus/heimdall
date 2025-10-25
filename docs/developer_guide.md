# Developer Guide

## Getting Started

This guide helps new developers quickly get up to speed with Heimdall development.

## Environment Setup

### Quick Setup

```bash
# Clone repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Run setup script
make dev-setup

# Start development environment
make dev-up

# Verify everything works
curl http://localhost:8000/health
```

### Python Development

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make type-check

# All checks
make check
```

## Project Structure

```
heimdall/
├── frontend/                    # React/TypeScript frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API service layer
│   │   └── utils/             # Utility functions
│   └── package.json
│
├── services/                    # Microservices
│   ├── api-gateway/           # FastAPI main service
│   │   ├── main.py
│   │   ├── routers/           # API endpoint modules
│   │   └── models/            # Pydantic models
│   ├── rf-acquisition/        # WebSDR data collection
│   ├── signal-processor/      # Signal processing
│   ├── ml-detector/           # ML inference service
│   └── celery-worker/         # Async task processing
│
├── db/                          # Database
│   ├── schema.sql             # SQL schema
│   ├── migrations/            # Alembic migrations
│   └── seeds/                 # Initial data
│
├── tests/                       # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
│
├── helm/                        # Kubernetes Helm charts
├── docker-compose.yml         # Local development setup
├── docker-compose.prod.yml    # Production setup
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── Makefile                   # Build automation
```

## Development Workflow

### Creating a New Feature

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write tests first** (TDD approach)
   ```bash
   # Create test file
   touch tests/unit/test_my_feature.py
   
   # Write tests
   # Run tests (should fail)
   pytest tests/unit/test_my_feature.py -v
   ```

3. **Implement feature**
   ```bash
   # Edit relevant service files
   # vim services/api-gateway/routers/my_router.py
   
   # Run tests (should pass)
   pytest tests/unit/test_my_feature.py -v
   ```

4. **Code quality checks**
   ```bash
   # Format code
   make format
   
   # Run linting
   make lint
   
   # Type checking
   make type-check
   ```

5. **Create pull request**
   ```bash
   git push origin feature/my-feature
   # Create PR on GitHub
   ```

### Adding a New API Endpoint

Example: Adding GET `/api/v1/results/{task_id}`

1. **Create model** (`services/api-gateway/models.py`):
   ```python
   from pydantic import BaseModel
   from datetime import datetime
   
   class LocationResult(BaseModel):
       latitude: float
       longitude: float
       uncertainty_m: float
   
   class TaskResult(BaseModel):
       task_id: str
       status: str
       location: LocationResult
       completed_at: datetime
   ```

2. **Create router** (`services/api-gateway/routers/results.py`):
   ```python
   from fastapi import APIRouter, HTTPException
   from heimdall.models import TaskResult
   
   router = APIRouter(prefix="/results", tags=["results"])
   
   @router.get("/{task_id}", response_model=TaskResult)
   async def get_result(task_id: str):
       """Get localization result for task."""
       result = db.query_result(task_id)
       if not result:
           raise HTTPException(status_code=404)
       return result
   ```

3. **Include router** in main app (`services/api-gateway/main.py`):
   ```python
   from heimdall.routers import results
   
   app.include_router(results.router, prefix="/api/v1")
   ```

4. **Test endpoint**:
   ```bash
   curl http://localhost:8000/api/v1/results/task-123
   ```

### Adding Database Schema

1. **Create migration**:
   ```bash
   alembic revision --autogenerate -m "Add result_confidence column"
   ```

2. **Review migration** (`db/migrations/versions/xyz.py`):
   ```python
   def upgrade():
       op.add_column('task_results', 
           sa.Column('confidence', sa.Float()))
   
   def downgrade():
       op.drop_column('task_results', 'confidence')
   ```

3. **Apply migration**:
   ```bash
   alembic upgrade head
   ```

4. **Verify**:
   ```bash
   docker-compose exec postgres \
     psql -U heimdall_user -d heimdall -c \
     "SELECT * FROM information_schema.columns WHERE table_name='task_results';"
   ```

### Adding Tests

Example unit test structure:

```python
import pytest
from unittest.mock import Mock, patch
from heimdall.signal.processor import SignalProcessor

class TestSignalProcessor:
    """Tests for SignalProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return SignalProcessor()
    
    def test_process_valid_signal(self, processor):
        """Test processing valid signal."""
        # Arrange
        mock_signal = Mock()
        mock_signal.shape = (48000, 2)
        
        # Act
        result = processor.process(mock_signal)
        
        # Assert
        assert result is not None
        assert hasattr(result, 'features')
    
    def test_process_invalid_signal_raises_error(self, processor):
        """Test that invalid signal raises error."""
        with pytest.raises(ValueError):
            processor.process(None)
```

Run tests:
```bash
pytest tests/unit/test_signal_processor.py -v

# With coverage
pytest tests/unit/test_signal_processor.py --cov=heimdall

# With debugging
pytest tests/unit/test_signal_processor.py -vv -s --pdb
```

## Debugging

### Using print statements

```python
def process_signal(signal):
    print(f"DEBUG: signal shape = {signal.shape}")
    result = expensive_operation(signal)
    print(f"DEBUG: result = {result[:5]}")  # First 5 items
    return result
```

### Using debugger

```python
import pdb

def my_function():
    x = get_value()
    pdb.set_trace()  # Execution pauses here
    y = process(x)
    return y

# Or run pytest with debugger
pytest tests/unit/test_my_feature.py --pdb
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def important_function():
    logger.debug("Starting operation")
    try:
        result = do_work()
        logger.info(f"Success: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        raise

# View logs
docker-compose logs -f api-gateway | grep important_function
```

## Common Tasks

### Running services

```bash
# Start all services
make dev-up

# Start specific service
docker-compose up -d api-gateway

# Stop all services
make dev-down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart api-gateway
```

### Database operations

```bash
# Connect to database
docker-compose exec postgres psql -U heimdall_user -d heimdall

# Run query
SELECT * FROM signal_measurements LIMIT 10;

# Backup
make db-backup

# Restore
make db-restore BACKUP_FILE=backup.sql

# Reset (WARNING: deletes data)
make db-reset
```

### Running tests

```bash
# All tests
make test

# Unit only
pytest tests/unit/ -v

# Integration only
pytest tests/integration/ -v

# E2E only
pytest tests/e2e/ -v

# Specific test
pytest tests/unit/test_signal_processor.py::test_feature_extraction -v

# With coverage
make test-coverage
```

## Performance Profiling

### Using cProfile

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = expensive_operation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Using memory_profiler

```python
from memory_profiler import profile

@profile
def my_function():
    large_list = [i for i in range(1000000)]
    return sum(large_list)

# Run with: python -m memory_profiler my_script.py
```

## Release Process

1. **Update version** in `__init__.py`:
   ```python
   __version__ = "1.0.0"
   ```

2. **Update CHANGELOG.md**:
   ```markdown
   ## [1.0.0] - 2025-10-22
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

3. **Create release tag**:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

4. **Build release artifacts**:
   ```bash
   docker build -t heimdall:1.0.0 .
   docker tag heimdall:1.0.0 heimdall:latest
   docker push heimdall:1.0.0
   ```

## Code Style & Conventions

### Python

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all functions
- Use meaningful variable names
- Write docstrings for all modules, functions, classes

```python
def calculate_uncertainty(predictions: np.ndarray) -> float:
    """
    Calculate Gaussian uncertainty from model predictions.
    
    Args:
        predictions: Model output array of shape (batch, 2)
                    First column: means, second: log-std
    
    Returns:
        Mean uncertainty in meters
        
    Raises:
        ValueError: If predictions shape invalid
    """
    if predictions.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2), got {predictions.shape}")
    
    log_std = predictions[:, 1]
    std = np.exp(log_std)
    return np.mean(std)
```

### Commit Messages

```
feat(api): add result caching endpoint

Implement GET /results endpoint with Redis caching
to improve repeated query performance.

- Add ResultSchema pydantic model
- Implement caching decorator
- Add comprehensive tests

Fixes #123
```

---

**Next Steps**: 
- [Contributing Guidelines](./contributing.md)
- [Testing Strategies](./testing_strategies.md)
- [Architecture Guide](./ARCHITECTURE.md)

**Questions?** Open an [issue](https://github.com/fulgidus/heimdall/issues) or check [FAQ](./faqs.md)
