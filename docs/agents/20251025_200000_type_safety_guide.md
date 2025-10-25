# Type Safety & Static Analysis Guide

**Date**: 2025-10-25  
**Author**: Heimdall DevOps Team  
**Status**: Active

## Overview

This guide covers type checking and static analysis practices for the Heimdall SDR project. All Python code should follow strict type safety standards to catch bugs early and improve code reliability.

## Type Checking with mypy

### Configuration

The project uses strict mypy configuration defined in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
disallow_untyped_defs = true
disallow_untyped_calls = true
```

### Running Type Checks

**Comprehensive check:**
```bash
make type-check
```

**Strict mode (verbose):**
```bash
make type-check-strict
```

**Watch mode (development):**
```bash
make type-check-watch
```

**Generate coverage report:**
```bash
make type-coverage
```

### Type Hint Best Practices

#### 1. Function Signatures

Always type function parameters and return values:

```python
def process_signal(iq_data: NDArray[np.complex64], sample_rate: int) -> float:
    """Process IQ signal and return SNR."""
    # Implementation
    return snr_db
```

#### 2. Class Attributes

Use dataclasses or Pydantic models with typed attributes:

```python
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

@dataclass
class IQData:
    """IQ signal data with metadata."""
    iq_samples: NDArray[np.complex64]
    sample_rate: int
    center_frequency_hz: int
    
    def __post_init__(self) -> None:
        """Validate data types."""
        if self.iq_samples.dtype != np.complex64:
            raise TypeError(f"Expected complex64, got {self.iq_samples.dtype}")
```

#### 3. Optional vs None

Use `Optional[T]` for values that can be None:

```python
from typing import Optional

def fetch_data(url: str, timeout: Optional[int] = None) -> dict[str, str]:
    """Fetch data with optional timeout."""
    # Implementation
```

#### 4. Generic Types

Use proper generic types for collections:

```python
# Good
def get_receivers() -> list[WebSDRConfig]:
    """Get list of WebSDR configurations."""
    return receivers

# Bad - avoid bare list
def get_receivers() -> list:
    return receivers
```

## Pydantic Validation

### API Request/Response Models

Use Pydantic models for all API contracts:

```python
from pydantic import BaseModel, Field, ConfigDict

class AcquisitionRequest(BaseModel):
    """Request to trigger RF acquisition."""
    
    frequency_mhz: float = Field(..., gt=2.0, lt=1000.0)
    duration_seconds: int = Field(..., ge=1, le=600)
    receiver_ids: Optional[list[str]] = None
    
    model_config = ConfigDict(strict=True)
```

### Validation Benefits

- Runtime type checking
- Automatic JSON serialization/deserialization
- OpenAPI schema generation
- Input validation with constraints

### Custom Validators

```python
from pydantic import field_validator

class SignalData(BaseModel):
    frequency_mhz: float
    
    @field_validator('frequency_mhz')
    @classmethod
    def validate_frequency(cls, v: float) -> float:
        """Validate frequency is in amateur radio bands."""
        if not (144.0 <= v <= 148.0 or 430.0 <= v <= 440.0):
            raise ValueError('Frequency must be in 2m or 70cm band')
        return v
```

## Static Analysis Tools

### pylint

Check for errors and code quality issues:

```bash
make lint-types
```

Configuration in `.pylintrc`:
- Focuses on errors (E) and fatal issues (F)
- Max line length: 100
- Reasonable complexity limits

### flake8

PEP 8 style checking:

```bash
flake8 services/
```

Configuration in `.flake8`:
- Max line length: 100
- Ignores E203, W503 (black compatibility)
- Per-file ignores for tests and `__init__.py`

## Pre-commit Hooks

Type checking runs automatically on commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks configured in `.pre-commit-config.yaml`:
- mypy (strict mode)
- pylint (errors only)
- black (formatting)
- ruff (linting)

## CI/CD Integration

### GitHub Actions

Type checking runs on every PR via `.github/workflows/type-safety.yml`:

1. Install dependencies
2. Run comprehensive type checks
3. Generate HTML report
4. Upload artifacts

### Required Checks

All PRs must pass:
- ✅ mypy --strict (0 errors)
- ✅ pylint E/F checks (0 errors)
- ✅ flake8 (0 errors)

## Common Type Errors and Fixes

### Error: "Function is missing a return type annotation"

```python
# Bad
def calculate_snr(signal, noise):
    return 10 * np.log10(signal / noise)

# Good
def calculate_snr(signal: float, noise: float) -> float:
    return 10 * np.log10(signal / noise)
```

### Error: "Need type annotation for 'x'"

```python
# Bad
results = []

# Good
results: list[MeasurementRecord] = []
```

### Error: "Argument has incompatible type"

```python
# Bad - implicit Any
def process(data):
    return data

result = process(123)  # mypy can't verify type

# Good - explicit types
def process(data: int) -> int:
    return data

result: int = process(123)  # mypy verifies type safety
```

### Error: "Incompatible return value type"

```python
# Bad
def get_config() -> dict[str, str]:
    return {"timeout": 30}  # int value, not str

# Good
def get_config() -> dict[str, int | str]:
    return {"timeout": 30, "url": "http://example.com"}
```

## Exception Handling and Typing

Type exceptions properly:

```python
from typing import NoReturn

def handle_critical_error(msg: str) -> NoReturn:
    """Raise exception and never return."""
    raise RuntimeError(msg)

def safe_divide(a: float, b: float) -> float | None:
    """Divide or return None on error."""
    try:
        return a / b
    except ZeroDivisionError:
        return None
```

## numpy and torch Typing

### numpy arrays

```python
from numpy.typing import NDArray
import numpy as np

def process_iq(data: NDArray[np.complex64]) -> NDArray[np.float32]:
    """Process IQ data to magnitude."""
    return np.abs(data).astype(np.float32)
```

### PyTorch tensors

```python
import torch
from torch import Tensor

def forward(x: Tensor) -> Tensor:
    """Forward pass through network."""
    return self.model(x)
```

## When to Use `Any`

Avoid `Any` unless absolutely necessary. Acceptable cases:

1. **Third-party library without stubs:**
```python
from typing import Any
import some_untyped_lib

def wrapper(data: dict[str, Any]) -> Any:
    return some_untyped_lib.process(data)
```

2. **Truly dynamic data:**
```python
def parse_json(text: str) -> Any:
    """Parse JSON with unknown structure."""
    return json.loads(text)
```

Add justification comment when using `Any`:

```python
# Using Any because external API returns dynamic structure
response: Any = api.fetch()
```

## Type Coverage Goals

Project targets:
- **Overall coverage**: 95%+
- **New code**: 100% (strictly enforced)
- **Tests**: Excluded from strict checking

Check coverage:
```bash
make type-coverage
```

## Troubleshooting

### "Module has no attribute X"

Add to mypy overrides in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["problematic_lib.*"]
ignore_missing_imports = true
```

### "Incompatible types in assignment"

Check if you need a type cast:

```python
from typing import cast

value = cast(int, some_function())  # Tell mypy the type
```

### Too many mypy errors

Use `# type: ignore[error-code]` sparingly:

```python
result = legacy_function()  # type: ignore[no-untyped-call]
```

Always include the specific error code.

## Resources

- [mypy documentation](https://mypy.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 - Variable Annotations](https://peps.python.org/pep-0526/)
- [numpy typing](https://numpy.org/doc/stable/reference/typing.html)

## Summary

- **Always** type function signatures
- **Use** Pydantic for API contracts
- **Run** type checks before committing
- **Fix** type errors immediately
- **Avoid** `Any` unless justified
- **Document** type ignore comments
- **Test** type safety in CI/CD

Type safety is not optional—it's a requirement for production code quality.
