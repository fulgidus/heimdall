# Training Service Tests

Comprehensive test suite for the training service.

## Structure

```
tests/
├── unit/                           # Unit tests
├── integration/                    # Integration tests
│   └── test_synthetic_pipeline_integration.py
├── test_feature_extractor_basic.py # Feature extraction tests
├── test_iq_generator.py            # IQ generation tests
├── test_performance.py             # Performance benchmarks
├── conftest.py                     # Shared fixtures
└── README.md                       # This file
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Integration tests (may require Docker)
pytest -m integration

# Performance benchmarks
pytest test_performance.py -v -m performance
```

## Test Categories

### Feature Extraction Tests
- `test_feature_extractor_basic.py` - Core feature extraction functionality
- Coverage: Signal detection, SNR calculation, frequency offset, spectral features

### IQ Generation Tests
- `test_iq_generator.py` - Synthetic IQ sample generation
- Coverage: Sample generation, multipath, fading, reproducibility

### Integration Tests
- `test_synthetic_pipeline_integration.py` - Full pipeline from generation to storage
- Coverage: Single/multi-receiver, reproducibility, feature structure

### Performance Tests
- `test_performance.py` - Performance benchmarks with strict targets
- Targets: IQ gen <50ms, extraction <100ms, end-to-end <150ms

## Coverage

Run with coverage report:
```bash
pytest --cov=src --cov-report=html
```

Target: ≥80% coverage for all modules

## See Also

- [Main Testing Guide](../../../docs/TESTING.md)
- [Contributing Guide](../../../CONTRIBUTING.md)
