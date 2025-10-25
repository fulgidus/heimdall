# Contributing Guidelines

## Welcome!

Thank you for your interest in contributing to Heimdall! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites
- Familiarity with [Git](https://git-scm.com/)
- Understanding of Python and/or JavaScript/React
- Development environment set up per [Installation Guide](./installation.md)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Create a feature branch
git checkout -b feature/your-feature-name

# Set up development tools
make dev-setup
```

## Contribution Types

### 1. Bug Reports

Report bugs by opening a GitHub Issue with:
- Clear, descriptive title
- Detailed description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### 2. Feature Requests

Suggest features by creating a GitHub Issue with:
- Clear description of the feature
- Use cases and motivation
- Proposed implementation (if applicable)
- Potential alternatives

### 3. Code Contributions

Make code changes through pull requests:

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make your changes
# Test thoroughly
git add .
git commit -m "Add my feature"
git push origin feature/my-feature

# Create pull request on GitHub
```

### 4. Documentation Improvements

Improve documentation by:
- Creating pull requests with changes
- Fixing typos and clarity issues
- Adding examples and use cases
- Improving architecture diagrams

## Code Standards

### Python Style

```python
# Follow PEP 8
# Use type hints
def process_signal(data: np.ndarray) -> Dict[str, float]:
    """Process radio signal data.
    
    Args:
        data: IQ samples from WebSDR
        
    Returns:
        Dictionary with processing results
    """
    # Implementation
    pass
```

### JavaScript/React Style

```javascript
// Use TypeScript
// Follow ESLint configuration
interface LocalizationResult {
  latitude: number;
  longitude: number;
  uncertainty_m: number;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  return <div>{/* Component */}</div>;
};
```

### Commit Messages

```
type(scope): subject

Body explaining changes in more detail.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

## Quality Standards

All code contributions must meet strict quality standards enforced by automated CI/CD checks:

### Code Quality Requirements

#### Minimum Coverage Threshold
- **Backend (Python)**: 80% code coverage minimum
- **Frontend (TypeScript)**: 80% code coverage minimum
- Coverage must not drop below threshold when adding new features

#### Python Quality Standards
- **Black**: All Python code must be formatted with Black (line length: 100)
- **Ruff**: Zero linting errors allowed (warnings acceptable with justification)
- **mypy**: Type checking must pass with project configuration
- **Testing**: All pytest tests must pass

#### TypeScript Quality Standards
- **ESLint**: Zero errors allowed (configured rules must pass)
- **TypeScript**: Strict mode enabled, no compilation errors
- **Prettier**: Code must be formatted according to project style
- **Testing**: All Vitest tests must pass

### Pre-commit Hooks Setup

Install pre-commit hooks to catch issues before pushing:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Pre-commit hooks will automatically:
- Format code with Black and Prettier
- Run Ruff linting with auto-fix
- Check for trailing whitespace
- Validate YAML, JSON, and TOML files
- Check for large files and merge conflicts

### Local Testing Before Push

**Always run local CI checks before pushing:**

```bash
# Run all quality checks (Python + TypeScript)
make quality-check

# Run security scans
make security-scan

# Run complete local CI simulation
make ci-local
```

Individual quality checks:

```bash
# Python quality checks
make lint-python           # Run Black and Ruff
make format-python         # Auto-format code
make type-check-python     # Run mypy
make test-python          # Run pytest with coverage

# TypeScript quality checks
make lint-typescript       # Run ESLint
make format-typescript     # Run Prettier
make type-check-typescript # Run tsc --noEmit
make test-typescript      # Run Vitest with coverage
```

### Pull Request Requirements

All PRs must meet these requirements to be merged:

1. ✅ **All CI checks pass**
   - Python quality workflow passes
   - TypeScript quality workflow passes
   - Security scan passes (or findings explained)
   
2. ✅ **Code coverage maintained**
   - Coverage reports show ≥80% coverage
   - No significant coverage drops
   
3. ✅ **No linting/formatting errors**
   - Black formatting check passes
   - Ruff linting passes
   - ESLint passes
   - Prettier check passes
   
4. ✅ **Type checking passes**
   - mypy reports no errors
   - TypeScript compilation successful
   
5. ✅ **All tests pass**
   - pytest suite passes
   - Vitest suite passes
   - No test skips without justification

6. ✅ **Code review approved**
   - At least one maintainer approval
   - All comments addressed or discussed

### Exception Process

In rare cases, quality checks may be skipped:

**When to skip:**
- Documentation-only changes (may skip some tests)
- Emergency hotfixes (with maintainer approval)
- Known false positives in security scans

**How to skip:**
1. Add label `skip-quality-checks` to PR (requires maintainer)
2. For security findings: Comment `@github-actions ignore-security-findings`
3. Document reason in PR description
4. Commit to fix in follow-up PR if appropriate

**Note:** Coverage threshold cannot be skipped. If coverage drops, either:
- Add tests to maintain coverage
- Justify why coverage drop is acceptable (rare)

### Security Standards

- **No hardcoded secrets**: Use environment variables
- **Dependency scanning**: All dependencies scanned for vulnerabilities
- **Container scanning**: All Docker images scanned with Trivy
- **Code scanning**: Bandit scans Python for security issues

Security findings are reported in PR comments. Address all HIGH and CRITICAL findings before merge.

### Pull Request Process

1. **Create descriptive PR**
   - Link related issues
   - Describe changes clearly
   - Highlight testing performed

2. **Code review**
   - Address reviewer feedback
   - Push updated commits
   - Maintain clean history

3. **Continuous Integration**
   - Ensure all backend checks pass (pytest, black, ruff)
   - Ensure all frontend checks pass (vitest, eslint, TypeScript)
   - Maintain test coverage >80%
   - No linting errors

4. **Merge**
   - Squash commits if appropriate
   - Update CHANGELOG
   - Close associated issues

## Testing Guidelines

### Running Backend Tests

```bash
# Run all backend tests
make test

# Run with coverage
make test-coverage

# Run specific test
pytest tests/test_signal_processor.py -v
```

### Running Frontend Tests

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run all frontend tests
npm run test

# Run tests once (CI mode)
npm run test:run

# Run linting
npm run lint

# Build frontend
npm run build
```

### Writing Tests

```python
import pytest
from heimdall.signal_processor import process_signal

@pytest.fixture
def sample_iq_data():
    """Fixture providing sample IQ data."""
    return np.random.randn(48000, 2)

def test_signal_processing(sample_iq_data):
    """Test signal processing pipeline."""
    result = process_signal(sample_iq_data)
    assert result is not None
    assert "magnitude" in result
    assert len(result["magnitude"]) > 0
```

## Documentation Standards

### Code Documentation

```python
def triangulate_location(
    signals: List[SignalMeasurement],
    stations: List[WebSDRStation]
) -> LocalizationResult:
    """
    Triangulate transmission source location from signal measurements.
    
    Uses multilateration with Least Squares estimation to compute
    the most likely transmission source location given signal
    measurements from multiple WebSDR stations.
    
    Args:
        signals: List of signal measurements from WebSDR receivers
        stations: List of WebSDR station locations and calibration
        
    Returns:
        LocalizationResult with coordinates and uncertainty
        
    Raises:
        ValueError: If insufficient signals for triangulation
        GeometryError: If station geometry insufficient
        
    Examples:
        >>> stations = [stn1, stn2, stn3]
        >>> signals = [measurement1, measurement2, measurement3]
        >>> result = triangulate_location(signals, stations)
        >>> print(f"Location: {result.location}")
    """
```

### Markdown Documentation

```markdown
## Clear Heading

Paragraph explaining the concept with links to related docs.

### Subheading

- Bullet point 1
- Bullet point 2

```code example```

**See also**: [Related Doc](./related.md)
```

## Community Guidelines

### Be Respectful

- Treat all community members with respect
- Assume good intent
- Provide constructive feedback
- Avoid dismissive language

### Be Helpful

- Help newer contributors
- Share knowledge freely
- Point to relevant documentation
- Be patient with learning questions

### Report Issues Appropriately

- For security issues: Email security@heimdall.org
- For bugs: Use GitHub Issues
- For discussions: Use GitHub Discussions
- For questions: Check FAQ first

## Recognition

Contributors are recognized in:
- [Contributors list](../docs/acknowledgements.md)
- Release notes for significant contributions
- Project README for major features

## License

By contributing, you agree that your contributions will be licensed under the [Creative Commons Non-Commercial License](../LICENSE).

---

**Questions?** Open an issue or check [FAQ](./faqs.md)
