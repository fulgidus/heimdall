# CI/CD Quick Reference Card

## 🚀 Quick Start

```bash
# One-time setup
make install-pre-commit

# Before each push
make ci-local
```

## 📋 Common Commands

### Python Quality
```bash
make lint-python            # Check formatting & linting
make format-python          # Auto-fix formatting
make type-check-python      # Run mypy
make test-python           # Run tests with coverage
```

### TypeScript Quality
```bash
make lint-typescript        # Run ESLint
make format-typescript      # Run Prettier
make type-check-typescript  # TypeScript compiler
make test-typescript       # Run Vitest with coverage
```

### Combined
```bash
make quality-check         # All quality checks
make security-scan         # All security scans
make ci-local             # Full CI simulation
```

## ✅ What Gets Checked

### Python (`.github/workflows/python-quality.yml`)
- ✅ Black formatting (line-length: 100)
- ✅ Ruff linting (imports, types, bugs)
- ✅ mypy type checking (strict mode)
- ✅ pytest coverage (80% minimum)

### TypeScript (`.github/workflows/typescript-quality.yml`)
- ✅ ESLint (code quality)
- ✅ TypeScript compilation (strict mode)
- ✅ Prettier formatting
- ✅ Vitest coverage (80% minimum)
- ✅ Bundle size tracking

### Security (`.github/workflows/security-scan.yml`)
- ✅ Bandit (Python security)
- ✅ Safety (dependency vulnerabilities)
- ✅ Trivy (container scanning)
- ✅ Dependabot (GitHub alerts)

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python tools config |
| `frontend/.eslintrc.json` | ESLint rules |
| `frontend/prettier.config.js` | Prettier formatting |
| `.pre-commit-config.yaml` | Pre-commit hooks |

## 🎯 Coverage Requirements

- Minimum: **80%**
- Scope: All Python & TypeScript code
- Reports: Auto-uploaded to Codecov
- Threshold: Enforced in CI

## 🚫 Skipping Checks

**Only when necessary** (requires maintainer approval):

1. Add label: `skip-quality-checks`
2. For security: Comment `@github-actions ignore-security-findings`
3. Document reason in PR description

## 📊 PR Comment Format

Workflows post automatic comments on PRs:

```
🐍 Python Code Quality Results

✅ Black formatting: PASS
✅ Ruff linting: PASS (0 errors)
❌ mypy type check: FAIL (3 errors)
✅ Coverage: 82% (meets threshold)

📋 View detailed report
⚠️ Action required: Please address the failing checks above.
```

## 🔍 Local Debugging

```bash
# Check specific service
cd services/rf-acquisition
pytest tests/ -v

# Check formatting only
black services/ --check

# Auto-fix linting
ruff check services/ --fix

# Type check specific file
mypy services/inference/src/main.py
```

## 📚 Documentation

- **Full Guide**: `docs/agents/20251025_201436_cicd_implementation_summary.md`
- **Contributing**: `docs/contributing.md` (Quality Standards section)
- **Workflows**: `.github/workflows/`

## 🆘 Troubleshooting

**Pre-commit hooks not running?**
```bash
make install-pre-commit
pre-commit run --all-files  # Test
```

**Coverage too low?**
```bash
# Generate coverage report
pytest services/ --cov=services --cov-report=html
open coverage_reports/html/index.html  # View report
```

**Type errors?**
```bash
# Check with relaxed config first
mypy services/ --config-file=pyproject.toml --no-strict-optional
```

**Prettier formatting issues?**
```bash
cd frontend
pnpm run format  # Auto-fix
pnpm run format:check  # Verify
```

---

**Questions?** Check `docs/contributing.md` or open an issue.
