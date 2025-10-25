# Project Standards

This document defines the coding and project organization standards for the Heimdall project.

## Core Directives

- Commit messages must be in English
- All documentation must be written in English
- Use a professional and concise tone
- Provide detailed explanations when requested
- Always keep technical documentation in `docs/index.md` updated
- Keep `AGENTS.md` updated with relevant changes
- When modifying configuration or setup files, ensure instructions in `README.md` reflect these changes
- When adding new dependencies or tools, update the "Technology Stack" section in `README.md`
- Keep `CHANGELOG.md` updated with a summary of changes made in each work session

## Terminal Command Rules

**MANDATORY - No exceptions:**

1. **NO arbitrary sleep commands**: Do NOT insert `sleep` statements anywhere in terminal commands unless explicitly required by the actual task logic (e.g., waiting for a specific asynchronous operation). Never use sleep for "safety" or "to be sure" - it wastes time and shows lazy thinking.
   - BANNED: `sleep 10`, `sleep 15`, `sleep 30`, etc.

**The rule applies to ALL agents in this project, without exception.**

## Frontend Package Manager Preference

**MANDATORY - Frontend projects use pnpm exclusively:**

- Use `pnpm install`, `pnpm add`, `pnpm script-name` for all frontend package management
- Do NOT use `npm` for frontend tasks - use `pnpm` instead
- pnpm is faster, uses less disk space, and provides better dependency management
- Applies to: `frontend/` directory and all frontend-related workflows
- Example: `pnpm install` (instead of `npm install`), `pnpm dev` (instead of `npm run dev`)
- This rule applies to ALL agents working on frontend code, without exception

## Script Organization

All utility scripts must be organized in the `/scripts/` directory:

### Script Location

- **`/scripts/` directory** - All utility, test, and automation scripts:
  - Python scripts (`.py`)
  - Shell scripts (`.sh`)
  - PowerShell scripts (`.ps1`)
  - Batch scripts (`.bat`)

### Root Directory Exceptions

- `conftest.py` - Pytest configuration (must remain in root)
- Any scripts required by CI/CD that must be in root

### Script Naming Conventions

- Use descriptive names: `test_health_endpoint.py`, `load_test.py`
- Use underscores for Python scripts: `health_check.py`
- Use hyphens for shell scripts: `health-check.sh`
- Include action in name: `generate_`, `test_`, `check_`, `monitor_`

### Script Categories in `/scripts/`

- **Testing scripts**: `test_*.py`, `test_*.sh`
- **Health checks**: `health-check.*`, `*_health_*.py`
- **Load testing**: `load_test*.py`, `performance_benchmark.py`
- **Setup/deployment**: `dev-setup.ps1`, `start-*.ps1`
- **Utilities**: `check_*.py`, `inspect_*.py`, `monitor_*.py`
- **Generation**: `generate_*.py`, `create_*.py`
- **Documentation**: `reorganize_docs.py`

## Code Quality Standards

### Python

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Maintain >80% test coverage
- Use Black for code formatting
- Use Ruff for linting
- Document all public APIs

### TypeScript/React

- Follow ESLint configuration
- Use TypeScript strict mode
- Maintain >80% test coverage
- Use Prettier for code formatting
- Document all public components and utilities

## Testing Standards

- Unit tests required for all business logic
- Integration tests for service-to-service communication
- E2E tests for critical user workflows
- Performance tests for latency-sensitive operations
- Security tests for authentication and authorization

## Version Control

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Follow conventional commits specification
- Create feature branches from `develop`
- Require PR reviews before merging
- Keep commit history clean and meaningful

## Deployment

- All services must have health check endpoints
- Use structured logging (JSON format)
- Implement graceful shutdown handlers
- Define resource limits for all containers
- Use secrets management for sensitive data

## Documentation Requirements

- All public APIs must have OpenAPI/Swagger documentation
- All configuration options must be documented
- All environment variables must be documented in `.env.example`
- All breaking changes must be documented in CHANGELOG.md
- All architectural decisions must be documented in ADRs (if applicable)
