# Documentation Standards

This document defines the documentation standards for the Heimdall project.

## Language Requirements

- All documentation MUST be in English
- Exception: Italian translation may exist in separate i18n files
- All technical documentation, guides, and API references must be in English only

## File Location and Naming

### Root Directory
Only these essential files:
- `README.md` - Main project README
- `AGENTS.md` - Project phase management guide
- `CHANGELOG.md` - Version history following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- `WEBSDRS.md` - WebSDR receiver configuration
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - Project license

### `/docs/` Directory
All public-facing documentation:
- `index.md` - Main documentation portal (must be in English)
- API references, architecture guides, tutorials, etc.
- All files in English only

### `/docs/agents/` Directory
Internal tracking and progress documents:
- Format: `YYYYMMDD_HHmmss_description.md`
- Example: `20251023_153000_phase6_completion_summary.md`
- Use lowercase with underscores for description
- All timestamps use 24-hour format

### `/docs/standards/` Directory
Project standards and conventions:
- `DOCUMENTATION_STANDARDS.md` (this file)
- `PROJECT_STANDARDS.md` - Coding and project standards
- `KNOWLEDGE_BASE.md` - Critical knowledge preservation

## Markdown File Standards

- Use proper markdown formatting (headers, lists, code blocks)
- Include table of contents for documents >500 lines
- Link to related documents using relative paths
- Follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format for CHANGELOG.md
- Use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for version numbers

## README.md Maintenance Guidelines

### Cross-Platform Compatibility
- Always use cross-platform commands
- Use `cp` instead of Windows-specific `copy`
- Verify commands work on Linux, macOS, and Windows (Git Bash)
- Add clarifying comments for each setup step

### Command Accuracy
- Test `cp .env.example .env` (creates configuration file)
- Test `docker-compose up -d` (starts all infrastructure)
- Test `make health-check` (verifies services are running)
- Ensure script paths are correct (e.g., `scripts/health-check.py`)

### Documentation Links
- Use correct paths: `docs/agents/YYYYMMDD_HHmmss_filename.md`
- Update links when files are renamed or moved
- Verify links point to existing files, not orphaned references

### Project Status Updates
- Update phase status headers to show current phase
- Mark completed phases with ✅ COMPLETE
- Add new phases as they are completed
- Keep phase descriptions accurate

### Architecture Accuracy
- Verify service names and descriptions
- Check technology stack is current
- Validate performance metrics reflect latest benchmarks
- Confirm deployment instructions match actual setup

## When to Update Documentation

- After completing a new phase
- When commands or setup procedures change
- When documentation is reorganized or files moved
- When performance benchmarks are updated
- During major architecture changes

## Preventing Orphaned Files

All documentation files MUST be discoverable and contextual. Orphaned files (not linked from anywhere) are NOT allowed.

### When Creating Tracking Documents in `/docs/agents/`

1. **Always add context**: Every new tracking document must include:
   - Clear title indicating the phase/task it relates to
   - Date and session information
   - Links to related documents (previous session, related phases)
   - Summary of what the document contains

2. **Link from appropriate locations**:
   - Add link in phase-specific index files (e.g., `PHASE6_INDEX.md`)
   - Reference from `AGENTS.md` in the relevant phase section
   - Link from related tracking documents
   - Update progress dashboards or status files

3. **Provide navigation aids**:
   - Include "Related Documents" section at end of file
   - Add "Previous Session" / "Next Session" links when applicable
   - Reference from completion summaries and handoff documents

4. **Integration requirements**:
   - New phase documents must be linked from phase descriptions in `AGENTS.md`
   - Session summaries must be linked from phase tracking documents
   - Completion reports must be referenced in phase status updates
   - All significant tracking files should appear in navigation chains

### Example of Proper Context and Linking

```markdown
# Phase 6 Session 2 Progress Report

**Related Documents:**
- [Phase 6 Start Guide](./PHASE6_START_HERE.md)
- [Phase 6 Session 1 Report](./PHASE6_SESSION1_FINAL_REPORT.md)
- [Phase 6 Index](./PHASE6_INDEX.md)
- [Previous: Phase 5 Completion](./PHASE5_COMPLETE_SESSION_REPORT.md)

**Session Info:** 2025-10-23 | Agent: copilot | Status: In Progress
```

## Orphan Prevention Tools

- Use `scripts/reorganize_docs.py --find-orphans` to identify unreachable files
- Review orphan reports regularly
- Before PR merge, ensure all new files are properly linked
- Update documentation index when adding significant new files

## Enforcement

These standards are mandatory for all contributions:
- Automated scripts enforce file naming conventions
- CI/CD checks verify documentation is in English
- Pull requests must follow these standards
- Use `scripts/reorganize_docs.py` to maintain compliance
