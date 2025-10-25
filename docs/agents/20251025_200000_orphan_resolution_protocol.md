# Orphan Documentation Resolution Protocol

**Created**: 2025-10-25 20:00:00 UTC  
**Purpose**: Standardized process for identifying and resolving orphaned documentation files  
**Related**: AGENTS.md - Documentation Standards section

---

## Overview

This protocol defines the process for detecting, evaluating, and resolving orphaned documentation files in the Heimdall project. Per AGENTS.md requirements: **"All documentation files MUST be discoverable and contextual. Orphaned files (not linked from anywhere) are NOT allowed."**

## What is an Orphaned File?

An **orphaned file** is a markdown document in `docs/` that is not reachable via links from any documentation entry point:
- `AGENTS.md` (root)
- `docs/index.md` (documentation portal)
- Other documentation files

### Why Orphans are Problematic

1. **Discoverability**: Users and contributors cannot find the information
2. **Maintenance**: Files may become outdated without being noticed
3. **Quality**: Suggests incomplete documentation structure
4. **Navigation**: Breaks the documentation hierarchy

---

## Automated Detection

### Running Orphan Detection

```bash
# Simple check
python scripts/audit_documentation.py

# Generate detailed reports
python scripts/audit_documentation.py --format=both --verbose

# Or use Makefile
make audit-docs
```

### Understanding Output

The audit script will produce:
- **audit_report.json**: Machine-readable format for CI/CD
- **audit_report.md**: Human-readable report with suggestions
- Console output with statistics

Example output:
```
📊 AUDIT STATISTICS:
   Total markdown files in docs/: 228
   Reachable from entry points: 203
   Orphaned files: 26
   Broken links: 100
```

### Exit Codes

- `0`: All files linked, no orphans
- `1`: Orphaned files detected (CI will fail)

---

## Resolution Decision Tree

When an orphaned file is identified, follow this decision process:

### Step 1: Review File Content

```bash
# View the orphaned file
cat docs/agents/ORPHANED_FILE.md

# Check file size and last modified
ls -lh docs/agents/ORPHANED_FILE.md
```

Ask yourself:
- Is the content still relevant?
- Is it duplicate information?
- Is it temporary/obsolete?

### Step 2: Decision - Keep or Delete?

#### KEEP if:
- ✅ Content is current and relevant
- ✅ Provides unique information
- ✅ Part of ongoing work/phase
- ✅ Referenced in recent work sessions

#### DELETE/ARCHIVE if:
- ❌ Content is obsolete
- ❌ Duplicate of other files
- ❌ Temporary tracking (completed work)
- ❌ Not updated in 6+ months and irrelevant

### Step 3A: If KEEPING - Add Appropriate Links

Determine the best linking location based on file type:

| File Type | Link From | Example |
|-----------|-----------|---------|
| Phase tracking | AGENTS.md → Phase N section | `[Phase 3 Status](docs/agents/20251022_080000_phase3_status.md)` |
| Implementation guide | Phase index file | `[Implementation Guide](20251022_080000_implementation_guide.md)` |
| Session report | Session tracking doc | `[Session Summary](20251022_080000_session_summary.md)` |
| API/Architecture | docs/index.md → Additional Resources | `[API Details](agents/api_implementation.md)` |
| Troubleshooting | docs/index.md → Troubleshooting | `[Debug Guide](agents/debug_guide.md)` |
| Testing | Developer guide or phase doc | `[E2E Testing](agents/e2e_testing.md)` |

#### Adding Links - Best Practices

1. **Use relative paths**:
   ```markdown
   # From AGENTS.md to docs/agents/file.md
   [Description](docs/agents/20251022_080000_file.md)
   
   # From docs/index.md to docs/agents/file.md
   [Description](agents/20251022_080000_file.md)
   
   # From docs/agents/index.md to docs/agents/file.md
   [Description](20251022_080000_file.md)
   ```

2. **Provide context** in link text:
   ```markdown
   # Good
   [Phase 3 Complete Summary](docs/agents/20251022_080000_phase3_complete_summary.md)
   
   # Bad (too generic)
   [Click here](docs/agents/20251022_080000_phase3_complete_summary.md)
   ```

3. **Group related links**:
   ```markdown
   ### Phase 3 Documentation
   - [Phase 3 Index](docs/agents/20251022_080000_phase3_index.md)
   - [Phase 3 Status](docs/agents/20251022_080000_phase3_status.md)
   - [Phase 3 Complete](docs/agents/20251022_080000_phase3_complete_summary.md)
   ```

### Step 3B: If DELETING - Proper Cleanup

```bash
# Move to archive (don't delete yet)
mkdir -p docs/archive/$(date +%Y%m)
mv docs/agents/ORPHANED_FILE.md docs/archive/$(date +%Y%m)/

# Or delete if truly obsolete
rm docs/agents/ORPHANED_FILE.md

# Commit the change
git add -A
git commit -m "docs: remove obsolete orphaned file ORPHANED_FILE.md"
```

---

## Example Walkthrough

### Scenario: Orphaned Phase 5 Session Report

```bash
# 1. Detect orphan
$ python scripts/audit_documentation.py
Orphaned files: 1
   • agents/20251022_080000_phase5_session_notes.md
```

### Step-by-Step Resolution

#### 1. Review Content
```bash
$ cat docs/agents/20251022_080000_phase5_session_notes.md
# Phase 5 Session Notes
Work done on 2025-10-22...
- Implemented MLflow tracking
- Created ONNX export
- Tests passing
```

**Decision**: Content is relevant → KEEP

#### 2. Identify Linking Location

This is a Phase 5 session report, so it should be linked from:
- AGENTS.md Phase 5 tracking section
- OR Phase 5 index document

#### 3. Add Link to AGENTS.md

Edit `AGENTS.md`:
```markdown
**📋 Tracking**:
- [Phase 5 Document Index](docs/agents/20251022_080000_phase5_document_index.md)
- [Phase 5 Session Notes](docs/agents/20251022_080000_phase5_session_notes.md) ← ADD THIS
- [Phase 5 Complete](docs/agents/20251022_080000_phase5_complete_final.md)
```

#### 4. Verify Fix

```bash
$ python scripts/audit_documentation.py
Orphaned files: 0
✅ AUDIT PASSED: No orphaned files found!
```

#### 5. Commit Changes

```bash
git add AGENTS.md
git commit -m "docs: link Phase 5 session notes from AGENTS.md"
```

---

## Creating Phase Index Files

For phases with many documents, create index files:

### Example: `docs/agents/PHASE5_INDEX.md`

```markdown
# Phase 5: Training Pipeline - Document Index

**Status**: ✅ COMPLETE  
**Main Reference**: [AGENTS.md Phase 5](../../AGENTS.md#-phase-5-training-pipeline)

## Quick Navigation

### Implementation Documents
- [Phase 5 Start Here](20251022_080000_phase5_start_here.md)
- [Architecture Design](20251022_080000_phase5_architecture_upgrade.md)
- [MLflow Implementation](20251022_080000_phase5_t5_6_mlflow_complete.md)

### Session Reports
- [Session 1 Summary](20251022_080000_phase5_session_1_summary.md)
- [Session 2 Summary](20251022_080000_phase5_session_2_summary.md)

### Completion Documents
- [Final Report](20251022_080000_phase5_complete_final.md)
- [Handoff to Phase 6](20251022_080000_phase5_handoff.md)
```

Then link the index from AGENTS.md:
```markdown
**📋 Tracking**:
- [Phase 5 Index](docs/agents/PHASE5_INDEX.md) - All Phase 5 documentation
```

---

## Prevention Best Practices

### When Creating New Documentation

1. **Always add link immediately** after creating file
2. **Update phase tracking** in AGENTS.md
3. **Use naming convention**: `YYYYMMDD_HHMMSS_description.md`
4. **Run audit before committing**:
   ```bash
   make audit-docs
   git add -A
   git commit -m "docs: add new documentation with proper linking"
   ```

### When Reorganizing Documentation

1. **Run audit before** reorganization
2. **Update all links** that point to moved files
3. **Run audit after** to verify
4. **Test in CI** before merging

### During Code Reviews

Reviewers should check:
- [ ] New docs are linked from entry points
- [ ] Links use correct relative paths
- [ ] `make audit-docs` passes
- [ ] CI workflow passes

---

## CI/CD Integration

### Automated Checks

The `.github/workflows/doc-audit.yml` workflow:
- Runs on PRs touching `docs/` or `AGENTS.md`
- Fails if orphans or broken links detected
- Posts comment on PR with findings
- Uploads detailed reports as artifacts

### Local Pre-Commit Check

Add to your workflow:
```bash
# Before committing docs changes
make check-docs

# If fails, review and fix
make audit-docs
# Fix issues...

# Verify
make check-docs
git commit -m "docs: fix orphaned files"
```

---

## Troubleshooting

### False Positives

**Q: File is linked but still shows as orphaned?**

A: Check that:
1. Link path is correct (relative from source file)
2. Link uses `.md` extension
3. File is actually in `docs/` directory
4. No typos in filename

### Common Mistakes

1. **Absolute paths**: Use relative paths, not `/docs/agents/file.md`
2. **Missing extension**: Always use `.md` in links
3. **Wrong directory**: Ensure file is in `docs/` tree
4. **Circular references**: Don't rely only on sibling files linking to each other

### Getting Help

If unsure about resolution:
1. Post in PR comments
2. Ask project maintainer (fulgidus)
3. Review similar phase documentation for patterns

---

## Summary Checklist

When resolving orphaned files:

- [ ] Run `make audit-docs` to identify orphans
- [ ] Review each orphaned file's content
- [ ] Decide: Keep and link, or archive/delete
- [ ] If keeping: Add link from appropriate parent document
- [ ] Verify fix with `make audit-docs`
- [ ] Commit changes with descriptive message
- [ ] Ensure CI passes before merging

---

## Related Documentation

- [AGENTS.md](../../AGENTS.md) - Project phase management guide
- [docs/index.md](../index.md) - Documentation portal
- [Documentation Standards](../../AGENTS.md#documentation-standards) - Naming conventions and organization

## Tools

- `scripts/audit_documentation.py` - Orphan detection
- `scripts/generate_doc_index.py` - Link validation
- `.github/workflows/doc-audit.yml` - CI automation
- `Makefile` - Convenience commands
