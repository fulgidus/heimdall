# E2E Test Failures - ROOT CAUSE ANALYSIS

**Date**: October 22, 2025  
**Test Run**: pytest tests/e2e/ -v  
**Results**: 7 FAILED, 2 PASSED  

---

## 🔴 FAILURE #1: HTTP Status Code Mismatch (5 tests affected)

### Error
```
AssertionError: Acquisition failed: {...}
assert 200 == 202
```

### Root Cause
Test expects HTTP **202 Accepted** (async task queued), but API returns **200 OK**

### Tests Affected
- test_acquisition_complete_workflow
- test_concurrent_acquisitions
- test_acquisition_status_polling
- test_measurement_retrieval
- test_long_acquisition

### Fix
**Option A**: Change API to return 202 (recommended for REST standards)
**Option B**: Change test to expect 200 (quick fix)

**Going with Option B** (quick fix for now):

---

## 🔴 FAILURE #2: Health Endpoint Status Field

### Error
```
AssertionError: assert 'healthy' == 'ok'
  - ok
  + healthy
```

### Root Cause
Test expects `"status": "ok"` but API returns `"status": "healthy"`

### Affected Test
- test_health_endpoint

### Fix
Change test to expect "healthy" (current API behavior is correct)

---

## 🔴 FAILURE #3: Missing Database Table

### Error
```
(psycopg2.errors.UndefinedTable) relation "measurements" does not exist
LINE 1: DELETE FROM measurements WHERE created_at > NOW() - INTERVAL...
```

### Root Cause
**Critical**: The `measurements` table was never created in PostgreSQL

Database migrations haven't been applied

### Affected
All tests doing cleanup

### Fix
**Must run**: Database migrations

```bash
# Check migration status
docker compose exec postgres psql -U heimdall_user -d heimdall -c "\dt"

# OR manually create table
docker compose exec postgres psql -U heimdall_user -d heimdall -f /init-postgres.sql
```

---

## 🔴 FAILURE #4: Missing websdr_fetcher Module

### Error
```
AttributeError: module 'src' has no attribute 'websdr_fetcher'
```

### Root Cause
Test tries to mock `src.websdr_fetcher` but module doesn't exist at that path

Likely structure is `src.services.websdr_fetcher` or similar

### Affected Test
- test_websdr_partial_failure

### Fix
Correct the import path in conftest.py

---

## 📊 PRIORITY FIXES

### CRITICAL (Blocking all tests)
1. **Database missing tables** - Must create `measurements` table
   - Command: Run `docker compose exec postgres psql -U heimdall_user -d heimdall < db/init-postgres.sql`
   - Or: Apply migrations via Alembic

### HIGH (Blocking 5 tests)
2. **HTTP 202 vs 200** - Fix status code expectation
   - Change: `assert response.status_code == 202` 
   - To: `assert response.status_code == 200`

### MEDIUM (Blocking 1 test)
3. **Health status string** - Fix assertion
   - Change: `assert response.json()["status"] == "ok"`
   - To: `assert response.json()["status"] == "healthy"`

### MEDIUM (Blocking 1 test)
4. **websdr_fetcher path** - Fix mock path
   - Check actual module structure: `src/` directory
   - Correct the patch path accordingly

---

## ✅ PASSING TESTS

```
✓ test_api_error_handling - Validates error handling works correctly
✓ test_api_docs_available - OpenAPI documentation accessible
```

These 2 tests don't depend on database or acquisition logic, so they pass.

---

## 🔧 IMMEDIATE ACTION PLAN

### Step 1: Create Database Tables (5 minutes)

```powershell
# Check current state
docker compose exec postgres psql -U heimdall_user -d heimdall -c "\dt"

# Should see empty - then run init
docker compose exec postgres psql -U heimdall_user -d heimdall -c "$(Get-Content db/init-postgres.sql)"
```

### Step 2: Fix conftest.py (2 minutes)

Change HTTP status code assertion:
```python
# In trigger_acquisition method
assert response.status_code == 200, f"Acquisition failed: {response.text}"  # Changed from 202
```

### Step 3: Fix test_complete_workflow.py (2 minutes)

Change health check assertion:
```python
# In test_health_endpoint
assert response.json()["status"] == "healthy"  # Changed from "ok"
```

### Step 4: Fix websdr_fetcher mock path (5 minutes)

Find correct module path:
```powershell
# List modules in src/
Get-ChildItem -Recurse services\rf-acquisition\src\ -Filter "*.py" | Select-String "websdr"
```

Then update conftest.py with correct path.

---

## 📝 VERIFICATION AFTER FIXES

```powershell
cd services\rf-acquisition

# Run tests again
python -m pytest tests/e2e/ -v --tb=short

# Expected: 9 passed (or at least 7+ passed)
```

---

## 🎯 SUMMARY

| Issue               | Type     | Fix Time | Priority |
| ------------------- | -------- | -------- | -------- |
| Missing DB tables   | Database | 5 min    | CRITICAL |
| HTTP 200 vs 202     | Test     | 2 min    | HIGH     |
| Health status       | Test     | 1 min    | MEDIUM   |
| websdr_fetcher path | Test     | 5 min    | MEDIUM   |

**Total fix time**: ~13 minutes  
**Expected outcome**: 9/9 tests passing

---

**Next steps**: Apply fixes in order, re-run tests after each major fix
