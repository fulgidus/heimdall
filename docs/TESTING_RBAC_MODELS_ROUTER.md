# Testing Guide: Models Router with RBAC

**Date**: 2025-11-08  
**Status**: Ready for Testing  
**Phase**: Phase 7 - RBAC Implementation - Task 12

---

## Overview

This document provides a comprehensive testing guide for the ML Models router with RBAC (Role-Based Access Control) functionality. The implementation has been completed and is ready for integration testing.

## Files Modified (14 files staged)

### New Files Created:
1. `db/migrations/04-add-rbac-schema.sql` - RBAC schema (users, constellations, sources, models, shares)
2. `db/migrations/05-migrate-existing-data.sql` - Data migration script
3. `db/migrations/README.md` - Migration instructions
4. `docs/RBAC_IMPLEMENTATION.md` - Complete RBAC documentation
5. `services/backend/src/models/constellation.py` - Constellation model
6. `services/backend/src/models/shares.py` - Share models (ConstellationShare, SourceShare, ModelShare)
7. `services/backend/src/routers/constellations.py` - Constellations API (797 lines)
8. `services/backend/src/routers/models.py` - **Models API (797 lines) - PRIMARY FOCUS**
9. `services/backend/src/routers/sources.py` - Sources API (1077 lines)
10. `services/backend/tests/integration/test_models_rbac.py` - **Comprehensive test suite**

### Modified Files:
1. `services/backend/src/main.py` - Registered models router
2. `services/backend/src/models/session.py` - Added owner_id field
3. `services/backend/src/routers/sessions.py` - Added RBAC checks
4. `services/common/auth/__init__.py` - Export RBAC functions
5. `services/common/auth/rbac.py` - **REWRITTEN (668 lines) for asyncpg compatibility**

---

## Critical Fix: asyncpg Compatibility

### Problem Discovered:
The original `rbac.py` was written using **SQLAlchemy ORM** syntax:
- Type hints: `AsyncSession` from SQLAlchemy
- Queries: `select()`, `and_()`, `result.scalar_one_or_none()`
- Would **fail at runtime** when used with asyncpg connection pools

### Solution Implemented:
Completely rewrote `services/common/auth/rbac.py`:
- Changed type hints: `AsyncSession` → `DbConnection` (asyncpg.Connection)
- Converted all queries to **raw SQL** with parameterized placeholders (`$1`, `$2`, etc.)
- Changed query execution: `fetchval()`, `fetchrow()`, `fetch()` instead of SQLAlchemy methods
- Removed all SQLAlchemy imports
- **Result**: 100% compatible with asyncpg connection pools used by all routers

### Functions Rewritten (asyncpg-compatible):
```python
# Constellation permissions
- can_view_constellation(conn, user_id, constellation_id, is_admin)
- can_edit_constellation(conn, user_id, constellation_id, is_admin)
- can_delete_constellation(conn, user_id, constellation_id, is_admin)
- get_user_constellations(conn, user_id, is_admin)

# Source permissions
- can_view_source(conn, user_id, source_id, is_admin)
- can_edit_source(conn, user_id, source_id, is_admin)
- can_delete_source(conn, user_id, source_id, is_admin)
- get_user_sources(conn, user_id, is_admin)

# Model permissions
- can_view_model(conn, user_id, model_id, is_admin)
- can_edit_model(conn, user_id, model_id, is_admin)
- can_delete_model(conn, user_id, model_id, is_admin)
- get_user_models(conn, user_id, is_admin)

# Helper functions
- is_constellation_owner(conn, constellation_id, user_id)
- is_source_owner(conn, source_id, user_id)
- is_model_owner(conn, model_id, user_id)
- get_constellation_permission(conn, constellation_id, user_id)
- get_source_permission(conn, source_id, user_id)
- get_model_permission(conn, model_id, user_id)
```

---

## Models Router API Endpoints

### Base URL: `/api/v1/models`

### 1. **GET /api/v1/models** - List Models
**Auth**: Required (any authenticated user)  
**Returns**: List of models user can access (owned + shared)

**Response Fields**:
- `id`: Model UUID
- `model_name`: Human-readable name
- `version`: Version number
- `model_type`: Type (e.g., "localization_cnn")
- `accuracy_meters`: Localization accuracy
- `is_active`: Whether model is active
- `is_production`: Whether model is in production
- `is_owner`: Boolean (true if user is owner)
- `permission`: String ("read", "edit", or null for owners)

**Permission Logic**:
- Admins: See all models
- Users: See owned models + models shared with them

---

### 2. **GET /api/v1/models/{model_id}** - View Model Details
**Auth**: Required  
**Permission**: View permission required

**Response**: Full model metadata including:
- Training metrics (accuracy, loss, epoch)
- MLflow tracking info (run_id, experiment_id)
- Model locations (ONNX, PyTorch)
- Hyperparameters (dict)
- Ownership info (`is_owner`, `permission`)

**Permission Logic**:
- Admins: Always succeed
- Owners: Full access
- Shared "read" or "edit": Can view
- Others: **403 Forbidden**

---

### 3. **PATCH /api/v1/models/{model_id}** - Update Model
**Auth**: Required  
**Permission**: Edit permission required

**Request Body**:
```json
{
  "model_name": "UpdatedModelName"
}
```

**Permission Logic**:
- Admins: Always succeed
- Owners: Can update
- Shared "edit": Can update
- Shared "read": **403 Forbidden**
- Others: **403 Forbidden**

---

### 4. **DELETE /api/v1/models/{model_id}** - Delete Model
**Auth**: Required  
**Permission**: Owner or admin only

**Permission Logic**:
- Admins: Can delete
- Owners: Can delete
- Shared "edit": **403 Forbidden** (cannot delete, only modify)
- Shared "read": **403 Forbidden**
- Others: **403 Forbidden**

**Note**: Deletion is restricted to owners/admins to prevent accidental data loss.

---

### 5. **POST /api/v1/models/{model_id}/deploy** - Deploy Model
**Auth**: Required  
**Permission**: Edit permission required

**Request Body**:
```json
{
  "is_active": true,
  "is_production": false
}
```

**Permission Logic**:
- Admins: Can deploy
- Owners: Can deploy
- Shared "edit": **Can deploy** (allows operators to activate models)
- Shared "read": **403 Forbidden**
- Others: **403 Forbidden**

**Note**: Shared "edit" users can deploy models to allow collaborative workflows.

---

### 6. **GET /api/v1/models/{model_id}/shares** - List Shares
**Auth**: Required  
**Permission**: Owner or admin only

**Response**:
```json
{
  "shares": [
    {
      "id": "uuid",
      "model_id": "uuid",
      "user_id": "keycloak-user-id",
      "permission": "read",
      "shared_by": "owner-user-id",
      "shared_at": "2025-11-08T12:00:00Z"
    }
  ]
}
```

**Permission Logic**:
- Admins: Can list all shares
- Owners: Can list all shares
- Shared users: **403 Forbidden** (privacy: cannot see who else has access)

---

### 7. **POST /api/v1/models/{model_id}/shares** - Create Share
**Auth**: Required  
**Permission**: Owner or admin only

**Request Body**:
```json
{
  "user_id": "keycloak-user-id",
  "permission": "read"  // or "edit"
}
```

**Permission Logic**:
- Admins: Can create shares
- Owners: Can create shares
- Shared users: **403 Forbidden** (cannot re-share)

**Validation**:
- `permission` must be "read" or "edit"
- `user_id` must be a valid Keycloak user ID
- Duplicate shares return **409 Conflict**

---

### 8. **PUT /api/v1/models/{model_id}/shares/{share_id}** - Update Share
**Auth**: Required  
**Permission**: Owner or admin only

**Request Body**:
```json
{
  "permission": "edit"  // upgrade from "read" to "edit"
}
```

**Permission Logic**: Same as create share

---

### 9. **DELETE /api/v1/models/{model_id}/shares/{share_id}** - Delete Share
**Auth**: Required  
**Permission**: Owner or admin only

**Permission Logic**: Same as create share

---

## Permission Matrix

| Action          | Admin | Owner | Shared "edit" | Shared "read" | Unauthorized |
|-----------------|-------|-------|---------------|---------------|--------------|
| List models     | ✅    | ✅    | ✅            | ✅            | ✅ (empty)   |
| View model      | ✅    | ✅    | ✅            | ✅            | ❌ 403       |
| Update model    | ✅    | ✅    | ✅            | ❌ 403        | ❌ 403       |
| Delete model    | ✅    | ✅    | ❌ 403        | ❌ 403        | ❌ 403       |
| Deploy model    | ✅    | ✅    | ✅            | ❌ 403        | ❌ 403       |
| List shares     | ✅    | ✅    | ❌ 403        | ❌ 403        | ❌ 403       |
| Create share    | ✅    | ✅    | ❌ 403        | ❌ 403        | ❌ 403       |
| Update share    | ✅    | ✅    | ❌ 403        | ❌ 403        | ❌ 403       |
| Delete share    | ✅    | ✅    | ❌ 403        | ❌ 403        | ❌ 403       |

---

## Test Suite

### Location: `services/backend/tests/integration/test_models_rbac.py`

### Test Coverage:

#### **List Models (GET /api/v1/models)**
- ✅ `test_list_models_as_owner` - Owner sees owned models with `is_owner=True`
- ✅ `test_list_models_as_admin` - Admin sees all models
- ✅ `test_list_models_as_shared_user` - Shared user sees shared models

#### **View Model Details (GET /api/v1/models/{id})**
- ✅ `test_view_model_as_owner` - Owner can view with full access
- ✅ `test_view_model_as_shared_read_user` - Shared read user can view with `permission="read"`
- ✅ `test_view_model_as_shared_edit_user` - Shared edit user can view with `permission="edit"`
- ✅ `test_view_model_as_unauthorized_user` - Unauthorized user gets 403
- ✅ `test_view_model_as_admin` - Admin can always view
- ✅ `test_view_nonexistent_model` - Returns 404 for missing models

#### **Update Model (PATCH /api/v1/models/{id})**
- ✅ `test_update_model_as_owner` - Owner can update
- ✅ `test_update_model_as_shared_edit_user` - Shared edit user can update
- ✅ `test_update_model_as_shared_read_user` - Shared read user gets 403
- ✅ `test_update_model_as_admin` - Admin can update

#### **Delete Model (DELETE /api/v1/models/{id})**
- ✅ `test_delete_model_as_owner` - Owner can delete
- ✅ `test_delete_model_as_shared_edit_user` - Shared edit user gets 403 (cannot delete)
- ✅ `test_delete_model_as_admin` - Admin can delete

#### **Deploy Model (POST /api/v1/models/{id}/deploy)**
- ✅ `test_deploy_model_as_owner` - Owner can deploy
- ✅ `test_deploy_model_as_shared_edit_user` - Shared edit user can deploy
- ✅ `test_deploy_model_as_shared_read_user` - Shared read user gets 403

#### **Share Management**
- ✅ `test_create_share_as_owner` - Owner can create shares
- ✅ `test_create_share_as_shared_edit_user` - Shared user cannot create shares (403)
- ✅ `test_list_shares_as_owner` - Owner can list all shares
- ✅ `test_list_shares_as_shared_user` - Shared user cannot list shares (403)
- ✅ `test_delete_share_as_owner` - Owner can delete shares
- ✅ `test_delete_share_as_shared_user` - Shared user cannot delete shares (403)

#### **Edge Cases**
- ✅ `test_create_duplicate_share` - Returns 409 Conflict
- ✅ `test_invalid_permission_level` - Returns 422 Unprocessable Entity

**Total Tests**: 26 test cases

---

## Running Tests

### Prerequisites:
1. **Database setup**: Run migrations 04 and 05
   ```bash
   psql -U heimdall -d heimdall_db -f db/migrations/04-add-rbac-schema.sql
   psql -U heimdall -d heimdall_db -f db/migrations/05-migrate-existing-data.sql
   ```

2. **Docker services**: Ensure all 13 containers are running
   ```bash
   docker-compose up -d
   ```

3. **Backend service**: Start the backend
   ```bash
   cd services/backend
   python -m uvicorn src.main:app --reload
   ```

### Run Tests:
```bash
# Run all RBAC tests
cd services/backend
pytest tests/integration/test_models_rbac.py -v

# Run specific test
pytest tests/integration/test_models_rbac.py::test_view_model_as_owner -v

# Run with coverage
pytest tests/integration/test_models_rbac.py --cov=src.routers.models --cov-report=html
```

---

## Manual Testing with curl

### 1. Get Keycloak Token:
```bash
# Set up environment
export KEYCLOAK_URL="http://localhost:8080"
export REALM="heimdall"
export CLIENT_ID="heimdall-backend"
export CLIENT_SECRET="your-secret"

# Login as owner
TOKEN=$(scripts/get-keycloak-token.sh owner@example.com password123)
```

### 2. List Models:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/models
```

### 3. View Model Details:
```bash
MODEL_ID="uuid-here"
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/models/$MODEL_ID
```

### 4. Update Model:
```bash
curl -X PATCH \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "NewName"}' \
  http://localhost:8000/api/v1/models/$MODEL_ID
```

### 5. Create Share:
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "shared-user-id", "permission": "read"}' \
  http://localhost:8000/api/v1/models/$MODEL_ID/shares
```

### 6. Test Permission Denied (as unauthorized user):
```bash
# Login as different user
TOKEN_UNAUTHORIZED=$(scripts/get-keycloak-token.sh unauthorized@example.com password123)

# Try to view model (should get 403)
curl -H "Authorization: Bearer $TOKEN_UNAUTHORIZED" \
  http://localhost:8000/api/v1/models/$MODEL_ID
```

---

## Expected Test Results

### Success Scenarios:
- ✅ Owner can perform all actions on their models
- ✅ Admins bypass all permission checks
- ✅ Shared "edit" users can view, update, and deploy (but not delete or share)
- ✅ Shared "read" users can only view
- ✅ Unauthorized users get 403 for all protected actions

### Error Scenarios:
- ✅ Unauthorized access returns **403 Forbidden**
- ✅ Missing models return **404 Not Found**
- ✅ Duplicate shares return **409 Conflict**
- ✅ Invalid permission levels return **422 Unprocessable Entity**

---

## Database Queries for Verification

### Check RBAC Tables:
```sql
-- List all users
SELECT * FROM rbac.users;

-- List all models with owners
SELECT id, model_name, owner_id FROM models WHERE owner_id IS NOT NULL;

-- List all model shares
SELECT 
  ms.id,
  m.model_name,
  ms.user_id,
  ms.permission,
  ms.shared_by,
  ms.shared_at
FROM rbac.model_shares ms
JOIN models m ON ms.model_id = m.id;

-- Check specific user's permissions
SELECT 
  m.id,
  m.model_name,
  CASE
    WHEN m.owner_id = 'user-id-here' THEN 'owner'
    ELSE ms.permission
  END as access_level
FROM models m
LEFT JOIN rbac.model_shares ms ON m.id = ms.model_id AND ms.user_id = 'user-id-here'
WHERE m.owner_id = 'user-id-here' OR ms.user_id = 'user-id-here';
```

---

## Integration with Frontend

### Frontend Requirements (Phase 7):
1. **Authentication**: JWT token from Keycloak in Authorization header
2. **User Context**: Display `is_owner` and `permission` fields in UI
3. **Conditional UI**:
   - Show "Edit" button only if `is_owner` or `permission === 'edit'`
   - Show "Delete" button only if `is_owner`
   - Show "Share" button only if `is_owner`
   - Show "Deploy" button if `is_owner` or `permission === 'edit'`

### Example Frontend Code:
```typescript
interface ModelMetadata {
  id: string;
  model_name: string;
  is_owner: boolean;
  permission: 'read' | 'edit' | null;
}

function ModelActionsMenu({ model }: { model: ModelMetadata }) {
  const canEdit = model.is_owner || model.permission === 'edit';
  const canDelete = model.is_owner;
  const canShare = model.is_owner;
  const canDeploy = model.is_owner || model.permission === 'edit';

  return (
    <div>
      {canEdit && <button onClick={handleEdit}>Edit</button>}
      {canDelete && <button onClick={handleDelete}>Delete</button>}
      {canShare && <button onClick={handleShare}>Share</button>}
      {canDeploy && <button onClick={handleDeploy}>Deploy</button>}
    </div>
  );
}
```

---

## Known Issues / TODO

### Completed:
- ✅ asyncpg compatibility fix in `rbac.py`
- ✅ Models router implementation
- ✅ Comprehensive test suite
- ✅ Permission matrix documentation

### Pending:
- ⏳ Run migrations on production database
- ⏳ Execute test suite against live backend
- ⏳ Manual testing with Keycloak authentication
- ⏳ Frontend integration (Phase 7 continuation)
- ⏳ Performance testing with large datasets

### Future Enhancements:
- 📋 Audit logging for permission checks
- 📋 Email notifications for share invitations
- 📋 Bulk share operations (share with multiple users)
- 📋 Share expiration dates
- 📋 Share access history

---

## Commit Checklist

Before committing, verify:
- ✅ All 14 files are staged
- ✅ No Python syntax errors
- ✅ Test file compiles successfully
- ✅ Documentation is complete
- ⏳ Tests pass against live backend
- ⏳ Manual testing complete

**DO NOT COMMIT** until all tests pass and manual verification is complete.

---

## Next Steps

1. **Run Migrations**:
   ```bash
   psql -U heimdall -d heimdall_db -f db/migrations/04-add-rbac-schema.sql
   psql -U heimdall -d heimdall_db -f db/migrations/05-migrate-existing-data.sql
   ```

2. **Start Backend**:
   ```bash
   cd services/backend
   python -m uvicorn src.main:app --reload
   ```

3. **Run Tests**:
   ```bash
   pytest tests/integration/test_models_rbac.py -v
   ```

4. **Manual Testing**:
   - Test with real Keycloak tokens
   - Verify permission checks work correctly
   - Test all edge cases (404, 403, 409, 422)

5. **Commit** (only after all tests pass):
   ```bash
   git add -A
   git commit -m "feat(backend): implement RBAC for models router with asyncpg compatibility"
   ```

6. **Continue Phase 7**:
   - Task 13: Implement sources router with RBAC
   - Task 14-16: Frontend authentication hooks
   - Task 17-20: Frontend sharing UI

---

## Contact

For questions or issues:
- **Owner**: fulgidus (alessio.corsi@gmail.com)
- **Documentation**: `/docs/RBAC_IMPLEMENTATION.md`
- **Architecture**: `/docs/ARCHITECTURE.md`
