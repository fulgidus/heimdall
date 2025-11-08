# 🔐 RBAC Implementation Plan - Heimdall SDR

**Status**: ✅ COMPLETE  
**Started**: 2025-11-08  
**Completed**: 2025-11-08  
**Owner**: fulgidus  
**Progress**: 24/24 tasks complete (100%)

---

## 📋 Executive Summary

Implementation of Role-Based Access Control (RBAC) for Heimdall with the introduction of **Constellations** - logical groupings of WebSDR stations that can be owned, shared, and managed by users with different permission levels.

### Key Concepts

- **Constellation**: A logical grouping of WebSDR stations
- **Ownership**: Resources (Constellations, Sources, Models) have an owner
- **Sharing**: Owners can share resources with other users (read/edit permissions)
- **Roles**: 3 roles from Keycloak - USER, OPERATOR, ADMIN

---

## 🎯 Goals

1. ✅ Introduce Constellation concept (grouping of SDR stations)
2. ✅ Implement ownership model for Constellations, Sources, Models
3. ✅ Add permission-based sharing (read/edit) for all owned resources
4. ✅ Enforce role-based access control (USER/OPERATOR/ADMIN)
5. ✅ Create frontend UI for managing Constellations and sharing
6. ✅ Maintain backward compatibility with existing data

---

## 🏗️ Architecture Decisions

### ✅ User Assignment to Constellations
**Decision**: Use `constellation_shares` table with `permission` field ('read'|'edit')

**Rationale**:
- Consistent schema pattern for all shareable resources
- Granular permissions (read-only vs edit access)
- Future-proof for additional permission levels
- Allows OPERATOR to share in read-only mode

### ✅ Backend RBAC Strategy
**Decision**: Hybrid approach
- **API Gateway**: Role-based route access (e.g., USER cannot access /admin routes)
- **Backend Services**: Ownership and sharing checks using `common/auth/rbac.py`

**Rationale**:
- Defense in depth security
- Gateway handles coarse-grained checks (fast)
- Services handle fine-grained checks (context-aware)
- Centralized RBAC utilities in `common/auth/rbac.py` (DRY)

### ✅ Frontend Guards Strategy
**Decision**: Dual approach
- **Route Guards**: Protect entire pages based on role
- **Component Guards**: Show/hide UI elements based on permissions

**Rationale**:
- Route guards prevent unauthorized page access
- Component guards provide better UX (show relevant controls only)
- Defense in depth at UI level

---

## 📊 Permission Matrix

| Resource | USER | OPERATOR | ADMIN |
|----------|------|----------|-------|
| **Constellations** | View assigned (read) | CRUD owned/shared (edit) | Full access |
| **Sources** | View public/shared | CRUD owned/shared | Full access |
| **Models** | View shared | CRUD owned/shared | Full access |
| **Sessions** | Start on assigned constellations | Full control on accessible | Full control |
| **WebSDRs** | View all (global) | View all + assign to constellations | Full access |
| **System Settings** | ❌ | ❌ | ✅ |

### Role Definitions

#### 🔵 USER
- Can view Constellation page and System Status
- Can view only constellations they're assigned to
- Can start localization sessions on assigned constellations
- Can set frequency and parameters for localization
- Can view history of assigned constellations
- **Cannot** create/edit constellations
- **Cannot** manage SDRs, sources, or models

#### 🟢 OPERATOR
- All USER permissions +
- Can create new constellations (becomes owner)
- Can edit owned constellations or those shared with 'edit' permission
- Can add/remove SDRs from constellations they can edit
- Can create and manage sources (owned or shared)
- Can create and manage models (owned or shared)
- Can start RF acquisitions
- Can start training sessions
- Can generate synthetic samples
- Can share owned resources with other users
- Sees only owned or shared resources (not all)

#### 🔴 ADMIN
- Full system access
- Can view and edit ALL constellations, sources, models
- Can manage users and their constellation assignments
- Can modify system settings
- Can perform all OPERATOR actions on any resource

---

## 🗄️ Database Schema

### New Tables

#### `constellations`
Represents a logical grouping of WebSDR stations.

```sql
CREATE TABLE constellations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id VARCHAR(255) NOT NULL,  -- Keycloak user ID (sub claim)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_constellations_owner ON constellations(owner_id);
```

#### `constellation_members`
Many-to-many relationship: Constellations ↔ WebSDR Stations.

```sql
CREATE TABLE constellation_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    constellation_id UUID NOT NULL REFERENCES constellations(id) ON DELETE CASCADE,
    websdr_station_id UUID NOT NULL REFERENCES websdr_stations(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    added_by VARCHAR(255),  -- User who added this SDR
    UNIQUE(constellation_id, websdr_station_id)
);

CREATE INDEX idx_constellation_members_constellation ON constellation_members(constellation_id);
CREATE INDEX idx_constellation_members_websdr ON constellation_members(websdr_station_id);
```

#### `constellation_shares`
User access permissions for constellations.

```sql
CREATE TABLE constellation_shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    constellation_id UUID NOT NULL REFERENCES constellations(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,  -- Keycloak user ID
    permission VARCHAR(20) NOT NULL CHECK (permission IN ('read', 'edit')),
    shared_by VARCHAR(255) NOT NULL,  -- User who created the share
    shared_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(constellation_id, user_id)
);

CREATE INDEX idx_constellation_shares_constellation ON constellation_shares(constellation_id);
CREATE INDEX idx_constellation_shares_user ON constellation_shares(user_id);
```

#### `source_shares`
User access permissions for known sources.

```sql
CREATE TABLE source_shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES known_sources(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    permission VARCHAR(20) NOT NULL CHECK (permission IN ('read', 'edit')),
    shared_by VARCHAR(255) NOT NULL,
    shared_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, user_id)
);

CREATE INDEX idx_source_shares_source ON source_shares(source_id);
CREATE INDEX idx_source_shares_user ON source_shares(user_id);
```

#### `model_shares`
User access permissions for ML models.

```sql
CREATE TABLE model_shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    permission VARCHAR(20) NOT NULL CHECK (permission IN ('read', 'edit')),
    shared_by VARCHAR(255) NOT NULL,
    shared_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, user_id)
);

CREATE INDEX idx_model_shares_model ON model_shares(model_id);
CREATE INDEX idx_model_shares_user ON model_shares(user_id);
```

### Modified Tables

#### `known_sources`
Add ownership and public visibility.

```sql
ALTER TABLE known_sources 
    ADD COLUMN owner_id VARCHAR(255),
    ADD COLUMN is_public BOOLEAN DEFAULT false;

CREATE INDEX idx_known_sources_owner ON known_sources(owner_id);
CREATE INDEX idx_known_sources_public ON known_sources(is_public);
```

#### `models`
Add ownership and description.

```sql
ALTER TABLE models 
    ADD COLUMN owner_id VARCHAR(255),
    ADD COLUMN description TEXT;

CREATE INDEX idx_models_owner ON models(owner_id);
```

#### `recording_sessions`
Link sessions to constellations.

```sql
ALTER TABLE recording_sessions 
    ADD COLUMN constellation_id UUID REFERENCES constellations(id);

CREATE INDEX idx_recording_sessions_constellation ON recording_sessions(constellation_id);
```

---

## 🔧 Implementation Tasks

### ✅ Phase 1: Database Foundation
**Goal**: Create database schema and migration scripts

- [x] **Task 1**: Create migration SQL script `04-add-rbac-schema.sql`
  - Create all new tables (constellations, shares)
  - Alter existing tables (add owner_id, is_public, constellation_id)
  - Create all indexes
  - **Files**: `db/migrations/04-add-rbac-schema.sql`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 13**: Create data migration script
  - Assign all existing sources/models to default admin user
  - Create a default "Global" constellation with all SDRs
  - **Files**: `db/migrations/05-migrate-existing-data.sql`
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 2: Backend Models
**Goal**: Create SQLAlchemy ORM models for new tables

- [x] **Task 2**: Create Constellation models
  - `Constellation` model
  - `ConstellationMember` model  
  - `ConstellationShare` model
  - **Files**: `services/backend/src/models/constellation.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 3**: Update existing models
  - Add `owner_id` to `KnownSource` model
  - Add `is_public` to `KnownSource` model
  - **Files**: `services/backend/src/models/session.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 4**: Create sharing models
  - `SourceShare` model
  - `ModelShare` model
  - **Files**: `services/backend/src/models/shares.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 5**: Update RecordingSession model
  - Add `constellation_id` foreign key
  - Add relationship to `Constellation`
  - **Files**: `services/backend/src/models/session.py`
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 3: Backend RBAC Utilities
**Goal**: Create reusable permission checking functions

- [x] **Task 6**: Create RBAC utility module
  - `can_view_constellation(db, user_id, constellation_id, is_admin)`
  - `can_edit_constellation(db, user_id, constellation_id, is_admin)`
  - `can_delete_constellation(db, user_id, constellation_id, is_admin)`
  - `can_view_source(db, user_id, source_id, is_admin)`
  - `can_edit_source(db, user_id, source_id, is_admin)`
  - `can_view_model(db, user_id, model_id, is_admin)`
  - `can_edit_model(db, user_id, model_id, is_admin)`
  - `get_user_constellations(db, user_id, is_admin)`
  - Added helper functions (ownership checks, permission lookups)
  - Added synchronous versions for non-async contexts
  - **Files**: `services/common/auth/rbac.py`, `services/common/auth/__init__.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 7**: Update User model
  - Add `is_user` property
  - Update `from_token_data()` to properly detect all roles
  - **Files**: `services/common/auth/models.py`
  - **Status**: ✅ Complete (2025-11-08) - Completed in previous session

### ✅ Phase 4: Backend API - Constellations
**Goal**: Create REST API for constellation management

- [x] **Task 8**: Create Constellation CRUD router
  - `GET /api/v1/constellations` - List accessible constellations
  - `POST /api/v1/constellations` - Create constellation (operator+)
  - `GET /api/v1/constellations/{id}` - Get constellation details
  - `PUT /api/v1/constellations/{id}` - Update constellation
  - `DELETE /api/v1/constellations/{id}` - Delete constellation
  - `POST /api/v1/constellations/{id}/members` - Add WebSDR
  - `DELETE /api/v1/constellations/{id}/members/{websdr_id}` - Remove WebSDR
  - **Files**: `services/backend/src/routers/constellations.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 9**: Create Constellation sharing endpoints
  - `GET /api/v1/constellations/{id}/shares` - List shares
  - `POST /api/v1/constellations/{id}/shares` - Share with user
  - `PUT /api/v1/constellations/{id}/shares/{user_id}` - Update permission
  - `DELETE /api/v1/constellations/{id}/shares/{user_id}` - Remove share
  - **Files**: `services/backend/src/routers/constellations.py`
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 5: Backend API - Existing Resources
**Goal**: Add RBAC checks to existing endpoints

- [x] **Task 10**: Update sessions router
  - Add constellation_id to session creation
  - Check constellation access before creating session
  - Filter sessions list by accessible constellations
  - **Files**: `services/backend/src/routers/sessions.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 11**: Create/update Sources router
  - Add owner_id to source creation
  - Implement sharing endpoints
  - Filter sources by ownership/sharing/public
  - **Files**: `services/backend/src/routers/sources.py` (new)
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 12**: Create/update Models router
  - Add owner_id to model creation
  - Implement sharing endpoints
  - Filter models by ownership/sharing
  - **Files**: `services/backend/src/routers/models.py` (new)
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 6: Frontend Authentication
**Goal**: Create authentication hooks and utilities

- [x] **Task 16**: Update authStore
  - Fix role extraction from JWT (admin/operator/user)
  - Ensure role hierarchy (admin includes operator, operator includes user)
  - Added `extractUserRole()` function implementing hierarchy
  - Updated `User` interface with `'operator'` role type and `roles: string[]` array
  - **Files**: `frontend/src/store/authStore.ts`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 14a**: Create useAuth hook
  - `isAdmin: boolean` - true only for admin
  - `isOperator: boolean` - true for operator OR admin
  - `isUser: boolean` - true for all authenticated users
  - `user: User | null`
  - **Files**: `frontend/src/hooks/useAuth.ts`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 14b**: Create usePermissions hook
  - Generic: `canView()`, `canEdit()`, `canDelete()`, `canShare()`, `canCreate()`
  - Constellation: `canViewConstellation()`, `canEditConstellation()`, `canDeleteConstellation()`
  - Source: `canViewSource()`, `canEditSource()`, `canDeleteSource()`
  - Model: `canViewModel()`, `canEditModel()`, `canDeleteModel()`
  - Special: `canAccessSystemSettings()`, `canStartTraining()`, `canGenerateSynthetic()`
  - Admin bypass logic for all checks
  - **Files**: `frontend/src/hooks/usePermissions.ts`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 15**: Create RequireRole route guard
  - HOC component to protect routes by role
  - Shows "Access Denied" UI with role requirement message
  - Redirects to /login if not authenticated
  - Supports optional custom fallback components
  - **Files**: `frontend/src/components/auth/RequireRole.tsx`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 16b**: Update App.tsx route guards
  - Wrapped all routes with `<RequireRole role="...">` guards
  - User+ routes: /dashboard, /localization, /analytics, /profile, /recording, /history, /system-status
  - Operator+ routes: /websdrs, /sources, /import-export, /training, /terrain, /audio-library, /data-ingestion
  - Admin-only routes: /settings
  - **Files**: `frontend/src/App.tsx`
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 7: Frontend UI
**Goal**: Create user-facing pages and components

- [x] **Task 17**: Create Constellations management page
  - List user's constellations (owned + shared)
  - Create new constellation (operator+)
  - Edit constellation (name, description)
  - Add/remove WebSDRs to constellation
  - Delete constellation (owner only)
  - Search and filter constellations
  - Permission-based UI controls
  - **Files**: `frontend/src/pages/Constellations.tsx`, `frontend/src/components/constellations/ConstellationCard.tsx`, `frontend/src/components/constellations/ConstellationForm.tsx`, `frontend/src/services/api/constellations.ts`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 19**: Create sharing UI component
  - Modal/drawer for sharing management
  - Search users by email/username (debounced, 300ms)
  - Set permission level (read/edit)
  - List current shares with avatars
  - Update permissions inline
  - Remove shares
  - Owner-only controls
  - Real-time updates with toast notifications
  - Generic design (reusable for sources/models)
  - **Files**: `frontend/src/components/sharing/ShareModal.tsx`, `frontend/src/components/sharing/UserSearch.tsx`, `frontend/src/components/sharing/ShareList.tsx`, `frontend/src/components/sharing/index.ts`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 17b**: Integrate ShareModal into Constellations page
  - Add share modal state management
  - Implement share handlers (add/update/remove)
  - Connect to constellation sharing APIs
  - **Files**: `frontend/src/pages/Constellations.tsx`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 18**: Update Dashboard
  - Filter displayed data by user's constellations
  - Show constellation selector dropdown
  - Update metrics to be constellation-specific
  - **Files**: `frontend/src/pages/Dashboard.tsx`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 20**: Add component-level guards
  - Hide edit/delete buttons if no permission
  - Hide admin-only settings
  - Hide operator-only controls for users
  - **Files**: 
    - `frontend/src/pages/Training/components/SyntheticTab/SyntheticTab.tsx`
    - `frontend/src/pages/Training/components/ModelsTab/ModelsTab.tsx`
    - `frontend/src/pages/Training/components/ModelsTab/ModelCard.tsx`
    - `frontend/src/pages/Training/components/JobsTab/JobsTab.tsx`
    - `frontend/src/pages/SourcesManagement.tsx`
    - `frontend/src/pages/Settings.tsx`
    - `frontend/src/pages/WebSDRManagement.tsx`
    - `frontend/src/pages/SessionHistory.tsx`
    - `frontend/src/components/widgets/QuickActionsWidget.tsx`
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 8: Testing
**Goal**: Ensure correctness and reliability

- [x] **Task 21**: Unit tests for RBAC utilities
  - Test `can_view_constellation()` with different roles
  - Test `can_edit_constellation()` with ownership/sharing
  - Test permission inheritance (admin can do everything)
  - 67 comprehensive test cases covering all RBAC functions
  - **Files**: `services/common/auth/tests/test_rbac.py`
  - **Status**: ✅ Complete (2025-11-08)

- [x] **Task 22**: Integration tests for APIs
  - Test constellation CRUD with different roles
  - Test sharing endpoints
  - Test permission enforcement (403 errors)
  - 50+ comprehensive integration test cases
  - **Files**: `services/backend/tests/integration/test_constellations_rbac.py`
  - **Status**: ✅ Complete (2025-11-08)

### ✅ Phase 9: Documentation
**Goal**: Document RBAC system for developers and users

- [x] **Task 23**: Update documentation
  - Add RBAC section to ARCHITECTURE.md ✅
  - Add Constellation guide to user docs (`docs/RBAC.md`) ✅
  - Update API documentation (`docs/API.md` Constellations section) ✅
  - Migration guide included in ARCHITECTURE.md ✅
  - **Files**: `docs/ARCHITECTURE.md`, `docs/RBAC.md`, `docs/API.md`
  - **Status**: ✅ Complete (2025-11-08)

---

## 📂 File Structure

```
heimdall/
├── db/
│   └── migrations/
│       ├── 04-add-rbac-schema.sql          [Task 1]
│       └── 05-migrate-existing-data.sql    [Task 13]
├── services/
│   ├── common/
│   │   └── auth/
│   │       ├── models.py                   [Task 7 - update]
│   │       ├── rbac.py                     [Task 6 - new]
│   │       └── tests/
│   │           └── test_rbac.py            [Task 21]
│   └── backend/
│       └── src/
│           ├── models/
│           │   ├── constellation.py        [Task 2 - new]
│           │   ├── shares.py               [Task 4 - new]
│           │   ├── db.py                   [Task 3 - update]
│           │   └── session.py              [Task 5 - update]
│           ├── routers/
│           │   ├── constellations.py       [Task 8, 9 - new]
│           │   ├── sources.py              [Task 11 - new]
│           │   ├── models.py               [Task 12 - new]
│           │   └── sessions.py             [Task 10 - update]
│           └── tests/
│               └── test_constellations.py  [Task 22]
└── frontend/
    └── src/
        ├── store/
        │   └── authStore.ts                [Task 16 - update]
        ├── hooks/
        │   ├── useAuth.ts                  [Task 14 - new]
        │   └── usePermissions.ts           [Task 14 - new]
        ├── components/
        │   ├── auth/
        │   │   └── RequireRole.tsx         [Task 15 - new]
        │   └── sharing/
        │       └── ShareModal.tsx          [Task 19 - new]
        └── pages/
            ├── Constellations.tsx          [Task 17 - new]
            └── Dashboard.tsx               [Task 18 - update]
```

---

## 🧪 Testing Strategy

### Unit Tests
- RBAC permission functions (can_view, can_edit, can_delete)
- User role detection and hierarchy
- Permission inheritance (admin > operator > user)

### Integration Tests
- Constellation CRUD operations with different roles
- Sharing workflows (create share, update permission, remove share)
- Permission enforcement (403 Forbidden responses)
- Cross-service RBAC (sessions requiring constellation access)

### End-to-End Tests
- Complete user workflow: login → create constellation → share → other user views
- Admin workflow: view all constellations, manage users
- Operator workflow: create resources, share with users

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] All database migrations tested locally
- [ ] Existing data migration verified
- [ ] Backend tests passing (unit + integration)
- [ ] Frontend tests passing
- [ ] API documentation updated

### Deployment Steps
1. **Backup database** (critical!)
2. **Run migration**: `04-add-rbac-schema.sql`
3. **Run data migration**: `05-migrate-existing-data.sql`
4. **Deploy backend** with new RBAC code
5. **Deploy frontend** with new UI
6. **Verify**: Check admin can access all data
7. **Verify**: Check operator can create constellations
8. **Verify**: Check user can view assigned constellations

### Post-Deployment
- [ ] Monitor logs for RBAC-related errors
- [ ] Verify existing sessions still work
- [ ] Verify new constellation creation works
- [ ] Verify sharing works
- [ ] Update documentation site

---

## 📝 Progress Tracking

### Completed Tasks
- ✅ Task 1: Database migration `04-add-rbac-schema.sql` (2025-11-08)
- ✅ Task 13: Data migration `05-migrate-existing-data.sql` (2025-11-08)
- ✅ Task 2: Constellation SQLAlchemy models (2025-11-08)
- ✅ Task 3: KnownSource model updated with owner_id and is_public (2025-11-08)
- ✅ Task 4: Sharing SQLAlchemy models (2025-11-08)
- ✅ Task 5: RecordingSession model with constellation_id (2025-11-08)
- ✅ Task 7: User model update (2025-11-08)
- ✅ Task 6: RBAC utilities module (asyncpg-compatible) (2025-11-08)
- ✅ Task 8: Constellation CRUD router (2025-11-08)
- ✅ Task 9: Constellation sharing endpoints (2025-11-08)
- ✅ Task 10: Sessions router with RBAC (2025-11-08)
- ✅ Task 11: Sources router with RBAC (2025-11-08)
- ✅ Task 12: Models router with RBAC (2025-11-08)
- ✅ Task 14a: useAuth hook (2025-11-08)
- ✅ Task 14b: usePermissions hook (2025-11-08)
- ✅ Task 15: RequireRole route guard (2025-11-08)
- ✅ Task 16: authStore role extraction and App.tsx route guards (2025-11-08)
- ✅ Task 17: Constellations management page (2025-11-08)
- ✅ Task 19: Sharing UI component (ShareModal) (2025-11-08)
- ✅ Task 17b: ShareModal integration into Constellations page (2025-11-08)
- ✅ Task 18: Dashboard constellation filtering (2025-11-08)
- ✅ Task 20: Component-level permission guards (2025-11-08)
- ✅ Task 21: RBAC unit tests (67 test cases) (2025-11-08)
- ✅ Task 22: Constellation integration tests (50+ test cases) (2025-11-08)
- ✅ Task 23: Documentation updates (ARCHITECTURE.md, RBAC.md, API.md) (2025-11-08)

### In Progress
_None - All tasks complete!_

### Blocked
_None_

### Completed
All 24 tasks completed successfully! ✅ RBAC implementation ready for production deployment.

---

## 🐛 Known Issues / Decisions Needed

### Issue 1: Default Admin User ID
**Question**: What is the Keycloak user ID (sub claim) for the default admin user?  
**Impact**: Need this for data migration (Task 13)  
**Resolution**: Check Keycloak realm export or query Keycloak API

### Issue 2: WebSDR Global Visibility
**Confirmed**: All WebSDR stations remain globally visible (not owned).  
Users with operator+ role can assign any WebSDR to their constellations.

### Issue 3: Backward Compatibility
**Strategy**: Create a default "Global" constellation containing all existing SDRs.  
Assign it to admin user. This preserves existing workflows.

---

## 🔗 Related Documents

- [AGENTS.md](../AGENTS.md) - Project roadmap
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [AUTHENTICATION_FIX.md](AUTHENTICATION_FIX.md) - Previous auth work
- Database schema: [init-postgres.sql](../db/init-postgres.sql)
- Backend models: [services/backend/src/models/](../services/backend/src/models/)
- Frontend auth: [frontend/src/store/authStore.ts](../frontend/src/store/authStore.ts)

---

## 📞 Questions / Clarifications

If you have questions about this implementation:
1. Check this document first
2. Check related documents above
3. Ask in project discussion

---

**Last Updated**: 2025-11-08 by fulgidus  
**Status**: ✅ COMPLETE - Ready for production deployment  
**Next Steps**: Follow deployment checklist above to deploy RBAC to production
