# Database Migrations

This directory contains SQL migration scripts for the Heimdall database schema.

## Migration Files

| File | Description | Status |
|------|-------------|--------|
| `001-add-missing-websdr-columns.sql` | Add missing columns to websdr_stations | Applied |
| `01-add-websdr-enhancements.sql` | WebSDR enhancements | Applied |
| `02-add-error-margin-to-known-sources.sql` | Add error margin to sources | Applied |
| `03-make-source-fields-optional.sql` | Make source fields optional | Applied |
| `04-add-rbac-schema.sql` | Add RBAC tables and columns | Ready |
| `05-migrate-existing-data.sql` | Migrate existing data to RBAC | Ready |

## How to Apply Migrations

### Prerequisites
1. PostgreSQL container must be running
2. Database `heimdall` must exist
3. User `heimdall_user` must have appropriate permissions

### Apply a Single Migration

```bash
# Connect to PostgreSQL container
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall < db/migrations/04-add-rbac-schema.sql
```

### Apply Multiple Migrations

```bash
# Apply RBAC schema
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall < db/migrations/04-add-rbac-schema.sql

# Apply data migration
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall < db/migrations/05-migrate-existing-data.sql
```

## Testing Migrations Locally

### 1. Backup Your Database First!

```bash
docker exec heimdall-postgres pg_dump -U heimdall_user heimdall > backup_$(date +%Y%m%d_%H%M%S).sql
```

### 2. Apply Migration 04 (RBAC Schema)

```bash
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall < db/migrations/04-add-rbac-schema.sql
```

Expected output:
- `NOTICE: Migration 04: All RBAC tables created successfully`
- `NOTICE: Migration 04: All RBAC columns added successfully`
- `NOTICE: Migration 04: RBAC Schema - COMPLETED`

### 3. Verify Tables Were Created

```bash
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall -c "\dt heimdall.constellation*"
```

Expected output:
```
 Schema   |         Name          | Type  |     Owner      
----------+-----------------------+-------+----------------
 heimdall | constellations        | table | heimdall_user
 heimdall | constellation_members | table | heimdall_user
 heimdall | constellation_shares  | table | heimdall_user
```

### 4. Update Admin User ID in Migration 05

**IMPORTANT**: Before applying migration 05, you MUST update the admin user ID!

1. Find your Keycloak admin user ID:
   ```bash
   # Method 1: Extract from JWT token after login
   # Login to frontend, open browser DevTools → Application → Local Storage
   # Copy the JWT token and decode it at https://jwt.io
   # The 'sub' claim is your user ID
   
   # Method 2: Query Keycloak database (if accessible)
   docker exec -i heimdall-keycloak psql -U keycloak -d keycloak \
     -c "SELECT id, username FROM user_entity WHERE username='admin';"
   ```

2. Edit `db/migrations/05-migrate-existing-data.sql`:
   ```sql
   -- Change this line:
   admin_user_id VARCHAR(255) := 'admin-default-id';  -- CHANGE THIS!
   
   -- To your actual admin user ID:
   admin_user_id VARCHAR(255) := '12345678-1234-1234-1234-123456789abc';
   ```

### 5. Apply Migration 05 (Data Migration)

```bash
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall < db/migrations/05-migrate-existing-data.sql
```

Expected output:
- `NOTICE: Migration 05: Assigned X existing sources to admin user`
- `NOTICE: Migration 05: Assigned X existing models to admin user`
- `NOTICE: Migration 05: Created Global constellation`
- `NOTICE: Migration 05: Added X / X WebSDR stations to Global constellation`
- `NOTICE: Migration 05: Data Migration - COMPLETED SUCCESSFULLY`

### 6. Verify Data Migration

```bash
# Check constellations
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "SELECT id, name, owner_id FROM heimdall.constellations;"

# Check constellation members
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "SELECT COUNT(*) as member_count FROM heimdall.constellation_members;"

# Check sources have owner_id
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "SELECT COUNT(*) as total, COUNT(owner_id) as with_owner FROM heimdall.known_sources;"

# Check models have owner_id
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "SELECT COUNT(*) as total, COUNT(owner_id) as with_owner FROM heimdall.models;"
```

## Rollback Procedure

If you need to rollback the migrations:

### Rollback Migration 05 (Data Migration)

```sql
-- Remove constellation data
DELETE FROM constellation_members;
DELETE FROM constellations WHERE name = 'Global';

-- Clear owner_id from sources and models
UPDATE known_sources SET owner_id = NULL, is_public = false;
UPDATE models SET owner_id = NULL;
UPDATE recording_sessions SET constellation_id = NULL;
```

### Rollback Migration 04 (Schema)

```sql
-- Drop new tables
DROP TABLE IF EXISTS model_shares CASCADE;
DROP TABLE IF EXISTS source_shares CASCADE;
DROP TABLE IF EXISTS constellation_shares CASCADE;
DROP TABLE IF EXISTS constellation_members CASCADE;
DROP TABLE IF EXISTS constellations CASCADE;

-- Remove columns from existing tables
ALTER TABLE known_sources DROP COLUMN IF EXISTS owner_id;
ALTER TABLE known_sources DROP COLUMN IF EXISTS is_public;
ALTER TABLE models DROP COLUMN IF EXISTS owner_id;
ALTER TABLE models DROP COLUMN IF EXISTS description;
ALTER TABLE recording_sessions DROP COLUMN IF EXISTS constellation_id;
```

## Production Deployment Checklist

- [ ] Backup production database
- [ ] Test migrations on staging environment
- [ ] Update admin user ID in migration 05
- [ ] Schedule maintenance window
- [ ] Apply migration 04
- [ ] Apply migration 05
- [ ] Verify all tables created
- [ ] Verify data migration successful
- [ ] Deploy backend services with RBAC code
- [ ] Deploy frontend with RBAC UI
- [ ] Test admin access
- [ ] Test operator access
- [ ] Test user access
- [ ] Monitor logs for errors

## Troubleshooting

### Error: "relation already exists"

This means the migration has already been applied. Check:
```bash
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "\d heimdall.constellations"
```

### Error: "column already exists"

The migration includes `IF NOT EXISTS` checks, so this shouldn't happen. If it does, manually check column existence:
```bash
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "\d heimdall.known_sources"
```

### Error: "Global constellation not found"

Check if the Global constellation was created:
```bash
docker exec -i heimdall-postgres psql -U heimdall_user -d heimdall \
  -c "SELECT * FROM heimdall.constellations WHERE name = 'Global';"
```

If not found, manually create it:
```sql
INSERT INTO constellations (name, description, owner_id)
VALUES ('Global', 'Default constellation', 'YOUR_ADMIN_USER_ID');
```

## Questions?

See:
- [RBAC Implementation Plan](../docs/RBAC_IMPLEMENTATION.md)
- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [Development Guide](../docs/DEVELOPMENT.md)
