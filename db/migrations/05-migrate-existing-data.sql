-- Migration 05: Migrate Existing Data to RBAC
-- Purpose: Assign existing resources to default admin and create Global constellation
-- Author: fulgidus
-- Date: 2025-11-08
-- Related: docs/RBAC_IMPLEMENTATION.md
-- Prerequisites: Migration 04 must be applied first

-- ============================================================================
-- CONFIGURATION
-- ============================================================================

-- Default admin user ID (Keycloak sub claim)
-- NOTE: Replace this with your actual admin user ID from Keycloak
-- To find it: Check Keycloak admin console → Users → Your Admin → ID field
-- Or extract from JWT token 'sub' claim after admin login
DO $$
DECLARE
    admin_user_id VARCHAR(255) := 'admin-default-id';  -- CHANGE THIS!
BEGIN
    -- Store in temporary table for use in subsequent statements
    CREATE TEMP TABLE IF NOT EXISTS migration_config (
        admin_user_id VARCHAR(255)
    );
    
    DELETE FROM migration_config;
    INSERT INTO migration_config VALUES (admin_user_id);
    
    RAISE NOTICE 'Migration 05: Using admin user ID: %', admin_user_id;
    RAISE WARNING 'IMPORTANT: Update admin_user_id in this script with your actual Keycloak admin user ID!';
END $$;

-- ============================================================================
-- PART 1: MIGRATE EXISTING SOURCES
-- ============================================================================

-- Assign all existing known_sources to admin user
DO $$
DECLARE
    admin_id VARCHAR(255);
    updated_count INT;
BEGIN
    SELECT admin_user_id INTO admin_id FROM migration_config;
    
    UPDATE known_sources
    SET owner_id = admin_id,
        is_public = true  -- Make existing sources public for backward compatibility
    WHERE owner_id IS NULL;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    
    RAISE NOTICE 'Migration 05: Assigned % existing sources to admin user', updated_count;
END $$;

-- ============================================================================
-- PART 2: MIGRATE EXISTING MODELS
-- ============================================================================

-- Assign all existing models to admin user
DO $$
DECLARE
    admin_id VARCHAR(255);
    updated_count INT;
BEGIN
    SELECT admin_user_id INTO admin_id FROM migration_config;
    
    UPDATE models
    SET owner_id = admin_id,
        description = COALESCE(description, 'Migrated model from pre-RBAC system')
    WHERE owner_id IS NULL;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    
    RAISE NOTICE 'Migration 05: Assigned % existing models to admin user', updated_count;
END $$;

-- ============================================================================
-- PART 3: CREATE DEFAULT "GLOBAL" CONSTELLATION
-- ============================================================================

-- Create a default constellation containing all WebSDR stations
-- This preserves backward compatibility for existing workflows
DO $$
DECLARE
    admin_id VARCHAR(255);
    global_constellation_id UUID;
    websdr_count INT;
    added_count INT := 0;
    websdr_record RECORD;
BEGIN
    SELECT admin_user_id INTO admin_id FROM migration_config;
    
    -- Create Global constellation if it doesn't exist
    INSERT INTO constellations (name, description, owner_id)
    VALUES (
        'Global',
        'Default constellation containing all WebSDR stations (created during RBAC migration)',
        admin_id
    )
    ON CONFLICT DO NOTHING
    RETURNING id INTO global_constellation_id;
    
    -- If constellation already existed, get its ID
    IF global_constellation_id IS NULL THEN
        SELECT id INTO global_constellation_id 
        FROM constellations 
        WHERE name = 'Global';
        
        RAISE NOTICE 'Migration 05: Global constellation already exists (id: %)', global_constellation_id;
    ELSE
        RAISE NOTICE 'Migration 05: Created Global constellation (id: %)', global_constellation_id;
    END IF;
    
    -- Add all WebSDR stations to Global constellation
    SELECT COUNT(*) INTO websdr_count FROM websdr_stations;
    
    FOR websdr_record IN SELECT id, name FROM websdr_stations LOOP
        INSERT INTO constellation_members (constellation_id, websdr_station_id, added_by)
        VALUES (global_constellation_id, websdr_record.id, admin_id)
        ON CONFLICT (constellation_id, websdr_station_id) DO NOTHING;
        
        IF FOUND THEN
            added_count := added_count + 1;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Migration 05: Added % / % WebSDR stations to Global constellation', added_count, websdr_count;
END $$;

-- ============================================================================
-- PART 4: LINK EXISTING SESSIONS TO GLOBAL CONSTELLATION
-- ============================================================================

-- Link existing recording sessions to the Global constellation
DO $$
DECLARE
    global_constellation_id UUID;
    updated_count INT;
BEGIN
    -- Get Global constellation ID
    SELECT id INTO global_constellation_id 
    FROM constellations 
    WHERE name = 'Global';
    
    IF global_constellation_id IS NULL THEN
        RAISE EXCEPTION 'Global constellation not found. Migration failed.';
    END IF;
    
    -- Update sessions that don't have a constellation assigned
    UPDATE recording_sessions
    SET constellation_id = global_constellation_id
    WHERE constellation_id IS NULL;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    
    RAISE NOTICE 'Migration 05: Linked % existing sessions to Global constellation', updated_count;
END $$;

-- ============================================================================
-- PART 5: VALIDATION
-- ============================================================================

-- Verify data migration
DO $$
DECLARE
    sources_without_owner INT;
    models_without_owner INT;
    sessions_without_constellation INT;
    global_exists BOOLEAN;
    global_member_count INT;
BEGIN
    -- Check sources
    SELECT COUNT(*) INTO sources_without_owner 
    FROM known_sources 
    WHERE owner_id IS NULL;
    
    IF sources_without_owner > 0 THEN
        RAISE WARNING 'Migration incomplete: % sources without owner', sources_without_owner;
    END IF;
    
    -- Check models
    SELECT COUNT(*) INTO models_without_owner 
    FROM models 
    WHERE owner_id IS NULL;
    
    IF models_without_owner > 0 THEN
        RAISE WARNING 'Migration incomplete: % models without owner', models_without_owner;
    END IF;
    
    -- Check Global constellation exists
    SELECT EXISTS(SELECT 1 FROM constellations WHERE name = 'Global') 
    INTO global_exists;
    
    IF NOT global_exists THEN
        RAISE EXCEPTION 'Migration failed: Global constellation not created';
    END IF;
    
    -- Check Global constellation has members
    SELECT COUNT(*) INTO global_member_count
    FROM constellation_members cm
    JOIN constellations c ON cm.constellation_id = c.id
    WHERE c.name = 'Global';
    
    IF global_member_count = 0 THEN
        RAISE WARNING 'Global constellation has no WebSDR members';
    END IF;
    
    -- Check sessions
    SELECT COUNT(*) INTO sessions_without_constellation
    FROM recording_sessions
    WHERE constellation_id IS NULL;
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migration 05: Data Migration - VALIDATION';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Sources without owner: %', sources_without_owner;
    RAISE NOTICE 'Models without owner: %', models_without_owner;
    RAISE NOTICE 'Sessions without constellation: %', sessions_without_constellation;
    RAISE NOTICE 'Global constellation exists: %', global_exists;
    RAISE NOTICE 'Global constellation members: %', global_member_count;
    RAISE NOTICE '============================================';
    
    IF sources_without_owner = 0 AND 
       models_without_owner = 0 AND 
       global_exists AND 
       global_member_count > 0 THEN
        RAISE NOTICE 'Migration 05: Data Migration - COMPLETED SUCCESSFULLY';
    ELSE
        RAISE WARNING 'Migration 05: Data Migration - COMPLETED WITH WARNINGS';
    END IF;
END $$;

-- ============================================================================
-- PART 6: SUMMARY
-- ============================================================================

DO $$
DECLARE
    source_count INT;
    model_count INT;
    session_count INT;
    websdr_count INT;
BEGIN
    SELECT COUNT(*) INTO source_count FROM known_sources;
    SELECT COUNT(*) INTO model_count FROM models;
    SELECT COUNT(*) INTO session_count FROM recording_sessions;
    SELECT COUNT(*) INTO websdr_count FROM websdr_stations;
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migration 05: RBAC Data Migration - SUMMARY';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migrated resources:';
    RAISE NOTICE '  - % known sources → admin user (public)', source_count;
    RAISE NOTICE '  - % ML models → admin user', model_count;
    RAISE NOTICE '  - % recording sessions → Global constellation', session_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Created constellations:';
    RAISE NOTICE '  - Global (%  WebSDR stations)', websdr_count;
    RAISE NOTICE '';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'IMPORTANT: Post-Migration Steps';
    RAISE NOTICE '============================================';
    RAISE NOTICE '1. Verify admin_user_id is correct in this script';
    RAISE NOTICE '2. Test admin user can access all resources';
    RAISE NOTICE '3. Create additional constellations as needed';
    RAISE NOTICE '4. Share constellations with operators/users';
    RAISE NOTICE '5. Deploy backend with RBAC code';
    RAISE NOTICE '6. Deploy frontend with RBAC UI';
    RAISE NOTICE '============================================';
END $$;

-- Cleanup temporary table
DROP TABLE IF EXISTS migration_config;
