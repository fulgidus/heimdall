-- Migration 04: Add RBAC Schema
-- Purpose: Implement Role-Based Access Control with Constellations
-- Author: fulgidus
-- Date: 2025-11-08
-- Related: docs/RBAC_IMPLEMENTATION.md

-- ============================================================================
-- PART 1: NEW TABLES
-- ============================================================================

-- Table: constellations
-- Purpose: Logical groupings of WebSDR stations that can be owned and shared
CREATE TABLE IF NOT EXISTS constellations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id VARCHAR(255) NOT NULL,  -- Keycloak user ID (sub claim)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_constellations_owner ON constellations(owner_id);
CREATE INDEX idx_constellations_name ON constellations(name);

COMMENT ON TABLE constellations IS 'Logical groupings of WebSDR stations with ownership';
COMMENT ON COLUMN constellations.owner_id IS 'Keycloak user ID from JWT sub claim';

-- Table: constellation_members
-- Purpose: Many-to-many relationship between Constellations and WebSDR Stations
CREATE TABLE IF NOT EXISTS constellation_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    constellation_id UUID NOT NULL REFERENCES constellations(id) ON DELETE CASCADE,
    websdr_station_id UUID NOT NULL REFERENCES websdr_stations(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    added_by VARCHAR(255),  -- User who added this SDR to the constellation
    UNIQUE(constellation_id, websdr_station_id)
);

CREATE INDEX idx_constellation_members_constellation ON constellation_members(constellation_id);
CREATE INDEX idx_constellation_members_websdr ON constellation_members(websdr_station_id);

COMMENT ON TABLE constellation_members IS 'Many-to-many: Constellations ↔ WebSDR Stations';

-- Table: constellation_shares
-- Purpose: User access permissions for constellations (read/edit)
CREATE TABLE IF NOT EXISTS constellation_shares (
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
CREATE INDEX idx_constellation_shares_permission ON constellation_shares(permission);

COMMENT ON TABLE constellation_shares IS 'User access permissions for constellations';
COMMENT ON COLUMN constellation_shares.permission IS 'Access level: read (view only) or edit (modify)';

-- Table: source_shares
-- Purpose: User access permissions for known sources
CREATE TABLE IF NOT EXISTS source_shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES known_sources(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,  -- Keycloak user ID
    permission VARCHAR(20) NOT NULL CHECK (permission IN ('read', 'edit')),
    shared_by VARCHAR(255) NOT NULL,  -- User who created the share
    shared_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, user_id)
);

CREATE INDEX idx_source_shares_source ON source_shares(source_id);
CREATE INDEX idx_source_shares_user ON source_shares(user_id);
CREATE INDEX idx_source_shares_permission ON source_shares(permission);

COMMENT ON TABLE source_shares IS 'User access permissions for known radio sources';

-- Table: model_shares
-- Purpose: User access permissions for ML models
CREATE TABLE IF NOT EXISTS model_shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,  -- Keycloak user ID
    permission VARCHAR(20) NOT NULL CHECK (permission IN ('read', 'edit')),
    shared_by VARCHAR(255) NOT NULL,  -- User who created the share
    shared_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, user_id)
);

CREATE INDEX idx_model_shares_model ON model_shares(model_id);
CREATE INDEX idx_model_shares_user ON model_shares(user_id);
CREATE INDEX idx_model_shares_permission ON model_shares(permission);

COMMENT ON TABLE model_shares IS 'User access permissions for ML models';

-- ============================================================================
-- PART 2: ALTER EXISTING TABLES
-- ============================================================================

-- Add ownership to known_sources
ALTER TABLE known_sources 
    ADD COLUMN IF NOT EXISTS owner_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_known_sources_owner ON known_sources(owner_id);
CREATE INDEX IF NOT EXISTS idx_known_sources_public ON known_sources(is_public);

COMMENT ON COLUMN known_sources.owner_id IS 'Keycloak user ID of the source owner';
COMMENT ON COLUMN known_sources.is_public IS 'Whether the source is visible to all users';

-- Add ownership and description to models
ALTER TABLE models 
    ADD COLUMN IF NOT EXISTS owner_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS description TEXT;

CREATE INDEX IF NOT EXISTS idx_models_owner ON models(owner_id);

COMMENT ON COLUMN models.owner_id IS 'Keycloak user ID of the model owner';
COMMENT ON COLUMN models.description IS 'Human-readable description of the model';

-- Link recording sessions to constellations
ALTER TABLE recording_sessions 
    ADD COLUMN IF NOT EXISTS constellation_id UUID REFERENCES constellations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_recording_sessions_constellation ON recording_sessions(constellation_id);

COMMENT ON COLUMN recording_sessions.constellation_id IS 'Constellation used for this recording session';

-- ============================================================================
-- PART 3: HELPER FUNCTIONS
-- ============================================================================

-- Function: Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to constellations table
DROP TRIGGER IF EXISTS trigger_constellations_updated_at ON constellations;
CREATE TRIGGER trigger_constellations_updated_at
    BEFORE UPDATE ON constellations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PART 4: GRANTS
-- ============================================================================

-- Grant permissions to heimdall_user
GRANT ALL PRIVILEGES ON TABLE constellations TO heimdall_user;
GRANT ALL PRIVILEGES ON TABLE constellation_members TO heimdall_user;
GRANT ALL PRIVILEGES ON TABLE constellation_shares TO heimdall_user;
GRANT ALL PRIVILEGES ON TABLE source_shares TO heimdall_user;
GRANT ALL PRIVILEGES ON TABLE model_shares TO heimdall_user;

-- ============================================================================
-- PART 5: VALIDATION
-- ============================================================================

-- Verify all tables exist
DO $$
DECLARE
    missing_tables TEXT[];
BEGIN
    SELECT ARRAY_AGG(table_name)
    INTO missing_tables
    FROM (
        VALUES 
            ('constellations'),
            ('constellation_members'),
            ('constellation_shares'),
            ('source_shares'),
            ('model_shares')
    ) AS expected(table_name)
    WHERE NOT EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_schema = 'heimdall' 
        AND table_name = expected.table_name
    );

    IF missing_tables IS NOT NULL THEN
        RAISE EXCEPTION 'Migration failed: Missing tables: %', missing_tables;
    END IF;

    RAISE NOTICE 'Migration 04: All RBAC tables created successfully';
END $$;

-- Verify all columns added
DO $$
DECLARE
    missing_columns TEXT[];
BEGIN
    SELECT ARRAY_AGG(table_name || '.' || column_name)
    INTO missing_columns
    FROM (
        VALUES 
            ('known_sources', 'owner_id'),
            ('known_sources', 'is_public'),
            ('models', 'owner_id'),
            ('models', 'description'),
            ('recording_sessions', 'constellation_id')
    ) AS expected(table_name, column_name)
    WHERE NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = expected.table_name 
        AND column_name = expected.column_name
    );

    IF missing_columns IS NOT NULL THEN
        RAISE EXCEPTION 'Migration failed: Missing columns: %', missing_columns;
    END IF;

    RAISE NOTICE 'Migration 04: All RBAC columns added successfully';
END $$;

-- Summary
RAISE NOTICE '============================================';
RAISE NOTICE 'Migration 04: RBAC Schema - COMPLETED';
RAISE NOTICE '============================================';
RAISE NOTICE 'Created tables:';
RAISE NOTICE '  - constellations';
RAISE NOTICE '  - constellation_members';
RAISE NOTICE '  - constellation_shares';
RAISE NOTICE '  - source_shares';
RAISE NOTICE '  - model_shares';
RAISE NOTICE 'Modified tables:';
RAISE NOTICE '  - known_sources (added owner_id, is_public)';
RAISE NOTICE '  - models (added owner_id, description)';
RAISE NOTICE '  - recording_sessions (added constellation_id)';
RAISE NOTICE '============================================';
RAISE NOTICE 'Next: Run 05-migrate-existing-data.sql';
RAISE NOTICE '============================================';
