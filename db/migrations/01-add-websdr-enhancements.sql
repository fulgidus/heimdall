-- Migration: Add retry_count and admin_email to websdr_stations
-- Add sdr_profiles table for tracking supported frequencies from health-check
-- Date: 2025-10-27
SET
    search_path TO heimdall,
    public;

-- Add new columns to websdr_stations
ALTER TABLE
    websdr_stations
ADD
    COLUMN IF NOT EXISTS retry_count INT DEFAULT 3,
ADD
    COLUMN IF NOT EXISTS admin_email VARCHAR(255),
ADD
    COLUMN IF NOT EXISTS location_description TEXT,
ADD
    COLUMN IF NOT EXISTS altitude_asl INT;

-- Create sdr_profiles table for storing SDR capabilities from health-check
CREATE TABLE IF NOT EXISTS sdr_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    websdr_station_id UUID NOT NULL REFERENCES websdr_stations(id) ON DELETE CASCADE,
    sdr_name VARCHAR(50) NOT NULL,
    -- e.g., "A)", "B)", "C)"
    sdr_type VARCHAR(100),
    -- e.g., "RtlSdrSource"
    profile_name VARCHAR(255) NOT NULL,
    -- e.g., "2m [144.00-146.00 Mhz]"
    center_freq_hz BIGINT NOT NULL,
    sample_rate_hz INT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX idx_sdr_profiles_websdr_station ON sdr_profiles(websdr_station_id);

CREATE INDEX idx_sdr_profiles_frequency ON sdr_profiles(center_freq_hz);

CREATE INDEX idx_sdr_profiles_active ON sdr_profiles(websdr_station_id, is_active);

COMMENT ON TABLE sdr_profiles IS 'SDR receiver profiles with supported frequencies (populated from health-check JSON)';

COMMENT ON COLUMN websdr_stations.retry_count IS 'Number of retry attempts for failed connections';

COMMENT ON COLUMN websdr_stations.admin_email IS 'Administrator contact email from health-check';

COMMENT ON COLUMN websdr_stations.location_description IS 'Full location description from health-check';

COMMENT ON COLUMN websdr_stations.altitude_asl IS 'Altitude above sea level in meters';