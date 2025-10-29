-- Make frequency_hz, latitude, and longitude optional in known_sources table
-- Amateur radio stations may not have these details initially known

-- Make frequency_hz optional
ALTER TABLE heimdall.known_sources 
ALTER COLUMN frequency_hz DROP NOT NULL;

-- Make latitude optional
ALTER TABLE heimdall.known_sources 
ALTER COLUMN latitude DROP NOT NULL;

-- Make longitude optional
ALTER TABLE heimdall.known_sources 
ALTER COLUMN longitude DROP NOT NULL;

-- Add comments to clarify optional fields
COMMENT ON COLUMN heimdall.known_sources.frequency_hz IS 'Frequency in Hz (optional - may be unknown for amateur stations)';
COMMENT ON COLUMN heimdall.known_sources.latitude IS 'Latitude in degrees (optional - may be unknown initially)';
COMMENT ON COLUMN heimdall.known_sources.longitude IS 'Longitude in degrees (optional - may be unknown initially)';
