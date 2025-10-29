-- Add error_margin_meters to known_sources table
-- This field represents the uncertainty radius around the source location

ALTER TABLE heimdall.known_sources 
ADD COLUMN IF NOT EXISTS error_margin_meters FLOAT DEFAULT 50.0 CHECK (error_margin_meters > 0);

COMMENT ON COLUMN heimdall.known_sources.error_margin_meters IS 'Error margin radius in meters (visualized as circle on map)';
