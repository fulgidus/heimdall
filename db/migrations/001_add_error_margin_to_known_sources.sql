-- Migration: Add error_margin_meters column to known_sources table
-- Date: 2025-10-30
-- Description: Add optional error_margin_meters column to track localization accuracy requirements
ALTER TABLE
    heimdall.known_sources
ADD
    COLUMN error_margin_meters FLOAT DEFAULT 50.0 CHECK (error_margin_meters > 0);

-- Create index for efficient queries
CREATE INDEX idx_known_sources_error_margin ON heimdall.known_sources(error_margin_meters);

-- Update schema comment
COMMENT ON COLUMN heimdall.known_sources.error_margin_meters IS 'Expected localization error radius in meters (default: 50m)';