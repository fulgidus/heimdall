-- Migration: Add missing WebSDR columns
-- Description: Add retry_count, admin_email, location_description, and altitude_asl columns
--              to websdr_stations table
BEGIN;

-- Add missing columns to websdr_stations table if they don't exist
ALTER TABLE
    heimdall.websdr_stations
ADD
    COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 3,
ADD
    COLUMN IF NOT EXISTS admin_email VARCHAR(255),
ADD
    COLUMN IF NOT EXISTS location_description TEXT,
ADD
    COLUMN IF NOT EXISTS altitude_asl INTEGER;

-- Commit the transaction
COMMIT;

-- Verify the columns were added
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM
    information_schema.columns
WHERE
    table_schema = 'heimdall'
    AND table_name = 'websdr_stations'
ORDER BY
    ordinal_position;