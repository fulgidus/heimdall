-- Migration: Add health tracking columns to synthetic_datasets table
-- Date: 2025-11-08
-- Description: Add columns for dataset integrity validation and repair tracking

-- Add health tracking columns to synthetic_datasets table
ALTER TABLE heimdall.synthetic_datasets 
ADD COLUMN IF NOT EXISTS health_status VARCHAR(20) DEFAULT 'unknown',
ADD COLUMN IF NOT EXISTS last_validated_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS validation_issues JSONB;

-- Create index for efficient health status queries
CREATE INDEX IF NOT EXISTS idx_synthetic_datasets_health_status 
ON heimdall.synthetic_datasets(health_status);

-- Create index for validation timestamp queries
CREATE INDEX IF NOT EXISTS idx_synthetic_datasets_last_validated 
ON heimdall.synthetic_datasets(last_validated_at);

-- Add check constraint for valid health status values
ALTER TABLE heimdall.synthetic_datasets 
ADD CONSTRAINT check_health_status 
CHECK (health_status IN ('unknown', 'healthy', 'warning', 'critical'));

-- Add comments for documentation
COMMENT ON COLUMN heimdall.synthetic_datasets.health_status IS 
'Dataset integrity status: unknown (not validated), healthy (<5% issues), warning (5-10% issues), critical (>10% issues)';

COMMENT ON COLUMN heimdall.synthetic_datasets.last_validated_at IS 
'Timestamp of the last dataset integrity validation';

COMMENT ON COLUMN heimdall.synthetic_datasets.validation_issues IS 
'JSON object containing validation results: {orphaned_iq_files: int, orphaned_features: int, total_issues: int}';
