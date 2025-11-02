-- Migration: Add pause/resume functionality to training jobs
-- Date: 2025-11-02
-- Description: Add 'paused' status and pause_checkpoint_path column for training job pause/resume

-- Add pause_checkpoint_path column to store pause checkpoint location
ALTER TABLE heimdall.training_jobs
ADD COLUMN IF NOT EXISTS pause_checkpoint_path VARCHAR(512);

-- Add comment to explain the field
COMMENT ON COLUMN heimdall.training_jobs.pause_checkpoint_path IS 'S3 path to the pause checkpoint (separate from best model checkpoint)';

-- Update status check constraint to include 'paused'
-- Drop existing constraint if it exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'valid_status' AND table_name = 'training_jobs'
    ) THEN
        ALTER TABLE heimdall.training_jobs DROP CONSTRAINT valid_status;
    END IF;
END
$$;

-- Add new constraint including 'paused'
ALTER TABLE heimdall.training_jobs
ADD CONSTRAINT valid_status CHECK (
    status IN ('pending', 'queued', 'running', 'paused', 'completed', 'failed', 'cancelled')
);

-- Create index for querying paused jobs
CREATE INDEX IF NOT EXISTS idx_training_jobs_paused ON heimdall.training_jobs(status) WHERE status = 'paused';
