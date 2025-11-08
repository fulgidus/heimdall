-- Migration: Add synthetic job continuation support
-- Date: 2025-11-02
-- Description: Add parent_job_id to track continuation chains for cancelled synthetic jobs

-- Add parent_job_id column to link continuation jobs
ALTER TABLE heimdall.training_jobs
ADD COLUMN IF NOT EXISTS parent_job_id UUID;

-- Add foreign key constraint
ALTER TABLE heimdall.training_jobs
ADD CONSTRAINT fk_parent_job 
    FOREIGN KEY (parent_job_id) 
    REFERENCES heimdall.training_jobs(id) 
    ON DELETE SET NULL;

-- Create index for efficient continuation lookups
CREATE INDEX IF NOT EXISTS idx_training_jobs_parent 
    ON heimdall.training_jobs(parent_job_id);

-- Add comment explaining the field
COMMENT ON COLUMN heimdall.training_jobs.parent_job_id IS 
    'Reference to parent job if this is a continuation of a cancelled synthetic job. Allows tracking continuation chains.';

-- Create index for finding cancelled synthetic jobs with progress
CREATE INDEX IF NOT EXISTS idx_training_jobs_cancelled_synthetic 
    ON heimdall.training_jobs(job_type, status, current_progress) 
    WHERE status = 'cancelled' AND job_type = 'synthetic_generation' AND current_progress > 0;
