-- Migration: Add dataset_id column to training_jobs table
-- Date: 2025-11-07
-- Description: Link training jobs to synthetic datasets for proper frontend display

-- Add dataset_id column to training_jobs table
ALTER TABLE heimdall.training_jobs 
ADD COLUMN IF NOT EXISTS dataset_id UUID 
REFERENCES heimdall.synthetic_datasets(id) ON DELETE SET NULL;

-- Create index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_training_jobs_dataset_id 
ON heimdall.training_jobs(dataset_id);

-- Backfill existing synthetic generation jobs with their dataset_id
-- This links jobs to datasets that were created by them
UPDATE heimdall.training_jobs tj
SET dataset_id = sd.id
FROM heimdall.synthetic_datasets sd
WHERE sd.created_by_job_id = tj.id
  AND tj.job_type = 'synthetic_generation'
  AND tj.dataset_id IS NULL;

-- Add comment for documentation
COMMENT ON COLUMN heimdall.training_jobs.dataset_id IS 
'UUID of the synthetic dataset created by this job (for synthetic_generation jobs)';
