-- Add progress tracking fields for training jobs
-- These fields support both epoch-based (training) and sample-based (synthetic data) progress tracking

ALTER TABLE heimdall.training_jobs
ADD COLUMN IF NOT EXISTS current_progress INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_progress INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS progress_message TEXT;

-- Add comment to explain the fields
COMMENT ON COLUMN heimdall.training_jobs.current_progress IS 'Current progress: samples for synthetic generation, epochs for training';
COMMENT ON COLUMN heimdall.training_jobs.total_progress IS 'Total progress: total samples for synthetic generation, total epochs for training';
COMMENT ON COLUMN heimdall.training_jobs.progress_message IS 'Human-readable progress message from the task';

-- Create index for querying progress
CREATE INDEX IF NOT EXISTS idx_training_jobs_progress ON heimdall.training_jobs(status, current_progress, total_progress);
