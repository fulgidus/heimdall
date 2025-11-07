-- Data Migration: Fix job_type for existing training_jobs
-- This updates rows where job_type is NULL by inferring the type from config

SET search_path TO heimdall, public;

-- Update job_type based on config content
UPDATE training_jobs 
SET job_type = CASE 
    -- Synthetic generation jobs have expand_dataset_id or num_samples
    WHEN config->>'expand_dataset_id' IS NOT NULL THEN 'synthetic_generation'
    WHEN config->>'num_samples' IS NOT NULL THEN 'synthetic_generation'
    -- Training jobs have epochs, dataset_ids, and model_architecture
    WHEN config->>'epochs' IS NOT NULL THEN 'training'
    WHEN config->>'dataset_ids' IS NOT NULL THEN 'training'
    -- Model export jobs (if any)
    WHEN config->>'model_id' IS NOT NULL AND config->>'export_path' IS NOT NULL THEN 'model_export'
    -- Default to training for ambiguous cases
    ELSE 'training'
END
WHERE job_type IS NULL;

-- Verify the update
DO $$
DECLARE
    updated_count INTEGER;
    null_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO updated_count FROM training_jobs WHERE job_type IS NOT NULL;
    SELECT COUNT(*) INTO null_count FROM training_jobs WHERE job_type IS NULL;
    
    RAISE NOTICE 'Migration complete: % jobs have job_type set, % jobs still have NULL job_type', updated_count, null_count;
END $$;
