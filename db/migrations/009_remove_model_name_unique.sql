-- Migration 009: Remove conflicting UNIQUE constraint on model_name
-- This constraint conflicts with models_name_version_unique which allows
-- multiple versions of the same model name

SET search_path TO heimdall, public;

-- Drop the old UNIQUE constraint on model_name only
-- Keep models_name_version_unique which allows versioning
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'models_model_name_key'
    ) THEN
        ALTER TABLE models DROP CONSTRAINT models_model_name_key;
        RAISE NOTICE 'Dropped models_model_name_key constraint to enable model versioning';
    END IF;
END $$;

-- Verify the correct constraint exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'models_name_version_unique'
    ) THEN
        RAISE EXCEPTION 'models_name_version_unique constraint missing! Database schema inconsistent.';
    END IF;
END $$;
