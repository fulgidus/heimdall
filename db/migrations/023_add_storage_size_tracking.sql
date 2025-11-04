-- Migration 023: Add Storage Size Tracking for Synthetic Datasets
-- Adds storage_size_bytes field to track total disk usage (PostgreSQL + MinIO)
-- Important for IQ-raw datasets which can be 100x larger than feature-based datasets

-- Set search path
SET search_path TO heimdall, public;

-- ============================================================================
-- ADD STORAGE_SIZE_BYTES TO SYNTHETIC_DATASETS
-- ============================================================================
-- Track total storage used by dataset (PostgreSQL tables + MinIO objects)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'synthetic_datasets' 
        AND column_name = 'storage_size_bytes'
    ) THEN
        ALTER TABLE synthetic_datasets 
        ADD COLUMN storage_size_bytes BIGINT DEFAULT NULL;
        
        COMMENT ON COLUMN synthetic_datasets.storage_size_bytes IS 
            'Total storage size in bytes (PostgreSQL + MinIO). NULL means not yet calculated.';
    END IF;
END$$;

-- Add index for filtering/sorting by storage size
CREATE INDEX IF NOT EXISTS idx_synthetic_datasets_storage_size 
ON synthetic_datasets(storage_size_bytes DESC NULLS LAST);

-- ============================================================================
-- HELPER FUNCTION: Calculate Dataset Storage Size
-- ============================================================================
-- Function to calculate storage size for a dataset (PostgreSQL only)
-- MinIO sizes must be calculated by application code

CREATE OR REPLACE FUNCTION heimdall.calculate_dataset_storage_size(
    p_dataset_id UUID
) RETURNS BIGINT AS $$
DECLARE
    v_storage_table TEXT;
    v_pg_size BIGINT := 0;
    v_iq_table_size BIGINT := 0;
BEGIN
    -- Get storage table name
    SELECT storage_table INTO v_storage_table
    FROM heimdall.synthetic_datasets
    WHERE id = p_dataset_id;
    
    IF v_storage_table IS NULL THEN
        RETURN 0;
    END IF;
    
    -- Calculate size of feature storage table (includes indexes)
    BEGIN
        EXECUTE format('SELECT pg_total_relation_size(%L)', 'heimdall.' || v_storage_table)
        INTO v_pg_size;
    EXCEPTION WHEN OTHERS THEN
        v_pg_size := 0;
    END;
    
    -- Calculate size of IQ samples table for this dataset (if iq_raw type)
    SELECT COALESCE(
        pg_total_relation_size('heimdall.synthetic_iq_samples'),
        0
    ) INTO v_iq_table_size
    FROM heimdall.synthetic_datasets
    WHERE id = p_dataset_id AND dataset_type = 'iq_raw';
    
    -- Note: This does NOT include MinIO object storage
    -- Application must add MinIO sizes separately
    RETURN COALESCE(v_pg_size, 0) + COALESCE(v_iq_table_size, 0);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION heimdall.calculate_dataset_storage_size(UUID) IS 
    'Calculate PostgreSQL storage size for dataset (excludes MinIO). Returns bytes.';

-- ============================================================================
-- HELPER VIEW: Dataset Storage Summary
-- ============================================================================
CREATE OR REPLACE VIEW v_dataset_storage_summary AS
SELECT 
    sd.id,
    sd.name,
    sd.dataset_type,
    sd.num_samples,
    sd.storage_size_bytes,
    -- Format as human-readable
    CASE 
        WHEN sd.storage_size_bytes IS NULL THEN 'Not calculated'
        WHEN sd.storage_size_bytes < 1024 THEN sd.storage_size_bytes || ' B'
        WHEN sd.storage_size_bytes < 1048576 THEN ROUND(sd.storage_size_bytes / 1024.0, 2) || ' KB'
        WHEN sd.storage_size_bytes < 1073741824 THEN ROUND(sd.storage_size_bytes / 1048576.0, 2) || ' MB'
        ELSE ROUND(sd.storage_size_bytes / 1073741824.0, 2) || ' GB'
    END AS storage_size_human,
    -- Per-sample storage cost
    CASE 
        WHEN sd.num_samples > 0 AND sd.storage_size_bytes IS NOT NULL 
        THEN ROUND(sd.storage_size_bytes::NUMERIC / sd.num_samples, 0)
        ELSE NULL
    END AS bytes_per_sample,
    sd.created_at
FROM synthetic_datasets sd
ORDER BY sd.storage_size_bytes DESC NULLS LAST;

COMMENT ON VIEW v_dataset_storage_summary IS 
    'Human-readable storage size summary for all datasets';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- This migration adds:
-- 1. storage_size_bytes field to synthetic_datasets table
-- 2. Helper function to calculate PostgreSQL storage size
-- 3. View for human-readable storage summary
--
-- Note: Application code must:
-- - Calculate MinIO object sizes separately
-- - Update storage_size_bytes periodically (especially for iq_raw datasets)
-- - Sum PostgreSQL + MinIO sizes for total storage
