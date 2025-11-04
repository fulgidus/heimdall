-- Migration: Rename metrics columns from _km to _m suffix
-- Date: 2025-11-04
-- Description: Standardize all distance metrics to SI units (meters)
-- The columns were originally named with _km suffix but now store meters
-- This migration renames them to accurately reflect their unit
--
-- NOTE: This migration is only needed for databases that ran the original
-- version of migration 014 (which used _km suffix). Fresh installations
-- running the updated migration 014 will have correct _m suffixes already
-- and this migration will effectively be a no-op.

-- Rename columns in training_metrics table
ALTER TABLE heimdall.training_metrics
    RENAME COLUMN train_rmse_km TO train_rmse_m;

ALTER TABLE heimdall.training_metrics
    RENAME COLUMN val_rmse_km TO val_rmse_m;

ALTER TABLE heimdall.training_metrics
    RENAME COLUMN val_rmse_good_geom_km TO val_rmse_good_geom_m;

ALTER TABLE heimdall.training_metrics
    RENAME COLUMN val_distance_p50_km TO val_distance_p50_m;

ALTER TABLE heimdall.training_metrics
    RENAME COLUMN val_distance_p68_km TO val_distance_p68_m;

ALTER TABLE heimdall.training_metrics
    RENAME COLUMN val_distance_p95_km TO val_distance_p95_m;

ALTER TABLE heimdall.training_metrics
    RENAME COLUMN mean_predicted_uncertainty_km TO mean_predicted_uncertainty_m;

-- Drop old indexes
DROP INDEX IF EXISTS heimdall.idx_training_metrics_val_rmse;
DROP INDEX IF EXISTS heimdall.idx_training_metrics_p68;

-- Recreate indexes with new column names
CREATE INDEX IF NOT EXISTS idx_training_metrics_val_rmse 
    ON heimdall.training_metrics(training_job_id, val_rmse_m);

CREATE INDEX IF NOT EXISTS idx_training_metrics_p68 
    ON heimdall.training_metrics(training_job_id, val_distance_p68_m);

-- Update column comments to reflect meters (SI unit)
COMMENT ON COLUMN heimdall.training_metrics.train_rmse_m IS 'Training set RMSE distance error in meters (SI unit)';
COMMENT ON COLUMN heimdall.training_metrics.val_rmse_m IS 'Validation set RMSE distance error in meters (SI unit)';
COMMENT ON COLUMN heimdall.training_metrics.val_rmse_good_geom_m IS 'Validation RMSE for samples with GDOP<5 (good geometry) in meters';
COMMENT ON COLUMN heimdall.training_metrics.val_distance_p50_m IS 'Validation 50th percentile (median) distance error in meters';
COMMENT ON COLUMN heimdall.training_metrics.val_distance_p68_m IS 'Validation 68th percentile distance error in meters (project KPI)';
COMMENT ON COLUMN heimdall.training_metrics.val_distance_p95_m IS 'Validation 95th percentile distance error in meters (worst-case)';
COMMENT ON COLUMN heimdall.training_metrics.mean_predicted_uncertainty_m IS 'Mean predicted uncertainty (std dev) from model in meters';
