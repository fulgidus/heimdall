-- Migration: Add advanced training metrics for localization model
-- Date: 2025-11-04
-- Description: Add distance error metrics, uncertainty, GDOP, and health metrics

-- Add new columns to training_metrics table
ALTER TABLE heimdall.training_metrics
    ADD COLUMN IF NOT EXISTS train_rmse_m FLOAT,
    ADD COLUMN IF NOT EXISTS val_rmse_m FLOAT,
    ADD COLUMN IF NOT EXISTS val_rmse_good_geom_m FLOAT,
    ADD COLUMN IF NOT EXISTS val_distance_p50_m FLOAT,
    ADD COLUMN IF NOT EXISTS val_distance_p68_m FLOAT,
    ADD COLUMN IF NOT EXISTS val_distance_p95_m FLOAT,
    ADD COLUMN IF NOT EXISTS mean_predicted_uncertainty_m FLOAT,
    ADD COLUMN IF NOT EXISTS uncertainty_calibration_error FLOAT,
    ADD COLUMN IF NOT EXISTS mean_gdop FLOAT,
    ADD COLUMN IF NOT EXISTS gdop_below_5_percent FLOAT,
    ADD COLUMN IF NOT EXISTS weight_norm FLOAT,
    ADD COLUMN IF NOT EXISTS batch_processing_time_ms FLOAT;

-- Create index for efficient querying of distance metrics
CREATE INDEX IF NOT EXISTS idx_training_metrics_val_rmse 
    ON heimdall.training_metrics(training_job_id, val_rmse_m);

CREATE INDEX IF NOT EXISTS idx_training_metrics_p68 
    ON heimdall.training_metrics(training_job_id, val_distance_p68_m);

-- Comment on new columns
COMMENT ON COLUMN heimdall.training_metrics.train_rmse_m IS 'Training set RMSE distance error in meters (SI unit)';
COMMENT ON COLUMN heimdall.training_metrics.val_rmse_m IS 'Validation set RMSE distance error in meters (SI unit)';
COMMENT ON COLUMN heimdall.training_metrics.val_rmse_good_geom_m IS 'Validation RMSE for samples with GDOP<5 (good geometry) in meters';
COMMENT ON COLUMN heimdall.training_metrics.val_distance_p50_m IS 'Validation 50th percentile (median) distance error in meters';
COMMENT ON COLUMN heimdall.training_metrics.val_distance_p68_m IS 'Validation 68th percentile distance error in meters (project KPI)';
COMMENT ON COLUMN heimdall.training_metrics.val_distance_p95_m IS 'Validation 95th percentile distance error in meters (worst-case)';
COMMENT ON COLUMN heimdall.training_metrics.mean_predicted_uncertainty_m IS 'Mean predicted uncertainty (std dev) from model in meters';
COMMENT ON COLUMN heimdall.training_metrics.uncertainty_calibration_error IS 'Difference between predicted uncertainty and actual error';
COMMENT ON COLUMN heimdall.training_metrics.mean_gdop IS 'Mean GDOP across validation set';
COMMENT ON COLUMN heimdall.training_metrics.gdop_below_5_percent IS 'Percentage of validation samples with GDOP<5';
COMMENT ON COLUMN heimdall.training_metrics.weight_norm IS 'L2 norm of model weights';
COMMENT ON COLUMN heimdall.training_metrics.batch_processing_time_ms IS 'Average batch processing time in milliseconds';
