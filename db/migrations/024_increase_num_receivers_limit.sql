-- Migration 024: Increase num_receivers_detected constraint for IQ-raw datasets
-- IQ-raw datasets with random receiver geometry can have 5-10+ receivers per sample
-- (vs fixed 7 receivers for feature_based datasets)

SET search_path TO heimdall, public;

-- Drop old constraint (max 7)
ALTER TABLE measurement_features 
DROP CONSTRAINT IF EXISTS measurement_features_num_receivers_detected_check;

-- Add new constraint with higher limit (max 15)
ALTER TABLE measurement_features 
ADD CONSTRAINT measurement_features_num_receivers_detected_check 
CHECK (num_receivers_detected >= 0 AND num_receivers_detected <= 15);

-- Add comment explaining the change
COMMENT ON CONSTRAINT measurement_features_num_receivers_detected_check ON measurement_features IS
    'Allow up to 15 receivers for iq_raw datasets with random receiver geometry (5-10 typical, vs fixed 7 for feature_based)';
