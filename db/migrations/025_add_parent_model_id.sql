-- Migration 025: Add parent_model_id to models table for tracking model evolution
-- This enables proper model lineage tracking when evolving from parent models

SET search_path TO heimdall, public;

-- Add parent_model_id column to models table
ALTER TABLE models 
ADD COLUMN IF NOT EXISTS parent_model_id UUID;

-- Add foreign key constraint to reference parent model
ALTER TABLE models
ADD CONSTRAINT models_parent_model_id_fkey 
FOREIGN KEY (parent_model_id) REFERENCES models(id) ON DELETE SET NULL;

-- Add index for efficient parent model lookups
CREATE INDEX IF NOT EXISTS idx_models_parent_model 
ON models(parent_model_id) WHERE parent_model_id IS NOT NULL;

-- Add comment explaining the column
COMMENT ON COLUMN models.parent_model_id IS 'Reference to parent model when this model was created via evolution/transfer learning';
