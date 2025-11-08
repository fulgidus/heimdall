-- Add new recording session states and create "Unknown" source

-- Update recording session status constraint to include new states
ALTER TABLE heimdall.recording_sessions 
DROP CONSTRAINT IF EXISTS valid_status;

ALTER TABLE heimdall.recording_sessions 
ADD CONSTRAINT valid_status CHECK (
    status IN ('pending', 'recording', 'source_assigned', 'in_progress', 'completed', 'failed')
);

-- Insert "Unknown" source if it doesn't exist
INSERT INTO heimdall.known_sources 
(name, description, is_validated, error_margin_meters)
VALUES ('Unknown', 'Placeholder for unknown or unidentified sources', false, 1000.0)
ON CONFLICT (name) DO NOTHING;

-- Add comments to clarify new statuses
COMMENT ON COLUMN heimdall.recording_sessions.status IS 
'Session status: pending (initial), recording (capturing data), source_assigned (source identified), in_progress (processing), completed (finished), failed (error occurred)';
