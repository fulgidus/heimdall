-- User Settings Table
-- Stores user preferences and application settings

CREATE TABLE IF NOT EXISTS heimdall.user_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL UNIQUE, -- Keycloak user ID (sub claim)
    
    -- General settings
    theme VARCHAR(50) DEFAULT 'dark',
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(100) DEFAULT 'UTC',
    auto_refresh BOOLEAN DEFAULT TRUE,
    refresh_interval INTEGER DEFAULT 30,
    
    -- API settings
    api_timeout INTEGER DEFAULT 30000,
    retry_attempts INTEGER DEFAULT 3,
    enable_caching BOOLEAN DEFAULT TRUE,
    
    -- Notification settings
    email_notifications BOOLEAN DEFAULT TRUE,
    system_alerts BOOLEAN DEFAULT TRUE,
    performance_warnings BOOLEAN DEFAULT TRUE,
    webhook_url VARCHAR(512),
    
    -- Advanced settings
    debug_mode BOOLEAN DEFAULT FALSE,
    log_level VARCHAR(20) DEFAULT 'info',
    max_concurrent_requests INTEGER DEFAULT 5,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on user_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_user_settings_user_id ON heimdall.user_settings(user_id);

-- Add update trigger to automatically update updated_at
CREATE OR REPLACE FUNCTION heimdall.update_user_settings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_user_settings_updated_at
    BEFORE UPDATE ON heimdall.user_settings
    FOR EACH ROW
    EXECUTE FUNCTION heimdall.update_user_settings_updated_at();
