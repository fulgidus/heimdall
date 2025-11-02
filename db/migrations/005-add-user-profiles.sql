-- Migration 005: Add user profiles table
-- Stores extended user profile information not managed by Keycloak

CREATE TABLE IF NOT EXISTS heimdall.user_profiles (
    user_id VARCHAR(255) PRIMARY KEY,  -- Keycloak user ID (sub claim from JWT)
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    phone VARCHAR(50),
    organization VARCHAR(255),
    location VARCHAR(255),
    bio TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_updated_at ON heimdall.user_profiles(updated_at DESC);

-- Add comment
COMMENT ON TABLE heimdall.user_profiles IS 'Extended user profile information complementing Keycloak user data';
