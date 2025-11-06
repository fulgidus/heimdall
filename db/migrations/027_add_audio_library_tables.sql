-- Migration: Audio Library with Preprocessing Support
-- Add audio_library table and audio_chunks table for preprocessed audio chunks

-- Table: audio_library - Audio files metadata with processing status
CREATE TABLE IF NOT EXISTS heimdall.audio_library (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL CHECK (category IN ('voice', 'music', 'documentary', 'conference', 'custom')),
    tags TEXT[], -- Array of tags for filtering
    
    -- Audio file metadata
    file_size_bytes BIGINT NOT NULL,
    duration_seconds FLOAT NOT NULL,
    sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    audio_format VARCHAR(20) NOT NULL, -- 'wav', 'mp3', 'flac', 'ogg'
    
    -- Storage locations
    minio_bucket VARCHAR(100) NOT NULL DEFAULT 'heimdall-audio-library',
    minio_path VARCHAR(512) NOT NULL, -- Path in MinIO bucket
    
    -- Processing status
    processing_status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (processing_status IN ('PENDING', 'PROCESSING', 'READY', 'FAILED')),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_error TEXT, -- Error message if processing failed
    
    -- Chunk statistics (filled after preprocessing)
    total_chunks INTEGER DEFAULT 0, -- Number of 1-second chunks generated
    chunks_bucket VARCHAR(100) DEFAULT 'heimdall-audio-chunks',
    
    -- Status flags
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for audio_library
CREATE INDEX idx_audio_library_category ON heimdall.audio_library(category) WHERE enabled = TRUE;
CREATE INDEX idx_audio_library_enabled ON heimdall.audio_library(enabled);
CREATE INDEX idx_audio_library_status ON heimdall.audio_library(processing_status);
CREATE INDEX idx_audio_library_tags ON heimdall.audio_library USING GIN(tags);

-- Table: audio_chunks - Preprocessed 1-second audio chunks ready for training
CREATE TABLE IF NOT EXISTS heimdall.audio_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audio_id UUID NOT NULL REFERENCES heimdall.audio_library(id) ON DELETE CASCADE,
    
    -- Chunk identification
    chunk_index INTEGER NOT NULL, -- 0-based index (chunk N corresponds to second N)
    
    -- Chunk metadata
    duration_seconds FLOAT NOT NULL DEFAULT 1.0,
    sample_rate INTEGER NOT NULL DEFAULT 200000, -- Resampled to 200 kHz
    num_samples INTEGER NOT NULL, -- Number of samples in chunk (typically 200000)
    
    -- Storage location (MinIO)
    minio_bucket VARCHAR(100) NOT NULL DEFAULT 'heimdall-audio-chunks',
    minio_path VARCHAR(512) NOT NULL, -- e.g., 'audio_id/chunk_0000.npy'
    file_size_bytes INTEGER NOT NULL,
    
    -- Preprocessing metadata
    original_offset_seconds FLOAT NOT NULL, -- Original offset in source audio file
    rms_amplitude FLOAT, -- RMS amplitude for normalization tracking
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint: one chunk per (audio_id, chunk_index)
    UNIQUE(audio_id, chunk_index)
);

-- Indexes for audio_chunks
CREATE INDEX idx_audio_chunks_audio_id ON heimdall.audio_chunks(audio_id);
CREATE INDEX idx_audio_chunks_lookup ON heimdall.audio_chunks(audio_id, chunk_index);

-- Update trigger for audio_library.updated_at
CREATE OR REPLACE FUNCTION heimdall.update_audio_library_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audio_library_update_timestamp
    BEFORE UPDATE ON heimdall.audio_library
    FOR EACH ROW
    EXECUTE FUNCTION heimdall.update_audio_library_timestamp();

-- Comments
COMMENT ON TABLE heimdall.audio_library IS 'Audio files library with preprocessing pipeline support';
COMMENT ON TABLE heimdall.audio_chunks IS 'Preprocessed 1-second audio chunks ready for training (resampled to 200kHz)';
COMMENT ON COLUMN heimdall.audio_library.processing_status IS 'PENDING: awaiting preprocessing, PROCESSING: in progress, READY: chunks available, FAILED: error during preprocessing';
COMMENT ON COLUMN heimdall.audio_chunks.chunk_index IS 'Sequential chunk index (0 = first second, 1 = second second, etc.)';
