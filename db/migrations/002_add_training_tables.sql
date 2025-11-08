-- Migration: Add training_jobs and training_metrics tables
-- Date: 2025-11-01
-- Description: Add tables for AI training job management with WebSocket monitoring

-- Table: training_jobs
CREATE TABLE IF NOT EXISTS heimdall.training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255) NOT NULL,
    celery_task_id VARCHAR(255) UNIQUE,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Training configuration
    config JSONB NOT NULL,

    -- Progress tracking
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,
    progress_percent FLOAT DEFAULT 0.0,

    -- Metrics
    train_loss FLOAT,
    val_loss FLOAT,
    train_accuracy FLOAT,
    val_accuracy FLOAT,
    learning_rate FLOAT,

    -- Best model info
    best_epoch INTEGER,
    best_val_loss FLOAT,

    -- Artifacts
    checkpoint_path VARCHAR(512),
    onnx_model_path VARCHAR(512),
    mlflow_run_id VARCHAR(255),

    -- Error handling
    error_message TEXT,

    -- Metadata
    dataset_size INTEGER,
    train_samples INTEGER,
    val_samples INTEGER,
    model_architecture VARCHAR(100),

    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON heimdall.training_jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_celery_task ON heimdall.training_jobs(celery_task_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created ON heimdall.training_jobs(created_at DESC);

-- Table: training_metrics (time-series data for detailed tracking)
CREATE TABLE IF NOT EXISTS heimdall.training_metrics (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    training_job_id UUID REFERENCES heimdall.training_jobs(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    batch INTEGER,

    -- Loss values
    train_loss FLOAT,
    val_loss FLOAT,

    -- Accuracy values
    train_accuracy FLOAT,
    val_accuracy FLOAT,

    -- Learning rate
    learning_rate FLOAT,

    -- Additional metrics
    gradient_norm FLOAT,

    -- Metadata
    phase VARCHAR(20), -- 'train', 'val', 'test'

    PRIMARY KEY (timestamp, id)
);

-- Convert to hypertable for efficient time-series queries
SELECT create_hypertable(
    'heimdall.training_metrics',
    'timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_training_metrics_job ON heimdall.training_metrics(training_job_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch ON heimdall.training_metrics(training_job_id, epoch);

-- Grant permissions
GRANT ALL PRIVILEGES ON heimdall.training_jobs TO heimdall_user;
GRANT ALL PRIVILEGES ON heimdall.training_metrics TO heimdall_user;
