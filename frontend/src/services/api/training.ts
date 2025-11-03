/**
 * Training API service
 *
 * Provides functions for:
 * - Training job management
 * - Synthetic data generation
 * - Model management
 */

import api from '@/lib/api';

// ============================================================================
// TYPES
// ============================================================================

export interface BaseJob {
    id: string;
    job_name: string;
    status: 'pending' | 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
    created_at: string;
    started_at?: string;
    completed_at?: string;
    config: Record<string, any>;
    progress_percent: number;
    error_message?: string;
    message?: string;  // Progress message from task
}

export interface TrainingJob extends BaseJob {
    job_type: 'training';
    current_epoch: number;
    total_epochs: number;
    train_loss?: number;
    val_loss?: number;
    model_architecture?: string;
}

export interface SyntheticGenerationJobAPI extends BaseJob {
    job_type: 'synthetic_generation';
    current?: number;  // Current samples generated
    total?: number;    // Total samples requested
    dataset_id?: string;
}

export type AnyTrainingJob = TrainingJob | SyntheticGenerationJobAPI;

export interface SyntheticDataset {
    id: string;
    name: string;
    description?: string;
    num_samples: number;
    // NOTE: train_count, val_count, test_count removed - splits calculated at training time
    config: Record<string, any>;
    quality_metrics?: Record<string, any>;
    storage_table: string;
    created_at: string;
    created_by_job_id?: string;
}

export interface TrainedModel {
    id: string;
    model_name: string;
    version: number;
    model_type?: string;
    synthetic_dataset_id?: string;
    mlflow_run_id?: string;
    onnx_model_location?: string;
    pytorch_model_location?: string;
    accuracy_meters?: number;
    loss_value?: number;
    epoch?: number;
    is_active: boolean;
    is_production: boolean;
    hyperparameters?: Record<string, any>;
    training_metrics?: Record<string, any>;
    test_metrics?: Record<string, any>;
    created_at: string;
    trained_by_job_id?: string;
}

export interface SyntheticDataRequest {
    name: string;
    description?: string;
    num_samples: number;
    inside_ratio?: number;
    train_ratio?: number;
    val_ratio?: number;
    test_ratio?: number;
    frequency_mhz?: number;
    tx_power_dbm?: number;
    min_snr_db?: number;
    min_receivers?: number;
    max_gdop?: number;
}

// ============================================================================
// TRAINING JOBS API
// ============================================================================

export async function createTrainingJob(config: Record<string, any>): Promise<AnyTrainingJob> {
    const response = await api.post('/v1/training/jobs', config);
    return response.data;
}

export async function listTrainingJobs(
    status?: string,
    limit: number = 50,
    offset: number = 0
): Promise<{ jobs: AnyTrainingJob[]; total: number }> {
    const params: Record<string, string> = {
        limit: limit.toString(),
        offset: offset.toString(),
    };

    if (status) {
        params.status = status;
    }

    const response = await api.get('/v1/training/jobs', { params });
    return response.data;
}

export async function getTrainingJob(jobId: string): Promise<AnyTrainingJob> {
    const response = await api.get(`/v1/training/jobs/${jobId}`);
    return response.data;
}

export async function cancelTrainingJob(jobId: string): Promise<{ status: string; job_id: string }> {
    const response = await api.post(`/v1/training/jobs/${jobId}/cancel`);
    return response.data;
}

export async function pauseTrainingJob(jobId: string): Promise<{ status: string; job_id: string; message: string }> {
    const response = await api.post(`/v1/training/jobs/${jobId}/pause`);
    return response.data;
}

export async function resumeTrainingJob(jobId: string): Promise<{ status: string; job_id: string; celery_task_id: string; message: string }> {
    const response = await api.post(`/v1/training/jobs/${jobId}/resume`);
    return response.data;
}

export async function continueSyntheticJob(jobId: string): Promise<{
    job_id: string;
    parent_job_id: string;
    dataset_id: string;
    dataset_name: string;
    status: string;
    created_at: string;
    samples_existing: number;
    samples_remaining: number;
    total_samples: number;
    status_url: string;
    message: string;
}> {
    const response = await api.post(`/v1/training/jobs/${jobId}/continue`);
    return response.data;
}

export async function deleteTrainingJob(jobId: string): Promise<void> {
    await api.delete(`/v1/training/jobs/${jobId}`);
}

// ============================================================================
// SYNTHETIC DATA API
// ============================================================================

export async function generateSyntheticData(request: SyntheticDataRequest): Promise<{ job_id: string }> {
    const response = await api.post('/v1/training/synthetic/generate', request);
    return response.data;
}

export async function listSyntheticDatasets(
    limit: number = 50,
    offset: number = 0
): Promise<{ datasets: SyntheticDataset[]; total: number }> {
    const params = {
        limit: limit.toString(),
        offset: offset.toString(),
    };

    const response = await api.get('/v1/training/synthetic/datasets', { params });
    return response.data;
}

export async function getSyntheticDataset(datasetId: string): Promise<SyntheticDataset> {
    const response = await api.get(`/v1/training/synthetic/datasets/${datasetId}`);
    return response.data;
}

export async function deleteSyntheticDataset(datasetId: string): Promise<void> {
    await api.delete(`/v1/training/synthetic/datasets/${datasetId}`);
}

export interface SyntheticSampleResponse {
    id: number;
    timestamp: string;
    tx_lat: number;
    tx_lon: number;
    tx_power_dbm: number;
    frequency_hz: number;
    receivers: Record<string, any>;
    gdop: number;
    num_receivers: number;
    split: string;
    created_at: string;
}

export async function getSyntheticDatasetSamples(
    datasetId: string,
    limit: number = 10,
    offset: number = 0,
    split?: string
): Promise<{ samples: SyntheticSampleResponse[]; total: number; limit: number; offset: number; dataset_id: string }> {
    const params: Record<string, string> = {
        limit: limit.toString(),
        offset: offset.toString(),
    };

    if (split) {
        params.split = split;
    }

    const response = await api.get(`/v1/training/synthetic/datasets/${datasetId}/samples`, { params });
    return response.data;
}

// ============================================================================
// MODELS API
// ============================================================================

export async function listModels(
    activeOnly: boolean = false,
    limit: number = 50,
    offset: number = 0
): Promise<{ models: TrainedModel[]; total: number }> {
    const params = {
        limit: limit.toString(),
        offset: offset.toString(),
        active_only: activeOnly.toString(),
    };

    const response = await api.get('/v1/training/models', { params });
    return response.data;
}

export async function getModel(modelId: string): Promise<TrainedModel> {
    const response = await api.get(`/v1/training/models/${modelId}`);
    return response.data;
}

export async function deployModel(modelId: string, setProduction: boolean = false): Promise<{ status: string }> {
    const response = await api.post(`/v1/training/models/${modelId}/deploy`, null, {
        params: { set_production: setProduction }
    });
    return response.data;
}

export async function deleteModel(modelId: string): Promise<void> {
    await api.delete(`/v1/training/models/${modelId}`);
}
