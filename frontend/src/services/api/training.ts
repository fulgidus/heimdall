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

export interface TrainingJob {
    id: string;
    job_name: string;
    status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
    created_at: string;
    started_at?: string;
    completed_at?: string;
    config: Record<string, any>;
    current_epoch: number;
    total_epochs: number;
    progress_percent: number;
    train_loss?: number;
    val_loss?: number;
    error_message?: string;
    model_architecture?: string;
    current?: number;  // For synthetic data generation: current samples
    total?: number;    // For synthetic data generation: total samples
    message?: string;  // Progress message from task
}

export interface SyntheticDataset {
    id: string;
    name: string;
    description?: string;
    num_samples: number;
    train_count?: number;
    val_count?: number;
    test_count?: number;
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

export async function createTrainingJob(config: Record<string, any>): Promise<TrainingJob> {
    const response = await api.post('/v1/training/jobs', config);
    return response.data;
}

export async function listTrainingJobs(
    status?: string,
    limit: number = 50,
    offset: number = 0
): Promise<{ jobs: TrainingJob[]; total: number }> {
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

export async function getTrainingJob(jobId: string): Promise<TrainingJob> {
    const response = await api.get(`/v1/training/jobs/${jobId}`);
    return response.data;
}

export async function cancelTrainingJob(jobId: string): Promise<{ status: string; job_id: string }> {
    const response = await api.post(`/v1/training/jobs/${jobId}/cancel`);
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
