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
    frequency_mhz: number;  // Required by backend
    tx_power_dbm: number;   // Required by backend
    min_snr_db: number;     // Required by backend
    min_receivers: number;  // Required by backend
    max_gdop?: number;
    dataset_type?: string;
    use_random_receivers?: boolean;
    use_srtm_terrain?: boolean;
    use_gpu?: boolean | null;
    seed?: number;
    tx_antenna_dist?: {
        whip: number;
        rubber_duck: number;
        portable_directional: number;
    };
    rx_antenna_dist?: {
        omni_vertical: number;
        yagi: number;
        collinear: number;
    };
}

// ============================================================================
// TRAINING JOBS API
// ============================================================================

export async function createTrainingJob(config: Record<string, any>): Promise<AnyTrainingJob> {
    const response = await api.post('/v1/jobs/training', config);
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

    const response = await api.get('/v1/jobs/training', { params });
    return response.data;
}

export async function getTrainingJob(jobId: string): Promise<AnyTrainingJob> {
    const response = await api.get(`/v1/jobs/training/${jobId}`);
    return response.data;
}

export async function cancelTrainingJob(jobId: string): Promise<{ status: string; job_id: string }> {
    const response = await api.post(`/v1/jobs/training/${jobId}/cancel`);
    return response.data;
}

export async function pauseTrainingJob(jobId: string): Promise<{ status: string; job_id: string; message: string }> {
    const response = await api.post(`/v1/jobs/training/${jobId}/pause`);
    return response.data;
}

export async function resumeTrainingJob(jobId: string): Promise<{ status: string; job_id: string; celery_task_id: string; message: string }> {
    const response = await api.post(`/v1/jobs/training/${jobId}/resume`);
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
    const response = await api.post(`/v1/jobs/training/${jobId}/continue`);
    return response.data;
}

export async function deleteTrainingJob(jobId: string): Promise<void> {
    await api.delete(`/v1/jobs/training/${jobId}`);
}

// ============================================================================
// SYNTHETIC DATA API
// ============================================================================

export async function generateSyntheticData(request: SyntheticDataRequest): Promise<{ job_id: string }> {
    const response = await api.post('/v1/jobs/synthetic/generate', request);
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

    const response = await api.get('/v1/jobs/synthetic/datasets', { params });
    return response.data;
}

export async function getSyntheticDataset(datasetId: string): Promise<SyntheticDataset> {
    const response = await api.get(`/v1/jobs/synthetic/datasets/${datasetId}`);
    return response.data;
}

export async function deleteSyntheticDataset(datasetId: string): Promise<void> {
    await api.delete(`/v1/jobs/synthetic/datasets/${datasetId}`);
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

    const response = await api.get(`/v1/jobs/synthetic/datasets/${datasetId}/samples`, { params });
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

    const response = await api.get('/v1/models', { params });
    return response.data;
}

export async function getModel(modelId: string): Promise<TrainedModel> {
    const response = await api.get(`/v1/models/${modelId}`);
    return response.data;
}

export async function deployModel(modelId: string, setProduction: boolean = false): Promise<{ status: string }> {
    const response = await api.post(`/v1/models/${modelId}/deploy`, null, {
        params: { set_production: setProduction }
    });
    return response.data;
}

export async function deleteModel(modelId: string): Promise<void> {
    await api.delete(`/v1/models/${modelId}`);
}

// ============================================================================
// MODEL ARCHITECTURES API
// ============================================================================

export type ModelCategory = 'spectrogram' | 'iq_raw_cnn' | 'transformer' | 'temporal' | 'hybrid' | 'features';
export type ComplexityLevel = 'low' | 'medium' | 'high' | 'very_high';
export type SpeedRating = 'very_fast' | 'fast' | 'moderate' | 'slow';
export type QualityRating = 'excellent' | 'good' | 'fair' | 'experimental';
export type DataType = 'spectrogram' | 'iq_raw' | 'hybrid' | 'features' | 'feature_based';

export interface ModelArchitecturePerformance {
    accuracy_meters_mean: number;
    accuracy_meters_min: number;
    accuracy_meters_max: number;
    inference_ms_mean: number;
    inference_ms_min: number;
    inference_ms_max: number;
}

export interface ModelArchitectureMetadata {
    description: string;
    input_format: string;
    typical_use_cases: string[];
    advantages: string[];
    disadvantages: string[];
    recommended_for: string[];
    not_recommended_for: string[];
    training_time_estimate: string;
    dataset_size_recommendation: string;
}

export interface ModelArchitectureStarRatings {
    accuracy: number;  // 1-5
    speed: number;     // 1-5
    efficiency: number; // 1-5
}

export interface ModelArchitectureBadges {
    recommended?: boolean;
    maximum_accuracy?: boolean;
    fastest?: boolean;
    best_ratio?: boolean;
    experimental?: boolean;
    production_ready?: boolean;
    memory_efficient?: boolean;
    gpu_optimized?: boolean;
}

export interface ModelArchitecture {
    id: string;
    display_name: string;
    category: ModelCategory;
    emoji: string;
    class_name: string;
    data_type: DataType;
    parameters_millions: number;
    model_size_mb: number;
    complexity: ComplexityLevel;
    speed_rating: SpeedRating;
    quality_rating: QualityRating;
    performance: ModelArchitecturePerformance;
    metadata: ModelArchitectureMetadata;
    star_ratings: ModelArchitectureStarRatings;
    badges: ModelArchitectureBadges;
    created_at: string;
}

export interface ModelArchitecturesResponse {
    architectures: ModelArchitecture[];
    total: number;
    filters_applied: {
        category?: ModelCategory;
        complexity?: ComplexityLevel;
        min_accuracy?: number;
        max_inference_ms?: number;
    };
}

export interface CompareModelsRequest {
    architecture_ids: string[];
}

export interface ModelComparisonDifference {
    metric: string;
    difference_percent: number;
    winner_id: string;
}

export interface ModelComparisonResponse {
    architectures: ModelArchitecture[];
    comparison_table: Record<string, Record<string, any>>;
    differences: ModelComparisonDifference[];
    recommendation: {
        best_accuracy_id: string;
        best_speed_id: string;
        best_efficiency_id: string;
        overall_best_id: string;
        reasoning: string;
    };
}

export interface RecommendedModelResponse {
    architecture: ModelArchitecture;
    reasoning: string;
    alternatives: Array<{
        architecture: ModelArchitecture;
        reason: string;
    }>;
}

export async function listModelArchitectures(
    category?: ModelCategory,
    complexity?: ComplexityLevel,
    minAccuracy?: number,
    maxInferenceMs?: number
): Promise<ModelArchitecturesResponse> {
    const params: Record<string, string> = {};

    if (category) params.category = category;
    if (complexity) params.complexity = complexity;
    if (minAccuracy !== undefined) params.min_accuracy = minAccuracy.toString();
    if (maxInferenceMs !== undefined) params.max_inference_ms = maxInferenceMs.toString();

    const response = await api.get('/v1/training/models/architectures', { params });
    return response.data;
}

export async function getModelArchitecture(architectureId: string): Promise<ModelArchitecture> {
    const response = await api.get(`/v1/training/models/architectures/${architectureId}`);
    return response.data;
}

export async function compareModelArchitectures(architectureIds: string[]): Promise<ModelComparisonResponse> {
    const response = await api.post('/v1/training/models/compare', { architecture_ids: architectureIds });
    return response.data;
}

export async function getRecommendedModelArchitecture(
    datasetSize?: number,
    prioritizeSpeed?: boolean
): Promise<RecommendedModelResponse> {
    const params: Record<string, string> = {};

    if (datasetSize !== undefined) params.dataset_size = datasetSize.toString();
    if (prioritizeSpeed !== undefined) params.prioritize_speed = prioritizeSpeed.toString();

    const response = await api.get('/v1/training/models/recommended', { params });
    return response.data;
}

export async function getModelArchitectureCard(architectureId: string): Promise<{ card: string }> {
    const response = await api.get(`/v1/training/models/architectures/${architectureId}/card`);
    return response.data;
}
