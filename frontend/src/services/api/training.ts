/**
 * Training API service
 * 
 * Provides functions for:
 * - Training job management
 * - Synthetic data generation
 * - Model management
 */

import { API_BASE_URL } from '@/lib/config';

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
  const response = await fetch(`${API_BASE_URL}/api/v1/training/jobs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create training job');
  }

  return response.json();
}

export async function listTrainingJobs(
  status?: string,
  limit: number = 50,
  offset: number = 0
): Promise<{ jobs: TrainingJob[]; total: number }> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });

  if (status) {
    params.append('status', status);
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/training/jobs?${params}`);

  if (!response.ok) {
    throw new Error('Failed to fetch training jobs');
  }

  return response.json();
}

export async function getTrainingJob(jobId: string): Promise<TrainingJob> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/jobs/${jobId}`);

  if (!response.ok) {
    throw new Error('Failed to fetch training job');
  }

  return response.json();
}

export async function deleteTrainingJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/jobs/${jobId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to delete training job');
  }
}

// ============================================================================
// SYNTHETIC DATA API
// ============================================================================

export async function generateSyntheticData(request: SyntheticDataRequest): Promise<{ job_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/synthetic/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate synthetic data');
  }

  return response.json();
}

export async function listSyntheticDatasets(
  limit: number = 50,
  offset: number = 0
): Promise<{ datasets: SyntheticDataset[]; total: number }> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });

  const response = await fetch(`${API_BASE_URL}/api/v1/training/synthetic/datasets?${params}`);

  if (!response.ok) {
    throw new Error('Failed to fetch synthetic datasets');
  }

  return response.json();
}

export async function getSyntheticDataset(datasetId: string): Promise<SyntheticDataset> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/synthetic/datasets/${datasetId}`);

  if (!response.ok) {
    throw new Error('Failed to fetch synthetic dataset');
  }

  return response.json();
}

export async function deleteSyntheticDataset(datasetId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/synthetic/datasets/${datasetId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to delete synthetic dataset');
  }
}

// ============================================================================
// MODELS API
// ============================================================================

export async function listModels(
  activeOnly: boolean = false,
  limit: number = 50,
  offset: number = 0
): Promise<{ models: TrainedModel[]; total: number }> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
    active_only: activeOnly.toString(),
  });

  const response = await fetch(`${API_BASE_URL}/api/v1/training/models?${params}`);

  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }

  return response.json();
}

export async function getModel(modelId: string): Promise<TrainedModel> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/models/${modelId}`);

  if (!response.ok) {
    throw new Error('Failed to fetch model');
  }

  return response.json();
}

export async function deployModel(modelId: string, setProduction: boolean = false): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/models/${modelId}/deploy?set_production=${setProduction}`, {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to deploy model');
  }

  return response.json();
}

export async function deleteModel(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/training/models/${modelId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete model');
  }
}
