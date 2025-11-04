/**
 * Training UI TypeScript Types
 * Defines all interfaces for Training Job management, metrics, and models
 */

export type TrainingJobStatus = 
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TrainingJobConfig {
  job_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  model_architecture: string;
  dataset_ids?: string[];
  validation_split?: number;
  early_stop_patience?: number;
  checkpoint_every_n_epochs?: number;
}

export interface TrainingJob {
  id: string;
  job_name: string;
  name?: string; // Deprecated - use job_name
  status: TrainingJobStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  config: TrainingJobConfig;
  progress_percent?: number;
  current_epoch?: number;
  total_epochs: number;
  estimated_completion?: string;
  error_message?: string;
}

export interface TrainingMetric {
  job_id: string;
  epoch: number;
  timestamp: string;
  train_loss: number;
  val_loss: number;
  train_accuracy?: number;
  val_accuracy?: number;
  learning_rate: number;
  epoch_duration_seconds?: number;
  // Advanced localization metrics (Phase 7 - Nov 2025) - All in meters (SI unit)
  train_rmse_m?: number;
  val_rmse_m?: number;
  val_rmse_good_geom_m?: number;
  val_distance_p50_m?: number;
  val_distance_p68_m?: number;  // Project KPI: ±30m @ 68% confidence
  val_distance_p95_m?: number;
  mean_predicted_uncertainty_m?: number;
  uncertainty_calibration_error?: number;
  mean_gdop?: number;
  gdop_below_5_percent?: number;
  gradient_norm?: number;
  weight_norm?: number;
}

export interface FinalMetrics {
  train_loss: number;
  val_loss: number;
  train_accuracy?: number;
  val_accuracy?: number;
}

export interface TrainedModel {
  id: string;
  model_name: string; // Backend field name
  name?: string; // Deprecated - use model_name
  version: number;
  model_type: string | null;
  architecture?: string; // Deprecated - derived from hyperparameters
  synthetic_dataset_id: string | null;
  mlflow_run_id: string | null;
  mlflow_experiment_id: number | null;
  onnx_model_location: string | null; // Backend field name
  onnx_path?: string; // Deprecated - use onnx_model_location
  pytorch_model_location: string | null;
  accuracy_meters: number | null;
  accuracy_sigma_meters: number | null;
  loss_value: number | null;
  epoch: number | null;
  is_active: boolean;
  is_production: boolean;
  hyperparameters: Record<string, any> | null;
  training_metrics: Record<string, any> | null;
  test_metrics: Record<string, any> | null;
  created_at: string;
  trained_by_job_id: string | null;
  training_job_id?: string; // Deprecated - use trained_by_job_id
  parameters_count?: number; // Deprecated - derived from model
  final_metrics?: FinalMetrics; // Deprecated - use training_metrics/test_metrics
}

export interface ExportOptions {
  include_config: boolean;
  include_metrics: boolean;
  include_normalization: boolean;
  include_samples: boolean;
  num_samples?: number;
  description?: string;
}

export interface TrainingWebSocketMessage {
  event: 'connected' | 'training_started' | 'training_progress' | 'training_completed' | 'training_failed';
  job_id: string;
  data: Partial<TrainingJob> | TrainingMetric;
  timestamp: string;
}

export interface CreateJobRequest {
  job_name: string;
  config: {
    dataset_ids: string[];
    epochs: number;
    batch_size: number;
    learning_rate: number;
    model_architecture?: string;
    validation_split?: number;
    early_stop_patience?: number;
  };
}

export interface CreateJobResponse {
  job_id: string;
  message: string;
}

// ============================================================================
// SYNTHETIC DATA GENERATION TYPES
// ============================================================================

export interface SyntheticDataset {
  id: string;
  name: string;
  description?: string;
  dataset_type?: 'feature_based' | 'iq_raw';
  num_samples: number;
  config: Record<string, any>;
  quality_metrics?: QualityMetrics;
  storage_table: string;
  storage_size_bytes?: number;  // Total storage (PostgreSQL + MinIO), null if not calculated
  created_at: string;
  created_by_job_id?: string;
}

export interface QualityMetrics {
  mean_snr_db?: number;
  mean_gdop?: number;
  mean_receivers?: number;
  min_distance_km?: number;
  max_distance_km?: number;
  mean_distance_km?: number;
  inside_count?: number;
  outside_count?: number;
  inside_ratio?: number;
}

export interface SyntheticDataRequest {
  name: string;
  description?: string;
  dataset_type?: 'feature_based' | 'iq_raw';
  num_samples: number;
  inside_ratio?: number;
  frequency_mhz?: number;
  tx_power_dbm?: number;
  min_snr_db?: number;
  min_receivers?: number;
  max_gdop?: number;
  use_srtm?: boolean;
  min_elevation_meters?: number;
  max_elevation_meters?: number;
  // Random receiver parameters (for iq_raw datasets)
  use_random_receivers?: boolean;
  min_receivers_count?: number;
  max_receivers_count?: number;
  receiver_seed?: number;
  area_lat_min?: number;
  area_lat_max?: number;
  area_lon_min?: number;
  area_lon_max?: number;
}

export interface SyntheticSample {
  id: string;
  timestamp: string;
  tx_lat: number;
  tx_lon: number;
  tx_power_dbm: number;
  frequency_hz: number;
  receivers: any[];
  gdop: number;
  num_receivers: number;
  split: 'train' | 'val' | 'test';
  created_at: string;
}

export interface ModelArchitecture {
  name: string;
  display_name: string;
  data_type: 'feature_based' | 'iq_raw' | 'both';
  description: string;
  default_params: Record<string, any>;
}

export interface ArchitecturesResponse {
  architectures: ModelArchitecture[];
  total: number;
}

export interface SyntheticSamplesResponse {
  samples: SyntheticSample[];
  total: number;
  limit: number;
  offset: number;
  dataset_id: string;
}

export type SyntheticJobStatus = 
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface SyntheticGenerationJob {
  id: string;
  name?: string;
  job_name: string;
  job_type: 'synthetic_generation';
  status: SyntheticJobStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  config: SyntheticDataRequest;
  progress_percent?: number;
  current?: number;
  total: number;
  estimated_completion?: string;
  error_message?: string;
  dataset_id?: string;
}

export interface ExpandDatasetRequest {
  dataset_id: string;
  num_additional_samples: number;
}
