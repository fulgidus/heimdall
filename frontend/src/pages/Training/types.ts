/**
 * Training UI TypeScript Types
 * Defines all interfaces for Training Job management, metrics, and models
 */

export type TrainingJobStatus = 
  | 'pending'
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
  early_stopping_patience?: number;
  checkpoint_every_n_epochs?: number;
}

export interface TrainingJob {
  id: string;
  name: string;
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
  // Advanced localization metrics (Phase 7 - Nov 2025)
  train_rmse_km?: number;
  val_rmse_km?: number;
  val_rmse_good_geom_km?: number;
  val_distance_p50_km?: number;
  val_distance_p68_km?: number;  // Project KPI: ±30m @ 68% confidence
  val_distance_p95_km?: number;
  mean_predicted_uncertainty_km?: number;
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
  name: string;
  version: string;
  architecture: string;
  created_at: string;
  training_job_id?: string;
  onnx_path: string;
  parameters_count?: number;
  final_metrics?: FinalMetrics;
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
  event: 'training_started' | 'training_progress' | 'training_completed' | 'training_failed';
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
    early_stopping_patience?: number;
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
  num_samples: number;
  config: Record<string, any>;
  quality_metrics?: QualityMetrics;
  storage_table: string;
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
}

export interface SyntheticSample {
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

export interface SyntheticSamplesResponse {
  samples: SyntheticSample[];
  total: number;
  limit: number;
  offset: number;
  dataset_id: string;
}

export type SyntheticJobStatus = 
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface SyntheticGenerationJob {
  id: string;
  name: string;
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
