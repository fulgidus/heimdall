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
  job_type: 'training' | 'synthetic_generation'; // Job type discriminator (required)
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
  checkpoint_path?: string; // Path to model checkpoint in MinIO
  pause_checkpoint_path?: string; // Path to pause checkpoint (if paused)
  train_loss?: number; // Current training loss (updated in real-time)
  val_loss?: number; // Current validation loss (updated in real-time)
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
  frequency_mhz: number;  // Required by backend
  tx_power_dbm: number;   // Required by backend
  min_snr_db: number;     // Required by backend
  min_receivers: number;  // Required by backend
  max_gdop?: number;
  use_srtm_terrain?: boolean;  // Backend expects 'use_srtm_terrain', not 'use_srtm'
  use_random_receivers?: boolean;
  use_gpu?: boolean | null;  // null = auto-detect, true = force GPU, false = force CPU
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
  // Simulation enhancement flags
  enable_meteorological?: boolean;  // Meteorological effects (tropospheric refraction/ducting)
  enable_sporadic_e?: boolean;      // Sporadic-E ionospheric propagation
  enable_knife_edge?: boolean;      // Knife-edge diffraction over obstacles
  enable_polarization?: boolean;    // Polarization mismatch loss
  enable_antenna_patterns?: boolean; // Realistic antenna radiation patterns
  // Audio library flags
  use_audio_library?: boolean;      // Use real audio from library instead of formant synthesis
  audio_library_fallback?: boolean; // Fallback to formant synthesis if audio library fails
}

export interface SyntheticSample {
  id: string | number;  // UUID string for IQ samples, int for feature samples
  timestamp: string;
  tx_lat: number;
  tx_lon: number;
  tx_power_dbm: number;
  frequency_hz: number;
  receivers: any[];  // Array for iq_raw, object for feature_based
  gdop: number;
  num_receivers: number;
  split?: 'train' | 'val' | 'test' | null;  // Only for feature_based datasets
  created_at: string;
  iq_available?: boolean;  // Whether IQ data is available for this sample
  iq_metadata?: IQMetadata;  // IQ metadata (sample_rate, duration, etc.)
  sample_idx?: number;  // Sample index (for IQ data lookup)
}

export interface IQMetadata {
  sample_rate_hz: number;
  duration_ms: number;
  center_frequency_hz: number;
}

export interface IQData {
  real_b64: string;  // Base64-encoded float32 array
  imag_b64: string;  // Base64-encoded float32 array
  length: number;
  dtype: string;
  // Decoded arrays (added by trainingStore after base64 decoding)
  i_samples?: Float32Array;
  q_samples?: Float32Array;
}

export interface ReceiverMetadata {
  rx_id: string;
  lat: number;
  lon: number;
  alt: number;
  distance_km: number;
  snr_db: number;
  rx_power_dbm: number;
  signal_present?: boolean;
  // Propagation details
  fspl_db?: number;
  terrain_loss_db?: number;
  knife_edge_loss_db?: number;
  atmospheric_absorption_db?: number;
  tropospheric_effect_db?: number;
  sporadic_e_enhancement_db?: number;
  polarization_loss_db?: number;
  // Antenna info
  tx_antenna_gain_db?: number;
  rx_antenna_gain_db?: number;
  tx_polarization?: string;
  rx_polarization?: string;
  tx_antenna_type?: string;
  rx_antenna_type?: string;
}

export interface IQDataResponse {
  dataset_id: string;
  sample_idx: number;
  rx_id: string;
  iq_data: IQData;
  iq_metadata: IQMetadata;
  rx_metadata: ReceiverMetadata;
  tx_metadata: {
    lat: number;
    lon: number;
    power_dbm: number;
    frequency_hz: number;
  };
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
  // Rejection statistics (added for debugging low success rates)
  total_attempted?: number;
  rejected_min_receivers?: number;
  rejected_min_snr?: number;
  rejected_gdop?: number;
  total_rejections?: number;
}

export interface ExpandDatasetRequest {
  dataset_id: string;
  num_additional_samples: number;
  use_gpu?: boolean | null;  // null = auto-detect, true = force GPU, false = force CPU
}
