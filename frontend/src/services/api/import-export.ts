import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export interface CreatorInfo {
  username: string;
  name: string;
}

export interface SectionSizes {
  settings: number;
  sources: number;
  websdrs: number;
  sessions: number;
  training_model: number;
  inference_model: number;
}

export interface HeimdallMetadata {
  version: string;
  created_at: string;
  creator: CreatorInfo;
  section_sizes: SectionSizes;
  description?: string;
}

export interface UserSettings {
  theme: string;
  language: string;
  default_frequency_mhz?: number;
  default_duration_seconds?: number;
  map_center_lat?: number;
  map_center_lon?: number;
  map_zoom?: number;
  auto_approve_sessions: boolean;
  notification_enabled: boolean;
  advanced_mode: boolean;
}

export interface ExportedSource {
  id: string;
  name: string;
  description?: string;
  frequency_hz: number;
  latitude: number;
  longitude: number;
  power_dbm?: number;
  source_type?: string;
  is_validated: boolean;
  error_margin_meters?: number;
  created_at: string;
  updated_at: string;
}

export interface ExportedWebSDR {
  id: string;
  name: string;
  url: string;
  country?: string;
  latitude: number;
  longitude: number;
  frequency_min_hz?: number;
  frequency_max_hz?: number;
  is_active: boolean;
  api_type?: string;
  rate_limit_ms?: number;
  timeout_seconds?: number;
  retry_count?: number;
  admin_email?: string;
  location_description?: string;
  altitude_asl?: number;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface ExportedSession {
  id: string;
  known_source_id: string;
  session_name: string;
  session_start: string;
  session_end?: string;
  duration_seconds?: number;
  celery_task_id?: string;
  status: string;
  approval_status: string;
  notes?: string;
  created_at: string;
  updated_at: string;
  measurements_count: number;
  source_name?: string;
  source_frequency?: number;
}

export interface ExportedModel {
  id: string;
  model_name: string;
  model_type?: string;
  training_dataset_id?: string;
  mlflow_run_id?: string;
  mlflow_experiment_id?: number;
  onnx_model_location?: string;
  pytorch_model_location?: string;
  accuracy_meters?: number;
  accuracy_sigma_meters?: number;
  loss_value?: number;
  epoch?: number;
  is_active: boolean;
  is_production: boolean;
  created_at: string;
  updated_at: string;
  model_data?: string;
}

export interface HeimdallSections {
  settings?: UserSettings;
  sources?: ExportedSource[];
  websdrs?: ExportedWebSDR[];
  sessions?: ExportedSession[];
  training_model?: ExportedModel;
  inference_model?: ExportedModel;
}

export interface HeimdallFile {
  metadata: HeimdallMetadata;
  sections: HeimdallSections;
}

export interface ExportRequest {
  creator: CreatorInfo;
  description?: string;
  include_settings?: boolean;
  include_sources?: boolean;
  include_websdrs?: boolean;
  include_sessions?: boolean;
  include_training_model?: boolean;
  include_inference_model?: boolean;
  session_ids?: string[];
}

export interface ImportRequest {
  file_content: HeimdallFile;
  import_settings?: boolean;
  import_sources?: boolean;
  import_websdrs?: boolean;
  import_sessions?: boolean;
  import_training_model?: boolean;
  import_inference_model?: boolean;
  overwrite_existing?: boolean;
}

export interface ImportResult {
  success: boolean;
  message: string;
  imported_counts: Record<string, number>;
  errors: string[];
  warnings: string[];
}

export interface ExportMetadataResponse {
  available_sources_count: number;
  available_websdrs_count: number;
  available_sessions_count: number;
  has_training_model: boolean;
  has_inference_model: boolean;
  estimated_size_bytes: number;
}

/**
 * Export selected data sections to .heimdall format
 */
export async function exportData(request: ExportRequest): Promise<HeimdallFile> {
  const response = await axios.post<HeimdallFile>(
    `${API_BASE_URL}/api/import-export/export`,
    request
  );
  return response.data;
}

/**
 * Import data from .heimdall file
 */
export async function importData(request: ImportRequest): Promise<ImportResult> {
  const response = await axios.post<ImportResult>(
    `${API_BASE_URL}/api/import-export/import`,
    request
  );
  return response.data;
}

/**
 * Get metadata about available data for export
 */
export async function getExportMetadata(): Promise<ExportMetadataResponse> {
  const response = await axios.get<ExportMetadataResponse>(
    `${API_BASE_URL}/api/import-export/export/metadata`
  );
  return response.data;
}

/**
 * Parse .heimdall file from JSON string
 */
export function parseHeimdallFile(content: string): HeimdallFile {
  return JSON.parse(content);
}

/**
 * Serialize .heimdall file to JSON string
 */
export function serializeHeimdallFile(file: HeimdallFile): string {
  return JSON.stringify(file, null, 2);
}

/**
 * Validate .heimdall file structure
 */
export function validateHeimdallFile(file: any): file is HeimdallFile {
  return (
    file &&
    typeof file === 'object' &&
    'metadata' in file &&
    'sections' in file &&
    file.metadata &&
    typeof file.metadata === 'object' &&
    'version' in file.metadata &&
    'created_at' in file.metadata &&
    'creator' in file.metadata
  );
}

/**
 * Format file size in human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
