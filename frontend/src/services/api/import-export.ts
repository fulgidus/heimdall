/**
 * Import/Export API Service
 *
 * Handles import and export of Heimdall system data:
 * - Export data to .heimdall JSON files
 * - Import data from .heimdall JSON files
 * - Get metadata about exportable data
 */

import api from '@/lib/api';

// Type definitions matching backend models

export interface CreatorInfo {
  username: string;
  name?: string;
}

export interface SectionSizes {
  settings: number;
  sources: number;
  websdrs: number;
  sessions: number;
  sample_sets: number;
  models: number;
  audio_library: number;
}

export interface ExportMetadata {
  version: string;
  created_at: string;
  creator: CreatorInfo;
  section_sizes: SectionSizes;
  description?: string;
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
  location_description?: string;
  latitude: number;
  longitude: number;
  altitude_meters?: number;
  country?: string;
  operator?: string;
  is_active: boolean;
  timeout_seconds: number;
  retry_count: number;
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
}

export interface ExportedSampleSet {
  id: string;
  name: string;
  description?: string;
  num_samples: number;
  config?: Record<string, unknown>;
  quality_metrics?: Record<string, unknown>;
  created_at: string;
  samples?: Record<string, unknown>[];
}

export interface ExportedModel {
  id: string;
  model_name: string;
  version: number;
  model_type: string;
  created_at: string;
  onnx_model_base64?: string;
  accuracy_meters?: number;
  hyperparameters?: Record<string, unknown>;
  training_metrics?: Record<string, unknown>;
}

export interface UserSettings {
  theme: string;
  default_frequency_mhz: number;
  default_duration_seconds: number;
  auto_approve_sessions: boolean;
}

export interface ExportSections {
  settings?: UserSettings;
  sources?: ExportedSource[];
  websdrs?: ExportedWebSDR[];
  sessions?: ExportedSession[];
  sample_sets?: ExportedSampleSet[];
  models?: ExportedModel[];
  audio_library?: ExportedAudioLibrary[];
}

export interface HeimdallFile {
  metadata: ExportMetadata;
  sections: ExportSections;
}

export interface ExportRequest {
  creator: CreatorInfo;
  description?: string;
  include_settings?: boolean;
  include_sources?: boolean;
  include_websdrs?: boolean;
  include_sessions?: boolean;
  sample_set_configs?: SampleSetExportConfig[] | null;
  model_ids?: string[] | null;
  audio_library_ids?: string[] | null;
}

export interface ExportResponse {
  file: HeimdallFile;
  size_bytes: number;
}

export interface ImportRequest {
  heimdall_file: HeimdallFile;
  import_settings?: boolean;
  import_sources?: boolean;
  import_websdrs?: boolean;
  import_sessions?: boolean;
  import_sample_sets?: boolean;
  import_models?: boolean;
  import_audio_library?: boolean;
  overwrite_existing?: boolean;
}

export interface ImportResponse {
  success: boolean;
  message: string;
  imported_counts: Record<string, number>;
  errors: string[];
}

export interface AvailableSampleSet {
  id: string;
  name: string;
  num_samples: number;
  num_iq_samples: number;
  created_at: string;
  estimated_size_bytes: number;
  estimated_size_per_feature: number;
  estimated_size_per_iq: number;
}

export interface SampleSetExportConfig {
  dataset_id: string;
  sample_offset: number;
  sample_limit: number | null;
}

export interface AvailableModel {
  id: string;
  model_name: string;
  version: number;
  created_at: string;
  has_onnx: boolean;
}

export interface ExportedAudioChunk {
  id: string;
  chunk_index: number;
  duration_seconds: number;
  sample_rate: number;
  num_samples: number;
  file_size_bytes: number;
  original_offset_seconds: number;
  rms_amplitude?: number;
  created_at: string;
  audio_data_base64?: string;
}

export interface ExportedAudioLibrary {
  id: string;
  filename: string;
  category: string;
  tags?: string[];
  file_size_bytes: number;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  audio_format: string;
  processing_status: string;
  total_chunks: number;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  chunks?: ExportedAudioChunk[];
}

export interface AvailableAudioLibrary {
  id: string;
  filename: string;
  category: string;
  duration_seconds: number;
  total_chunks: number;
  file_size_bytes: number;
  chunks_total_bytes: number;
  created_at: string;
}

export interface MetadataResponse {
  sources_count: number;
  websdrs_count: number;
  sessions_count: number;
  sample_sets: AvailableSampleSet[];
  models: AvailableModel[];
  audio_library: AvailableAudioLibrary[];
  estimated_sizes: SectionSizes;
}

/**
 * Get metadata about available data for export
 */
export async function getExportMetadata(): Promise<MetadataResponse> {
  const response = await api.get('/import-export/export/metadata');
  return response.data;
}

/**
 * Export selected data sections to .heimdall file format
 */
export async function exportData(request: ExportRequest): Promise<ExportResponse> {
  const response = await api.post('/import-export/export', request);
  return response.data;
}

/**
 * Import data from .heimdall file format
 */
export async function importData(request: ImportRequest): Promise<ImportResponse> {
  const response = await api.post('/import-export/import', request);
  return response.data;
}

/**
 * Download .heimdall file to user's computer (browser)
 */
export function downloadHeimdallFile(file: HeimdallFile, filename?: string): void {
  const json = JSON.stringify(file, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `heimdall-export-${new Date().toISOString().split('T')[0]}.heimdall`;
  document.body.appendChild(a);
  a.click();
  
  // Cleanup after a short delay to ensure download starts
  setTimeout(() => {
    try {
      if (a.parentNode === document.body) {
        document.body.removeChild(a);
      }
    } catch (error) {
      // Silent fail - element already removed or not in body
      console.debug('Download link cleanup: already removed');
    }
    URL.revokeObjectURL(url);
  }, 100);
}

/**
 * Load .heimdall file from user's computer (browser)
 */
export function loadHeimdallFile(): Promise<HeimdallFile> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.heimdall,.json';

    let fileSelected = false;

    input.onchange = (e: Event) => {
      fileSelected = true;
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];

      if (!file) {
        reject(new Error('No file selected'));
        return;
      }

      const reader = new FileReader();

      reader.onload = (readerEvent: ProgressEvent<FileReader>) => {
        try {
          const content = readerEvent.target?.result as string;
          const heimdallFile = JSON.parse(content) as HeimdallFile;
          resolve(heimdallFile);
        } catch (_error) {
          reject(new Error('Invalid .heimdall file format'));
        }
      };

      reader.onerror = () => {
        reject(new Error('Error reading file'));
      };

      reader.readAsText(file);
    };

    // Handle cancel - user closed dialog without selecting a file
    input.oncancel = () => {
      reject(new Error('File selection cancelled'));
    };

    // Fallback for browsers that don't support oncancel
    setTimeout(() => {
      if (!fileSelected && !input.files?.length) {
        reject(new Error('File selection cancelled'));
      }
    }, 100);

    input.click();
  });
}

// ===== ASYNC EXPORT WITH STREAMING PROGRESS =====

/**
 * WebSocket event types for export progress
 */
export interface ExportProgressEvent {
  event: 'export:progress';
  task_id: string;
  stage: 'settings' | 'sources' | 'websdrs' | 'sample_sets' | 'finalizing';
  current: number;
  total: number;
  message: string;
  percentage?: number;
}

export interface ExportCompletedEvent {
  event: 'export:completed';
  task_id: string;
  status: 'completed' | 'failed';
  download_url?: string;
  file_size_bytes?: number;
  error_message?: string;
}

export type ExportWebSocketEvent = ExportProgressEvent | ExportCompletedEvent;

/**
 * Response from starting an async export
 */
export interface AsyncExportResponse {
  task_id: string;
  message: string;
}

/**
 * Start an async export job (returns immediately with task_id)
 * Progress is tracked via WebSocket events
 */
export async function exportDataAsync(request: ExportRequest): Promise<AsyncExportResponse> {
  const response = await api.post('/import-export/export/async', request);
  return response.data;
}

/**
 * Download exported file from MinIO via task_id
 * This should be called after receiving export:completed event
 */
export function downloadExportedFile(taskId: string): void {
  const downloadUrl = `/api/import-export/download/${taskId}`;
  
  // Use window.location for automatic download
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = `heimdall-export-${taskId.slice(0, 8)}.heimdall`;
  document.body.appendChild(a);
  a.click();
  
  // Cleanup
  setTimeout(() => {
    if (a.parentNode === document.body) {
      document.body.removeChild(a);
    }
  }, 100);
}
