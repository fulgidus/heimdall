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
  training_model: number;
  inference_model: number;
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

export interface ExportedModel {
  model_type: string;
  model_name: string;
  version: string;
  created_at: string;
  file_path?: string;
  metrics?: Record<string, unknown>;
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
  training_model?: ExportedModel;
  inference_model?: ExportedModel;
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
  include_training_model?: boolean;
  include_inference_model?: boolean;
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
  import_training_model?: boolean;
  import_inference_model?: boolean;
  overwrite_existing?: boolean;
}

export interface ImportResponse {
  success: boolean;
  message: string;
  imported_counts: Record<string, number>;
  errors: string[];
}

export interface MetadataResponse {
  sources_count: number;
  websdrs_count: number;
  sessions_count: number;
  has_training_model: boolean;
  has_inference_model: boolean;
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
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Load .heimdall file from user's computer (browser)
 */
export function loadHeimdallFile(): Promise<HeimdallFile> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.heimdall,.json';

    input.onchange = (e: Event) => {
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

    input.click();
  });
}
