/**
 * Audio Library API Client
 * Handles operations for managing audio samples used in ML training
 */

import api from '../../lib/api';

/**
 * Audio sample category types
 */
export type AudioCategory = 'voice' | 'music' | 'documentary' | 'conference' | 'custom';

/**
 * Audio sample metadata
 */
export interface AudioSample {
  id: string;
  filename: string;
  category: AudioCategory;
  description?: string;
  tags: string[];
  duration_seconds: number;
  sample_rate: number;
  file_size_bytes: number;
  uploaded_at: string;
  enabled: boolean;
}

/**
 * Audio library statistics
 */
export interface AudioLibraryStats {
  total_files: number;
  total_duration_seconds: number;
  total_size_bytes: number;
  enabled_files: number;
  by_category: Record<AudioCategory, number>;
}

/**
 * Category weights for proportional sampling
 */
export interface CategoryWeights {
  voice: number;
  music: number;
  documentary: number;
  conference: number;
  custom: number;
}

/**
 * Upload parameters
 */
export interface AudioUploadParams {
  file: File;
  category: AudioCategory;
  description?: string;
  tags?: string[];
}

/**
 * List parameters
 */
export interface AudioListParams {
  category?: AudioCategory;
  enabled?: boolean;
  limit?: number;
  offset?: number;
}

/**
 * Upload an audio file to the library
 */
export const uploadAudioSample = async (params: AudioUploadParams): Promise<AudioSample> => {
  const formData = new FormData();
  formData.append('file', params.file);
  formData.append('category', params.category);
  
  if (params.description) {
    formData.append('description', params.description);
  }
  
  if (params.tags && params.tags.length > 0) {
    formData.append('tags', params.tags.join(','));
  }

  const response = await api.post<AudioSample>('/v1/audio-library/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // 60 seconds for file upload
  });

  return response.data;
};

/**
 * List audio samples with optional filters
 */
export const listAudioSamples = async (params?: AudioListParams): Promise<AudioSample[]> => {
  const response = await api.get<{total: number; files: AudioSample[]}>('/v1/audio-library/list', {
    params,
  });

  return response.data.files;
};

/**
 * Get audio library statistics
 */
export const getAudioLibraryStats = async (): Promise<AudioLibraryStats> => {
  const response = await api.get<AudioLibraryStats>('/v1/audio-library/stats');
  return response.data;
};

/**
 * Download an audio sample
 */
export const downloadAudioSample = async (id: string): Promise<Blob> => {
  const response = await api.get<Blob>(`/v1/audio-library/${id}/download`, {
    responseType: 'blob',
  });

  return response.data;
};

/**
 * Enable or disable an audio sample
 */
export const toggleAudioSampleEnabled = async (id: string, enabled: boolean): Promise<AudioSample> => {
  const response = await api.patch<AudioSample>(`/v1/audio-library/${id}/enable`, null, {
    params: { enabled },
  });

  return response.data;
};

/**
 * Delete an audio sample
 */
export const deleteAudioSample = async (id: string): Promise<void> => {
  await api.delete(`/v1/audio-library/${id}`);
};

/**
 * Get a random audio sample (used by training pipeline)
 */
export const getRandomAudioSample = async (category?: AudioCategory): Promise<Blob> => {
  const response = await api.get<Blob>('/v1/audio-library/random', {
    params: category ? { category } : undefined,
    responseType: 'blob',
  });

  return response.data;
};

/**
 * Helper: Format file size for display
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

/**
 * Helper: Format duration for display
 */
export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Get category weights for training sample selection
 */
export const getCategoryWeights = async (): Promise<CategoryWeights> => {
  const response = await api.get<CategoryWeights>('/v1/audio-library/weights');
  return response.data;
};

/**
 * Update category weights for training sample selection
 */
export const updateCategoryWeights = async (weights: CategoryWeights): Promise<CategoryWeights> => {
  const response = await api.put<CategoryWeights>('/v1/audio-library/weights', weights);
  return response.data;
};

/**
 * Helper: Download file to user's computer
 */
export const downloadFile = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};
