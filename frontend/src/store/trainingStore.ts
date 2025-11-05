/**
 * Training Store
 * 
 * Manages training job state, metrics, and models with real-time WebSocket updates
 */

import { create } from 'zustand';
import api from '../lib/api';
import type {
  TrainingJob,
  TrainingMetric,
  TrainedModel,
  CreateJobRequest,
  CreateJobResponse,
  ExportOptions,
  SyntheticDataset,
  SyntheticDataRequest,
  SyntheticSamplesResponse,
  SyntheticGenerationJob,
  ExpandDatasetRequest,
} from '../pages/Training/types';

interface TrainingStore {
  // State
  jobs: TrainingJob[];
  metrics: Map<string, TrainingMetric[]>;
  models: TrainedModel[];
  datasets: SyntheticDataset[];
  generationJobs: SyntheticGenerationJob[];
  wsConnected: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions - Jobs
  fetchJobs: () => Promise<void>;
  createJob: (jobConfig: CreateJobRequest) => Promise<string>;
  cancelJob: (jobId: string) => Promise<void>;
  pauseJob: (jobId: string) => Promise<void>;
  resumeJob: (jobId: string) => Promise<void>;
  deleteJob: (jobId: string) => Promise<void>;

  // Actions - Metrics
  fetchMetrics: (jobId: string) => Promise<void>;

  // Actions - Models
  fetchModels: () => Promise<void>;
  downloadModel: (modelId: string, options: ExportOptions) => Promise<void>;
  importModel: (file: File) => Promise<void>;
  deleteModel: (modelId: string) => Promise<void>;
  updateModelName: (modelId: string, newName: string) => Promise<void>;
  setModelActive: (modelId: string) => Promise<void>;
  setModelProduction: (modelId: string) => Promise<void>;
  evolveTraining: (parentModelId: string, additionalEpochs: number, earlyStopPatience: number) => Promise<string>;

  // Actions - Synthetic Datasets
  fetchDatasets: (silent?: boolean) => Promise<void>;
  generateSyntheticData: (request: SyntheticDataRequest) => Promise<string>;
  deleteDataset: (datasetId: string) => Promise<void>;
  fetchDatasetSamples: (datasetId: string, limit?: number) => Promise<SyntheticSamplesResponse>;
  expandDataset: (request: ExpandDatasetRequest) => Promise<string>;

  // Actions - Synthetic Generation Jobs
  fetchGenerationJobs: (silent?: boolean) => Promise<void>;
  pauseGenerationJob: (jobId: string) => Promise<void>;
  resumeGenerationJob: (jobId: string) => Promise<void>;
  cancelGenerationJob: (jobId: string) => Promise<void>;
  deleteGenerationJob: (jobId: string) => Promise<void>;

  // WebSocket handlers
  handleJobUpdate: (job: Partial<TrainingJob> & { id: string }) => void;
  handleMetricUpdate: (metric: TrainingMetric) => void;
  handleGenerationJobUpdate: (jobUpdate: Partial<SyntheticGenerationJob> & { id: string }) => void;
  setWsConnected: (connected: boolean) => void;

  // Utility
  clearError: () => void;
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  // Initial State
  jobs: [],
  metrics: new Map(),
  models: [],
  datasets: [],
  generationJobs: [],
  wsConnected: false,
  isLoading: false,
  error: null,

  // Fetch all training jobs
  fetchJobs: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.get('/v1/jobs/training');
      set({ jobs: response.data.jobs || response.data, isLoading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch training jobs';
      set({ error: errorMessage, isLoading: false });
      console.error('Training jobs fetch error:', error);
    }
  },

  // Create a new training job
  createJob: async (jobConfig: CreateJobRequest) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.post<CreateJobResponse>('/v1/jobs/training', jobConfig);
      set({ isLoading: false });
      
      // Refresh jobs list
      await get().fetchJobs();
      
      return response.data.job_id;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create training job';
      set({ error: errorMessage, isLoading: false });
      console.error('Training job creation error:', error);
      throw error;
    }
  },

  // Cancel a running or paused job
  cancelJob: async (jobId: string) => {
    console.log('[trainingStore] cancelJob called:', { jobId: jobId.slice(0, 8) });
    set({ error: null });
    try {
      const endpoint = `/v1/jobs/training/${jobId}/cancel`;
      console.log('[trainingStore] Calling POST', endpoint);
      await api.post(endpoint);
      console.log('[trainingStore] Cancel API call successful');
      
      // Update job status locally
      set(state => ({
        jobs: state.jobs.map(job =>
          job.id === jobId ? { ...job, status: 'cancelled' as const } : job
        ),
      }));
    } catch (error: any) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to cancel training job';
      set({ error: errorMessage });
      console.error('[trainingStore] Training job cancellation error:', error);
      console.error('[trainingStore] Error response:', error?.response?.data);
      throw error;
    }
  },

  // Pause a running job
  pauseJob: async (jobId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/jobs/training/${jobId}/pause`);
      
      // Update job status locally
      set(state => ({
        jobs: state.jobs.map(job =>
          job.id === jobId ? { ...job, status: 'paused' as const } : job
        ),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to pause training job';
      set({ error: errorMessage });
      console.error('Training job pause error:', error);
      throw error;
    }
  },

  // Resume a paused job
  resumeJob: async (jobId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/jobs/training/${jobId}/resume`);
      
      // Update job status locally
      set(state => ({
        jobs: state.jobs.map(job =>
          job.id === jobId ? { ...job, status: 'running' as const } : job
        ),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to resume training job';
      set({ error: errorMessage });
      console.error('Training job resume error:', error);
      throw error;
    }
  },

  // Delete a completed/failed/cancelled job
  deleteJob: async (jobId: string) => {
    console.log('[trainingStore] deleteJob called:', { jobId: jobId.slice(0, 8) });
    set({ error: null });
    try {
      const endpoint = `/v1/jobs/training/${jobId}`;
      console.log('[trainingStore] Calling DELETE', endpoint);
      await api.delete(endpoint);
      console.log('[trainingStore] Delete API call successful');
      
      // Remove job from list
      set(state => ({
        jobs: state.jobs.filter(job => job.id !== jobId),
      }));
    } catch (error: any) {
      // If job not found (404), remove it from local state anyway
      if (error?.status === 404 || error?.response?.status === 404) {
        console.warn(`[trainingStore] Job ${jobId} not found on server, removing from local state`);
        set(state => ({
          jobs: state.jobs.filter(job => job.id !== jobId),
        }));
        return; // Don't throw error for 404
      }
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete training job';
      set({ error: errorMessage });
      console.error('Training job deletion error:', error);
      throw error;
    }
  },

  // Fetch metrics for a specific job
  fetchMetrics: async (jobId: string) => {
    set({ error: null });
    try {
      const response = await api.get(`/v1/jobs/training/${jobId}/metrics`);
      const metrics = response.data.metrics || response.data;
      
      set(state => {
        const newMetrics = new Map(state.metrics);
        newMetrics.set(jobId, metrics);
        return { metrics: newMetrics };
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch training metrics';
      set({ error: errorMessage });
      console.error('Training metrics fetch error:', error);
    }
  },

  // Fetch all trained models
  fetchModels: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.get('/v1/models');
      set({ models: response.data.models || response.data, isLoading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch trained models';
      set({ error: errorMessage, isLoading: false });
      console.error('Trained models fetch error:', error);
    }
  },

  // Download a model as .heimdall bundle
  downloadModel: async (modelId: string, options: ExportOptions) => {
    set({ error: null });
    try {
      const params = new URLSearchParams();
      params.append('include_config', options.include_config.toString());
      params.append('include_metrics', options.include_metrics.toString());
      params.append('include_normalization', options.include_normalization.toString());
      params.append('include_samples', options.include_samples.toString());
      if (options.num_samples) params.append('num_samples', options.num_samples.toString());
      if (options.description) params.append('description', options.description);

      const response = await api.get(`/v1/models/${modelId}/export?${params.toString()}`, {
        responseType: 'blob',
      });

      // Extract filename from Content-Disposition header or use default
      const contentDisposition = response.headers['content-disposition'];
      let filename = `model-${modelId}.heimdall`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) filename = filenameMatch[1];
      }

      // Create blob and trigger download
      const blob = new Blob([response.data], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      
      // Defensive cleanup: check parent before removing
      try {
        if (link.parentNode === document.body) {
          document.body.removeChild(link);
        }
      } catch (error) {
        // Silent fail - element already removed
        console.debug('Download link cleanup: already removed');
      }
      
      window.URL.revokeObjectURL(url);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to download model';
      set({ error: errorMessage });
      console.error('Model download error:', error);
      throw error;
    }
  },

  // Import a model from .heimdall bundle
  importModel: async (file: File) => {
    set({ isLoading: true, error: null });
    try {
      const formData = new FormData();
      formData.append('file', file);

      await api.post('/v1/models/import', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      set({ isLoading: false });
      
      // Refresh models list
      await get().fetchModels();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to import model';
      set({ error: errorMessage, isLoading: false });
      console.error('Model import error:', error);
      throw error;
    }
  },

  // Delete a model
  deleteModel: async (modelId: string) => {
    set({ error: null });
    try {
      await api.delete(`/v1/models/${modelId}`);
      
      // Remove model from list
      set(state => ({
        models: state.models.filter(model => model.id !== modelId),
      }));
    } catch (error: any) {
      // If model not found (404), remove it from local state anyway
      if (error?.status === 404 || error?.response?.status === 404) {
        console.warn(`Model ${modelId} not found on server, removing from local state`);
        set(state => ({
          models: state.models.filter(model => model.id !== modelId),
        }));
        return; // Don't throw error for 404
      }
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete model';
      set({ error: errorMessage });
      console.error('Model deletion error:', error);
      throw error;
    }
  },

  // Update model name
  updateModelName: async (modelId: string, newName: string) => {
    set({ error: null });
    try {
      const params = new URLSearchParams();
      params.append('model_name', newName);

      await api.patch(`/v1/models/${modelId}?${params.toString()}`);
      
      // Update model in local state
      set(state => ({
        models: state.models.map(model =>
          model.id === modelId ? { ...model, model_name: newName } : model
        ),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update model name';
      set({ error: errorMessage });
      console.error('Model name update error:', error);
      throw error;
    }
  },

  // Set a model as active
  setModelActive: async (modelId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/models/${modelId}/deploy`, null, {
        params: { set_production: false }
      });
      
      // Refresh models list to get updated state
      await get().fetchModels();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to set model as active';
      set({ error: errorMessage });
      console.error('Model activation error:', error);
      throw error;
    }
  },

  // Set a model as production
  setModelProduction: async (modelId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/models/${modelId}/deploy`, null, {
        params: { set_production: true }
      });
      
      // Refresh models list to get updated state
      await get().fetchModels();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to set model as production';
      set({ error: errorMessage });
      console.error('Model production setting error:', error);
      throw error;
    }
  },

  // Evolve a model by training additional epochs with parent weights
  evolveTraining: async (parentModelId: string, additionalEpochs: number, earlyStopPatience: number) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.post('/v1/training/evolve', {
        parent_model_id: parentModelId,
        additional_epochs: additionalEpochs,
        early_stop_patience: earlyStopPatience,
      });
      set({ isLoading: false });
      
      // Refresh jobs list
      await get().fetchJobs();
      
      return response.data.id;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to evolve training';
      set({ error: errorMessage, isLoading: false });
      console.error('Training evolution error:', error);
      throw error;
    }
  },

  // Handle WebSocket job update
  handleJobUpdate: (jobUpdate: Partial<TrainingJob> & { id: string }) => {
    console.log('[TrainingStore] handleJobUpdate called with:', jobUpdate);
    
    set(state => {
      const existingJobIndex = state.jobs.findIndex(job => job.id === jobUpdate.id);
      
      if (existingJobIndex >= 0) {
        // Update existing job
        const updatedJobs = [...state.jobs];
        const existingJob = updatedJobs[existingJobIndex];
        
        // Merge updates while preserving existing fields
        updatedJobs[existingJobIndex] = {
          ...existingJob,
          ...jobUpdate,
          // Ensure critical fields are updated
          progress_percent: jobUpdate.progress_percent ?? existingJob.progress_percent,
          current_epoch: jobUpdate.current_epoch ?? existingJob.current_epoch,
          total_epochs: jobUpdate.total_epochs ?? existingJob.total_epochs,
          estimated_completion: jobUpdate.estimated_completion ?? existingJob.estimated_completion,
        };
        
        console.log('[TrainingStore] Job updated:', {
          id: jobUpdate.id,
          old_progress: existingJob.progress_percent,
          new_progress: updatedJobs[existingJobIndex].progress_percent,
          old_epoch: existingJob.current_epoch,
          new_epoch: updatedJobs[existingJobIndex].current_epoch,
        });
        
        return { jobs: updatedJobs };
      } else {
        // Add new job (if it doesn't exist)
        console.log('[TrainingStore] New job added:', jobUpdate.id);
        return { jobs: [jobUpdate as TrainingJob, ...state.jobs] };
      }
    });
  },

  // Handle WebSocket metric update
  handleMetricUpdate: (metric: TrainingMetric) => {
    set(state => {
      const newMetrics = new Map(state.metrics);
      const existingMetrics = newMetrics.get(metric.job_id) || [];
      
      // Append new metric (avoid duplicates by epoch)
      const isDuplicate = existingMetrics.some(m => m.epoch === metric.epoch);
      if (!isDuplicate) {
        newMetrics.set(metric.job_id, [...existingMetrics, metric]);
      }
      
      return { metrics: newMetrics };
    });
  },

  // Handle WebSocket generation job update
  handleGenerationJobUpdate: (jobUpdate: Partial<SyntheticGenerationJob> & { id: string }) => {
    console.log('[TrainingStore] handleGenerationJobUpdate called with:', jobUpdate);
    
    set(state => {
      const existingJobIndex = state.generationJobs.findIndex(job => job.id === jobUpdate.id);
      
      if (existingJobIndex >= 0) {
        // Update existing job
        const updatedJobs = [...state.generationJobs];
        const existingJob = updatedJobs[existingJobIndex];
        
        // Merge updates while preserving existing fields
        updatedJobs[existingJobIndex] = {
          ...existingJob,
          ...jobUpdate,
          // Ensure critical fields are updated
          current: jobUpdate.current ?? existingJob.current,
          total: jobUpdate.total ?? existingJob.total,
          progress_percent: jobUpdate.progress_percent ?? existingJob.progress_percent,
        };
        
        console.log('[TrainingStore] Generation job updated:', {
          id: jobUpdate.id,
          old_status: existingJob.status,
          new_status: updatedJobs[existingJobIndex].status,
          old_current: existingJob.current,
          new_current: updatedJobs[existingJobIndex].current,
          old_progress: existingJob.progress_percent,
          new_progress: updatedJobs[existingJobIndex].progress_percent,
        });
        
        return { generationJobs: updatedJobs };
      } else {
        // Add new job (if it doesn't exist)
        console.log('[TrainingStore] New generation job added:', jobUpdate.id);
        return { generationJobs: [jobUpdate as SyntheticGenerationJob, ...state.generationJobs] };
      }
    });
  },

  // Set WebSocket connection status
  setWsConnected: (connected: boolean) => {
    set({ wsConnected: connected });
  },

  // Fetch all synthetic datasets
  fetchDatasets: async (silent = false) => {
    if (!silent) {
      set({ isLoading: true, error: null });
    }
    try {
      const response = await api.get('/v1/jobs/synthetic/datasets');
      set({ datasets: response.data.datasets || response.data, isLoading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch synthetic datasets';
      set({ error: errorMessage, isLoading: false });
      console.error('Synthetic datasets fetch error:', error);
    }
  },

  // Generate synthetic data
  generateSyntheticData: async (request: SyntheticDataRequest) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.post('/v1/jobs/synthetic/generate', request);
      set({ isLoading: false });
      
      // Refresh datasets list
      await get().fetchDatasets();
      
      return response.data.job_id;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate synthetic data';
      set({ error: errorMessage, isLoading: false });
      console.error('Synthetic data generation error:', error);
      throw error;
    }
  },

  // Delete a synthetic dataset
  deleteDataset: async (datasetId: string) => {
    set({ error: null });
    try {
      await api.delete(`/v1/jobs/synthetic/datasets/${datasetId}`);
      
      // Remove dataset from list
      set(state => ({
        datasets: state.datasets.filter(dataset => dataset.id !== datasetId),
      }));
    } catch (error: any) {
      // If dataset not found (404), remove it from local state anyway
      if (error?.status === 404 || error?.response?.status === 404) {
        console.warn(`Dataset ${datasetId} not found on server, removing from local state`);
        set(state => ({
          datasets: state.datasets.filter(dataset => dataset.id !== datasetId),
        }));
        return; // Don't throw error for 404
      }
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete dataset';
      set({ error: errorMessage });
      console.error('Dataset deletion error:', error);
      throw error;
    }
  },

  // Fetch samples from a synthetic dataset
  fetchDatasetSamples: async (datasetId: string, limit: number = 10) => {
    set({ error: null });
    try {
      const response = await api.get(`/v1/jobs/synthetic/datasets/${datasetId}/samples`, {
        params: { limit },
      });
      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch dataset samples';
      set({ error: errorMessage });
      console.error('Dataset samples fetch error:', error);
      throw error;
    }
  },

  // Expand an existing dataset with more samples
  expandDataset: async (request: ExpandDatasetRequest) => {
    set({ isLoading: true, error: null });
    try {
      // Get the original dataset to verify it exists
      const dataset = get().datasets.find(d => d.id === request.dataset_id);
      if (!dataset) {
        throw new Error('Dataset not found');
      }

      // IMPORTANT: Do NOT send config parameters from frontend!
      // Backend will automatically inherit parameters (frequency, power, SNR, etc.) 
      // from the original dataset's config stored in the database.
      // This ensures consistency and prevents parameter mismatches.
      const generationRequest = {
        name: `${dataset.name} (Expansion)`,
        description: `Additional ${request.num_additional_samples} samples for ${dataset.name}`,
        num_samples: request.num_additional_samples,
        expand_dataset_id: request.dataset_id, // Signal this is an expansion
        use_gpu: request.use_gpu, // Pass GPU/CPU preference (null = auto, true = GPU, false = CPU)
      };

      // Use the same endpoint, but it will add to the existing dataset
      const response = await api.post('/v1/jobs/synthetic/generate', generationRequest);
      
      set({ isLoading: false });
      
      // Refresh generation jobs list
      await get().fetchGenerationJobs();
      
      return response.data.job_id;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to expand dataset';
      set({ error: errorMessage, isLoading: false });
      console.error('Dataset expansion error:', error);
      throw error;
    }
  },

  // Fetch all synthetic generation jobs
  fetchGenerationJobs: async (silent = false) => {
    if (!silent) {
      set({ isLoading: true, error: null });
    }
    try {
      const response = await api.get('/v1/jobs/synthetic');
      const jobs = response.data.jobs || response.data;
      set({ 
        generationJobs: jobs,
        isLoading: false 
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch generation jobs';
      set({ error: errorMessage, isLoading: false });
      console.error('Generation jobs fetch error:', error);
    }
  },

  // Pause a running generation job
  pauseGenerationJob: async (jobId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/jobs/synthetic/${jobId}/pause`);
      
      // Update job status locally
      set(state => ({
        generationJobs: state.generationJobs.map(job =>
          job.id === jobId ? { ...job, status: 'paused' as const } : job
        ),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to pause generation job';
      set({ error: errorMessage });
      console.error('Generation job pause error:', error);
      throw error;
    }
  },

  // Resume a paused generation job
  resumeGenerationJob: async (jobId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/jobs/synthetic/${jobId}/resume`);
      
      // Update job status locally
      set(state => ({
        generationJobs: state.generationJobs.map(job =>
          job.id === jobId ? { ...job, status: 'running' as const } : job
        ),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to resume generation job';
      set({ error: errorMessage });
      console.error('Generation job resume error:', error);
      throw error;
    }
  },

  // Cancel a running or paused generation job
  cancelGenerationJob: async (jobId: string) => {
    set({ error: null });
    try {
      await api.post(`/v1/jobs/synthetic/${jobId}/cancel`);
      
      // Update job status locally
      set(state => ({
        generationJobs: state.generationJobs.map(job =>
          job.id === jobId ? { ...job, status: 'cancelled' as const } : job
        ),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to cancel generation job';
      set({ error: errorMessage });
      console.error('Generation job cancellation error:', error);
      throw error;
    }
  },

  // Delete a completed/failed/cancelled generation job
  deleteGenerationJob: async (jobId: string) => {
    set({ error: null });
    try {
      await api.delete(`/v1/jobs/synthetic/${jobId}`);
      
      // Remove job from list
      set(state => ({
        generationJobs: state.generationJobs.filter(job => job.id !== jobId),
      }));
    } catch (error: any) {
      // If job not found (404), remove it from local state anyway
      if (error?.status === 404 || error?.response?.status === 404) {
        console.warn(`Job ${jobId} not found on server, removing from local state`);
        set(state => ({
          generationJobs: state.generationJobs.filter(job => job.id !== jobId),
        }));
        return; // Don't throw error for 404
      }
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete generation job';
      set({ error: errorMessage });
      console.error('Generation job deletion error:', error);
      throw error;
    }
  },

  // Clear error
  clearError: () => set({ error: null }),
}));
