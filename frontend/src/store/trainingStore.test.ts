/**
 * Training Store Tests
 *
 * Comprehensive test suite for the trainingStore Zustand store
 * Tests all actions: jobs, metrics, models, export/import, and WebSocket management
 * Truth-first approach: Tests real Zustand store behavior with mocked API responses
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/trainingStore');

// Import after unmocking
import { useTrainingStore } from './trainingStore';
import { trainingService } from '@/services/api/training';

// Mock the training service
vi.mock('@/services/api/training', () => ({
  trainingService: {
    listJobs: vi.fn(),
    getJob: vi.fn(),
    createJob: vi.fn(),
    pauseJob: vi.fn(),
    resumeJob: vi.fn(),
    cancelJob: vi.fn(),
    deleteJob: vi.fn(),
    getJobMetrics: vi.fn(),
    listModels: vi.fn(),
    getModel: vi.fn(),
    deleteModel: vi.fn(),
    exportModel: vi.fn(),
    importModel: vi.fn(),
  },
}));

describe('Training Store (Zustand)', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useTrainingStore.setState({
      jobs: [],
      currentJob: null,
      models: [],
      metrics: new Map(),
      activeConnections: new Map(),
      isLoading: false,
      error: null,
    });
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Clean up WebSocket connections
    const state = useTrainingStore.getState();
    state.activeConnections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });
  });

  describe('Store Initialization', () => {
    it('should initialize with default state', () => {
      const state = useTrainingStore.getState();
      expect(state.jobs).toEqual([]);
      expect(state.currentJob).toBe(null);
      expect(state.models).toEqual([]);
      expect(state.metrics).toBeInstanceOf(Map);
      expect(state.metrics.size).toBe(0);
      expect(state.activeConnections).toBeInstanceOf(Map);
      expect(state.activeConnections.size).toBe(0);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
    });

    it('should have all required actions', () => {
      const state = useTrainingStore.getState();
      expect(typeof state.fetchJobs).toBe('function');
      expect(typeof state.fetchJob).toBe('function');
      expect(typeof state.createJob).toBe('function');
      expect(typeof state.pauseJob).toBe('function');
      expect(typeof state.resumeJob).toBe('function');
      expect(typeof state.cancelJob).toBe('function');
      expect(typeof state.deleteJob).toBe('function');
      expect(typeof state.fetchJobMetrics).toBe('function');
      expect(typeof state.fetchModels).toBe('function');
      expect(typeof state.fetchModel).toBe('function');
      expect(typeof state.deleteModel).toBe('function');
      expect(typeof state.downloadModel).toBe('function');
      expect(typeof state.importModel).toBe('function');
      expect(typeof state.connectToJob).toBe('function');
      expect(typeof state.disconnectFromJob).toBe('function');
      expect(typeof state.clearError).toBe('function');
    });
  });

  describe('fetchJobs Action', () => {
    it('should fetch jobs successfully', async () => {
      const mockJobs = [
        {
          id: 'job-1',
          name: 'Test Job 1',
          status: 'running',
          config: {
            model_type: 'localization',
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
          },
          created_at: '2025-01-01T00:00:00Z',
          started_at: '2025-01-01T00:01:00Z',
          completed_at: null,
          current_epoch: 10,
          total_epochs: 100,
          progress: 0.1,
        },
        {
          id: 'job-2',
          name: 'Test Job 2',
          status: 'completed',
          config: {
            model_type: 'localization',
            epochs: 50,
            batch_size: 16,
            learning_rate: 0.0005,
          },
          created_at: '2025-01-02T00:00:00Z',
          started_at: '2025-01-02T00:01:00Z',
          completed_at: '2025-01-02T02:00:00Z',
          current_epoch: 50,
          total_epochs: 50,
          progress: 1.0,
        },
      ];

      vi.mocked(trainingService.listJobs).mockResolvedValue(mockJobs);

      await useTrainingStore.getState().fetchJobs();

      const state = useTrainingStore.getState();
      expect(state.jobs).toEqual(mockJobs);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(trainingService.listJobs).toHaveBeenCalledOnce();
    });

    it('should set loading state during fetch', async () => {
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      const promise = useTrainingStore.getState().fetchJobs();

      // Should be loading immediately
      expect(useTrainingStore.getState().isLoading).toBe(true);

      await promise;

      // Should not be loading after completion
      expect(useTrainingStore.getState().isLoading).toBe(false);
    });

    it('should handle fetch error gracefully', async () => {
      const errorMessage = 'Failed to fetch jobs';
      vi.mocked(trainingService.listJobs).mockRejectedValue(new Error(errorMessage));

      await useTrainingStore.getState().fetchJobs();

      const state = useTrainingStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
      expect(state.jobs).toEqual([]);
    });
  });

  describe('fetchJob Action (Single Job)', () => {
    it('should fetch single job successfully', async () => {
      const mockJob = {
        id: 'job-1',
        name: 'Test Job 1',
        status: 'running',
        config: {
          model_type: 'localization',
          epochs: 100,
          batch_size: 32,
          learning_rate: 0.001,
        },
        created_at: '2025-01-01T00:00:00Z',
        started_at: '2025-01-01T00:01:00Z',
        completed_at: null,
        current_epoch: 10,
        total_epochs: 100,
        progress: 0.1,
      };

      vi.mocked(trainingService.getJob).mockResolvedValue(mockJob);

      await useTrainingStore.getState().fetchJob('job-1');

      const state = useTrainingStore.getState();
      expect(state.currentJob).toEqual(mockJob);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(trainingService.getJob).toHaveBeenCalledWith('job-1');
    });

    it('should handle fetch job error', async () => {
      const errorMessage = 'Job not found';
      vi.mocked(trainingService.getJob).mockRejectedValue(new Error(errorMessage));

      await useTrainingStore.getState().fetchJob('invalid-job-id');

      const state = useTrainingStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.currentJob).toBe(null);
      expect(state.isLoading).toBe(false);
    });
  });

  describe('createJob Action', () => {
    it('should create job successfully', async () => {
      const newJobConfig = {
        name: 'New Training Job',
        config: {
          model_type: 'localization',
          epochs: 100,
          batch_size: 32,
          learning_rate: 0.001,
        },
      };

      const createdJob = {
        id: 'job-3',
        ...newJobConfig,
        status: 'pending',
        created_at: '2025-01-03T00:00:00Z',
        started_at: null,
        completed_at: null,
        current_epoch: 0,
        total_epochs: 100,
        progress: 0,
      };

      vi.mocked(trainingService.createJob).mockResolvedValue(createdJob);
      vi.mocked(trainingService.listJobs).mockResolvedValue([createdJob]);

      const result = await useTrainingStore.getState().createJob(newJobConfig);

      expect(result).toEqual(createdJob);
      expect(trainingService.createJob).toHaveBeenCalledWith(newJobConfig);
      // Should refresh jobs list after creation
      expect(trainingService.listJobs).toHaveBeenCalled();
    });

    it('should handle create job error', async () => {
      const errorMessage = 'Failed to create job';
      vi.mocked(trainingService.createJob).mockRejectedValue(new Error(errorMessage));

      const newJobConfig = {
        name: 'New Training Job',
        config: {
          model_type: 'localization',
          epochs: 100,
          batch_size: 32,
          learning_rate: 0.001,
        },
      };

      await expect(useTrainingStore.getState().createJob(newJobConfig)).rejects.toThrow(
        errorMessage
      );

      const state = useTrainingStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
    });
  });

  describe('Job Control Actions', () => {
    it('should pause job successfully', async () => {
      vi.mocked(trainingService.pauseJob).mockResolvedValue(undefined);
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      await useTrainingStore.getState().pauseJob('job-1');

      expect(trainingService.pauseJob).toHaveBeenCalledWith('job-1');
      // Should refresh jobs list after pause
      expect(trainingService.listJobs).toHaveBeenCalled();
    });

    it('should resume job successfully', async () => {
      vi.mocked(trainingService.resumeJob).mockResolvedValue(undefined);
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      await useTrainingStore.getState().resumeJob('job-1');

      expect(trainingService.resumeJob).toHaveBeenCalledWith('job-1');
      expect(trainingService.listJobs).toHaveBeenCalled();
    });

    it('should cancel job successfully', async () => {
      vi.mocked(trainingService.cancelJob).mockResolvedValue(undefined);
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      await useTrainingStore.getState().cancelJob('job-1');

      expect(trainingService.cancelJob).toHaveBeenCalledWith('job-1');
      expect(trainingService.listJobs).toHaveBeenCalled();
    });

    it('should handle job control errors', async () => {
      const errorMessage = 'Failed to pause job';
      vi.mocked(trainingService.pauseJob).mockRejectedValue(new Error(errorMessage));

      await useTrainingStore.getState().pauseJob('job-1');

      const state = useTrainingStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
    });
  });

  describe('deleteJob Action', () => {
    it('should delete job successfully', async () => {
      vi.mocked(trainingService.deleteJob).mockResolvedValue(undefined);
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      await useTrainingStore.getState().deleteJob('job-1');

      expect(trainingService.deleteJob).toHaveBeenCalledWith('job-1');
      // Should refresh jobs list after deletion
      expect(trainingService.listJobs).toHaveBeenCalled();
    });

    it('should clear current job if deleted', async () => {
      useTrainingStore.setState({
        currentJob: {
          id: 'job-1',
          name: 'Test Job',
          status: 'completed',
          config: { model_type: 'localization', epochs: 100 },
          created_at: '2025-01-01T00:00:00Z',
        },
      });

      vi.mocked(trainingService.deleteJob).mockResolvedValue(undefined);
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      await useTrainingStore.getState().deleteJob('job-1');

      expect(useTrainingStore.getState().currentJob).toBe(null);
    });
  });

  describe('Metrics Management', () => {
    it('should fetch job metrics successfully', async () => {
      const mockMetrics = {
        epochs: [1, 2, 3, 4, 5],
        train_loss: [0.5, 0.4, 0.3, 0.25, 0.2],
        val_loss: [0.6, 0.5, 0.4, 0.35, 0.3],
        train_accuracy: [0.8, 0.85, 0.9, 0.92, 0.95],
        val_accuracy: [0.75, 0.8, 0.85, 0.88, 0.9],
      };

      vi.mocked(trainingService.getJobMetrics).mockResolvedValue(mockMetrics);

      await useTrainingStore.getState().fetchJobMetrics('job-1');

      const state = useTrainingStore.getState();
      expect(state.metrics.get('job-1')).toEqual(mockMetrics);
      expect(trainingService.getJobMetrics).toHaveBeenCalledWith('job-1');
    });

    it('should handle metrics fetch error', async () => {
      const errorMessage = 'Failed to fetch metrics';
      vi.mocked(trainingService.getJobMetrics).mockRejectedValue(new Error(errorMessage));

      await useTrainingStore.getState().fetchJobMetrics('job-1');

      const state = useTrainingStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.metrics.has('job-1')).toBe(false);
    });
  });

  describe('Models Management', () => {
    it('should fetch models successfully', async () => {
      const mockModels = [
        {
          id: 'model-1',
          name: 'Model v1.0',
          version: '1.0.0',
          job_id: 'job-1',
          created_at: '2025-01-01T00:00:00Z',
          metrics: {
            accuracy: 0.95,
            loss: 0.2,
          },
          size_bytes: 1024000,
        },
        {
          id: 'model-2',
          name: 'Model v2.0',
          version: '2.0.0',
          job_id: 'job-2',
          created_at: '2025-01-02T00:00:00Z',
          metrics: {
            accuracy: 0.97,
            loss: 0.15,
          },
          size_bytes: 2048000,
        },
      ];

      vi.mocked(trainingService.listModels).mockResolvedValue(mockModels);

      await useTrainingStore.getState().fetchModels();

      const state = useTrainingStore.getState();
      expect(state.models).toEqual(mockModels);
      expect(trainingService.listModels).toHaveBeenCalledOnce();
    });

    it('should fetch single model successfully', async () => {
      const mockModel = {
        id: 'model-1',
        name: 'Model v1.0',
        version: '1.0.0',
        job_id: 'job-1',
        created_at: '2025-01-01T00:00:00Z',
        metrics: {
          accuracy: 0.95,
          loss: 0.2,
        },
        size_bytes: 1024000,
      };

      vi.mocked(trainingService.getModel).mockResolvedValue(mockModel);

      const result = await useTrainingStore.getState().fetchModel('model-1');

      expect(result).toEqual(mockModel);
      expect(trainingService.getModel).toHaveBeenCalledWith('model-1');
    });

    it('should delete model successfully', async () => {
      vi.mocked(trainingService.deleteModel).mockResolvedValue(undefined);
      vi.mocked(trainingService.listModels).mockResolvedValue([]);

      await useTrainingStore.getState().deleteModel('model-1');

      expect(trainingService.deleteModel).toHaveBeenCalledWith('model-1');
      // Should refresh models list after deletion
      expect(trainingService.listModels).toHaveBeenCalled();
    });
  });

  describe('Model Export/Import', () => {
    it('should download model successfully', async () => {
      const mockBlob = new Blob(['model data'], { type: 'application/octet-stream' });
      vi.mocked(trainingService.exportModel).mockResolvedValue(mockBlob);

      // Mock URL.createObjectURL and link click
      const createObjectURLSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url');
      const revokeObjectURLSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      const clickSpy = vi.fn();

      const originalCreateElement = document.createElement.bind(document);
      vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
        const element = originalCreateElement(tagName);
        if (tagName === 'a') {
          element.click = clickSpy;
        }
        return element;
      });

      await useTrainingStore.getState().downloadModel('model-1', {
        include_config: true,
        include_metrics: true,
      });

      expect(trainingService.exportModel).toHaveBeenCalledWith('model-1', {
        include_config: true,
        include_metrics: true,
      });
      expect(createObjectURLSpy).toHaveBeenCalledWith(mockBlob);
      expect(clickSpy).toHaveBeenCalled();
      expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url');

      // Cleanup
      createObjectURLSpy.mockRestore();
      revokeObjectURLSpy.mockRestore();
    });

    it('should import model successfully', async () => {
      const mockFile = new File(['model data'], 'model.heimdall', {
        type: 'application/octet-stream',
      });

      const mockImportedModel = {
        id: 'model-3',
        name: 'Imported Model',
        version: '1.0.0',
        created_at: '2025-01-03T00:00:00Z',
        metrics: { accuracy: 0.96, loss: 0.18 },
        size_bytes: 1536000,
      };

      vi.mocked(trainingService.importModel).mockResolvedValue(mockImportedModel);
      vi.mocked(trainingService.listModels).mockResolvedValue([mockImportedModel]);

      const result = await useTrainingStore.getState().importModel(mockFile);

      expect(result).toEqual(mockImportedModel);
      expect(trainingService.importModel).toHaveBeenCalled();
      // Should refresh models list after import
      expect(trainingService.listModels).toHaveBeenCalled();
    });

    it('should handle import error', async () => {
      const mockFile = new File(['invalid data'], 'model.heimdall', {
        type: 'application/octet-stream',
      });
      const errorMessage = 'Invalid model file';
      vi.mocked(trainingService.importModel).mockRejectedValue(new Error(errorMessage));

      await expect(useTrainingStore.getState().importModel(mockFile)).rejects.toThrow(
        errorMessage
      );

      const state = useTrainingStore.getState();
      expect(state.error).toBe(errorMessage);
    });
  });

  describe('Error Handling', () => {
    it('should clear error', () => {
      useTrainingStore.setState({ error: 'Some error' });
      useTrainingStore.getState().clearError();
      expect(useTrainingStore.getState().error).toBe(null);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty jobs list', async () => {
      vi.mocked(trainingService.listJobs).mockResolvedValue([]);

      await useTrainingStore.getState().fetchJobs();

      const state = useTrainingStore.getState();
      expect(state.jobs).toEqual([]);
    });

    it('should handle empty models list', async () => {
      vi.mocked(trainingService.listModels).mockResolvedValue([]);

      await useTrainingStore.getState().fetchModels();

      const state = useTrainingStore.getState();
      expect(state.models).toEqual([]);
    });

    it('should handle multiple metrics for different jobs', async () => {
      const mockMetrics1 = {
        epochs: [1, 2],
        train_loss: [0.5, 0.4],
        val_loss: [0.6, 0.5],
      };
      const mockMetrics2 = {
        epochs: [1, 2, 3],
        train_loss: [0.4, 0.3, 0.2],
        val_loss: [0.5, 0.4, 0.3],
      };

      vi.mocked(trainingService.getJobMetrics)
        .mockResolvedValueOnce(mockMetrics1)
        .mockResolvedValueOnce(mockMetrics2);

      await useTrainingStore.getState().fetchJobMetrics('job-1');
      await useTrainingStore.getState().fetchJobMetrics('job-2');

      const state = useTrainingStore.getState();
      expect(state.metrics.get('job-1')).toEqual(mockMetrics1);
      expect(state.metrics.get('job-2')).toEqual(mockMetrics2);
      expect(state.metrics.size).toBe(2);
    });
  });
});
