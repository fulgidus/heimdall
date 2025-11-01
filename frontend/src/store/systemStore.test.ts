/**
 * System Store Tests
 *
 * Comprehensive test suite for the systemStore Zustand store
 * Tests all actions: service health checking, model performance, selectors
 * Truth-first approach: Tests real Zustand store behavior with mocked API responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/systemStore');

// Import after unmocking
import { useSystemStore } from './systemStore';
import { systemService, inferenceService } from '@/services/api';

// Mock the API services
vi.mock('@/services/api', () => ({
  systemService: {
    checkAllServicesHealth: vi.fn(),
    checkServiceHealth: vi.fn(),
  },
  inferenceService: {
    getModelPerformance: vi.fn(),
  },
  webSDRService: {},
  acquisitionService: {},
  analyticsService: {},
  sessionService: {},
}));

describe('System Store (Zustand)', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useSystemStore.setState({
      servicesHealth: {},
      modelPerformance: null,
      isLoading: false,
      error: null,
      lastCheck: null,
    });
    vi.clearAllMocks();
  });

  describe('Store Initialization', () => {
    it('should initialize with default state', () => {
      const state = useSystemStore.getState();
      expect(state.servicesHealth).toEqual({});
      expect(state.modelPerformance).toBe(null);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(state.lastCheck).toBe(null);
    });

    it('should have all required actions', () => {
      const state = useSystemStore.getState();
      expect(typeof state.checkAllServices).toBe('function');
      expect(typeof state.checkService).toBe('function');
      expect(typeof state.fetchModelPerformance).toBe('function');
      expect(typeof state.isServiceHealthy).toBe('function');
      expect(typeof state.getServiceStatus).toBe('function');
      expect(typeof state.refreshAll).toBe('function');
    });
  });

  describe('checkAllServices Action', () => {
    it('should check all services successfully', async () => {
      const mockServicesHealth = {
        'api-gateway': { status: 'healthy', latency_ms: 10 },
        backend: { status: 'healthy', latency_ms: 50 },
        training: { status: 'degraded', latency_ms: 300 },
        inference: { status: 'healthy', latency_ms: 45 },
      };

      vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue(mockServicesHealth);

      const beforeCheck = new Date();
      await useSystemStore.getState().checkAllServices();
      const afterCheck = new Date();

      const state = useSystemStore.getState();
      expect(state.servicesHealth).toEqual(mockServicesHealth);
      expect(state.lastCheck).toBeTruthy();
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);

      if (state.lastCheck) {
        const checkTime = new Date(state.lastCheck);
        expect(checkTime.getTime()).toBeGreaterThanOrEqual(beforeCheck.getTime());
        expect(checkTime.getTime()).toBeLessThanOrEqual(afterCheck.getTime());
      }
    });

    it('should set loading state during check', async () => {
      vi.mocked(systemService.checkAllServicesHealth).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve({}), 50))
      );

      const promise = useSystemStore.getState().checkAllServices();

      // Should be loading immediately
      expect(useSystemStore.getState().isLoading).toBe(true);

      await promise;

      // Should not be loading after completion
      expect(useSystemStore.getState().isLoading).toBe(false);
    });

    it('should handle check error gracefully', async () => {
      const errorMessage = 'Services unavailable';
      vi.mocked(systemService.checkAllServicesHealth).mockRejectedValue(new Error(errorMessage));

      await useSystemStore.getState().checkAllServices();

      const state = useSystemStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
    });

    it('should clear error on successful check', async () => {
      useSystemStore.setState({ error: 'Previous error' });

      vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({});

      await useSystemStore.getState().checkAllServices();

      const state = useSystemStore.getState();
      expect(state.error).toBe(null);
    });
  });

  describe('checkService Action', () => {
    it('should check single service successfully', async () => {
      const mockServiceHealth = { status: 'healthy', latency_ms: 25 };

      vi.mocked(systemService.checkServiceHealth).mockResolvedValue(mockServiceHealth);

      await useSystemStore.getState().checkService('api-gateway');

      const state = useSystemStore.getState();
      expect(state.servicesHealth['api-gateway']).toEqual(mockServiceHealth);
      expect(systemService.checkServiceHealth).toHaveBeenCalledWith('api-gateway');
    });

    it('should update existing service health', async () => {
      // Set initial state
      useSystemStore.setState({
        servicesHealth: {
          'api-gateway': { status: 'degraded', latency_ms: 100 },
          backend: { status: 'healthy', latency_ms: 50 },
        },
      });

      const updatedHealth = { status: 'healthy', latency_ms: 20 };
      vi.mocked(systemService.checkServiceHealth).mockResolvedValue(updatedHealth);

      await useSystemStore.getState().checkService('api-gateway');

      const state = useSystemStore.getState();
      expect(state.servicesHealth['api-gateway']).toEqual(updatedHealth);
      // Other services should remain unchanged
      expect(state.servicesHealth['backend']).toEqual({ status: 'healthy', latency_ms: 50 });
    });

    it('should handle service check error gracefully', async () => {
      const errorMessage = 'Service not found';
      vi.mocked(systemService.checkServiceHealth).mockRejectedValue(new Error(errorMessage));

      // Should not throw or set error state (errors are logged only)
      await useSystemStore.getState().checkService('unknown-service');

      const state = useSystemStore.getState();
      // Error should not be set for single service checks
      expect(state.error).toBe(null);
    });
  });

  describe('fetchModelPerformance Action', () => {
    it('should fetch model performance successfully', async () => {
      const mockPerformance = {
        accuracy: 0.85,
        latency_avg_ms: 150,
        predictions_total: 1000,
        predictions_successful: 850,
      };

      vi.mocked(inferenceService.getModelPerformance).mockResolvedValue(mockPerformance);

      await useSystemStore.getState().fetchModelPerformance();

      const state = useSystemStore.getState();
      expect(state.modelPerformance).toEqual(mockPerformance);
    });

    it('should handle fetch error gracefully', async () => {
      const errorMessage = 'Model service unavailable';
      vi.mocked(inferenceService.getModelPerformance).mockRejectedValue(new Error(errorMessage));

      // Should not throw or set error state (not critical)
      await useSystemStore.getState().fetchModelPerformance();

      const state = useSystemStore.getState();
      expect(state.modelPerformance).toBe(null);
      expect(state.error).toBe(null);
    });

    it('should update existing model performance', async () => {
      const initialPerformance = {
        accuracy: 0.8,
        latency_avg_ms: 200,
        predictions_total: 500,
        predictions_successful: 400,
      };
      useSystemStore.setState({ modelPerformance: initialPerformance });

      const updatedPerformance = {
        accuracy: 0.85,
        latency_avg_ms: 150,
        predictions_total: 1000,
        predictions_successful: 850,
      };
      vi.mocked(inferenceService.getModelPerformance).mockResolvedValue(updatedPerformance);

      await useSystemStore.getState().fetchModelPerformance();

      const state = useSystemStore.getState();
      expect(state.modelPerformance).toEqual(updatedPerformance);
    });
  });

  describe('Selector Functions', () => {
    beforeEach(() => {
      useSystemStore.setState({
        servicesHealth: {
          'api-gateway': { status: 'healthy', latency_ms: 10 },
          backend: { status: 'degraded', latency_ms: 250 },
          training: { status: 'unhealthy', latency_ms: 1000 },
        },
      });
    });

    describe('isServiceHealthy', () => {
      it('should return true for healthy service', () => {
        const isHealthy = useSystemStore.getState().isServiceHealthy('api-gateway');
        expect(isHealthy).toBe(true);
      });

      it('should return false for degraded service', () => {
        const isHealthy = useSystemStore.getState().isServiceHealthy('backend');
        expect(isHealthy).toBe(false);
      });

      it('should return false for unhealthy service', () => {
        const isHealthy = useSystemStore.getState().isServiceHealthy('training');
        expect(isHealthy).toBe(false);
      });

      it('should return false for unknown service', () => {
        const isHealthy = useSystemStore.getState().isServiceHealthy('unknown-service');
        expect(isHealthy).toBe(false);
      });
    });

    describe('getServiceStatus', () => {
      it('should return status for existing service', () => {
        const status = useSystemStore.getState().getServiceStatus('api-gateway');
        expect(status).toEqual({ status: 'healthy', latency_ms: 10 });
      });

      it('should return null for non-existent service', () => {
        const status = useSystemStore.getState().getServiceStatus('unknown-service');
        expect(status).toBe(null);
      });
    });
  });

  describe('refreshAll Action', () => {
    it('should refresh all system data', async () => {
      const mockServicesHealth = {
        'api-gateway': { status: 'healthy', latency_ms: 10 },
      };
      const mockModelPerformance = {
        accuracy: 0.85,
        latency_avg_ms: 150,
        predictions_total: 1000,
        predictions_successful: 850,
      };

      vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue(mockServicesHealth);
      vi.mocked(inferenceService.getModelPerformance).mockResolvedValue(mockModelPerformance);

      await useSystemStore.getState().refreshAll();

      const state = useSystemStore.getState();
      expect(state.servicesHealth).toEqual(mockServicesHealth);
      expect(state.modelPerformance).toEqual(mockModelPerformance);
      expect(systemService.checkAllServicesHealth).toHaveBeenCalledOnce();
      expect(inferenceService.getModelPerformance).toHaveBeenCalledOnce();
    });

    it('should handle partial failures', async () => {
      vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({
        'api-gateway': { status: 'healthy', latency_ms: 10 },
      });
      vi.mocked(inferenceService.getModelPerformance).mockRejectedValue(
        new Error('Model service unavailable')
      );

      await useSystemStore.getState().refreshAll();

      const state = useSystemStore.getState();
      // Services health should be updated even if model performance fails
      expect(state.servicesHealth).toEqual({
        'api-gateway': { status: 'healthy', latency_ms: 10 },
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty services health', async () => {
      vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({});

      await useSystemStore.getState().checkAllServices();

      const state = useSystemStore.getState();
      expect(state.servicesHealth).toEqual({});
      expect(state.error).toBe(null);
    });

    it('should handle null model performance', async () => {
      vi.mocked(inferenceService.getModelPerformance).mockResolvedValue(null);

      await useSystemStore.getState().fetchModelPerformance();

      const state = useSystemStore.getState();
      expect(state.modelPerformance).toBe(null);
    });

    it('should handle non-Error exceptions', async () => {
      vi.mocked(systemService.checkAllServicesHealth).mockRejectedValue('String error');

      await useSystemStore.getState().checkAllServices();

      const state = useSystemStore.getState();
      expect(state.error).toBe('Failed to check services');
    });

    it('should maintain state integrity across multiple operations', async () => {
      // First check
      vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({
        'api-gateway': { status: 'healthy', latency_ms: 10 },
      });
      await useSystemStore.getState().checkAllServices();
      expect(useSystemStore.getState().servicesHealth['api-gateway']).toBeDefined();

      // Single service check
      vi.mocked(systemService.checkServiceHealth).mockResolvedValue({
        status: 'healthy',
        latency_ms: 50,
      });
      await useSystemStore.getState().checkService('backend');

      const state = useSystemStore.getState();
      expect(state.servicesHealth['api-gateway']).toBeDefined();
      expect(state.servicesHealth['backend']).toBeDefined();
    });

    it('should handle concurrent service checks', async () => {
      vi.mocked(systemService.checkServiceHealth).mockResolvedValue({
        status: 'healthy',
        latency_ms: 50,
      });

      // Start multiple concurrent checks
      await Promise.all([
        useSystemStore.getState().checkService('service1'),
        useSystemStore.getState().checkService('service2'),
        useSystemStore.getState().checkService('service3'),
      ]);

      const state = useSystemStore.getState();
      expect(Object.keys(state.servicesHealth)).toHaveLength(3);
    });
  });
});
