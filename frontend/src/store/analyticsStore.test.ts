/**
 * Analytics Store Tests
 *
 * Comprehensive test suite for the analyticsStore Zustand store
 * Tests all actions: loading, fetching, error handling, and state management
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useAnalyticsStore } from './analyticsStore';
import { analyticsService } from '@/services/api';

// Mock the analytics service BEFORE importing the store
vi.mock('@/services/api', () => ({
  analyticsService: {
    getPredictionMetrics: vi.fn(),
    getWebSDRPerformance: vi.fn(),
    getSystemPerformance: vi.fn(),
    getAccuracyDistribution: vi.fn(),
  },
  webSDRService: {},
  acquisitionService: {},
  inferenceService: {},
  systemService: {},
  sessionService: {},
}));

describe('Analytics Store (Zustand)', () => {
  beforeEach(() => {
    // Reset store state
    useAnalyticsStore.setState({
      isLoading: false,
      error: null,
      predictionMetrics: null,
      websdrPerformance: [],
      systemPerformance: null,
      accuracyDistribution: null,
      timeRange: '7d',
    });
    vi.clearAllMocks();
  });

  describe('Store Initialization', () => {
    it('should initialize with default state', () => {
      const state = useAnalyticsStore.getState();
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(state.timeRange).toBe('7d');
      expect(state.predictionMetrics).toBe(null);
    });

    it('should have all required actions', () => {
      const state = useAnalyticsStore.getState();
      expect(typeof state.setTimeRange).toBe('function');
      expect(typeof state.fetchAllAnalytics).toBe('function');
      expect(typeof state.setError).toBe('function');
    });
  });

  describe('setTimeRange Action', () => {
    it('should update time range', () => {
      useAnalyticsStore.getState().setTimeRange('30d');
      expect(useAnalyticsStore.getState().timeRange).toBe('30d');
    });

    it('should accept valid time ranges', () => {
      const validRanges = ['24h', '7d', '30d', '90d'];
      validRanges.forEach(range => {
        useAnalyticsStore.getState().setTimeRange(range);
        expect(useAnalyticsStore.getState().timeRange).toBe(range);
      });
    });
  });

  describe('fetchAnalytics Action', () => {
    it('should set loading state when fetching starts', async () => {
      const mockMetrics = {
        total_predictions: [],
        successful_predictions: [],
        failed_predictions: [],
        average_confidence: [],
        average_uncertainty: [],
      };

      vi.mocked(analyticsService.getPredictionMetrics).mockResolvedValue(mockMetrics);

      const promise = useAnalyticsStore.getState().fetchAllAnalytics();
      // State should show loading immediately
      expect(useAnalyticsStore.getState().isLoading).toBe(true);

      await promise;
      expect(useAnalyticsStore.getState().isLoading).toBe(false);
    });

    it('should fetch all analytics data successfully', async () => {
      const mockPredictionMetrics = {
        total_predictions: [
          { timestamp: '2025-01-01T00:00:00', value: 100 },
          { timestamp: '2025-01-02T00:00:00', value: 150 },
        ],
        successful_predictions: [
          { timestamp: '2025-01-01T00:00:00', value: 85 },
          { timestamp: '2025-01-02T00:00:00', value: 127 },
        ],
        failed_predictions: [
          { timestamp: '2025-01-01T00:00:00', value: 15 },
          { timestamp: '2025-01-02T00:00:00', value: 23 },
        ],
        average_confidence: [
          { timestamp: '2025-01-01T00:00:00', value: 0.85 },
          { timestamp: '2025-01-02T00:00:00', value: 0.88 },
        ],
        average_uncertainty: [
          { timestamp: '2025-01-01T00:00:00', value: 25.5 },
          { timestamp: '2025-01-02T00:00:00', value: 24.8 },
        ],
      };

      const mockWebSDRPerformance = [
        {
          websdr_id: 1,
          name: 'WebSDR Italy 1',
          uptime_percentage: 99.5,
          average_snr: 15.2,
          total_acquisitions: 450,
          successful_acquisitions: 425,
        },
      ];

      const mockSystemPerformance = {
        cpu_usage: [
          { timestamp: '2025-01-01T00:00:00', value: 45 },
          { timestamp: '2025-01-02T00:00:00', value: 52 },
        ],
        memory_usage: [
          { timestamp: '2025-01-01T00:00:00', value: 2048 },
          { timestamp: '2025-01-02T00:00:00', value: 2256 },
        ],
        api_response_times: [
          { timestamp: '2025-01-01T00:00:00', value: 145 },
          { timestamp: '2025-01-02T00:00:00', value: 128 },
        ],
        active_tasks: [
          { timestamp: '2025-01-01T00:00:00', value: 3 },
          { timestamp: '2025-01-02T00:00:00', value: 5 },
        ],
      };

      const mockAccuracyDistribution = {
        accuracy_ranges: ['<10m', '10-20m', '20-30m', '30-50m', '50-100m', '>100m'],
        counts: [15, 45, 120, 80, 35, 10],
      };

      vi.mocked(analyticsService.getPredictionMetrics).mockResolvedValue(mockPredictionMetrics);
      vi.mocked(analyticsService.getWebSDRPerformance).mockResolvedValue(mockWebSDRPerformance);
      vi.mocked(analyticsService.getSystemPerformance).mockResolvedValue(mockSystemPerformance);
      vi.mocked(analyticsService.getAccuracyDistribution).mockResolvedValue(
        mockAccuracyDistribution
      );

      await useAnalyticsStore.getState().fetchAllAnalytics();

      const state = useAnalyticsStore.getState();
      expect(state.predictionMetrics).toEqual(mockPredictionMetrics);
      expect(state.websdrPerformance).toEqual(mockWebSDRPerformance);
      expect(state.systemPerformance).toEqual(mockSystemPerformance);
      expect(state.accuracyDistribution).toEqual(mockAccuracyDistribution);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
    });

    it('should handle errors from prediction metrics API', async () => {
      const mockError = new Error('Network error');
      vi.mocked(analyticsService.getPredictionMetrics).mockRejectedValue(mockError);
      vi.mocked(analyticsService.getWebSDRPerformance).mockResolvedValue([]);
      vi.mocked(analyticsService.getSystemPerformance).mockResolvedValue({
        cpu_usage: [],
        memory_usage: [],
        api_response_times: [],
        active_tasks: [],
      });
      vi.mocked(analyticsService.getAccuracyDistribution).mockResolvedValue({
        accuracy_ranges: [],
        counts: [],
      });

      await useAnalyticsStore.getState().fetchAllAnalytics();

      const state = useAnalyticsStore.getState();
      expect(state.isLoading).toBe(false);
      // When one metric fails, predictionMetrics will be null (due to error in fetchPredictionMetrics)
      // and other metrics will be successfully fetched
      expect(state.predictionMetrics).toBe(null);
      expect(state.websdrPerformance).toEqual([]);
      expect(state.systemPerformance).toBeTruthy();
    });

    it('should handle errors from WebSDR performance API', async () => {
      const mockError = new Error('Backend service unavailable');
      vi.mocked(analyticsService.getPredictionMetrics).mockResolvedValue({
        total_predictions: [],
        successful_predictions: [],
        failed_predictions: [],
        average_confidence: [],
        average_uncertainty: [],
      });
      vi.mocked(analyticsService.getWebSDRPerformance).mockRejectedValue(mockError);
      vi.mocked(analyticsService.getSystemPerformance).mockResolvedValue({
        cpu_usage: [],
        memory_usage: [],
        api_response_times: [],
        active_tasks: [],
      });
      vi.mocked(analyticsService.getAccuracyDistribution).mockResolvedValue({
        accuracy_ranges: [],
        counts: [],
      });

      await useAnalyticsStore.getState().fetchAllAnalytics();

      const state = useAnalyticsStore.getState();
      expect(state.isLoading).toBe(false);
      // When WebSDR performance fails, it will be empty array
      // and other metrics will be successfully fetched
      expect(state.predictionMetrics).toBeTruthy();
      expect(state.websdrPerformance).toEqual([]);
      expect(state.systemPerformance).toBeTruthy();
    });

    it('should fetch analytics with specific time range', async () => {
      vi.mocked(analyticsService.getPredictionMetrics).mockResolvedValue({
        total_predictions: [],
        successful_predictions: [],
        failed_predictions: [],
        average_confidence: [],
        average_uncertainty: [],
      });
      vi.mocked(analyticsService.getWebSDRPerformance).mockResolvedValue([]);
      vi.mocked(analyticsService.getSystemPerformance).mockResolvedValue({
        cpu_usage: [],
        memory_usage: [],
        api_response_times: [],
        active_tasks: [],
      });
      vi.mocked(analyticsService.getAccuracyDistribution).mockResolvedValue({
        accuracy_ranges: [],
        counts: [],
      });

      useAnalyticsStore.getState().setTimeRange('30d');
      await useAnalyticsStore.getState().fetchAllAnalytics();

      expect(analyticsService.getPredictionMetrics).toHaveBeenCalledWith('30d');
      expect(analyticsService.getWebSDRPerformance).toHaveBeenCalledWith('30d');
      expect(analyticsService.getSystemPerformance).toHaveBeenCalledWith('30d');
      expect(analyticsService.getAccuracyDistribution).toHaveBeenCalledWith('30d');
    });
  });

  describe('setError Action', () => {
    it('should set error message', () => {
      const errorMessage = 'Something went wrong';
      useAnalyticsStore.getState().setError(errorMessage);
      expect(useAnalyticsStore.getState().error).toBe(errorMessage);
    });

    it('should clear error when passed null', () => {
      useAnalyticsStore.getState().setError('Test error');
      expect(useAnalyticsStore.getState().error).toBe('Test error');
      useAnalyticsStore.getState().setError(null);
      expect(useAnalyticsStore.getState().error).toBe(null);
    });
  });

  describe('State Persistence', () => {
    it('should maintain state across multiple actions', async () => {
      const mockMetrics = {
        total_predictions: [{ timestamp: '2025-01-01T00:00:00', value: 100 }],
        successful_predictions: [{ timestamp: '2025-01-01T00:00:00', value: 85 }],
        failed_predictions: [{ timestamp: '2025-01-01T00:00:00', value: 15 }],
        average_confidence: [{ timestamp: '2025-01-01T00:00:00', value: 0.85 }],
        average_uncertainty: [{ timestamp: '2025-01-01T00:00:00', value: 25.5 }],
      };

      vi.mocked(analyticsService.getPredictionMetrics).mockResolvedValue(mockMetrics);
      vi.mocked(analyticsService.getWebSDRPerformance).mockResolvedValue([]);
      vi.mocked(analyticsService.getSystemPerformance).mockResolvedValue({
        cpu_usage: [],
        memory_usage: [],
        api_response_times: [],
        active_tasks: [],
      });
      vi.mocked(analyticsService.getAccuracyDistribution).mockResolvedValue({
        accuracy_ranges: [],
        counts: [],
      });

      useAnalyticsStore.getState().setTimeRange('24h');
      await useAnalyticsStore.getState().fetchAllAnalytics();

      let state = useAnalyticsStore.getState();
      expect(state.timeRange).toBe('24h');
      expect(state.predictionMetrics).toEqual(mockMetrics);

      // Change time range, state should persist other values
      useAnalyticsStore.getState().setTimeRange('30d');
      state = useAnalyticsStore.getState();
      expect(state.timeRange).toBe('30d');
      expect(state.predictionMetrics).toEqual(mockMetrics); // Should still be there
    });
  });
});
