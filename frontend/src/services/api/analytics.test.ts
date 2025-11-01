/**
 * Analytics API Service Tests
 *
 * Test suite for all analytics API endpoints
 * Verifies correct HTTP calls, error handling, and response parsing
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import api from '@/lib/api';
import {
  getPredictionMetrics,
  getWebSDRPerformance,
  getSystemPerformance,
  getAccuracyDistribution,
  type PredictionMetrics,
  type WebSDRPerformance,
  type SystemPerformance,
} from './analytics';

// Mock axios instance
vi.mock('@/lib/api', () => ({
  default: {
    get: vi.fn(),
  },
}));

describe('Analytics API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getPredictionMetrics', () => {
    it('should fetch prediction metrics with correct URL and params', async () => {
      const mockMetrics: PredictionMetrics = {
        total_predictions: [{ timestamp: '2025-01-01T00:00:00', value: 100 }],
        successful_predictions: [{ timestamp: '2025-01-01T00:00:00', value: 85 }],
        failed_predictions: [{ timestamp: '2025-01-01T00:00:00', value: 15 }],
        average_confidence: [{ timestamp: '2025-01-01T00:00:00', value: 0.85 }],
        average_uncertainty: [{ timestamp: '2025-01-01T00:00:00', value: 25.5 }],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockMetrics });

      const result = await getPredictionMetrics('7d');

      expect(api.get).toHaveBeenCalledWith('/api/v1/analytics/predictions/metrics', {
        params: { time_range: '7d' },
      });
      expect(result).toEqual(mockMetrics);
    });

    it('should use default time range when not specified', async () => {
      const mockMetrics: PredictionMetrics = {
        total_predictions: [],
        successful_predictions: [],
        failed_predictions: [],
        average_confidence: [],
        average_uncertainty: [],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockMetrics });

      await getPredictionMetrics();

      expect(api.get).toHaveBeenCalledWith('/api/v1/analytics/predictions/metrics', {
        params: { time_range: '7d' },
      });
    });

    it('should handle API errors gracefully', async () => {
      const error = new Error('Network Error');
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getPredictionMetrics('7d')).rejects.toThrow('Network Error');
    });

    it('should accept various time ranges', async () => {
      const mockMetrics: PredictionMetrics = {
        total_predictions: [],
        successful_predictions: [],
        failed_predictions: [],
        average_confidence: [],
        average_uncertainty: [],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockMetrics });

      const timeRanges = ['24h', '7d', '30d', '90d'];

      for (const range of timeRanges) {
        await getPredictionMetrics(range);
        expect(api.get).toHaveBeenCalledWith('/api/v1/analytics/predictions/metrics', {
          params: { time_range: range },
        });
      }
    });
  });

  describe('getWebSDRPerformance', () => {
    it('should fetch WebSDR performance with correct URL and params', async () => {
      const mockPerformance: WebSDRPerformance[] = [
        {
          websdr_id: 1,
          name: 'WebSDR Italy 1',
          uptime_percentage: 99.5,
          average_snr: 15.2,
          total_acquisitions: 450,
          successful_acquisitions: 425,
        },
        {
          websdr_id: 2,
          name: 'WebSDR Italy 2',
          uptime_percentage: 98.8,
          average_snr: 14.5,
          total_acquisitions: 420,
          successful_acquisitions: 410,
        },
      ];

      vi.mocked(api.get).mockResolvedValue({ data: mockPerformance });

      const result = await getWebSDRPerformance('7d');

      expect(api.get).toHaveBeenCalledWith('/api/v1/analytics/websdr/performance', {
        params: { time_range: '7d' },
      });
      expect(result).toEqual(mockPerformance);
      expect(result).toHaveLength(2);
    });

    it('should return empty array when no WebSDRs available', async () => {
      vi.mocked(api.get).mockResolvedValue({ data: [] });

      const result = await getWebSDRPerformance('7d');

      expect(result).toEqual([]);
      expect(result).toHaveLength(0);
    });

    it('should handle performance API errors', async () => {
      const error = new Error('Service Unavailable');
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getWebSDRPerformance('7d')).rejects.toThrow('Service Unavailable');
    });

    it('should parse WebSDR performance metrics correctly', async () => {
      const mockPerformance: WebSDRPerformance[] = [
        {
          websdr_id: 1,
          name: 'Test WebSDR',
          uptime_percentage: 99.9,
          average_snr: 16.0,
          total_acquisitions: 1000,
          successful_acquisitions: 999,
        },
      ];

      vi.mocked(api.get).mockResolvedValue({ data: mockPerformance });

      const result = await getWebSDRPerformance('30d');

      expect(result[0].websdr_id).toBe(1);
      expect(result[0].uptime_percentage).toBe(99.9);
      expect(result[0].total_acquisitions).toBe(1000);
    });
  });

  describe('getSystemPerformance', () => {
    it('should fetch system performance with correct URL and params', async () => {
      const mockPerformance: SystemPerformance = {
        cpu_usage: [{ timestamp: '2025-01-01T00:00:00', value: 45 }],
        memory_usage: [{ timestamp: '2025-01-01T00:00:00', value: 2048 }],
        api_response_times: [{ timestamp: '2025-01-01T00:00:00', value: 145 }],
        active_tasks: [{ timestamp: '2025-01-01T00:00:00', value: 3 }],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockPerformance });

      const result = await getSystemPerformance('7d');

      expect(api.get).toHaveBeenCalledWith('/api/v1/analytics/system/performance', {
        params: { time_range: '7d' },
      });
      expect(result).toEqual(mockPerformance);
    });

    it('should handle system performance API errors', async () => {
      const error = new Error('Backend Error');
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getSystemPerformance('7d')).rejects.toThrow('Backend Error');
    });

    it('should include all performance metrics', async () => {
      const mockPerformance: SystemPerformance = {
        cpu_usage: [{ timestamp: '2025-01-01T00:00:00', value: 50 }],
        memory_usage: [{ timestamp: '2025-01-01T00:00:00', value: 2560 }],
        api_response_times: [{ timestamp: '2025-01-01T00:00:00', value: 200 }],
        active_tasks: [{ timestamp: '2025-01-01T00:00:00', value: 5 }],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockPerformance });

      const result = await getSystemPerformance('7d');

      expect(result.cpu_usage).toBeDefined();
      expect(result.memory_usage).toBeDefined();
      expect(result.api_response_times).toBeDefined();
      expect(result.active_tasks).toBeDefined();
    });
  });

  describe('getAccuracyDistribution', () => {
    it('should fetch accuracy distribution with correct URL and params', async () => {
      const mockDistribution = {
        accuracy_ranges: ['<10m', '10-20m', '20-30m', '30-50m', '50-100m', '>100m'],
        counts: [15, 45, 120, 80, 35, 10],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockDistribution });

      const result = await getAccuracyDistribution('7d');

      expect(api.get).toHaveBeenCalledWith(
        '/api/v1/analytics/localizations/accuracy-distribution',
        {
          params: { time_range: '7d' },
        }
      );
      expect(result).toEqual(mockDistribution);
    });

    it('should verify accuracy range labels', async () => {
      const mockDistribution = {
        accuracy_ranges: ['<10m', '10-20m', '20-30m', '30-50m', '50-100m', '>100m'],
        counts: [10, 20, 30, 25, 10, 5],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockDistribution });

      const result = await getAccuracyDistribution('7d');

      expect(result.accuracy_ranges).toHaveLength(6);
      expect(result.counts).toHaveLength(6);
      expect(result.counts.reduce((a, b) => a + b, 0)).toBe(100);
    });

    it('should handle accuracy distribution API errors', async () => {
      const error = new Error('Distribution Service Error');
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getAccuracyDistribution('7d')).rejects.toThrow('Distribution Service Error');
    });
  });

  describe('Error Handling', () => {
    it('should handle 401 Unauthorized errors', async () => {
      interface ErrorWithResponse extends Error {
        response?: { status: number };
      }
      const error: ErrorWithResponse = new Error('Unauthorized');
      error.response = { status: 401 };
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getPredictionMetrics('7d')).rejects.toThrow('Unauthorized');
    });

    it('should handle 404 Not Found errors', async () => {
      interface ErrorWithResponse extends Error {
        response?: { status: number };
      }
      const error: ErrorWithResponse = new Error('Not Found');
      error.response = { status: 404 };
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getPredictionMetrics('7d')).rejects.toThrow('Not Found');
    });

    it('should handle 500 Server errors', async () => {
      interface ErrorWithResponse extends Error {
        response?: { status: number };
      }
      const error: ErrorWithResponse = new Error('Internal Server Error');
      error.response = { status: 500 };
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getPredictionMetrics('7d')).rejects.toThrow('Internal Server Error');
    });

    it('should handle network timeouts', async () => {
      interface ErrorWithCode extends Error {
        code?: string;
      }
      const error: ErrorWithCode = new Error('Request Timeout');
      error.code = 'ECONNABORTED';
      vi.mocked(api.get).mockRejectedValue(error);

      await expect(getPredictionMetrics('7d')).rejects.toThrow('Request Timeout');
    });
  });
});
