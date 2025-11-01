/**
 * System API Service Tests
 *
 * Comprehensive test suite for System API client
 * Tests service health checking, error handling, and parallel operations
 * Truth-first approach: Tests real API client with mocked HTTP responses
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import MockAdapter from 'axios-mock-adapter';
import api from '@/lib/api';
import { checkServiceHealth, checkAllServicesHealth, getAPIGatewayStatus } from './system';
import type { ServiceHealth } from './types';

// Mock the auth store
vi.mock('@/store', () => ({
  useAuthStore: {
    getState: vi.fn(() => ({ token: null })),
  },
}));

// Create axios mock adapter
let mock: MockAdapter;

describe('System API Service', () => {
  beforeEach(() => {
    mock = new MockAdapter(api);
  });

  afterEach(() => {
    mock.reset();
    mock.restore();
  });

  describe('checkServiceHealth', () => {
    it('should check single service health successfully', async () => {
      const mockHealth: ServiceHealth = {
        status: 'healthy',
        service: 'backend',
        version: '1.0.0',
        timestamp: '2025-01-01T00:00:00Z',
        details: {},
      };

      mock.onGet('/backend/health').reply(200, mockHealth);

      const result = await checkServiceHealth('backend');

      expect(result).toEqual(mockHealth);
      expect(result.status).toBe('healthy');
      expect(result.service).toBe('backend');
    });

    it('should check degraded service', async () => {
      const mockHealth: ServiceHealth = {
        status: 'degraded',
        service: 'training',
        version: '1.0.0',
        timestamp: '2025-01-01T00:00:00Z',
        details: { latency_ms: 500 },
      };

      mock.onGet('/training/health').reply(200, mockHealth);

      const result = await checkServiceHealth('training');

      expect(result.status).toBe('degraded');
    });

    it('should handle service health error', async () => {
      mock.onGet('/unknown-service/health').reply(404, {
        detail: 'Service not found',
      });

      await expect(checkServiceHealth('unknown-service')).rejects.toThrow();
    });
  });

  describe('checkAllServicesHealth', () => {
    it('should check all services successfully', async () => {
      mock.onGet('/backend/health').reply(200, {
        status: 'healthy',
        service: 'backend',
        version: '1.0.0',
        timestamp: '2025-01-01T00:00:00Z',
        details: {},
      });

      mock.onGet('/training/health').reply(200, {
        status: 'healthy',
        service: 'training',
        version: '1.0.0',
        timestamp: '2025-01-01T00:00:00Z',
        details: {},
      });

      mock.onGet('/inference/health').reply(200, {
        status: 'healthy',
        service: 'inference',
        version: '1.0.0',
        timestamp: '2025-01-01T00:00:00Z',
        details: {},
      });

      const result = await checkAllServicesHealth();

      expect(Object.keys(result)).toHaveLength(3);
      expect(result['backend'].status).toBe('healthy');
      expect(result['training'].status).toBe('healthy');
      expect(result['inference'].status).toBe('healthy');
    });

    it('should handle partial failures gracefully', async () => {
      mock.onGet('/backend/health').reply(500, {
        detail: 'Internal error',
      });

      mock.onGet('/training/health').reply(200, {
        status: 'healthy',
        service: 'training',
        version: '1.0.0',
        timestamp: '2025-01-01T00:00:00Z',
        details: {},
      });

      mock.onGet('/inference/health').networkError();

      const result = await checkAllServicesHealth();

      expect(Object.keys(result)).toHaveLength(3);
      expect(result['backend'].status).toBe('unhealthy');
      expect(result['inference'].status).toBe('unhealthy');
      expect(result['training'].status).toBe('healthy');
    });

    it('should mark all services as unhealthy on complete failure', async () => {
      mock.onGet(/\/.*\/health/).networkError();

      const result = await checkAllServicesHealth();

      expect(Object.keys(result)).toHaveLength(3);
      Object.values(result).forEach(health => {
        expect(health.status).toBe('unhealthy');
      });
    });
  });

  describe('getAPIGatewayStatus', () => {
    it('should get API gateway root status', async () => {
      const mockStatus = {
        name: 'Heimdall API Gateway',
        version: '1.0.0',
        status: 'running',
      };

      mock.onGet('/').reply(200, mockStatus);

      const result = await getAPIGatewayStatus();

      expect(result).toEqual(mockStatus);
    });

    it('should handle empty status', async () => {
      mock.onGet('/').reply(200, {});

      const result = await getAPIGatewayStatus();

      expect(result).toEqual({});
    });

    it('should handle gateway error', async () => {
      mock.onGet('/').reply(503, {
        detail: 'Service temporarily unavailable',
      });

      await expect(getAPIGatewayStatus()).rejects.toThrow();
    });
  });
});
