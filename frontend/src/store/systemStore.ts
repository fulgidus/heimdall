/**
 * System Store
 *
 * Manages system-level state and health monitoring
 */

import { create } from 'zustand';
import type { ServiceHealth, ModelPerformanceMetrics } from '@/services/api/types';
import { systemService, inferenceService } from '@/services/api';

interface SystemStore {
  servicesHealth: Record<string, ServiceHealth>;
  modelPerformance: ModelPerformanceMetrics | null;
  isLoading: boolean;
  error: string | null;
  lastCheck: Date | null;

  checkAllServices: () => Promise<void>;
  checkService: (serviceName: string) => Promise<void>;
  fetchModelPerformance: () => Promise<void>;
  updateServicesHealthFromWebSocket: (healthStatus: Record<string, ServiceHealth>) => void;

  isServiceHealthy: (serviceName: string) => boolean;
  getServiceStatus: (serviceName: string) => ServiceHealth | null;

  refreshAll: () => Promise<void>;
}

export const useSystemStore = create<SystemStore>((set, get) => ({
  servicesHealth: {},
  modelPerformance: null,
  isLoading: false,
  error: null,
  lastCheck: null,

  checkAllServices: async () => {
    set({ isLoading: true, error: null });
    try {
      const servicesHealth = await systemService.checkAllServicesHealth();
      set({
        servicesHealth,
        lastCheck: new Date(),
        isLoading: false,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to check services';
      set({ error: errorMessage, isLoading: false });
      console.error('Service health check error:', error);
    }
  },

  checkService: async (serviceName: string) => {
    try {
      const health = await systemService.checkServiceHealth(serviceName);
      set(state => ({
        servicesHealth: {
          ...state.servicesHealth,
          [serviceName]: health,
        },
      }));
    } catch (error) {
      console.error(`Failed to check ${serviceName} health:`, error);
    }
  },

  fetchModelPerformance: async () => {
    try {
      const modelPerformance = await inferenceService.getModelPerformance();
      set({ modelPerformance });
    } catch (error) {
      console.error('Failed to fetch model performance:', error);
      // Model service might not be available, this is not critical
    }
  },

  isServiceHealthy: (serviceName: string) => {
    const health = get().servicesHealth[serviceName];
    return health?.status === 'healthy';
  },

  getServiceStatus: (serviceName: string) => {
    return get().servicesHealth[serviceName] || null;
  },

  updateServicesHealthFromWebSocket: (healthStatus: Record<string, ServiceHealth>) => {
    set({
      servicesHealth: healthStatus,
      lastCheck: new Date(),
      error: null,
    });
  },

  refreshAll: async () => {
    await Promise.all([get().checkAllServices(), get().fetchModelPerformance()]);
  },
}));
