import { create } from 'zustand';
import { analyticsService } from '@/services/api';
import type {
  PredictionMetrics,
  WebSDRPerformance,
  SystemPerformance,
} from '@/services/api/analytics';

interface AnalyticsState {
  // Data
  predictionMetrics: PredictionMetrics | null;
  websdrPerformance: WebSDRPerformance[];
  systemPerformance: SystemPerformance | null;
  accuracyDistribution: {
    accuracy_ranges: string[];
    counts: number[];
  } | null;

  // UI State
  isLoading: boolean;
  error: string | null;
  timeRange: string;

  // Actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setTimeRange: (timeRange: string) => void;

  // API Actions
  fetchPredictionMetrics: (timeRange?: string) => Promise<void>;
  fetchWebSDRPerformance: (timeRange?: string) => Promise<void>;
  fetchSystemPerformance: (timeRange?: string) => Promise<void>;
  fetchAccuracyDistribution: (timeRange?: string) => Promise<void>;
  fetchAllAnalytics: (timeRange?: string) => Promise<void>;
  refreshData: () => Promise<void>;
}

export const useAnalyticsStore = create<AnalyticsState>((set, get) => ({
  // Initial state
  predictionMetrics: null,
  websdrPerformance: [],
  systemPerformance: null,
  accuracyDistribution: null,
  isLoading: false,
  error: null,
  timeRange: '7d',

  // Basic setters
  setLoading: loading => set({ isLoading: loading }),
  setError: error => set({ error }),
  setTimeRange: timeRange => set({ timeRange }),

  // API Actions
  fetchPredictionMetrics: async timeRange => {
    const range = timeRange || get().timeRange;
    try {
      const metrics = await analyticsService.getPredictionMetrics(range);
      set({
        predictionMetrics: metrics,
        error: null,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to fetch prediction metrics';
      set({ error: errorMessage });
      console.error('Failed to fetch prediction metrics:', error);
    }
  },

  fetchWebSDRPerformance: async timeRange => {
    const range = timeRange || get().timeRange;
    try {
      const performance = await analyticsService.getWebSDRPerformance(range);
      set({
        websdrPerformance: performance,
        error: null,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to fetch WebSDR performance';
      set({ error: errorMessage });
      console.error('Failed to fetch WebSDR performance:', error);
    }
  },

  fetchSystemPerformance: async timeRange => {
    const range = timeRange || get().timeRange;
    try {
      const performance = await analyticsService.getSystemPerformance(range);
      set({
        systemPerformance: performance,
        error: null,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to fetch system performance';
      set({ error: errorMessage });
      console.error('Failed to fetch system performance:', error);
    }
  },

  fetchAccuracyDistribution: async timeRange => {
    const range = timeRange || get().timeRange;
    try {
      const distribution = await analyticsService.getAccuracyDistribution(range);
      set({
        accuracyDistribution: distribution,
        error: null,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to fetch accuracy distribution';
      set({ error: errorMessage });
      console.error('Failed to fetch accuracy distribution:', error);
    }
  },

  fetchAllAnalytics: async timeRange => {
    const range = timeRange || get().timeRange;
    set({ isLoading: true, error: null });

    try {
      await Promise.allSettled([
        get().fetchPredictionMetrics(range),
        get().fetchWebSDRPerformance(range),
        get().fetchSystemPerformance(range),
        get().fetchAccuracyDistribution(range),
      ]);

      set({ timeRange: range });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to fetch analytics data';
      set({ error: errorMessage });
      console.error('Failed to fetch all analytics:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  refreshData: async () => {
    await get().fetchAllAnalytics();
  },
}));
