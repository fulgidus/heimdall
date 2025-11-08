/**
 * React Query Hooks for Metrics Endpoints
 *
 * Provides hooks for fetching system, model, and accuracy metrics
 */

import { useQuery } from '@tanstack/react-query';
import analyticsService from '@/services/api/analytics';
import inferenceService from '@/services/api/inference';

/**
 * Hook to fetch dashboard metrics
 * Refetches every 5 seconds for real-time updates
 */
export const useDashboardMetrics = () => {
  return useQuery({
    queryKey: ['metrics', 'dashboard'],
    queryFn: () => analyticsService.getDashboardMetrics(),
    refetchInterval: 5000,
    staleTime: 3000,
  });
};

/**
 * Hook to fetch model information
 * Refetches every 10 seconds
 */
export const useModelInfo = () => {
  return useQuery({
    queryKey: ['metrics', 'model'],
    queryFn: () => inferenceService.getModelInfo(),
    refetchInterval: 10000,
    staleTime: 8000,
  });
};

/**
 * Hook to fetch prediction metrics
 * Refetches every 30 seconds
 */
export const usePredictionMetrics = (timeRange: string = '7d') => {
  return useQuery({
    queryKey: ['metrics', 'predictions', timeRange],
    queryFn: () => analyticsService.getPredictionMetrics(timeRange),
    refetchInterval: 30000,
    staleTime: 25000,
  });
};

/**
 * Hook to fetch WebSDR performance metrics
 */
export const useWebSDRPerformance = (timeRange: string = '7d') => {
  return useQuery({
    queryKey: ['metrics', 'websdr-performance', timeRange],
    queryFn: () => analyticsService.getWebSDRPerformance(timeRange),
    refetchInterval: 30000,
    staleTime: 25000,
  });
};

/**
 * Hook to fetch system performance metrics
 */
export const useSystemPerformance = (timeRange: string = '7d') => {
  return useQuery({
    queryKey: ['metrics', 'system-performance', timeRange],
    queryFn: () => analyticsService.getSystemPerformance(timeRange),
    refetchInterval: 30000,
    staleTime: 25000,
  });
};

/**
 * Hook to fetch accuracy distribution
 */
export const useAccuracyDistribution = (timeRange: string = '7d') => {
  return useQuery({
    queryKey: ['metrics', 'accuracy-distribution', timeRange],
    queryFn: () => analyticsService.getAccuracyDistribution(timeRange),
    refetchInterval: 60000,
    staleTime: 50000,
  });
};
