/**
 * React Query Hooks for Health Endpoints
 *
 * Provides hooks for checking system and service health
 */

import { useQuery } from '@tanstack/react-query';
import systemService from '@/services/api/system';

/**
 * Hook to fetch overall system health
 * Refetches every 30 seconds
 */
export const useSystemHealth = () => {
  return useQuery({
    queryKey: ['health', 'system'],
    queryFn: () => systemService.checkAllServicesHealth(),
    refetchInterval: 30000,
    staleTime: 25000,
    retry: 2, // Retry health checks more times
  });
};

/**
 * Hook to fetch specific service health
 * Refetches every 30 seconds
 */
export const useServiceHealth = (serviceName: string) => {
  return useQuery({
    queryKey: ['health', 'service', serviceName],
    queryFn: () => systemService.checkServiceHealth(serviceName),
    refetchInterval: 30000,
    staleTime: 25000,
    enabled: !!serviceName,
  });
};

/**
 * Hook to fetch API Gateway status
 */
export const useAPIGatewayStatus = () => {
  return useQuery({
    queryKey: ['health', 'api-gateway'],
    queryFn: () => systemService.getAPIGatewayStatus(),
    refetchInterval: 60000,
    staleTime: 50000,
  });
};
