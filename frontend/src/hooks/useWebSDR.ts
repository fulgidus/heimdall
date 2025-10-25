/**
 * React Query Hooks for WebSDR Endpoints
 * 
 * Provides hooks for fetching WebSDR status and configuration
 */

import { useQuery } from '@tanstack/react-query';
import webSDRService from '@/services/api/websdr';

/**
 * Hook to fetch all WebSDRs
 * Refetches every 30 seconds
 */
export const useWebSDRs = () => {
    return useQuery({
        queryKey: ['websdrs', 'list'],
        queryFn: () => webSDRService.getWebSDRs(),
        refetchInterval: 30000,
        staleTime: 25000,
    });
};

/**
 * Hook to fetch WebSDR health status
 * Refetches every 10 seconds for real-time updates
 */
export const useWebSDRHealth = () => {
    return useQuery({
        queryKey: ['websdrs', 'health'],
        queryFn: () => webSDRService.checkWebSDRHealth(),
        refetchInterval: 10000,
        staleTime: 8000,
    });
};

/**
 * Hook to fetch a specific WebSDR by ID
 */
export const useWebSDR = (websdrId: number | null) => {
    return useQuery({
        queryKey: ['websdrs', websdrId],
        queryFn: async () => {
            if (websdrId === null) throw new Error('No WebSDR ID provided');
            const websdrs = await webSDRService.getWebSDRs();
            const websdr = websdrs.find((w) => w.id === websdrId);
            if (!websdr) throw new Error(`WebSDR ${websdrId} not found`);
            return websdr;
        },
        enabled: websdrId !== null,
        staleTime: 30000,
    });
};
