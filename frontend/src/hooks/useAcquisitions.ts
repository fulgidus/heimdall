/**
 * React Query Hooks for Acquisition Endpoints
 * 
 * Provides hooks for fetching acquisitions and triggering new ones
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import acquisitionService from '@/services/api/acquisition';

/**
 * Hook to fetch recent acquisitions
 * Refetches every 10 seconds
 */
export const useRecentAcquisitions = (limit: number = 10) => {
    return useQuery({
        queryKey: ['acquisitions', 'recent', limit],
        queryFn: () => acquisitionService.getRecentAcquisitions(limit),
        refetchInterval: 10000,
        staleTime: 8000,
    });
};

/**
 * Hook to fetch acquisition status by task ID
 * Polls every 5 seconds while task is active
 */
export const useAcquisitionStatus = (taskId: string | null) => {
    return useQuery({
        queryKey: ['acquisitions', 'status', taskId],
        queryFn: () => {
            if (!taskId) throw new Error('No task ID provided');
            return acquisitionService.getAcquisitionStatus(taskId);
        },
        enabled: !!taskId,
        refetchInterval: (query) => {
            const data = query.state.data as { status?: string } | undefined;
            // Only poll if status is pending or in_progress
            if (data?.status === 'pending' || data?.status === 'in_progress') {
                return 5000;
            }
            return false;
        },
        staleTime: 2000,
    });
};

/**
 * Hook to trigger a new acquisition
 * Invalidates acquisition queries on success
 */
export const useTriggerAcquisition = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (params: {
            frequency_mhz: number;
            duration_seconds: number;
            websdr_ids?: number[];
        }) => acquisitionService.triggerAcquisition(params),
        onSuccess: () => {
            // Invalidate and refetch acquisitions
            queryClient.invalidateQueries({ queryKey: ['acquisitions'] });
        },
    });
};

/**
 * Hook to get acquisition by ID
 */
export const useAcquisition = (acquisitionId: string | null) => {
    return useQuery({
        queryKey: ['acquisitions', acquisitionId],
        queryFn: () => {
            if (!acquisitionId) throw new Error('No acquisition ID provided');
            return acquisitionService.getAcquisition(acquisitionId);
        },
        enabled: !!acquisitionId,
        staleTime: 10000,
    });
};
