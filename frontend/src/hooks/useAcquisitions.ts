/**
 * React Query Hooks for Acquisition Endpoints
 * 
 * Provides hooks for fetching acquisitions and triggering new ones
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import acquisitionService from '@/services/api/acquisition';
import type { AcquisitionRequest } from '@/services/api/types';

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
            if (data?.status === 'PENDING' || data?.status === 'IN_PROGRESS' || data?.status === 'STARTED') {
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
        mutationFn: (params: AcquisitionRequest) => 
            acquisitionService.triggerAcquisition(params),
        onSuccess: () => {
            // Invalidate and refetch acquisitions
            queryClient.invalidateQueries({ queryKey: ['acquisitions'] });
        },
    });
};
