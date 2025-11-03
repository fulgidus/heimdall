/**
 * Acquisition WebSocket Integration Hook
 * 
 * Connects WebSocket events to the Acquisition store for real-time updates
 */

import { useEffect } from 'react';
import { useWebSocket } from '@/contexts/WebSocketContext';
import { useAcquisitionStore } from '@/store/acquisitionStore';
import type { AcquisitionStatusResponse } from '@/services/api/types';

export const useAcquisitionWebSocket = () => {
  const { subscribe, isConnected } = useWebSocket();
  const store = useAcquisitionStore();

  useEffect(() => {
    if (!isConnected) return;

    // Subscribe to acquisition progress updates
    const unsubscribeProgress = subscribe('acquisition:progress', (data: any) => {
      console.log('[useAcquisitionWebSocket] Received acquisition:progress:', data);
      if (data.task_id && data.status) {
        store.updateTaskFromWebSocket(data.task_id, data.status as AcquisitionStatusResponse);
      }
    });

    // Subscribe to acquisition completion
    const unsubscribeComplete = subscribe('acquisition:complete', (data: any) => {
      console.log('[useAcquisitionWebSocket] Received acquisition:complete:', data);
      if (data.task_id && data.status) {
        store.updateTaskFromWebSocket(data.task_id, data.status as AcquisitionStatusResponse);
      }
    });

    // Subscribe to acquisition errors
    const unsubscribeError = subscribe('acquisition:error', (data: any) => {
      console.log('[useAcquisitionWebSocket] Received acquisition:error:', data);
      if (data.task_id && data.status) {
        store.updateTaskFromWebSocket(data.task_id, data.status as AcquisitionStatusResponse);
      }
    });

    // Cleanup subscriptions on unmount
    return () => {
      unsubscribeProgress();
      unsubscribeComplete();
      unsubscribeError();
    };
  }, [isConnected, subscribe]); // store is stable in Zustand, no need to include in deps

  return {
    isConnected,
  };
};
