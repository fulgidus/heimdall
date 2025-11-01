/**
 * WebSDR WebSocket Integration Hook
 * 
 * Connects WebSocket events to the WebSDR store for real-time updates
 */

import { useEffect, useRef } from 'react';
import { useWebSocket } from '@/contexts/WebSocketContext';
import { useWebSDRStore } from '@/store/websdrStore';
import type { WebSDRConfig, WebSDRHealthStatus } from '@/services/api/types';

export const useWebSDRWebSocket = () => {
  const { subscribe, isConnected, manager } = useWebSocket();
  const store = useWebSDRStore();
  const hasRequestedData = useRef(false);

  useEffect(() => {
    // Update store connection status
    store.setWebSocketConnected(isConnected);
  }, [isConnected, store]);

  useEffect(() => {
    if (!isConnected || !manager) return;

    // Request initial WebSDR data when connected
    if (!hasRequestedData.current) {
      console.log('[useWebSDRWebSocket] Requesting initial WebSDR data');
      manager.send('get_data', { data_type: 'websdrs' });
      hasRequestedData.current = true;
    }

    // Subscribe to WebSDR data events
    const unsubscribeWebSDRs = subscribe('websdrs_data', (data: any) => {
      console.log('[useWebSDRWebSocket] Received websdrs_data:', data);
      if (Array.isArray(data.data)) {
        store.setWebSDRsFromWebSocket(data.data as WebSDRConfig[]);
      }
    });

    // Subscribe to WebSDR updates
    const unsubscribeUpdate = subscribe('websdr:update', (data: any) => {
      console.log('[useWebSDRWebSocket] Received websdr:update:', data);
      if (data.websdr) {
        store.updateWebSDRFromWebSocket(data.websdr as WebSDRConfig);
      }
    });

    // Subscribe to WebSDR creation
    const unsubscribeCreate = subscribe('websdr:created', (data: any) => {
      console.log('[useWebSDRWebSocket] Received websdr:created:', data);
      if (data.websdr) {
        store.updateWebSDRFromWebSocket(data.websdr as WebSDRConfig);
      }
    });

    // Subscribe to WebSDR deletion
    const unsubscribeDelete = subscribe('websdr:deleted', (data: any) => {
      console.log('[useWebSDRWebSocket] Received websdr:deleted:', data);
      if (data.id) {
        // Trigger a full refresh to remove deleted item
        store.fetchWebSDRs();
      }
    });

    // Subscribe to health status updates
    const unsubscribeHealth = subscribe('websdrs:health', (data: any) => {
      console.log('[useWebSDRWebSocket] Received websdrs:health:', data);
      if (data.health_status) {
        store.updateHealthFromWebSocket(data.health_status as Record<string, WebSDRHealthStatus>);
      }
    });

    // Cleanup subscriptions on unmount
    return () => {
      unsubscribeWebSDRs();
      unsubscribeUpdate();
      unsubscribeCreate();
      unsubscribeDelete();
      unsubscribeHealth();
      hasRequestedData.current = false;
    };
  }, [isConnected, manager, subscribe, store]);

  return {
    isConnected,
    requestRefresh: () => {
      if (manager && isConnected) {
        manager.send('get_data', { data_type: 'websdrs' });
      }
    },
  };
};
