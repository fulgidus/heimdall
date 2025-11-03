/**
 * Session WebSocket Integration Hook
 * 
 * Connects WebSocket events to the Session store for real-time updates
 */

import { useEffect } from 'react';
import { useWebSocket } from '@/contexts/WebSocketContext';
import { useSessionStore } from '@/store/sessionStore';
import type { RecordingSessionWithDetails } from '@/services/api/session';

export const useSessionWebSocket = () => {
  const { subscribe, isConnected } = useWebSocket();
  const store = useSessionStore();

  useEffect(() => {
    if (!isConnected) return;

    // Subscribe to session status updates
    const unsubscribeStatus = subscribe('session:status_update', (data: any) => {
      console.log('[useSessionWebSocket] Received session:status_update:', data);
      if (data.session) {
        store.updateSessionFromWebSocket(data.session as RecordingSessionWithDetails);
      } else if (data.data?.session) {
        store.updateSessionFromWebSocket(data.data.session as RecordingSessionWithDetails);
      }
    });

    // Subscribe to session created events
    const unsubscribeCreated = subscribe('session:created', (data: any) => {
      console.log('[useSessionWebSocket] Received session:created:', data);
      if (data.session) {
        store.updateSessionFromWebSocket(data.session as RecordingSessionWithDetails);
      } else if (data.data?.session) {
        store.updateSessionFromWebSocket(data.data.session as RecordingSessionWithDetails);
      }
    });

    // Subscribe to session started events
    const unsubscribeStarted = subscribe('session:started', (data: any) => {
      console.log('[useSessionWebSocket] Received session:started:', data);
      if (data.session) {
        store.updateSessionFromWebSocket(data.session as RecordingSessionWithDetails);
      } else if (data.data?.session) {
        store.updateSessionFromWebSocket(data.data.session as RecordingSessionWithDetails);
      }
    });

    // Subscribe to session completed events
    const unsubscribeCompleted = subscribe('session:completed', (data: any) => {
      console.log('[useSessionWebSocket] Received session:completed:', data);
      if (data.session) {
        store.updateSessionFromWebSocket(data.session as RecordingSessionWithDetails);
      } else if (data.data?.session) {
        store.updateSessionFromWebSocket(data.data.session as RecordingSessionWithDetails);
      }
    });

    // Subscribe to session source assigned events
    const unsubscribeSourceAssigned = subscribe('session:source_assigned', (data: any) => {
      console.log('[useSessionWebSocket] Received session:source_assigned:', data);
      if (data.session) {
        store.updateSessionFromWebSocket(data.session as RecordingSessionWithDetails);
      } else if (data.data?.session) {
        store.updateSessionFromWebSocket(data.data.session as RecordingSessionWithDetails);
      }
    });

    // Cleanup subscriptions on unmount
    return () => {
      unsubscribeStatus();
      unsubscribeCreated();
      unsubscribeStarted();
      unsubscribeCompleted();
      unsubscribeSourceAssigned();
    };
  }, [isConnected, subscribe]); // store is stable in Zustand, no need to include in deps

  return {
    isConnected,
  };
};
