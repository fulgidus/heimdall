/**
 * Training WebSocket Hook
 * 
 * Connects to training job WebSocket endpoint for real-time updates
 */

import { useEffect, useRef } from 'react';
import { useTrainingStore } from '../../../store/trainingStore';
import type { TrainingWebSocketMessage, TrainingMetric, TrainingJob } from '../types';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 
  `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8001`;

export const useTrainingWebSocket = (jobId: string | null) => {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const store = useTrainingStore();
  
  // Extract store methods to stable refs to prevent reconnection loops
  const storeRef = useRef(store);
  storeRef.current = store;

  useEffect(() => {
    // Don't connect if no jobId
    if (!jobId) {
      return;
    }

    const connect = () => {
      try {
        const wsUrl = `${WS_BASE_URL}/ws/training/${jobId}`;
        console.log('[useTrainingWebSocket] Connecting to:', wsUrl);

        const socket = new WebSocket(wsUrl);
        socketRef.current = socket;

        socket.onopen = () => {
          console.log('[useTrainingWebSocket] Connected to training job:', jobId);
          storeRef.current.setWsConnected(true);
          reconnectAttempts.current = 0;
        };

        socket.onmessage = (event) => {
          try {
            const message: TrainingWebSocketMessage = JSON.parse(event.data);
            console.log('[useTrainingWebSocket] Received message:', message);

            switch (message.event) {
              case 'connected':
                // Initial connection confirmation from server
                console.log('[useTrainingWebSocket] Server confirmed connection');
                break;

              case 'training_started':
                storeRef.current.handleJobUpdate({
                  id: message.job_id,
                  status: 'running',
                  started_at: message.timestamp,
                  ...message.data as Partial<TrainingJob>,
                });
                break;

              case 'training_progress':
                // Could be job update or metric update
                if ('epoch' in message.data) {
                  // It's a metric update
                  storeRef.current.handleMetricUpdate(message.data as TrainingMetric);
                } else {
                  // It's a job update
                  storeRef.current.handleJobUpdate({
                    id: message.job_id,
                    ...message.data as Partial<TrainingJob>,
                  });
                }
                break;

              case 'training_completed':
                storeRef.current.handleJobUpdate({
                  id: message.job_id,
                  status: 'completed',
                  completed_at: message.timestamp,
                  progress_percent: 100,
                  ...message.data as Partial<TrainingJob>,
                });
                break;

              case 'training_failed':
                storeRef.current.handleJobUpdate({
                  id: message.job_id,
                  status: 'failed',
                  completed_at: message.timestamp,
                  error_message: (message.data as any).error_message,
                  ...message.data as Partial<TrainingJob>,
                });
                break;

              default:
                console.warn('[useTrainingWebSocket] Unknown event type:', message.event);
            }
          } catch (error) {
            console.error('[useTrainingWebSocket] Error parsing message:', error);
          }
        };

        socket.onerror = (error) => {
          console.error('[useTrainingWebSocket] WebSocket error:', error);
        };

        socket.onclose = () => {
          console.log('[useTrainingWebSocket] Disconnected from training job:', jobId);
          storeRef.current.setWsConnected(false);
          socketRef.current = null;

          // Attempt to reconnect with exponential backoff
          if (reconnectAttempts.current < maxReconnectAttempts) {
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
            console.log(`[useTrainingWebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttempts.current++;
              connect();
            }, delay);
          } else {
            console.error('[useTrainingWebSocket] Max reconnection attempts reached');
          }
        };
      } catch (error) {
        console.error('[useTrainingWebSocket] Error creating WebSocket:', error);
      }
    };

    connect();

    // Cleanup on unmount or jobId change
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }

      storeRef.current.setWsConnected(false);
    };
  }, [jobId]); // Only reconnect when jobId changes

  return {
    isConnected: store.wsConnected,
  };
};
