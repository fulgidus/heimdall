/**
 * System WebSocket Integration Hook
 * 
 * Connects WebSocket events to the System store for real-time microservices health updates
 */

/* eslint-disable react-hooks/exhaustive-deps */

import { useEffect } from 'react';
import { useWebSocket } from '@/contexts/WebSocketContext';
import { useSystemStore } from '@/store/systemStore';
import type { ServiceHealth } from '@/services/api/types';

export const useSystemWebSocket = () => {
    const { subscribe, isConnected } = useWebSocket();
    const store = useSystemStore();

    useEffect(() => {
        if (!isConnected) return;

        // Subscribe to service health status updates
        const unsubscribeHealth = subscribe('services:health', (data: any) => {
            console.log('[useSystemWebSocket] Received services:health:', data);
            if (data.health_status) {
                store.updateServicesHealthFromWebSocket(data.health_status as Record<string, ServiceHealth>);
            }
        });

        // Cleanup subscriptions on unmount
        return () => {
            unsubscribeHealth();
        };
    }, [isConnected, subscribe]); // store is stable in Zustand, no need to include in deps

    return {
        isConnected,
    };
};
