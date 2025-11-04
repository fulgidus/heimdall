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

        // Subscribe to comprehensive system health updates (includes all components)
        const unsubscribeComprehensive = subscribe('system:comprehensive_health', (data: any) => {
            console.log('[useSystemWebSocket] Received system:comprehensive_health:', data);
            if (data.components) {
                store.updateComprehensiveHealthFromWebSocket(data.components as Record<string, ServiceHealth>);
            }
        });

        // Legacy: also subscribe to individual service health updates for backwards compatibility
        const unsubscribeHealth = subscribe('services:health', (data: any) => {
            console.log('[useSystemWebSocket] Received services:health:', data);
            if (data.health_status) {
                store.updateServicesHealthFromWebSocket(data.health_status as Record<string, ServiceHealth>);
            }
        });

        // Cleanup subscriptions on unmount
        return () => {
            unsubscribeComprehensive();
            unsubscribeHealth();
        };
    }, [isConnected, subscribe]); // store is stable in Zustand, no need to include in deps

    return {
        isConnected,
    };
};
