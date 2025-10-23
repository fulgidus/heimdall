/**
 * WebSDR Store
 * 
 * Manages WebSDR receivers state and operations
 */

import { create } from 'zustand';
import type { WebSDRConfig, WebSDRHealthStatus } from '@/services/api/types';
import { webSDRService } from '@/services/api';

interface WebSDRStore {
    websdrs: WebSDRConfig[];
    healthStatus: Record<number, WebSDRHealthStatus>;
    isLoading: boolean;
    error: string | null;
    lastHealthCheck: Date | null;
    
    fetchWebSDRs: () => Promise<void>;
    checkHealth: () => Promise<void>;
    getActiveWebSDRs: () => WebSDRConfig[];
    getWebSDRById: (id: number) => WebSDRConfig | undefined;
    isWebSDROnline: (id: number) => boolean;
    
    refreshAll: () => Promise<void>;
}

export const useWebSDRStore = create<WebSDRStore>((set, get) => ({
    websdrs: [],
    healthStatus: {},
    isLoading: false,
    error: null,
    lastHealthCheck: null,

    fetchWebSDRs: async () => {
        set({ isLoading: true, error: null });
        try {
            const websdrs = await webSDRService.getWebSDRs();
            set({ websdrs, isLoading: false });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to fetch WebSDRs';
            set({ error: errorMessage, isLoading: false });
            console.error('WebSDR fetch error:', error);
        }
    },

    checkHealth: async () => {
        try {
            const healthStatus = await webSDRService.checkWebSDRHealth();
            set({ 
                healthStatus, 
                lastHealthCheck: new Date(),
                error: null,
            });
        } catch (error) {
            console.error('Health check error:', error);
            // Don't set error for health checks, just log it
        }
    },

    getActiveWebSDRs: () => {
        return get().websdrs.filter(w => w.is_active);
    },

    getWebSDRById: (id: number) => {
        return get().websdrs.find(w => w.id === id);
    },

    isWebSDROnline: (id: number) => {
        const health = get().healthStatus[id];
        return health?.status === 'online';
    },

    refreshAll: async () => {
        await Promise.all([
            get().fetchWebSDRs(),
            get().checkHealth(),
        ]);
    },
}));
