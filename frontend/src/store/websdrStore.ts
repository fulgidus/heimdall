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
    healthStatus: Record<string, WebSDRHealthStatus>;  // UUID keys
    isLoading: boolean;
    error: string | null;
    lastHealthCheck: Date | null;

    fetchWebSDRs: () => Promise<void>;
    checkHealth: () => Promise<void>;
    getActiveWebSDRs: () => WebSDRConfig[];
    getWebSDRById: (id: string) => WebSDRConfig | undefined;  // UUID parameter
    isWebSDROnline: (id: string) => boolean;  // UUID parameter

    // CRUD operations
    createWebSDR: (data: Omit<WebSDRConfig, 'id'>) => Promise<WebSDRConfig>;
    updateWebSDR: (id: string, data: Partial<WebSDRConfig>) => Promise<WebSDRConfig>;  // UUID parameter
    deleteWebSDR: (id: string, hardDelete?: boolean) => Promise<void>;  // UUID parameter

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

    getWebSDRById: (id: string) => {
        return get().websdrs.find(w => w.id === id);
    },

    isWebSDROnline: (id: string) => {
        const health = get().healthStatus[id];
        return health?.status === 'online';
    },

    createWebSDR: async (data: Omit<WebSDRConfig, 'id'>) => {
        set({ isLoading: true, error: null });
        try {
            const newWebSDR = await webSDRService.createWebSDR(data);
            // Add to local state
            set(state => ({
                websdrs: [...state.websdrs, newWebSDR],
                isLoading: false
            }));
            return newWebSDR;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to create WebSDR';
            set({ error: errorMessage, isLoading: false });
            console.error('WebSDR creation error:', error);
            throw error;
        }
    },

    updateWebSDR: async (id: string, data: Partial<WebSDRConfig>) => {
        set({ isLoading: true, error: null });
        try {
            const updatedWebSDR = await webSDRService.updateWebSDR(id, data);
            // Update in local state
            set(state => ({
                websdrs: state.websdrs.map(w => w.id === id ? updatedWebSDR : w),
                isLoading: false
            }));
            return updatedWebSDR;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to update WebSDR';
            set({ error: errorMessage, isLoading: false });
            console.error('WebSDR update error:', error);
            throw error;
        }
    },

    deleteWebSDR: async (id: string, hardDelete: boolean = false) => {
        set({ isLoading: true, error: null });
        try {
            await webSDRService.deleteWebSDR(id, hardDelete);
            // Remove from local state or update is_active
            if (hardDelete) {
                set(state => ({
                    websdrs: state.websdrs.filter(w => w.id !== id),
                    isLoading: false
                }));
            } else {
                // Soft delete: mark as inactive
                set(state => ({
                    websdrs: state.websdrs.map(w =>
                        w.id === id ? { ...w, is_active: false } : w
                    ),
                    isLoading: false
                }));
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to delete WebSDR';
            set({ error: errorMessage, isLoading: false });
            console.error('WebSDR deletion error:', error);
            throw error;
        }
    },

    refreshAll: async () => {
        await Promise.all([
            get().fetchWebSDRs(),
            get().checkHealth(),
        ]);
    },
}));
