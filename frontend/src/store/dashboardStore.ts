import { create } from 'zustand';
import type {
    WebSDRConfig,
    WebSDRHealthStatus,
    ModelInfo,
    ServiceHealth
} from '@/services/api/types';
import {
    webSDRService,
    inferenceService,
    systemService
} from '@/services/api';

interface DashboardMetrics {
    activeWebSDRs: number;
    totalWebSDRs: number;
    signalDetections: number;
    systemUptime: number;
    averageAccuracy: number;
}

interface DashboardData {
    websdrs: WebSDRConfig[];
    websdrsHealth: Record<number, WebSDRHealthStatus>;
    modelInfo: ModelInfo | null;
    servicesHealth: Record<string, ServiceHealth>;
}

interface DashboardStore {
    metrics: DashboardMetrics;
    data: DashboardData;
    isLoading: boolean;
    error: string | null;
    lastUpdate: Date | null;

    setMetrics: (metrics: DashboardMetrics) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;

    fetchDashboardData: () => Promise<void>;
    fetchWebSDRs: () => Promise<void>;
    fetchModelInfo: () => Promise<void>;
    fetchServicesHealth: () => Promise<void>;

    refreshAll: () => Promise<void>;
}

export const useDashboardStore = create<DashboardStore>((set, get) => ({
    metrics: {
        activeWebSDRs: 0,
        totalWebSDRs: 0,
        signalDetections: 0,
        systemUptime: 0,
        averageAccuracy: 0,
    },
    data: {
        websdrs: [],
        websdrsHealth: {},
        modelInfo: null,
        servicesHealth: {},
    },
    isLoading: false,
    error: null,
    lastUpdate: null,

    setMetrics: (metrics) => set({ metrics }),
    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),

    fetchWebSDRs: async () => {
        try {
            // Load WebSDRs FAST - don't wait for health check
            const websdrs = await webSDRService.getWebSDRs();

            set((state) => ({
                data: {
                    ...state.data,
                    websdrs,
                },
                metrics: {
                    ...state.metrics,
                    totalWebSDRs: websdrs.length,
                    activeWebSDRs: websdrs.filter(w => w.is_active).length,
                },
            }));

            // Load health check in background (doesn't block UI)
            webSDRService.checkWebSDRHealth()
                .then(health => {
                    set((state) => ({
                        data: {
                            ...state.data,
                            websdrsHealth: health,
                        },
                    }));
                })
                .catch(error => {
                    console.warn('Health check failed (non-critical):', error);
                });
        } catch (error) {
            console.error('Failed to fetch WebSDRs:', error);
            throw error;
        }
    },

    fetchModelInfo: async () => {
        try {
            const modelInfo = await inferenceService.getModelInfo();

            set((state) => ({
                data: {
                    ...state.data,
                    modelInfo,
                },
                metrics: {
                    ...state.metrics,
                    averageAccuracy: modelInfo.accuracy ? modelInfo.accuracy * 100 : 0,
                    systemUptime: modelInfo.uptime_seconds,
                },
            }));
        } catch (error) {
            console.error('Failed to fetch model info:', error);
            // Model service might not be available, this is not critical
        }
    },

    fetchServicesHealth: async () => {
        try {
            const servicesHealth = await systemService.checkAllServicesHealth();

            set((state) => ({
                data: {
                    ...state.data,
                    servicesHealth,
                },
            }));
        } catch (error) {
            console.error('Failed to fetch services health:', error);
            throw error;
        }
    },

    fetchDashboardData: async () => {
        set({ isLoading: true, error: null });

        try {
            await Promise.allSettled([
                get().fetchWebSDRs(),
                get().fetchModelInfo(),
                get().fetchServicesHealth(),
            ]);

            set({
                lastUpdate: new Date(),
                error: null,
            });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to fetch dashboard data';
            set({ error: errorMessage });
            console.error('Dashboard data fetch error:', error);
        } finally {
            set({ isLoading: false });
        }
    },

    refreshAll: async () => {
        await get().fetchDashboardData();
    },
}));
