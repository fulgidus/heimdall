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
    systemService,
    analyticsService
} from '@/services/api';
import { WebSocketManager, ConnectionState, createWebSocketManager } from '@/lib/websocket';

interface DashboardMetrics {
    activeWebSDRs: number;
    totalWebSDRs: number;
    signalDetections: number;
    systemUptime: number;
    averageAccuracy: number;
}

interface DashboardData {
    websdrs: WebSDRConfig[];
    websdrsHealth: Record<string, WebSDRHealthStatus>;  // UUID keys from backend
    modelInfo: ModelInfo | null;
    servicesHealth: Record<string, ServiceHealth>;
}

interface DashboardStore {
    metrics: DashboardMetrics;
    data: DashboardData;
    isLoading: boolean;
    error: string | null;
    lastUpdate: Date | null;
    retryCount: number;
    retryDelay: number;
    // WebSocket state
    wsManager: WebSocketManager | null;
    wsConnectionState: ConnectionState;
    wsEnabled: boolean;

    setMetrics: (metrics: DashboardMetrics) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    resetRetry: () => void;
    incrementRetry: () => void;

    fetchDashboardData: () => Promise<void>;
    fetchWebSDRs: () => Promise<void>;
    fetchModelInfo: () => Promise<void>;
    fetchServicesHealth: () => Promise<void>;

    refreshAll: () => Promise<void>;

    // WebSocket methods
    connectWebSocket: () => Promise<void>;
    disconnectWebSocket: () => void;
    setWebSocketState: (state: ConnectionState) => void;
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
    retryCount: 0,
    retryDelay: 1000,
    // WebSocket state
    wsManager: null,
    wsConnectionState: ConnectionState.DISCONNECTED,
    wsEnabled: true, // WebSocket enabled for real-time updates

    setMetrics: (metrics) => set({ metrics }),
    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),

    resetRetry: () => set({ retryCount: 0, retryDelay: 1000 }),

    incrementRetry: () => set((state) => ({
        retryCount: state.retryCount + 1,
        retryDelay: Math.min(state.retryDelay * 2, 30000), // Max 30 seconds
    })),

    fetchWebSDRs: async () => {
        try {
            // Load WebSDRs FAST - don't wait for health check
            const websdrs = await webSDRService.getWebSDRs();

            // Ensure websdrs is an array (defensive programming)
            if (!Array.isArray(websdrs)) {
                console.error('❌ fetchWebSDRs: websdrs is not an array:', typeof websdrs);
                set({ error: 'Invalid WebSDRs response format' });
                return;
            }

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
            // Fetch all data in parallel
            const [metricsData] = await Promise.allSettled([
                analyticsService.getDashboardMetrics(),
            ]);

            // Update metrics from analytics endpoint
            if (metricsData.status === 'fulfilled') {
                set((state) => ({
                    metrics: {
                        ...state.metrics,
                        signalDetections: metricsData.value.signalDetections,
                        systemUptime: metricsData.value.systemUptime,
                        averageAccuracy: metricsData.value.modelAccuracy * 100,
                    },
                }));
            }

            // Fetch other data sources
            await Promise.allSettled([
                get().fetchWebSDRs(),
                get().fetchModelInfo(),
                get().fetchServicesHealth(),
            ]);

            set({
                lastUpdate: new Date(),
                error: null,
            });

            // Reset retry count on success
            get().resetRetry();
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to fetch dashboard data';
            set({ error: errorMessage });
            console.error('Dashboard data fetch error:', error);

            // Increment retry count for exponential backoff
            get().incrementRetry();
        } finally {
            set({ isLoading: false });
        }
    },

    refreshAll: async () => {
        await get().fetchDashboardData();
    },

    // WebSocket methods
    setWebSocketState: (state: ConnectionState) => {
        set({ wsConnectionState: state });
    },

    connectWebSocket: async () => {
        const { wsManager, wsEnabled, wsConnectionState } = get();

        if (!wsEnabled) {
            console.log('[Dashboard] WebSocket disabled, using polling');
            return;
        }

        // Guard: Don't reconnect if already connected or connecting
        if (wsManager && (wsConnectionState === ConnectionState.CONNECTED || wsConnectionState === ConnectionState.CONNECTING)) {
            console.log('[Dashboard] WebSocket already initialized and connected/connecting');
            return;
        }

        // Guard: If manager exists but is disconnected, clean it up first
        if (wsManager && wsConnectionState === ConnectionState.DISCONNECTED) {
            console.log('[Dashboard] Cleaning up disconnected WebSocket manager');
            wsManager.disconnect();
            set({ wsManager: null });
        }

        try {
            // Use configured WebSocket URL from environment or construct from browser location
            // Envoy proxies WebSocket requests to backend service at /ws
            const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const hostname = window.location.hostname;
            const port = window.location.port || (protocol === 'wss' ? '443' : '80');
            // WebSocket endpoint: Envoy routes /ws to backend /ws
            const wsUrl = import.meta.env.VITE_SOCKET_URL || `${protocol}://${hostname}:${port}/ws`;
            console.log('[Dashboard] Connecting to WebSocket:', wsUrl);

            const manager = createWebSocketManager(wsUrl);

            // Subscribe to connection state changes
            manager.onStateChange((state) => {
                get().setWebSocketState(state);
            });

            // Subscribe to real-time events
            manager.subscribe('services:health', (data) => {
                console.log('[Dashboard] Received services health update:', data);
                set((state) => ({
                    data: {
                        ...state.data,
                        servicesHealth: data,
                    },
                    lastUpdate: new Date(),
                }));
            });

            manager.subscribe('websdrs_update', (data) => {
                console.log('[Dashboard] Received WebSDR status update:', data);
                set((state) => ({
                    data: {
                        ...state.data,
                        websdrsHealth: data,
                    },
                    lastUpdate: new Date(),
                }));
            });

            manager.subscribe('signals:detected', (data) => {
                console.log('[Dashboard] Received signal detection:', data);
                set((state) => ({
                    metrics: {
                        ...state.metrics,
                        signalDetections: (state.metrics.signalDetections || 0) + 1,
                    },
                    lastUpdate: new Date(),
                }));
            });

            manager.subscribe('localizations:updated', (data) => {
                console.log('[Dashboard] Received localization update:', data);
                // Handle localization updates (could update a separate store)
                set({ lastUpdate: new Date() });
            });

            // Store manager and attempt connection
            set({ wsManager: manager });

            await manager.connect();
            console.log('[Dashboard] WebSocket connected successfully');
        } catch (error) {
            console.error('[Dashboard] WebSocket connection failed:', error);
            // Disable WebSocket and fallback to polling
            set({ wsEnabled: false, wsManager: null });
        }
    },

    disconnectWebSocket: () => {
        const { wsManager, wsConnectionState } = get();
        
        // Guard: Don't disconnect if already disconnected
        if (!wsManager || wsConnectionState === ConnectionState.DISCONNECTED) {
            console.log('[Dashboard] Already disconnected, skipping');
            return;
        }
        
        if (wsManager) {
            console.log('[Dashboard] Disconnecting WebSocket');
            wsManager.disconnect();
            set({ wsManager: null, wsConnectionState: ConnectionState.DISCONNECTED });
        }
    },
}));
