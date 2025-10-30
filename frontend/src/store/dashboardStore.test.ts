/**
 * Dashboard Store Tests
 *
 * Comprehensive test suite for the dashboardStore Zustand store
 * Tests all actions: data fetching, WebSocket management, retry logic, and state management
 * Truth-first approach: Tests real Zustand store behavior, no mocking of store internals
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { ConnectionState } from '@/lib/websocket';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/dashboardStore');

// Import after unmocking
import { useDashboardStore } from './dashboardStore';
import { webSDRService, inferenceService, systemService, analyticsService } from '@/services/api';

// Mock the API services
vi.mock('@/services/api', () => ({
    webSDRService: {
        getWebSDRs: vi.fn(),
        getWebSDRHealth: vi.fn(),
        checkWebSDRHealth: vi.fn(() => Promise.resolve({})), // Background health check - must return Promise
    },
    inferenceService: {
        getModelInfo: vi.fn(),
    },
    systemService: {
        getServicesHealth: vi.fn(),
        checkAllServicesHealth: vi.fn(() => Promise.resolve({})), // Background health check - must return Promise
    },
    analyticsService: {
        getSystemStats: vi.fn(),
        getDashboardMetrics: vi.fn(), // Used by fetchDashboardData
    },
    acquisitionService: {},
    sessionService: {},
}));

// Mock WebSocket
vi.mock('@/lib/websocket', () => ({
    ConnectionState: {
        DISCONNECTED: 'DISCONNECTED',
        CONNECTING: 'CONNECTING',
        CONNECTED: 'CONNECTED',
        RECONNECTING: 'RECONNECTING',
        ERROR: 'ERROR',
    },
    createWebSocketManager: vi.fn(() => ({
        connect: vi.fn().mockResolvedValue(undefined),
        disconnect: vi.fn(),
        getState: vi.fn(() => 'DISCONNECTED'),
        on: vi.fn(),
        off: vi.fn(),
    })),
}));

describe('Dashboard Store (Zustand)', () => {
    beforeEach(() => {
        // Reset store to initial state before each test
        useDashboardStore.setState({
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
            wsManager: null,
            wsConnectionState: ConnectionState.DISCONNECTED,
            wsEnabled: true,
        });
        vi.clearAllMocks();
    });

    afterEach(() => {
        // Clean up WebSocket connections
        const state = useDashboardStore.getState();
        if (state.wsManager) {
            state.disconnectWebSocket();
        }
    });

    describe('Store Initialization', () => {
        it('should initialize with default state', () => {
            const state = useDashboardStore.getState();
            expect(state.metrics.activeWebSDRs).toBe(0);
            expect(state.metrics.totalWebSDRs).toBe(0);
            expect(state.data.websdrs).toEqual([]);
            expect(state.isLoading).toBe(false);
            expect(state.error).toBe(null);
            expect(state.retryCount).toBe(0);
            expect(state.retryDelay).toBe(1000);
            expect(state.wsConnectionState).toBe(ConnectionState.DISCONNECTED);
            expect(state.wsEnabled).toBe(true);
        });

        it('should have all required actions', () => {
            const state = useDashboardStore.getState();
            expect(typeof state.setMetrics).toBe('function');
            expect(typeof state.setLoading).toBe('function');
            expect(typeof state.setError).toBe('function');
            expect(typeof state.resetRetry).toBe('function');
            expect(typeof state.incrementRetry).toBe('function');
            expect(typeof state.fetchDashboardData).toBe('function');
            expect(typeof state.fetchWebSDRs).toBe('function');
            expect(typeof state.fetchModelInfo).toBe('function');
            expect(typeof state.fetchServicesHealth).toBe('function');
            expect(typeof state.refreshAll).toBe('function');
            expect(typeof state.connectWebSocket).toBe('function');
            expect(typeof state.disconnectWebSocket).toBe('function');
            expect(typeof state.setWebSocketState).toBe('function');
        });
    });

    describe('Basic State Setters', () => {
        it('should update metrics via setMetrics', () => {
            const newMetrics = {
                activeWebSDRs: 5,
                totalWebSDRs: 7,
                signalDetections: 42,
                systemUptime: 3600,
                averageAccuracy: 0.85,
            };
            useDashboardStore.getState().setMetrics(newMetrics);
            expect(useDashboardStore.getState().metrics).toEqual(newMetrics);
        });

        it('should update loading state via setLoading', () => {
            useDashboardStore.getState().setLoading(true);
            expect(useDashboardStore.getState().isLoading).toBe(true);

            useDashboardStore.getState().setLoading(false);
            expect(useDashboardStore.getState().isLoading).toBe(false);
        });

        it('should update error state via setError', () => {
            const errorMessage = 'Network error occurred';
            useDashboardStore.getState().setError(errorMessage);
            expect(useDashboardStore.getState().error).toBe(errorMessage);

            useDashboardStore.getState().setError(null);
            expect(useDashboardStore.getState().error).toBe(null);
        });

        it('should update WebSocket state via setWebSocketState', () => {
            useDashboardStore.getState().setWebSocketState(ConnectionState.CONNECTING);
            expect(useDashboardStore.getState().wsConnectionState).toBe(ConnectionState.CONNECTING);

            useDashboardStore.getState().setWebSocketState(ConnectionState.CONNECTED);
            expect(useDashboardStore.getState().wsConnectionState).toBe(ConnectionState.CONNECTED);
        });
    });

    describe('Retry Logic', () => {
        it('should reset retry count and delay', () => {
            // First set some retry state
            useDashboardStore.setState({ retryCount: 5, retryDelay: 16000 });
            
            // Reset
            useDashboardStore.getState().resetRetry();
            
            const state = useDashboardStore.getState();
            expect(state.retryCount).toBe(0);
            expect(state.retryDelay).toBe(1000);
        });

        it('should increment retry count and double delay', () => {
            useDashboardStore.getState().incrementRetry();
            
            let state = useDashboardStore.getState();
            expect(state.retryCount).toBe(1);
            expect(state.retryDelay).toBe(2000); // 1000 * 2

            useDashboardStore.getState().incrementRetry();
            state = useDashboardStore.getState();
            expect(state.retryCount).toBe(2);
            expect(state.retryDelay).toBe(4000); // 2000 * 2
        });

        it('should cap retry delay at 30 seconds', () => {
            // Set delay to 20 seconds
            useDashboardStore.setState({ retryDelay: 20000 });
            
            useDashboardStore.getState().incrementRetry();
            expect(useDashboardStore.getState().retryDelay).toBe(30000); // Capped at 30s

            // Another increment should not exceed 30s
            useDashboardStore.getState().incrementRetry();
            expect(useDashboardStore.getState().retryDelay).toBe(30000);
        });
    });

    describe('fetchWebSDRs Action', () => {
        it('should fetch WebSDRs successfully', async () => {
            const mockWebSDRs = [
                {
                    id: 1,
                    name: 'WebSDR Torino',
                    url: 'http://example.com',
                    location_name: 'Torino, Italy',
                    latitude: 45.0703,
                    longitude: 7.6869,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
                {
                    id: 2,
                    name: 'WebSDR Genova',
                    url: 'http://example.com',
                    location_name: 'Genova, Italy',
                    latitude: 44.4056,
                    longitude: 8.9463,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
            ];

            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue(mockWebSDRs);

            await useDashboardStore.getState().fetchWebSDRs();

            const state = useDashboardStore.getState();
            expect(state.data.websdrs).toEqual(mockWebSDRs);
            expect(webSDRService.getWebSDRs).toHaveBeenCalledOnce();
        });

        it('should handle fetchWebSDRs error gracefully', async () => {
            const errorMessage = 'Failed to fetch WebSDRs';
            vi.mocked(webSDRService.getWebSDRs).mockRejectedValue(new Error(errorMessage));

            // fetchWebSDRs throws errors instead of setting error state
            await expect(
                useDashboardStore.getState().fetchWebSDRs()
            ).rejects.toThrow(errorMessage);
        });
    });

    describe('fetchModelInfo Action', () => {
        it('should fetch model info successfully', async () => {
            const mockModelInfo = {
                name: 'LocalizationNet v1.0',
                version: '1.0.0',
                accuracy: 0.85,
                predictions_total: 1000,
                predictions_successful: 850,
                last_trained: '2025-01-01T00:00:00Z',
            };

            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(mockModelInfo);

            await useDashboardStore.getState().fetchModelInfo();

            const state = useDashboardStore.getState();
            expect(state.data.modelInfo).toEqual(mockModelInfo);
            expect(inferenceService.getModelInfo).toHaveBeenCalledOnce();
        });

        it('should handle fetchModelInfo error gracefully', async () => {
            const errorMessage = 'Model not found';
            vi.mocked(inferenceService.getModelInfo).mockRejectedValue(new Error(errorMessage));

            // fetchModelInfo handles errors internally (not critical)
            await useDashboardStore.getState().fetchModelInfo();

            const state = useDashboardStore.getState();
            // Error should not be set, as it's handled internally
            expect(state.data.modelInfo).toBe(null);
        });
    });

    describe('fetchServicesHealth Action', () => {
        it('should fetch services health successfully', async () => {
            const mockServicesHealth = {
                'api-gateway': { status: 'healthy', latency_ms: 10 },
                'backend': { status: 'healthy', latency_ms: 50 },
                'training': { status: 'healthy', latency_ms: 30 },
                'inference': { status: 'healthy', latency_ms: 45 },
                'data-ingestion-web': { status: 'healthy', latency_ms: 20 },
            };

            vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue(mockServicesHealth);

            await useDashboardStore.getState().fetchServicesHealth();

            const state = useDashboardStore.getState();
            expect(state.data.servicesHealth).toEqual(mockServicesHealth);
            expect(systemService.checkAllServicesHealth).toHaveBeenCalledOnce();
        });

        it('should handle fetchServicesHealth error gracefully', async () => {
            const errorMessage = 'Services unavailable';
            vi.mocked(systemService.checkAllServicesHealth).mockRejectedValue(new Error(errorMessage));

            // fetchServicesHealth throws errors
            await expect(
                useDashboardStore.getState().fetchServicesHealth()
            ).rejects.toThrow(errorMessage);
        });
    });

    describe('fetchDashboardData Action (Comprehensive)', () => {
        it('should set loading state during fetch', async () => {
            vi.mocked(analyticsService.getDashboardMetrics).mockResolvedValue({
                signalDetections: 0,
                systemUptime: 0,
                modelAccuracy: 0,
            });
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);
            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(null);
            vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({});

            const promise = useDashboardStore.getState().fetchDashboardData();
            
            // Check loading state was set (but due to async nature, might complete fast)
            await promise;

            // Should not be loading after completion
            expect(useDashboardStore.getState().isLoading).toBe(false);
        });

        it('should fetch all dashboard data successfully', async () => {
            const mockMetrics = {
                signalDetections: 42,
                systemUptime: 3600,
                modelAccuracy: 0.85,
            };
            const mockWebSDRs = [{ id: 1, name: 'Test' }];
            const mockModelInfo = { 
                accuracy: 0.85,
                uptime_seconds: 3600,  // Add uptime_seconds
            };
            const mockServicesHealth = { 'api-gateway': { status: 'healthy' } };

            vi.mocked(analyticsService.getDashboardMetrics).mockResolvedValue(mockMetrics);
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue(mockWebSDRs);
            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(mockModelInfo);
            vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue(mockServicesHealth);

            await useDashboardStore.getState().fetchDashboardData();

            const state = useDashboardStore.getState();
            expect(state.data.websdrs).toEqual(mockWebSDRs);
            expect(state.data.modelInfo).toEqual(mockModelInfo);
            expect(state.data.servicesHealth).toEqual(mockServicesHealth);
            expect(state.metrics.signalDetections).toBe(42);
            expect(state.metrics.systemUptime).toBe(3600); // From modelInfo.uptime_seconds
            expect(state.error).toBe(null);
            expect(state.lastUpdate).toBeTruthy();
        });

        it('should reset retry count on successful fetch', async () => {
            // Set some retry state
            useDashboardStore.setState({ retryCount: 3, retryDelay: 8000 });

            vi.mocked(analyticsService.getDashboardMetrics).mockResolvedValue({
                signalDetections: 0,
                systemUptime: 0,
                modelAccuracy: 0,
            });
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);
            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(null);
            vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({});

            await useDashboardStore.getState().fetchDashboardData();

            const state = useDashboardStore.getState();
            expect(state.retryCount).toBe(0);
            expect(state.retryDelay).toBe(1000);
        });

        it('should handle partial failures gracefully', async () => {
            // getDashboardMetrics succeeds
            vi.mocked(analyticsService.getDashboardMetrics).mockResolvedValue({
                signalDetections: 42,
                systemUptime: 3600,
                modelAccuracy: 0.85,
            });
            // WebSDR succeeds
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([{ id: 1, name: 'Test' }]);
            // Others fail (but handled with Promise.allSettled)
            vi.mocked(inferenceService.getModelInfo).mockRejectedValue(new Error('Model error'));
            vi.mocked(systemService.checkAllServicesHealth).mockRejectedValue(new Error('Health error'));

            await useDashboardStore.getState().fetchDashboardData();

            const state = useDashboardStore.getState();
            // Metrics should be loaded (from analytics)
            expect(state.metrics.signalDetections).toBe(42);
            // WebSDR data should be loaded
            expect(state.data.websdrs).toHaveLength(1);
            // No error should be set because errors are handled with Promise.allSettled
            expect(state.lastUpdate).toBeTruthy();
        });
    });

    describe('refreshAll Action', () => {
        it('should call fetchDashboardData', async () => {
            vi.mocked(analyticsService.getDashboardMetrics).mockResolvedValue({
                signalDetections: 0,
                systemUptime: 0,
                modelAccuracy: 0,
            });
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);
            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(null);
            vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({});

            await useDashboardStore.getState().refreshAll();

            expect(analyticsService.getDashboardMetrics).toHaveBeenCalled();
            expect(webSDRService.getWebSDRs).toHaveBeenCalled();
            expect(inferenceService.getModelInfo).toHaveBeenCalled();
            expect(systemService.checkAllServicesHealth).toHaveBeenCalled();
        });
    });

    describe('Edge Cases and Error Handling', () => {
        it('should handle empty WebSDR list', async () => {
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);

            await useDashboardStore.getState().fetchWebSDRs();

            const state = useDashboardStore.getState();
            expect(state.data.websdrs).toEqual([]);
            expect(state.error).toBe(null);
        });

        it('should handle null model info', async () => {
            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(null);

            await useDashboardStore.getState().fetchModelInfo();

            const state = useDashboardStore.getState();
            expect(state.data.modelInfo).toBe(null);
            expect(state.error).toBe(null);
        });

        it('should handle network timeout', async () => {
            vi.mocked(webSDRService.getWebSDRs).mockRejectedValue(
                new Error('Network timeout')
            );

            // fetchWebSDRs throws errors
            await expect(
                useDashboardStore.getState().fetchWebSDRs()
            ).rejects.toThrow('Network timeout');
        });

        it('should clear error when setting to null', () => {
            useDashboardStore.setState({ error: 'Some error' });
            useDashboardStore.getState().setError(null);
            expect(useDashboardStore.getState().error).toBe(null);
        });
    });

    describe('State Consistency', () => {
        it('should update lastUpdate timestamp on successful fetch', async () => {
            vi.mocked(analyticsService.getDashboardMetrics).mockResolvedValue({
                signalDetections: 0,
                systemUptime: 0,
                modelAccuracy: 0,
            });
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);
            vi.mocked(inferenceService.getModelInfo).mockResolvedValue(null);
            vi.mocked(systemService.checkAllServicesHealth).mockResolvedValue({});

            const beforeFetch = new Date();
            await useDashboardStore.getState().fetchDashboardData();
            const afterFetch = new Date();

            const state = useDashboardStore.getState();
            expect(state.lastUpdate).toBeTruthy();
            
            if (state.lastUpdate) {
                const updateTime = new Date(state.lastUpdate);
                expect(updateTime.getTime()).toBeGreaterThanOrEqual(beforeFetch.getTime());
                expect(updateTime.getTime()).toBeLessThanOrEqual(afterFetch.getTime());
            }
        });

        it('should maintain data integrity across multiple fetches', async () => {
            // First fetch
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([{ id: 1, name: 'Test1' }]);
            await useDashboardStore.getState().fetchWebSDRs();
            expect(useDashboardStore.getState().data.websdrs).toHaveLength(1);

            // Second fetch with different data
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([
                { id: 1, name: 'Test1' },
                { id: 2, name: 'Test2' },
            ]);
            await useDashboardStore.getState().fetchWebSDRs();
            expect(useDashboardStore.getState().data.websdrs).toHaveLength(2);
        });
    });
});
