/**
 * WebSDR Store Tests
 *
 * Comprehensive test suite for the websdrStore Zustand store
 * Tests all actions: WebSDR fetching, health checking, selectors, and state management
 * Truth-first approach: Tests real Zustand store behavior with mocked API responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/websdrStore');

// Import after unmocking
import { useWebSDRStore } from './websdrStore';
import { webSDRService } from '@/services/api';

// Mock the API services
vi.mock('@/services/api', () => ({
    webSDRService: {
        getWebSDRs: vi.fn(),
        checkWebSDRHealth: vi.fn(),
    },
    acquisitionService: {},
    inferenceService: {},
    systemService: {},
    analyticsService: {},
    sessionService: {},
}));

describe('WebSDR Store (Zustand)', () => {
    beforeEach(() => {
        // Reset store to initial state before each test
        useWebSDRStore.setState({
            websdrs: [],
            healthStatus: {},
            isLoading: false,
            error: null,
            lastHealthCheck: null,
        });
        vi.clearAllMocks();
    });

    describe('Store Initialization', () => {
        it('should initialize with default state', () => {
            const state = useWebSDRStore.getState();
            expect(state.websdrs).toEqual([]);
            expect(state.healthStatus).toEqual({});
            expect(state.isLoading).toBe(false);
            expect(state.error).toBe(null);
            expect(state.lastHealthCheck).toBe(null);
        });

        it('should have all required actions', () => {
            const state = useWebSDRStore.getState();
            expect(typeof state.fetchWebSDRs).toBe('function');
            expect(typeof state.checkHealth).toBe('function');
            expect(typeof state.getActiveWebSDRs).toBe('function');
            expect(typeof state.getWebSDRById).toBe('function');
            expect(typeof state.isWebSDROnline).toBe('function');
            expect(typeof state.refreshAll).toBe('function');
        });
    });

    describe('fetchWebSDRs Action', () => {
        it('should fetch WebSDRs successfully', async () => {
            const mockWebSDRs = [
                {
                    id: 1,
                    name: 'WebSDR Torino',
                    url: 'http://websdr-torino.example.com',
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
                    url: 'http://websdr-genova.example.com',
                    location_name: 'Genova, Italy',
                    latitude: 44.4056,
                    longitude: 8.9463,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
            ];

            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue(mockWebSDRs);

            await useWebSDRStore.getState().fetchWebSDRs();

            const state = useWebSDRStore.getState();
            expect(state.websdrs).toEqual(mockWebSDRs);
            expect(state.isLoading).toBe(false);
            expect(state.error).toBe(null);
            expect(webSDRService.getWebSDRs).toHaveBeenCalledOnce();
        });

        it('should set loading state during fetch', async () => {
            vi.mocked(webSDRService.getWebSDRs).mockImplementation(
                () => new Promise((resolve) => setTimeout(() => resolve([]), 50))
            );

            const promise = useWebSDRStore.getState().fetchWebSDRs();
            
            // Should be loading immediately
            expect(useWebSDRStore.getState().isLoading).toBe(true);

            await promise;

            // Should not be loading after completion
            expect(useWebSDRStore.getState().isLoading).toBe(false);
        });

        it('should handle fetch error gracefully', async () => {
            const errorMessage = 'Network error';
            vi.mocked(webSDRService.getWebSDRs).mockRejectedValue(new Error(errorMessage));

            await useWebSDRStore.getState().fetchWebSDRs();

            const state = useWebSDRStore.getState();
            expect(state.error).toBe(errorMessage);
            expect(state.isLoading).toBe(false);
        });

        it('should clear error on successful fetch', async () => {
            // First set an error
            useWebSDRStore.setState({ error: 'Previous error' });

            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);

            await useWebSDRStore.getState().fetchWebSDRs();

            const state = useWebSDRStore.getState();
            expect(state.error).toBe(null);
        });
    });

    describe('checkHealth Action', () => {
        it('should check health successfully', async () => {
            const mockHealthStatus = {
                1: { status: 'online', response_time_ms: 150 },
                2: { status: 'online', response_time_ms: 180 },
                3: { status: 'offline', response_time_ms: null },
            };

            vi.mocked(webSDRService.checkWebSDRHealth).mockResolvedValue(mockHealthStatus);

            const beforeCheck = new Date();
            await useWebSDRStore.getState().checkHealth();
            const afterCheck = new Date();

            const state = useWebSDRStore.getState();
            expect(state.healthStatus).toEqual(mockHealthStatus);
            expect(state.lastHealthCheck).toBeTruthy();
            
            if (state.lastHealthCheck) {
                const checkTime = new Date(state.lastHealthCheck);
                expect(checkTime.getTime()).toBeGreaterThanOrEqual(beforeCheck.getTime());
                expect(checkTime.getTime()).toBeLessThanOrEqual(afterCheck.getTime());
            }
        });

        it('should handle health check error gracefully', async () => {
            const errorMessage = 'Health check failed';
            vi.mocked(webSDRService.checkWebSDRHealth).mockRejectedValue(new Error(errorMessage));

            // Health check errors should not set error state (logged only)
            await useWebSDRStore.getState().checkHealth();

            const state = useWebSDRStore.getState();
            // Error should not be set for health checks
            expect(state.error).toBe(null);
        });

        it('should clear error on successful health check', async () => {
            useWebSDRStore.setState({ error: 'Previous error' });

            vi.mocked(webSDRService.checkWebSDRHealth).mockResolvedValue({});

            await useWebSDRStore.getState().checkHealth();

            const state = useWebSDRStore.getState();
            expect(state.error).toBe(null);
        });
    });

    describe('Selector Functions', () => {
        beforeEach(() => {
            // Setup test data
            useWebSDRStore.setState({
                websdrs: [
                    { id: 1, name: 'WebSDR 1', is_active: true },
                    { id: 2, name: 'WebSDR 2', is_active: false },
                    { id: 3, name: 'WebSDR 3', is_active: true },
                ],
                healthStatus: {
                    1: { status: 'online', response_time_ms: 150 },
                    2: { status: 'offline', response_time_ms: null },
                    3: { status: 'online', response_time_ms: 180 },
                },
            });
        });

        describe('getActiveWebSDRs', () => {
            it('should return only active WebSDRs', () => {
                const active = useWebSDRStore.getState().getActiveWebSDRs();
                
                expect(active).toHaveLength(2);
                expect(active[0].id).toBe(1);
                expect(active[1].id).toBe(3);
                expect(active.every(w => w.is_active)).toBe(true);
            });

            it('should return empty array when no active WebSDRs', () => {
                useWebSDRStore.setState({
                    websdrs: [
                        { id: 1, name: 'WebSDR 1', is_active: false },
                        { id: 2, name: 'WebSDR 2', is_active: false },
                    ],
                });

                const active = useWebSDRStore.getState().getActiveWebSDRs();
                expect(active).toEqual([]);
            });
        });

        describe('getWebSDRById', () => {
            it('should return WebSDR by id', () => {
                const websdr = useWebSDRStore.getState().getWebSDRById(2);
                
                expect(websdr).toBeDefined();
                expect(websdr?.id).toBe(2);
                expect(websdr?.name).toBe('WebSDR 2');
            });

            it('should return undefined for non-existent id', () => {
                const websdr = useWebSDRStore.getState().getWebSDRById(999);
                expect(websdr).toBeUndefined();
            });
        });

        describe('isWebSDROnline', () => {
            it('should return true for online WebSDR', () => {
                const isOnline = useWebSDRStore.getState().isWebSDROnline(1);
                expect(isOnline).toBe(true);
            });

            it('should return false for offline WebSDR', () => {
                const isOnline = useWebSDRStore.getState().isWebSDROnline(2);
                expect(isOnline).toBe(false);
            });

            it('should return false for WebSDR without health status', () => {
                const isOnline = useWebSDRStore.getState().isWebSDROnline(999);
                expect(isOnline).toBe(false);
            });
        });
    });

    describe('refreshAll Action', () => {
        it('should fetch both WebSDRs and health status', async () => {
            const mockWebSDRs = [
                { id: 1, name: 'WebSDR 1', is_active: true },
            ];
            const mockHealth = {
                1: { status: 'online', response_time_ms: 150 },
            };

            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue(mockWebSDRs);
            vi.mocked(webSDRService.checkWebSDRHealth).mockResolvedValue(mockHealth);

            await useWebSDRStore.getState().refreshAll();

            const state = useWebSDRStore.getState();
            expect(state.websdrs).toEqual(mockWebSDRs);
            expect(state.healthStatus).toEqual(mockHealth);
            expect(webSDRService.getWebSDRs).toHaveBeenCalledOnce();
            expect(webSDRService.checkWebSDRHealth).toHaveBeenCalledOnce();
        });

        it('should handle partial failures', async () => {
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);
            vi.mocked(webSDRService.checkWebSDRHealth).mockRejectedValue(new Error('Health check failed'));

            // Should not throw
            await useWebSDRStore.getState().refreshAll();

            const state = useWebSDRStore.getState();
            // WebSDRs should be fetched even if health check fails
            expect(state.websdrs).toEqual([]);
        });
    });

    describe('Edge Cases', () => {
        it('should handle empty WebSDR list', async () => {
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([]);

            await useWebSDRStore.getState().fetchWebSDRs();

            const state = useWebSDRStore.getState();
            expect(state.websdrs).toEqual([]);
            expect(state.error).toBe(null);
        });

        it('should handle empty health status', async () => {
            vi.mocked(webSDRService.checkWebSDRHealth).mockResolvedValue({});

            await useWebSDRStore.getState().checkHealth();

            const state = useWebSDRStore.getState();
            expect(state.healthStatus).toEqual({});
        });

        it('should handle non-Error exceptions', async () => {
            vi.mocked(webSDRService.getWebSDRs).mockRejectedValue('String error');

            await useWebSDRStore.getState().fetchWebSDRs();

            const state = useWebSDRStore.getState();
            expect(state.error).toBe('Failed to fetch WebSDRs');
        });

        it('should maintain state integrity across multiple operations', async () => {
            // First fetch
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([
                { id: 1, name: 'WebSDR 1', is_active: true },
            ]);
            await useWebSDRStore.getState().fetchWebSDRs();
            expect(useWebSDRStore.getState().websdrs).toHaveLength(1);

            // Health check
            vi.mocked(webSDRService.checkWebSDRHealth).mockResolvedValue({
                1: { status: 'online', response_time_ms: 150 },
            });
            await useWebSDRStore.getState().checkHealth();
            expect(useWebSDRStore.getState().healthStatus[1]).toBeDefined();

            // Second fetch with different data
            vi.mocked(webSDRService.getWebSDRs).mockResolvedValue([
                { id: 1, name: 'WebSDR 1', is_active: true },
                { id: 2, name: 'WebSDR 2', is_active: false },
            ]);
            await useWebSDRStore.getState().fetchWebSDRs();
            
            const state = useWebSDRStore.getState();
            expect(state.websdrs).toHaveLength(2);
            // Health status should persist
            expect(state.healthStatus[1]).toBeDefined();
        });
    });
});
