/**
 * Localization Store Tests
 *
 * Comprehensive test suite for the localizationStore Zustand store
 * Tests all actions: localization fetching, prediction, state management
 * Truth-first approach: Tests real Zustand store behavior with mocked API responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/localizationStore');

// Import after unmocking
import { useLocalizationStore } from './localizationStore';
import { inferenceService } from '@/services/api';

// Mock the API services
vi.mock('@/services/api', () => ({
    inferenceService: {
        getRecentLocalizations: vi.fn(),
        predictLocalization: vi.fn(),
    },
    webSDRService: {},
    acquisitionService: {},
    systemService: {},
    analyticsService: {},
    sessionService: {},
}));

describe('Localization Store (Zustand)', () => {
    beforeEach(() => {
        // Reset store to initial state before each test
        useLocalizationStore.setState({
            recentLocalizations: [],
            currentPrediction: null,
            isLoading: false,
            isPredicting: false,
            error: null,
            selectedResultId: null,
        });
        vi.clearAllMocks();
    });

    describe('Store Initialization', () => {
        it('should initialize with default state', () => {
            const state = useLocalizationStore.getState();
            expect(state.recentLocalizations).toEqual([]);
            expect(state.currentPrediction).toBe(null);
            expect(state.isLoading).toBe(false);
            expect(state.isPredicting).toBe(false);
            expect(state.error).toBe(null);
            expect(state.selectedResultId).toBe(null);
        });

        it('should have all required actions', () => {
            const state = useLocalizationStore.getState();
            expect(typeof state.setLoading).toBe('function');
            expect(typeof state.setPredicting).toBe('function');
            expect(typeof state.setError).toBe('function');
            expect(typeof state.setSelectedResult).toBe('function');
            expect(typeof state.fetchRecentLocalizations).toBe('function');
            expect(typeof state.predictLocalization).toBe('function');
            expect(typeof state.clearCurrentPrediction).toBe('function');
            expect(typeof state.refreshData).toBe('function');
        });
    });

    describe('Basic Setters', () => {
        it('should set loading state', () => {
            useLocalizationStore.getState().setLoading(true);
            expect(useLocalizationStore.getState().isLoading).toBe(true);

            useLocalizationStore.getState().setLoading(false);
            expect(useLocalizationStore.getState().isLoading).toBe(false);
        });

        it('should set predicting state', () => {
            useLocalizationStore.getState().setPredicting(true);
            expect(useLocalizationStore.getState().isPredicting).toBe(true);

            useLocalizationStore.getState().setPredicting(false);
            expect(useLocalizationStore.getState().isPredicting).toBe(false);
        });

        it('should set error', () => {
            const errorMessage = 'Test error';
            useLocalizationStore.getState().setError(errorMessage);
            expect(useLocalizationStore.getState().error).toBe(errorMessage);

            useLocalizationStore.getState().setError(null);
            expect(useLocalizationStore.getState().error).toBe(null);
        });

        it('should set selected result', () => {
            useLocalizationStore.getState().setSelectedResult('result-123');
            expect(useLocalizationStore.getState().selectedResultId).toBe('result-123');

            useLocalizationStore.getState().setSelectedResult(null);
            expect(useLocalizationStore.getState().selectedResultId).toBe(null);
        });
    });

    describe('fetchRecentLocalizations Action', () => {
        it('should fetch recent localizations successfully', async () => {
            const mockLocalizations = [
                {
                    id: '1',
                    latitude: 45.0703,
                    longitude: 7.6869,
                    uncertainty_meters: 25.5,
                    timestamp: '2025-01-01T00:00:00Z',
                },
                {
                    id: '2',
                    latitude: 44.4056,
                    longitude: 8.9463,
                    uncertainty_meters: 30.2,
                    timestamp: '2025-01-02T00:00:00Z',
                },
            ];

            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue(mockLocalizations);

            await useLocalizationStore.getState().fetchRecentLocalizations();

            const state = useLocalizationStore.getState();
            expect(state.recentLocalizations).toEqual(mockLocalizations);
            expect(state.isLoading).toBe(false);
            expect(state.error).toBe(null);
            expect(inferenceService.getRecentLocalizations).toHaveBeenCalledWith(10);
        });

        it('should fetch with custom limit', async () => {
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([]);

            await useLocalizationStore.getState().fetchRecentLocalizations(20);

            expect(inferenceService.getRecentLocalizations).toHaveBeenCalledWith(20);
        });

        it('should set loading state during fetch', async () => {
            vi.mocked(inferenceService.getRecentLocalizations).mockImplementation(
                () => new Promise((resolve) => setTimeout(() => resolve([]), 50))
            );

            const promise = useLocalizationStore.getState().fetchRecentLocalizations();
            
            // Should be loading immediately
            expect(useLocalizationStore.getState().isLoading).toBe(true);

            await promise;

            // Should not be loading after completion
            expect(useLocalizationStore.getState().isLoading).toBe(false);
        });

        it('should handle fetch error gracefully', async () => {
            const errorMessage = 'Failed to fetch localizations';
            vi.mocked(inferenceService.getRecentLocalizations).mockRejectedValue(new Error(errorMessage));

            await useLocalizationStore.getState().fetchRecentLocalizations();

            const state = useLocalizationStore.getState();
            expect(state.error).toBe(errorMessage);
            expect(state.isLoading).toBe(false);
        });

        it('should clear error on successful fetch', async () => {
            useLocalizationStore.setState({ error: 'Previous error' });

            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([]);

            await useLocalizationStore.getState().fetchRecentLocalizations();

            const state = useLocalizationStore.getState();
            expect(state.error).toBe(null);
        });
    });

    describe('predictLocalization Action', () => {
        it('should predict localization successfully', async () => {
            const mockRequest = {
                session_id: 123,
                frequency_mhz: 145.5,
            };

            const mockPrediction = {
                latitude: 45.0703,
                longitude: 7.6869,
                uncertainty_meters: 25.5,
                confidence: 0.85,
                timestamp: '2025-01-01T00:00:00Z',
            };

            const mockLocalizations = [mockPrediction];

            vi.mocked(inferenceService.predictLocalization).mockResolvedValue(mockPrediction);
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue(mockLocalizations);

            const result = await useLocalizationStore.getState().predictLocalization(mockRequest);

            expect(result).toEqual(mockPrediction);
            expect(inferenceService.predictLocalization).toHaveBeenCalledWith(mockRequest);
            
            const state = useLocalizationStore.getState();
            expect(state.currentPrediction).toEqual(mockPrediction);
            expect(state.isPredicting).toBe(false);
            expect(state.error).toBe(null);
            // Should refresh recent localizations
            expect(inferenceService.getRecentLocalizations).toHaveBeenCalled();
        });

        it('should set predicting state during prediction', async () => {
            vi.mocked(inferenceService.predictLocalization).mockImplementation(
                () => new Promise((resolve) => setTimeout(() => resolve({
                    latitude: 45.0703,
                    longitude: 7.6869,
                    uncertainty_meters: 25.5,
                    confidence: 0.85,
                    timestamp: '2025-01-01T00:00:00Z',
                }), 50))
            );
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([]);

            const promise = useLocalizationStore.getState().predictLocalization({
                session_id: 123,
                frequency_mhz: 145.5,
            });
            
            // Should be predicting immediately
            expect(useLocalizationStore.getState().isPredicting).toBe(true);

            await promise;

            // Should not be predicting after completion
            expect(useLocalizationStore.getState().isPredicting).toBe(false);
        });

        it('should handle prediction error', async () => {
            const errorMessage = 'Prediction failed';
            vi.mocked(inferenceService.predictLocalization).mockRejectedValue(new Error(errorMessage));

            await expect(
                useLocalizationStore.getState().predictLocalization({
                    session_id: 123,
                    frequency_mhz: 145.5,
                })
            ).rejects.toThrow(errorMessage);

            const state = useLocalizationStore.getState();
            expect(state.error).toBe(errorMessage);
            expect(state.isPredicting).toBe(false);
        });

        it('should clear error on successful prediction', async () => {
            useLocalizationStore.setState({ error: 'Previous error' });

            const mockPrediction = {
                latitude: 45.0703,
                longitude: 7.6869,
                uncertainty_meters: 25.5,
                confidence: 0.85,
                timestamp: '2025-01-01T00:00:00Z',
            };

            vi.mocked(inferenceService.predictLocalization).mockResolvedValue(mockPrediction);
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([]);

            await useLocalizationStore.getState().predictLocalization({
                session_id: 123,
                frequency_mhz: 145.5,
            });

            const state = useLocalizationStore.getState();
            expect(state.error).toBe(null);
        });
    });

    describe('clearCurrentPrediction Action', () => {
        it('should clear current prediction', () => {
            useLocalizationStore.setState({
                currentPrediction: {
                    latitude: 45.0703,
                    longitude: 7.6869,
                    uncertainty_meters: 25.5,
                    confidence: 0.85,
                    timestamp: '2025-01-01T00:00:00Z',
                },
            });

            useLocalizationStore.getState().clearCurrentPrediction();

            expect(useLocalizationStore.getState().currentPrediction).toBe(null);
        });
    });

    describe('refreshData Action', () => {
        it('should refresh recent localizations', async () => {
            const mockLocalizations = [
                {
                    id: '1',
                    latitude: 45.0703,
                    longitude: 7.6869,
                    uncertainty_meters: 25.5,
                    timestamp: '2025-01-01T00:00:00Z',
                },
            ];

            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue(mockLocalizations);

            await useLocalizationStore.getState().refreshData();

            const state = useLocalizationStore.getState();
            expect(state.recentLocalizations).toEqual(mockLocalizations);
            expect(inferenceService.getRecentLocalizations).toHaveBeenCalled();
        });
    });

    describe('Edge Cases', () => {
        it('should handle empty recent localizations', async () => {
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([]);

            await useLocalizationStore.getState().fetchRecentLocalizations();

            const state = useLocalizationStore.getState();
            expect(state.recentLocalizations).toEqual([]);
            expect(state.error).toBe(null);
        });

        it('should handle non-Error exceptions in fetch', async () => {
            vi.mocked(inferenceService.getRecentLocalizations).mockRejectedValue('String error');

            await useLocalizationStore.getState().fetchRecentLocalizations();

            const state = useLocalizationStore.getState();
            expect(state.error).toBe('Failed to fetch localizations');
        });

        it('should handle non-Error exceptions in prediction', async () => {
            vi.mocked(inferenceService.predictLocalization).mockRejectedValue('String error');

            await expect(
                useLocalizationStore.getState().predictLocalization({
                    session_id: 123,
                    frequency_mhz: 145.5,
                })
            ).rejects.toThrow();

            const state = useLocalizationStore.getState();
            expect(state.error).toBe('Prediction failed');
        });

        it('should maintain state integrity across multiple predictions', async () => {
            const prediction1 = {
                latitude: 45.0703,
                longitude: 7.6869,
                uncertainty_meters: 25.5,
                confidence: 0.85,
                timestamp: '2025-01-01T00:00:00Z',
            };

            const prediction2 = {
                latitude: 44.4056,
                longitude: 8.9463,
                uncertainty_meters: 30.2,
                confidence: 0.80,
                timestamp: '2025-01-02T00:00:00Z',
            };

            vi.mocked(inferenceService.predictLocalization).mockResolvedValueOnce(prediction1);
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([prediction1]);

            await useLocalizationStore.getState().predictLocalization({
                session_id: 123,
                frequency_mhz: 145.5,
            });

            expect(useLocalizationStore.getState().currentPrediction).toEqual(prediction1);

            // Second prediction should replace the first
            vi.mocked(inferenceService.predictLocalization).mockResolvedValueOnce(prediction2);
            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([prediction2, prediction1]);

            await useLocalizationStore.getState().predictLocalization({
                session_id: 124,
                frequency_mhz: 435.0,
            });

            const state = useLocalizationStore.getState();
            expect(state.currentPrediction).toEqual(prediction2);
            expect(state.recentLocalizations).toHaveLength(2);
        });

        it('should handle concurrent fetch and predict operations', async () => {
            const mockPrediction = {
                latitude: 45.0703,
                longitude: 7.6869,
                uncertainty_meters: 25.5,
                confidence: 0.85,
                timestamp: '2025-01-01T00:00:00Z',
            };

            vi.mocked(inferenceService.getRecentLocalizations).mockResolvedValue([]);
            vi.mocked(inferenceService.predictLocalization).mockResolvedValue(mockPrediction);

            // Start both operations concurrently
            await Promise.all([
                useLocalizationStore.getState().fetchRecentLocalizations(),
                useLocalizationStore.getState().predictLocalization({
                    session_id: 123,
                    frequency_mhz: 145.5,
                }),
            ]);

            // Both operations should complete successfully
            const state = useLocalizationStore.getState();
            expect(state.currentPrediction).toEqual(mockPrediction);
            expect(state.isLoading).toBe(false);
            expect(state.isPredicting).toBe(false);
        });
    });
});
