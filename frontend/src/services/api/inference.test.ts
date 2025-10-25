/**
 * Inference API Service Integration Tests
 * 
 * Tests HTTP integration with the inference service.
 * Uses axios-mock-adapter to mock HTTP responses while testing real axios client behavior.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import MockAdapter from 'axios-mock-adapter';
import api from '@/lib/api';
import {
    getModelInfo,
    getModelPerformance,
    predictLocalization,
    predictLocalizationBatch,
    getRecentLocalizations,
    type PredictionRequest,
    type PredictionResponse,
    type BatchPredictionRequest,
} from './inference';
import type { ModelInfo, ModelPerformanceMetrics, LocalizationResult } from './types';

// Mock the auth store
vi.mock('@/store', () => ({
    useAuthStore: {
        getState: vi.fn(() => ({ token: null })),
    },
}));

let mock: MockAdapter;

beforeEach(() => {
    mock = new MockAdapter(api);
});

afterEach(() => {
    mock.reset();
    mock.restore();
    vi.clearAllMocks();
});

describe('getModelInfo', () => {
    it('should fetch model information successfully', async () => {
        const mockModelInfo: ModelInfo = {
            name: 'LocalizationNet-v1',
            version: '1.0.0',
            architecture: 'ResNet-18',
            parameters: 11689512,
            input_shape: [1, 128, 512],
            training_date: '2025-10-15T10:00:00Z',
            accuracy_meters: 25.5,
        };

        mock.onGet('/api/v1/analytics/model/info').reply(200, mockModelInfo);

        const result = await getModelInfo();

        expect(result).toEqual(mockModelInfo);
        expect(result.name).toBe('LocalizationNet-v1');
        expect(result.parameters).toBeGreaterThan(0);
    });

    it('should handle 500 error when model info not available', async () => {
        mock.onGet('/api/v1/analytics/model/info').reply(500, {
            detail: 'Model not loaded',
        });

        await expect(getModelInfo()).rejects.toThrow();
    });

    it('should handle network errors', async () => {
        mock.onGet('/api/v1/analytics/model/info').networkError();

        await expect(getModelInfo()).rejects.toThrow();
    });
});

describe('getModelPerformance', () => {
    it('should fetch model performance metrics successfully', async () => {
        const mockPerformance: ModelPerformanceMetrics = {
            total_predictions: 1543,
            average_inference_time_ms: 125.3,
            cache_hit_rate: 0.65,
            accuracy_meters: 28.7,
            last_updated: '2025-10-25T12:00:00Z',
        };

        mock.onGet('/api/v1/analytics/model/performance').reply(200, mockPerformance);

        const result = await getModelPerformance();

        expect(result.total_predictions).toBe(1543);
        expect(result.average_inference_time_ms).toBeGreaterThan(0);
        expect(result.cache_hit_rate).toBeGreaterThanOrEqual(0);
        expect(result.cache_hit_rate).toBeLessThanOrEqual(1);
    });

    it('should handle empty metrics (new deployment)', async () => {
        const mockPerformance: ModelPerformanceMetrics = {
            total_predictions: 0,
            average_inference_time_ms: 0,
            cache_hit_rate: 0,
            accuracy_meters: 0,
            last_updated: '2025-10-25T12:00:00Z',
        };

        mock.onGet('/api/v1/analytics/model/performance').reply(200, mockPerformance);

        const result = await getModelPerformance();

        expect(result.total_predictions).toBe(0);
    });

    it('should handle 503 error when metrics service unavailable', async () => {
        mock.onGet('/api/v1/analytics/model/performance').reply(503);

        await expect(getModelPerformance()).rejects.toThrow();
    });
});

describe('predictLocalization', () => {
    it('should make a successful prediction', async () => {
        const request: PredictionRequest = {
            iq_data: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            session_id: 'session-123',
            cache_enabled: true,
        };

        const mockResponse: PredictionResponse = {
            position: {
                latitude: 45.0642,
                longitude: 7.6603,
            },
            uncertainty: {
                sigma_x: 25.3,
                sigma_y: 30.1,
                theta: 0.45,
                confidence_interval: 0.68,
            },
            confidence: 0.85,
            model_version: '1.0.0',
            inference_time_ms: 142,
            timestamp: '2025-10-25T12:30:00Z',
            session_id: 'session-123',
            _cache_hit: false,
        };

        mock.onPost('/api/v1/inference/predict').reply(200, mockResponse);

        const result = await predictLocalization(request);

        expect(result.position.latitude).toBeCloseTo(45.0642, 4);
        expect(result.position.longitude).toBeCloseTo(7.6603, 4);
        expect(result.uncertainty.sigma_x).toBeGreaterThan(0);
        expect(result.confidence).toBeGreaterThan(0);
        expect(result.confidence).toBeLessThanOrEqual(1);
        expect(result.inference_time_ms).toBeGreaterThan(0);
    });

    it('should handle cache hit response', async () => {
        const request: PredictionRequest = {
            iq_data: [[0.1, 0.2]],
            cache_enabled: true,
        };

        const mockResponse: PredictionResponse = {
            position: { latitude: 45.0, longitude: 7.6 },
            uncertainty: { sigma_x: 25, sigma_y: 30, theta: 0.5, confidence_interval: 0.68 },
            confidence: 0.9,
            model_version: '1.0.0',
            inference_time_ms: 5, // Much faster due to cache
            timestamp: '2025-10-25T12:30:00Z',
            session_id: 'cached-session',
            _cache_hit: true,
        };

        mock.onPost('/api/v1/inference/predict').reply(200, mockResponse);

        const result = await predictLocalization(request);

        expect(result._cache_hit).toBe(true);
        expect(result.inference_time_ms).toBeLessThan(20); // Cache should be fast
    });

    it('should handle 400 error for invalid IQ data', async () => {
        const request: PredictionRequest = {
            iq_data: [], // Empty IQ data
        };

        mock.onPost('/api/v1/inference/predict').reply(400, {
            detail: 'IQ data is required and must not be empty',
        });

        await expect(predictLocalization(request)).rejects.toThrow();
    });

    it('should handle 500 error during inference', async () => {
        const request: PredictionRequest = {
            iq_data: [[0.1, 0.2]],
        };

        mock.onPost('/api/v1/inference/predict').reply(500, {
            detail: 'Model inference failed',
        });

        await expect(predictLocalization(request)).rejects.toThrow();
    });

    it('should handle timeout errors', async () => {
        const request: PredictionRequest = {
            iq_data: [[0.1, 0.2]],
        };

        mock.onPost('/api/v1/inference/predict').timeout();

        await expect(predictLocalization(request)).rejects.toThrow();
    });

    it('should validate uncertainty values are reasonable', async () => {
        const request: PredictionRequest = {
            iq_data: [[0.1, 0.2]],
        };

        const mockResponse: PredictionResponse = {
            position: { latitude: 45.0, longitude: 7.6 },
            uncertainty: { 
                sigma_x: 15.5,  // Lower uncertainty = higher confidence
                sigma_y: 18.2, 
                theta: 0.3, 
                confidence_interval: 0.95  // 95% CI
            },
            confidence: 0.92,
            model_version: '1.0.0',
            inference_time_ms: 120,
            timestamp: '2025-10-25T12:30:00Z',
            session_id: 'test-session',
            _cache_hit: false,
        };

        mock.onPost('/api/v1/inference/predict').reply(200, mockResponse);

        const result = await predictLocalization(request);

        // Test uncertainty values make sense
        expect(result.uncertainty.sigma_x).toBeGreaterThan(0);
        expect(result.uncertainty.sigma_y).toBeGreaterThan(0);
        expect(result.uncertainty.confidence_interval).toBeGreaterThan(0);
        expect(result.uncertainty.confidence_interval).toBeLessThanOrEqual(1);
    });
});

describe('predictLocalizationBatch', () => {
    it('should make batch predictions successfully', async () => {
        const request: BatchPredictionRequest = {
            predictions: [
                { iq_data: [[0.1, 0.2]] },
                { iq_data: [[0.3, 0.4]] },
                { iq_data: [[0.5, 0.6]] },
            ],
        };

        const mockResponse = {
            predictions: [
                {
                    position: { latitude: 45.0, longitude: 7.6 },
                    uncertainty: { sigma_x: 25, sigma_y: 30, theta: 0.5, confidence_interval: 0.68 },
                    confidence: 0.85,
                    model_version: '1.0.0',
                    inference_time_ms: 120,
                    timestamp: '2025-10-25T12:30:00Z',
                    session_id: 'batch-1',
                    _cache_hit: false,
                },
                {
                    position: { latitude: 45.1, longitude: 7.7 },
                    uncertainty: { sigma_x: 20, sigma_y: 25, theta: 0.4, confidence_interval: 0.68 },
                    confidence: 0.88,
                    model_version: '1.0.0',
                    inference_time_ms: 115,
                    timestamp: '2025-10-25T12:30:01Z',
                    session_id: 'batch-2',
                    _cache_hit: false,
                },
                {
                    position: { latitude: 45.2, longitude: 7.8 },
                    uncertainty: { sigma_x: 22, sigma_y: 28, theta: 0.45, confidence_interval: 0.68 },
                    confidence: 0.87,
                    model_version: '1.0.0',
                    inference_time_ms: 118,
                    timestamp: '2025-10-25T12:30:02Z',
                    session_id: 'batch-3',
                    _cache_hit: false,
                },
            ],
            batch_id: 'batch-abc-123',
            total_time_ms: 353,
        };

        mock.onPost('/api/v1/inference/predict/batch').reply(200, mockResponse);

        const result = await predictLocalizationBatch(request);

        expect(result.predictions).toHaveLength(3);
        expect(result.batch_id).toBe('batch-abc-123');
        expect(result.total_time_ms).toBeGreaterThan(0);
        
        // Batch time should be less than or equal to sum of individual times (parallel or sequential)
        const individualSum = result.predictions.reduce((sum, p) => sum + p.inference_time_ms, 0);
        expect(result.total_time_ms).toBeLessThanOrEqual(individualSum);
    });

    it('should handle empty batch', async () => {
        const request: BatchPredictionRequest = {
            predictions: [],
        };

        mock.onPost('/api/v1/inference/predict/batch').reply(400, {
            detail: 'Batch must contain at least one prediction',
        });

        await expect(predictLocalizationBatch(request)).rejects.toThrow();
    });

    it('should handle partial failures in batch', async () => {
        const request: BatchPredictionRequest = {
            predictions: [
                { iq_data: [[0.1, 0.2]] },
                { iq_data: [[0.3, 0.4]] },
            ],
        };

        // Backend might return 207 Multi-Status or 500 depending on implementation
        mock.onPost('/api/v1/inference/predict/batch').reply(500, {
            detail: 'Some predictions failed',
        });

        await expect(predictLocalizationBatch(request)).rejects.toThrow();
    });
});

describe('getRecentLocalizations', () => {
    it('should fetch recent localizations with default limit', async () => {
        const mockLocalizations: LocalizationResult[] = [
            {
                id: 1,
                session_id: 'session-1',
                latitude: 45.0,
                longitude: 7.6,
                uncertainty_meters: 25.5,
                confidence: 0.85,
                timestamp: '2025-10-25T12:00:00Z',
            },
            {
                id: 2,
                session_id: 'session-2',
                latitude: 45.1,
                longitude: 7.7,
                uncertainty_meters: 30.2,
                confidence: 0.82,
                timestamp: '2025-10-25T11:50:00Z',
            },
        ];

        mock.onGet('/api/v1/analytics/localizations/recent').reply(200, mockLocalizations);

        const result = await getRecentLocalizations();

        expect(result).toHaveLength(2);
        expect(result[0].confidence).toBeGreaterThan(0);
        expect(result[0].uncertainty_meters).toBeGreaterThan(0);
    });

    it('should fetch recent localizations with custom limit', async () => {
        const mockLocalizations: LocalizationResult[] = Array.from({ length: 5 }, (_, i) => ({
            id: i + 1,
            session_id: `session-${i + 1}`,
            latitude: 45.0 + i * 0.01,
            longitude: 7.6 + i * 0.01,
            uncertainty_meters: 25 + i,
            confidence: 0.85 - i * 0.01,
            timestamp: new Date(Date.now() - i * 60000).toISOString(),
        }));

        mock.onGet('/api/v1/analytics/localizations/recent', { params: { limit: 5 } }).reply(200, mockLocalizations);

        const result = await getRecentLocalizations(5);

        expect(result).toHaveLength(5);
        // Verify ordering (most recent first)
        for (let i = 1; i < result.length; i++) {
            const prev = new Date(result[i - 1].timestamp);
            const curr = new Date(result[i].timestamp);
            expect(prev.getTime()).toBeGreaterThanOrEqual(curr.getTime());
        }
    });

    it('should handle empty results (no recent localizations)', async () => {
        mock.onGet('/api/v1/analytics/localizations/recent').reply(200, []);

        const result = await getRecentLocalizations();

        expect(result).toHaveLength(0);
    });

    it('should handle 500 error from analytics service', async () => {
        mock.onGet('/api/v1/analytics/localizations/recent').reply(500);

        await expect(getRecentLocalizations()).rejects.toThrow();
    });

    it('should handle network errors', async () => {
        mock.onGet('/api/v1/analytics/localizations/recent').networkError();

        await expect(getRecentLocalizations()).rejects.toThrow();
    });
});

describe('Edge Cases and Real-World Scenarios', () => {
    it('should handle concurrent prediction requests', async () => {
        const mockResponse: PredictionResponse = {
            position: { latitude: 45.0, longitude: 7.6 },
            uncertainty: { sigma_x: 25, sigma_y: 30, theta: 0.5, confidence_interval: 0.68 },
            confidence: 0.85,
            model_version: '1.0.0',
            inference_time_ms: 120,
            timestamp: '2025-10-25T12:30:00Z',
            session_id: 'concurrent',
            _cache_hit: false,
        };

        mock.onPost('/api/v1/inference/predict').reply(200, mockResponse);

        const requests = Array.from({ length: 5 }, () => 
            predictLocalization({ iq_data: [[0.1, 0.2]] })
        );

        const results = await Promise.all(requests);

        expect(results).toHaveLength(5);
        results.forEach(result => {
            expect(result.position).toBeDefined();
            expect(result.confidence).toBeGreaterThan(0);
        });
    });

    it('should validate position coordinates are valid', async () => {
        const request: PredictionRequest = {
            iq_data: [[0.1, 0.2]],
        };

        const mockResponse: PredictionResponse = {
            position: { 
                latitude: 45.0642,  // Valid latitude (-90 to 90)
                longitude: 7.6603   // Valid longitude (-180 to 180)
            },
            uncertainty: { sigma_x: 25, sigma_y: 30, theta: 0.5, confidence_interval: 0.68 },
            confidence: 0.85,
            model_version: '1.0.0',
            inference_time_ms: 120,
            timestamp: '2025-10-25T12:30:00Z',
            session_id: 'test',
            _cache_hit: false,
        };

        mock.onPost('/api/v1/inference/predict').reply(200, mockResponse);

        const result = await predictLocalization(request);

        expect(result.position.latitude).toBeGreaterThanOrEqual(-90);
        expect(result.position.latitude).toBeLessThanOrEqual(90);
        expect(result.position.longitude).toBeGreaterThanOrEqual(-180);
        expect(result.position.longitude).toBeLessThanOrEqual(180);
    });
});
