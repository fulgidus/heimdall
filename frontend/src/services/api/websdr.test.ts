/**
 * WebSDR API Service Tests
 *
 * Comprehensive test suite for WebSDR API client
 * Tests HTTP requests, responses, error handling, and data transformations
 * Truth-first approach: Tests real API client with mocked HTTP responses
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import MockAdapter from 'axios-mock-adapter';
import api from '@/lib/api';
import {
    getWebSDRs,
    checkWebSDRHealth,
    getWebSDRConfig,
    getActiveWebSDRs,
} from './websdr';
import type { WebSDRConfig, WebSDRHealthStatus } from './types';

// Mock the auth store
vi.mock('@/store', () => ({
    useAuthStore: {
        getState: vi.fn(() => ({ token: null })),
    },
}));

// Create axios mock adapter
let mock: MockAdapter;

describe('WebSDR API Service', () => {
    beforeEach(() => {
        mock = new MockAdapter(api);
    });

    afterEach(() => {
        mock.reset();
        mock.restore();
    });

    describe('getWebSDRs', () => {
        it('should fetch all WebSDRs successfully', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: '550e8400-e29b-41d4-a716-446655440001',
                    name: 'WebSDR Torino',
                    url: 'http://websdr-torino.example.com',
                    latitude: 45.0703,
                    longitude: 7.6869,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
                {
                    id: '550e8400-e29b-41d4-a716-446655440002',
                    name: 'WebSDR Genova',
                    url: 'http://websdr-genova.example.com',
                    latitude: 44.4056,
                    longitude: 8.9463,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, mockWebSDRs);

            const result = await getWebSDRs();

            expect(result).toEqual(mockWebSDRs);
            expect(result).toHaveLength(2);
            expect(result[0].name).toBe('WebSDR Torino');
        });

        it('should return empty array when no WebSDRs', async () => {
            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, []);

            const result = await getWebSDRs();

            expect(result).toEqual([]);
            expect(result).toHaveLength(0);
        });

        it('should handle 500 error', async () => {
            mock.onGet('/api/v1/acquisition/websdrs-all').reply(500, {
                detail: 'Internal server error',
            });

            await expect(getWebSDRs()).rejects.toThrow();
        });

        it('should handle network error', async () => {
            mock.onGet('/api/v1/acquisition/websdrs-all').networkError();

            await expect(getWebSDRs()).rejects.toThrow();
        });
    });

    describe('checkWebSDRHealth', () => {
        it('should check WebSDR health successfully', async () => {
            const mockHealth: Record<string, WebSDRHealthStatus> = {
                '550e8400-e29b-41d4-a716-446655440001': {
                    websdr_id: '550e8400-e29b-41d4-a716-446655440001',
                    name: 'WebSDR 1',
                    status: 'online',
                    response_time_ms: 150,
                    last_check: '2024-11-01T00:00:00Z',
                },
                '550e8400-e29b-41d4-a716-446655440002': {
                    websdr_id: '550e8400-e29b-41d4-a716-446655440002',
                    name: 'WebSDR 2',
                    status: 'online',
                    response_time_ms: 180,
                    last_check: '2024-11-01T00:00:00Z',
                },
                '550e8400-e29b-41d4-a716-446655440003': {
                    websdr_id: '550e8400-e29b-41d4-a716-446655440003',
                    name: 'WebSDR 3',
                    status: 'offline',
                    last_check: '2024-11-01T00:00:00Z',
                },
            };

            mock.onGet('/api/v1/acquisition/websdrs/health').reply(200, mockHealth);

            const result = await checkWebSDRHealth();

            expect(result).toEqual(mockHealth);
            expect(result['550e8400-e29b-41d4-a716-446655440001'].status).toBe('online');
            expect(result['550e8400-e29b-41d4-a716-446655440003'].status).toBe('offline');
        });

        it('should return empty object when no health data', async () => {
            mock.onGet('/api/v1/acquisition/websdrs/health').reply(200, {});

            const result = await checkWebSDRHealth();

            expect(result).toEqual({});
        });

        it('should handle 503 service unavailable', async () => {
            mock.onGet('/api/v1/acquisition/websdrs/health').reply(503, {
                detail: 'Service temporarily unavailable',
            });

            await expect(checkWebSDRHealth()).rejects.toThrow();
        });
    });

    describe('getWebSDRConfig', () => {
        const mockWebSDRs: WebSDRConfig[] = [
            {
                id: '550e8400-e29b-41d4-a716-446655440003',
                name: 'WebSDR 1',
                url: 'http://websdr1.example.com',
                latitude: 45.0,
                longitude: 7.0,
                is_active: true,
                timeout_seconds: 30,
                retry_count: 3,
            },
            {
                id: '550e8400-e29b-41d4-a716-446655440004',
                name: 'WebSDR 2',
                url: 'http://websdr2.example.com',
                latitude: 44.0,
                longitude: 8.0,
                is_active: false,
                timeout_seconds: 30,
                retry_count: 3,
            },
        ];

        beforeEach(() => {
            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, mockWebSDRs);
        });

        it('should get specific WebSDR config by id', async () => {
            const result = await getWebSDRConfig('550e8400-e29b-41d4-a716-446655440003');

            expect(result).toEqual(mockWebSDRs[0]);
            expect(result.id).toBe('550e8400-e29b-41d4-a716-446655440003');
            expect(result.name).toBe('WebSDR 1');
        });

        it('should get second WebSDR config', async () => {
            const result = await getWebSDRConfig('550e8400-e29b-41d4-a716-446655440004');

            expect(result).toEqual(mockWebSDRs[1]);
            expect(result.id).toBe('550e8400-e29b-41d4-a716-446655440004');
        });

        it('should throw error when WebSDR not found', async () => {
            await expect(getWebSDRConfig('550e8400-e29b-41d4-a716-446655449999')).rejects.toThrow('WebSDR with id 550e8400-e29b-41d4-a716-446655449999 not found');
        });

        it('should throw error when id is invalid', async () => {
            await expect(getWebSDRConfig('invalid-uuid')).rejects.toThrow('WebSDR with id invalid-uuid not found');
        });
    });

    describe('getActiveWebSDRs', () => {
        it('should filter active WebSDRs only', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: '550e8400-e29b-41d4-a716-446655440005',
                    name: 'Active 1',
                    url: 'http://active1.example.com',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
                {
                    id: '550e8400-e29b-41d4-a716-446655440006',
                    name: 'Inactive 1',
                    url: 'http://inactive1.example.com',
                    latitude: 44.0,
                    longitude: 8.0,
                    is_active: false,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
                {
                    id: '550e8400-e29b-41d4-a716-446655440007',
                    name: 'Active 2',
                    url: 'http://active2.example.com',
                    latitude: 43.0,
                    longitude: 9.0,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, mockWebSDRs);

            const result = await getActiveWebSDRs();

            expect(result).toHaveLength(2);
            expect(result[0].name).toBe('Active 1');
            expect(result[1].name).toBe('Active 2');
            expect(result.every(w => w.is_active)).toBe(true);
        });

        it('should return empty array when no active WebSDRs', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: '550e8400-e29b-41d4-a716-446655440008',
                    name: 'Inactive 1',
                    url: 'http://inactive1.example.com',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: false,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, mockWebSDRs);

            const result = await getActiveWebSDRs();

            expect(result).toEqual([]);
        });

        it('should return all WebSDRs when all are active', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: '550e8400-e29b-41d4-a716-446655440009',
                    name: 'Active 1',
                    url: 'http://active1.example.com',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
                {
                    id: '550e8400-e29b-41d4-a716-446655440010',
                    name: 'Active 2',
                    url: 'http://active2.example.com',
                    latitude: 44.0,
                    longitude: 8.0,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, mockWebSDRs);

            const result = await getActiveWebSDRs();

            expect(result).toHaveLength(2);
            expect(result).toEqual(mockWebSDRs);
        });
    });

    describe('Edge Cases', () => {
        it('should handle concurrent WebSDR requests', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: '550e8400-e29b-41d4-a716-446655440011',
                    name: 'WebSDR 1',
                    url: 'http://websdr1.example.com',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: true,
                    timeout_seconds: 30,
                    retry_count: 3,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs-all').reply(200, mockWebSDRs);

            const [result1, result2, result3] = await Promise.all([
                getWebSDRs(),
                getActiveWebSDRs(),
                getWebSDRConfig('550e8400-e29b-41d4-a716-446655440011'),
            ]);

            expect(result1).toHaveLength(1);
            expect(result2).toHaveLength(1);
            expect(result3.id).toBe('550e8400-e29b-41d4-a716-446655440011');
        });

        it('should handle timeout error', async () => {
            mock.onGet('/api/v1/acquisition/websdrs-all').timeout();

            await expect(getWebSDRs()).rejects.toThrow();
        });
    });
});
