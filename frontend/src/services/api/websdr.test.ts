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

            mock.onGet('/api/v1/acquisition/websdrs').reply(200, mockWebSDRs);

            const result = await getWebSDRs();

            expect(result).toEqual(mockWebSDRs);
            expect(result).toHaveLength(2);
            expect(result[0].name).toBe('WebSDR Torino');
        });

        it('should return empty array when no WebSDRs', async () => {
            mock.onGet('/api/v1/acquisition/websdrs').reply(200, []);

            const result = await getWebSDRs();

            expect(result).toEqual([]);
            expect(result).toHaveLength(0);
        });

        it('should handle 500 error', async () => {
            mock.onGet('/api/v1/acquisition/websdrs').reply(500, {
                detail: 'Internal server error',
            });

            await expect(getWebSDRs()).rejects.toThrow();
        });

        it('should handle network error', async () => {
            mock.onGet('/api/v1/acquisition/websdrs').networkError();

            await expect(getWebSDRs()).rejects.toThrow();
        });
    });

    describe('checkWebSDRHealth', () => {
        it('should check WebSDR health successfully', async () => {
            const mockHealth: Record<number, WebSDRHealthStatus> = {
                1: {
                    status: 'online',
                    response_time_ms: 150,
                },
                2: {
                    status: 'online',
                    response_time_ms: 180,
                },
                3: {
                    status: 'offline',
                    response_time_ms: null,
                },
            };

            mock.onGet('/api/v1/acquisition/websdrs/health').reply(200, mockHealth);

            const result = await checkWebSDRHealth();

            expect(result).toEqual(mockHealth);
            expect(result[1].status).toBe('online');
            expect(result[3].status).toBe('offline');
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
                id: 1,
                name: 'WebSDR 1',
                url: 'http://websdr1.example.com',
                location_name: 'Location 1',
                latitude: 45.0,
                longitude: 7.0,
                is_active: true,
                frequency_min_mhz: 140,
                frequency_max_mhz: 450,
            },
            {
                id: 2,
                name: 'WebSDR 2',
                url: 'http://websdr2.example.com',
                location_name: 'Location 2',
                latitude: 44.0,
                longitude: 8.0,
                is_active: false,
                frequency_min_mhz: 140,
                frequency_max_mhz: 450,
            },
        ];

        beforeEach(() => {
            mock.onGet('/api/v1/acquisition/websdrs').reply(200, mockWebSDRs);
        });

        it('should get specific WebSDR config by id', async () => {
            const result = await getWebSDRConfig(1);

            expect(result).toEqual(mockWebSDRs[0]);
            expect(result.id).toBe(1);
            expect(result.name).toBe('WebSDR 1');
        });

        it('should get second WebSDR config', async () => {
            const result = await getWebSDRConfig(2);

            expect(result).toEqual(mockWebSDRs[1]);
            expect(result.id).toBe(2);
        });

        it('should throw error when WebSDR not found', async () => {
            await expect(getWebSDRConfig(999)).rejects.toThrow('WebSDR with id 999 not found');
        });

        it('should throw error when id is 0', async () => {
            await expect(getWebSDRConfig(0)).rejects.toThrow('WebSDR with id 0 not found');
        });
    });

    describe('getActiveWebSDRs', () => {
        it('should filter active WebSDRs only', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: 1,
                    name: 'Active 1',
                    url: 'http://active1.example.com',
                    location_name: 'Location 1',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
                {
                    id: 2,
                    name: 'Inactive 1',
                    url: 'http://inactive1.example.com',
                    location_name: 'Location 2',
                    latitude: 44.0,
                    longitude: 8.0,
                    is_active: false,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
                {
                    id: 3,
                    name: 'Active 2',
                    url: 'http://active2.example.com',
                    location_name: 'Location 3',
                    latitude: 43.0,
                    longitude: 9.0,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs').reply(200, mockWebSDRs);

            const result = await getActiveWebSDRs();

            expect(result).toHaveLength(2);
            expect(result[0].name).toBe('Active 1');
            expect(result[1].name).toBe('Active 2');
            expect(result.every(w => w.is_active)).toBe(true);
        });

        it('should return empty array when no active WebSDRs', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: 1,
                    name: 'Inactive 1',
                    url: 'http://inactive1.example.com',
                    location_name: 'Location 1',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: false,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs').reply(200, mockWebSDRs);

            const result = await getActiveWebSDRs();

            expect(result).toEqual([]);
        });

        it('should return all WebSDRs when all are active', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: 1,
                    name: 'Active 1',
                    url: 'http://active1.example.com',
                    location_name: 'Location 1',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
                {
                    id: 2,
                    name: 'Active 2',
                    url: 'http://active2.example.com',
                    location_name: 'Location 2',
                    latitude: 44.0,
                    longitude: 8.0,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs').reply(200, mockWebSDRs);

            const result = await getActiveWebSDRs();

            expect(result).toHaveLength(2);
            expect(result).toEqual(mockWebSDRs);
        });
    });

    describe('Edge Cases', () => {
        it('should handle concurrent WebSDR requests', async () => {
            const mockWebSDRs: WebSDRConfig[] = [
                {
                    id: 1,
                    name: 'WebSDR 1',
                    url: 'http://websdr1.example.com',
                    location_name: 'Location 1',
                    latitude: 45.0,
                    longitude: 7.0,
                    is_active: true,
                    frequency_min_mhz: 140,
                    frequency_max_mhz: 450,
                },
            ];

            mock.onGet('/api/v1/acquisition/websdrs').reply(200, mockWebSDRs);

            const [result1, result2, result3] = await Promise.all([
                getWebSDRs(),
                getActiveWebSDRs(),
                getWebSDRConfig(1),
            ]);

            expect(result1).toHaveLength(1);
            expect(result2).toHaveLength(1);
            expect(result3.id).toBe(1);
        });

        it('should handle timeout error', async () => {
            mock.onGet('/api/v1/acquisition/websdrs').timeout();

            await expect(getWebSDRs()).rejects.toThrow();
        });
    });
});
