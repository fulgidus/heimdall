/**
 * Acquisition API Service Tests
 *
 * Comprehensive test suite for acquisition API client
 * Tests HTTP requests, responses, error handling, and polling logic
 * Truth-first approach: Tests real API client with mocked HTTP responses
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import MockAdapter from 'axios-mock-adapter';
import api from '@/lib/api';
import {
    triggerAcquisition,
    getAcquisitionStatus,
    pollAcquisitionStatus,
} from './acquisition';
import type {
    AcquisitionRequest,
    AcquisitionTaskResponse,
    AcquisitionStatusResponse,
} from './types';

// Mock the auth store
vi.mock('@/store', () => ({
    useAuthStore: {
        getState: vi.fn(() => ({ token: null })),
    },
}));

// Create axios mock adapter
let mock: MockAdapter;

describe('Acquisition API Service', () => {
    beforeEach(() => {
        mock = new MockAdapter(api);
    });

    afterEach(() => {
        mock.reset();
        mock.restore();
        vi.clearAllTimers();
        vi.useRealTimers();
    });

    describe('triggerAcquisition', () => {
        it('should trigger acquisition successfully', async () => {
            const request: AcquisitionRequest = {
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            };

            const expectedResponse: AcquisitionTaskResponse = {
                task_id: 'task-abc123',
                message: 'Acquisition task started successfully',
            };

            mock.onPost('/api/v1/acquisition/acquire', request).reply(200, expectedResponse);

            const result = await triggerAcquisition(request);

            expect(result).toEqual(expectedResponse);
            expect(result.task_id).toBe('task-abc123');
            expect(result.message).toBe('Acquisition task started successfully');
        });

        it('should handle POST request with all fields', async () => {
            const request: AcquisitionRequest = {
                known_source_id: 2,
                frequency_mhz: 435.0,
                duration_seconds: 120,
            };

            const expectedResponse: AcquisitionTaskResponse = {
                task_id: 'task-xyz789',
                message: 'Started',
            };

            mock.onPost('/api/v1/acquisition/acquire').reply((config) => {
                const data = JSON.parse(config.data);
                expect(data.known_source_id).toBe(2);
                expect(data.frequency_mhz).toBe(435.0);
                expect(data.duration_seconds).toBe(120);
                return [200, expectedResponse];
            });

            const result = await triggerAcquisition(request);
            expect(result.task_id).toBe('task-xyz789');
        });

        it('should handle 400 bad request error', async () => {
            const request: AcquisitionRequest = {
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            };

            mock.onPost('/api/v1/acquisition/acquire').reply(400, {
                detail: 'Invalid frequency range',
            });

            await expect(triggerAcquisition(request)).rejects.toThrow();
        });

        it('should handle 500 server error', async () => {
            const request: AcquisitionRequest = {
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            };

            mock.onPost('/api/v1/acquisition/acquire').reply(500, {
                detail: 'Internal server error',
            });

            await expect(triggerAcquisition(request)).rejects.toThrow();
        });

        it('should handle network error', async () => {
            const request: AcquisitionRequest = {
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            };

            mock.onPost('/api/v1/acquisition/acquire').networkError();

            await expect(triggerAcquisition(request)).rejects.toThrow();
        });

        it('should handle timeout error', async () => {
            const request: AcquisitionRequest = {
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            };

            mock.onPost('/api/v1/acquisition/acquire').timeout();

            await expect(triggerAcquisition(request)).rejects.toThrow();
        });
    });

    describe('getAcquisitionStatus', () => {
        it('should get acquisition status successfully', async () => {
            const taskId = 'task-abc123';
            const expectedStatus: AcquisitionStatusResponse = {
                task_id: taskId,
                status: 'PROGRESS',
                progress: 50,
                message: 'Processing WebSDR data',
                measurements_collected: 3,
            };

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(200, expectedStatus);

            const result = await getAcquisitionStatus(taskId);

            expect(result).toEqual(expectedStatus);
            expect(result.task_id).toBe(taskId);
            expect(result.status).toBe('PROGRESS');
            expect(result.progress).toBe(50);
        });

        it('should get pending status', async () => {
            const taskId = 'task-pending';
            const expectedStatus: AcquisitionStatusResponse = {
                task_id: taskId,
                status: 'PENDING',
                progress: 0,
                message: 'Task queued',
                measurements_collected: 0,
            };

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(200, expectedStatus);

            const result = await getAcquisitionStatus(taskId);
            expect(result.status).toBe('PENDING');
            expect(result.progress).toBe(0);
        });

        it('should get success status', async () => {
            const taskId = 'task-success';
            const expectedStatus: AcquisitionStatusResponse = {
                task_id: taskId,
                status: 'SUCCESS',
                progress: 100,
                message: 'Acquisition completed successfully',
                measurements_collected: 7,
            };

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(200, expectedStatus);

            const result = await getAcquisitionStatus(taskId);
            expect(result.status).toBe('SUCCESS');
            expect(result.progress).toBe(100);
            expect(result.measurements_collected).toBe(7);
        });

        it('should get failure status', async () => {
            const taskId = 'task-failed';
            const expectedStatus: AcquisitionStatusResponse = {
                task_id: taskId,
                status: 'FAILURE',
                progress: 30,
                message: 'WebSDR connection failed',
                measurements_collected: 2,
            };

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(200, expectedStatus);

            const result = await getAcquisitionStatus(taskId);
            expect(result.status).toBe('FAILURE');
            expect(result.message).toContain('failed');
        });

        it('should handle 404 task not found', async () => {
            const taskId = 'task-notfound';

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(404, {
                detail: 'Task not found',
            });

            await expect(getAcquisitionStatus(taskId)).rejects.toThrow();
        });

        it('should handle network error', async () => {
            const taskId = 'task-network-error';

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).networkError();

            await expect(getAcquisitionStatus(taskId)).rejects.toThrow();
        });
    });

    describe('pollAcquisitionStatus', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        it('should poll until success status', async () => {
            const taskId = 'task-poll-success';
            let callCount = 0;

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(() => {
                callCount++;
                if (callCount === 1) {
                    return [200, {
                        task_id: taskId,
                        status: 'PENDING',
                        progress: 0,
                        message: 'Starting',
                        measurements_collected: 0,
                    }];
                } else if (callCount === 2) {
                    return [200, {
                        task_id: taskId,
                        status: 'PROGRESS',
                        progress: 50,
                        message: 'Processing',
                        measurements_collected: 3,
                    }];
                } else {
                    return [200, {
                        task_id: taskId,
                        status: 'SUCCESS',
                        progress: 100,
                        message: 'Completed',
                        measurements_collected: 7,
                    }];
                }
            });

            const pollPromise = pollAcquisitionStatus(taskId);

            // First poll
            await vi.runOnlyPendingTimersAsync();
            
            // Second poll
            await vi.advanceTimersByTimeAsync(2000);
            
            // Third poll
            await vi.advanceTimersByTimeAsync(2000);

            const result = await pollPromise;

            expect(result.status).toBe('SUCCESS');
            expect(result.progress).toBe(100);
            expect(callCount).toBe(3);
        });

        it('should poll until failure status', async () => {
            const taskId = 'task-poll-fail';
            let callCount = 0;

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(() => {
                callCount++;
                if (callCount === 1) {
                    return [200, {
                        task_id: taskId,
                        status: 'PROGRESS',
                        progress: 30,
                        message: 'Processing',
                        measurements_collected: 2,
                    }];
                } else {
                    return [200, {
                        task_id: taskId,
                        status: 'FAILURE',
                        progress: 30,
                        message: 'Connection failed',
                        measurements_collected: 2,
                    }];
                }
            });

            const pollPromise = pollAcquisitionStatus(taskId);

            await vi.runOnlyPendingTimersAsync();
            await vi.advanceTimersByTimeAsync(2000);

            const result = await pollPromise;

            expect(result.status).toBe('FAILURE');
            expect(callCount).toBe(2);
        });

        it('should poll until revoked status', async () => {
            const taskId = 'task-poll-revoked';
            let callCount = 0;

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(() => {
                callCount++;
                if (callCount === 1) {
                    return [200, {
                        task_id: taskId,
                        status: 'PROGRESS',
                        progress: 10,
                        message: 'Starting',
                        measurements_collected: 0,
                    }];
                } else {
                    return [200, {
                        task_id: taskId,
                        status: 'REVOKED',
                        progress: 10,
                        message: 'Task cancelled',
                        measurements_collected: 0,
                    }];
                }
            });

            const pollPromise = pollAcquisitionStatus(taskId);

            await vi.runOnlyPendingTimersAsync();
            await vi.advanceTimersByTimeAsync(2000);

            const result = await pollPromise;

            expect(result.status).toBe('REVOKED');
        });

        it('should call progress callback on each poll', async () => {
            const taskId = 'task-progress-callback';
            const progressCallback = vi.fn();
            let callCount = 0;

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(() => {
                callCount++;
                if (callCount === 1) {
                    return [200, {
                        task_id: taskId,
                        status: 'PROGRESS',
                        progress: 25,
                        message: 'Progress 1',
                        measurements_collected: 1,
                    }];
                } else if (callCount === 2) {
                    return [200, {
                        task_id: taskId,
                        status: 'PROGRESS',
                        progress: 75,
                        message: 'Progress 2',
                        measurements_collected: 5,
                    }];
                } else {
                    return [200, {
                        task_id: taskId,
                        status: 'SUCCESS',
                        progress: 100,
                        message: 'Complete',
                        measurements_collected: 7,
                    }];
                }
            });

            const pollPromise = pollAcquisitionStatus(taskId, progressCallback);

            await vi.runOnlyPendingTimersAsync();
            await vi.advanceTimersByTimeAsync(2000);
            await vi.advanceTimersByTimeAsync(2000);

            await pollPromise;

            expect(progressCallback).toHaveBeenCalledTimes(3);
            expect(progressCallback).toHaveBeenNthCalledWith(1, expect.objectContaining({ progress: 25 }));
            expect(progressCallback).toHaveBeenNthCalledWith(2, expect.objectContaining({ progress: 75 }));
            expect(progressCallback).toHaveBeenNthCalledWith(3, expect.objectContaining({ progress: 100 }));
        });

        it('should use custom poll interval', async () => {
            const taskId = 'task-custom-interval';
            const customInterval = 5000;
            let callCount = 0;

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(() => {
                callCount++;
                if (callCount === 1) {
                    return [200, {
                        task_id: taskId,
                        status: 'PROGRESS',
                        progress: 50,
                        message: 'Processing',
                        measurements_collected: 3,
                    }];
                } else {
                    return [200, {
                        task_id: taskId,
                        status: 'SUCCESS',
                        progress: 100,
                        message: 'Done',
                        measurements_collected: 7,
                    }];
                }
            });

            const pollPromise = pollAcquisitionStatus(taskId, undefined, customInterval);

            await vi.runOnlyPendingTimersAsync();
            await vi.advanceTimersByTimeAsync(customInterval);

            await pollPromise;

            expect(callCount).toBe(2);
        });

        it('should reject on polling error', async () => {
            const taskId = 'task-poll-error';

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).reply(500, {
                detail: 'Server error during polling',
            });

            await expect(pollAcquisitionStatus(taskId)).rejects.toThrow('Request failed');
        });

        it('should reject on network error during polling', async () => {
            const taskId = 'task-poll-network-error';

            mock.onGet(`/api/v1/acquisition/status/${taskId}`).networkError();

            await expect(pollAcquisitionStatus(taskId)).rejects.toThrow('Network Error');
        });
    });
});
