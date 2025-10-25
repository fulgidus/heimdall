/**
 * Acquisition Store Tests
 *
 * Comprehensive test suite for the acquisitionStore Zustand store
 * Tests all actions: acquisition management, task polling, status tracking
 * Truth-first approach: Tests real Zustand store behavior with mocked API responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/acquisitionStore');

// Import after unmocking
import { useAcquisitionStore } from './acquisitionStore';
import { acquisitionService } from '@/services/api';

// Mock the API services
vi.mock('@/services/api', () => ({
    acquisitionService: {
        triggerAcquisition: vi.fn(),
        getAcquisitionStatus: vi.fn(),
        pollAcquisitionStatus: vi.fn(),
    },
    webSDRService: {},
    inferenceService: {},
    systemService: {},
    analyticsService: {},
    sessionService: {},
}));

describe('Acquisition Store (Zustand)', () => {
    beforeEach(() => {
        // Reset store to initial state before each test
        useAcquisitionStore.setState({
            activeTasks: new Map(),
            recentAcquisitions: [],
            isLoading: false,
            error: null,
        });
        vi.clearAllMocks();
    });

    describe('Store Initialization', () => {
        it('should initialize with default state', () => {
            const state = useAcquisitionStore.getState();
            expect(state.activeTasks).toBeInstanceOf(Map);
            expect(state.activeTasks.size).toBe(0);
            expect(state.recentAcquisitions).toEqual([]);
            expect(state.isLoading).toBe(false);
            expect(state.error).toBe(null);
        });

        it('should have all required actions', () => {
            const state = useAcquisitionStore.getState();
            expect(typeof state.startAcquisition).toBe('function');
            expect(typeof state.getTaskStatus).toBe('function');
            expect(typeof state.pollTask).toBe('function');
            expect(typeof state.addActiveTask).toBe('function');
            expect(typeof state.updateTaskStatus).toBe('function');
            expect(typeof state.removeActiveTask).toBe('function');
            expect(typeof state.clearError).toBe('function');
        });
    });

    describe('startAcquisition Action', () => {
        it('should start acquisition successfully', async () => {
            const mockRequest = {
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            };

            const mockResponse = {
                task_id: 'task-123',
                message: 'Acquisition started',
            };

            vi.mocked(acquisitionService.triggerAcquisition).mockResolvedValue(mockResponse);

            const result = await useAcquisitionStore.getState().startAcquisition(mockRequest);

            expect(result).toEqual(mockResponse);
            expect(acquisitionService.triggerAcquisition).toHaveBeenCalledWith(mockRequest);

            const state = useAcquisitionStore.getState();
            expect(state.isLoading).toBe(false);
            expect(state.error).toBe(null);
            expect(state.activeTasks.has('task-123')).toBe(true);
        });

        it('should set loading state during acquisition start', async () => {
            vi.mocked(acquisitionService.triggerAcquisition).mockImplementation(
                () => new Promise((resolve) => setTimeout(() => resolve({
                    task_id: 'task-123',
                    message: 'Started',
                }), 50))
            );

            const promise = useAcquisitionStore.getState().startAcquisition({
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            });
            
            // Should be loading immediately
            expect(useAcquisitionStore.getState().isLoading).toBe(true);

            await promise;

            // Should not be loading after completion
            expect(useAcquisitionStore.getState().isLoading).toBe(false);
        });

        it('should handle start acquisition error', async () => {
            const errorMessage = 'Failed to start acquisition';
            vi.mocked(acquisitionService.triggerAcquisition).mockRejectedValue(new Error(errorMessage));

            await expect(
                useAcquisitionStore.getState().startAcquisition({
                    known_source_id: 1,
                    frequency_mhz: 145.5,
                    duration_seconds: 60,
                })
            ).rejects.toThrow(errorMessage);

            const state = useAcquisitionStore.getState();
            expect(state.error).toBe(errorMessage);
            expect(state.isLoading).toBe(false);
        });

        it('should add task to active tasks with correct initial status', async () => {
            const mockResponse = {
                task_id: 'task-456',
                message: 'Acquisition started',
            };

            vi.mocked(acquisitionService.triggerAcquisition).mockResolvedValue(mockResponse);

            await useAcquisitionStore.getState().startAcquisition({
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            });

            const state = useAcquisitionStore.getState();
            const activeTask = state.activeTasks.get('task-456');
            
            expect(activeTask).toBeDefined();
            expect(activeTask?.taskId).toBe('task-456');
            expect(activeTask?.status.status).toBe('PENDING');
            expect(activeTask?.status.progress).toBe(0);
            expect(activeTask?.status.measurements_collected).toBe(0);
            expect(activeTask?.startedAt).toBeInstanceOf(Date);
        });
    });

    describe('getTaskStatus Action', () => {
        it('should get task status successfully', async () => {
            const mockStatus = {
                task_id: 'task-123',
                status: 'IN_PROGRESS',
                progress: 50,
                message: 'Processing...',
                measurements_collected: 3,
            };

            vi.mocked(acquisitionService.getAcquisitionStatus).mockResolvedValue(mockStatus);

            const result = await useAcquisitionStore.getState().getTaskStatus('task-123');

            expect(result).toEqual(mockStatus);
            expect(acquisitionService.getAcquisitionStatus).toHaveBeenCalledWith('task-123');
        });

        it('should update task status in store', async () => {
            // Add a task first
            useAcquisitionStore.getState().addActiveTask('task-123', {
                task_id: 'task-123',
                status: 'PENDING',
                progress: 0,
                message: 'Starting...',
                measurements_collected: 0,
            });

            const mockStatus = {
                task_id: 'task-123',
                status: 'IN_PROGRESS',
                progress: 50,
                message: 'Processing...',
                measurements_collected: 3,
            };

            vi.mocked(acquisitionService.getAcquisitionStatus).mockResolvedValue(mockStatus);

            await useAcquisitionStore.getState().getTaskStatus('task-123');

            const state = useAcquisitionStore.getState();
            const task = state.activeTasks.get('task-123');
            expect(task?.status).toEqual(mockStatus);
        });

        it('should handle get status error', async () => {
            const errorMessage = 'Task not found';
            vi.mocked(acquisitionService.getAcquisitionStatus).mockRejectedValue(new Error(errorMessage));

            await expect(
                useAcquisitionStore.getState().getTaskStatus('task-999')
            ).rejects.toThrow(errorMessage);
        });
    });

    describe('pollTask Action', () => {
        it('should poll task and move to recent acquisitions on completion', async () => {
            const finalStatus = {
                task_id: 'task-123',
                status: 'COMPLETED',
                progress: 100,
                message: 'Completed successfully',
                measurements_collected: 7,
            };

            vi.mocked(acquisitionService.pollAcquisitionStatus).mockResolvedValue(finalStatus);

            // Add task to active tasks first
            useAcquisitionStore.getState().addActiveTask('task-123', {
                task_id: 'task-123',
                status: 'PENDING',
                progress: 0,
                message: 'Starting...',
                measurements_collected: 0,
            });

            const result = await useAcquisitionStore.getState().pollTask('task-123');

            expect(result).toEqual(finalStatus);

            const state = useAcquisitionStore.getState();
            // Task should be removed from active tasks
            expect(state.activeTasks.has('task-123')).toBe(false);
            // Task should be in recent acquisitions
            expect(state.recentAcquisitions).toHaveLength(1);
            expect(state.recentAcquisitions[0]).toEqual(finalStatus);
        });

        it('should call progress callback during polling', async () => {
            const progressCallback = vi.fn();
            const finalStatus = {
                task_id: 'task-123',
                status: 'COMPLETED',
                progress: 100,
                message: 'Done',
                measurements_collected: 7,
            };

            vi.mocked(acquisitionService.pollAcquisitionStatus).mockImplementation(
                async (taskId, onProgress) => {
                    // Simulate progress updates
                    onProgress?.({
                        task_id: taskId,
                        status: 'IN_PROGRESS',
                        progress: 25,
                        message: 'Progress...',
                        measurements_collected: 2,
                    });
                    onProgress?.({
                        task_id: taskId,
                        status: 'IN_PROGRESS',
                        progress: 75,
                        message: 'Almost done...',
                        measurements_collected: 5,
                    });
                    return finalStatus;
                }
            );

            useAcquisitionStore.getState().addActiveTask('task-123', {
                task_id: 'task-123',
                status: 'PENDING',
                progress: 0,
                message: 'Starting...',
                measurements_collected: 0,
            });

            await useAcquisitionStore.getState().pollTask('task-123', progressCallback);

            // Progress callback should be called for each update
            expect(progressCallback).toHaveBeenCalledTimes(2);
        });

        it('should limit recent acquisitions to 10', async () => {
            // Add 10 recent acquisitions
            useAcquisitionStore.setState({
                recentAcquisitions: Array.from({ length: 10 }, (_, i) => ({
                    task_id: `task-old-${i}`,
                    status: 'COMPLETED',
                    progress: 100,
                    message: 'Completed',
                    measurements_collected: 7,
                })),
            });

            const newFinalStatus = {
                task_id: 'task-new',
                status: 'COMPLETED',
                progress: 100,
                message: 'Completed',
                measurements_collected: 7,
            };

            vi.mocked(acquisitionService.pollAcquisitionStatus).mockResolvedValue(newFinalStatus);

            useAcquisitionStore.getState().addActiveTask('task-new', {
                task_id: 'task-new',
                status: 'PENDING',
                progress: 0,
                message: 'Starting...',
                measurements_collected: 0,
            });

            await useAcquisitionStore.getState().pollTask('task-new');

            const state = useAcquisitionStore.getState();
            // Should still have only 10 items
            expect(state.recentAcquisitions).toHaveLength(10);
            // New task should be first
            expect(state.recentAcquisitions[0].task_id).toBe('task-new');
            // Oldest task should be removed
            expect(state.recentAcquisitions.some(a => a.task_id === 'task-old-9')).toBe(false);
        });
    });

    describe('Task Management Actions', () => {
        describe('addActiveTask', () => {
            it('should add task to active tasks', () => {
                const status = {
                    task_id: 'task-123',
                    status: 'PENDING',
                    progress: 0,
                    message: 'Starting...',
                    measurements_collected: 0,
                };

                useAcquisitionStore.getState().addActiveTask('task-123', status);

                const state = useAcquisitionStore.getState();
                expect(state.activeTasks.size).toBe(1);
                expect(state.activeTasks.has('task-123')).toBe(true);
                
                const task = state.activeTasks.get('task-123');
                expect(task?.taskId).toBe('task-123');
                expect(task?.status).toEqual(status);
                expect(task?.startedAt).toBeInstanceOf(Date);
            });

            it('should handle multiple active tasks', () => {
                useAcquisitionStore.getState().addActiveTask('task-1', {
                    task_id: 'task-1',
                    status: 'PENDING',
                    progress: 0,
                    message: 'Task 1',
                    measurements_collected: 0,
                });

                useAcquisitionStore.getState().addActiveTask('task-2', {
                    task_id: 'task-2',
                    status: 'PENDING',
                    progress: 0,
                    message: 'Task 2',
                    measurements_collected: 0,
                });

                const state = useAcquisitionStore.getState();
                expect(state.activeTasks.size).toBe(2);
            });
        });

        describe('updateTaskStatus', () => {
            it('should update existing task status', () => {
                useAcquisitionStore.getState().addActiveTask('task-123', {
                    task_id: 'task-123',
                    status: 'PENDING',
                    progress: 0,
                    message: 'Starting...',
                    measurements_collected: 0,
                });

                const updatedStatus = {
                    task_id: 'task-123',
                    status: 'IN_PROGRESS',
                    progress: 50,
                    message: 'Processing...',
                    measurements_collected: 3,
                };

                useAcquisitionStore.getState().updateTaskStatus('task-123', updatedStatus);

                const state = useAcquisitionStore.getState();
                const task = state.activeTasks.get('task-123');
                expect(task?.status).toEqual(updatedStatus);
            });

            it('should not add task if it does not exist', () => {
                useAcquisitionStore.getState().updateTaskStatus('task-999', {
                    task_id: 'task-999',
                    status: 'IN_PROGRESS',
                    progress: 50,
                    message: 'Processing...',
                    measurements_collected: 3,
                });

                const state = useAcquisitionStore.getState();
                expect(state.activeTasks.has('task-999')).toBe(false);
            });
        });

        describe('removeActiveTask', () => {
            it('should remove task from active tasks', () => {
                useAcquisitionStore.getState().addActiveTask('task-123', {
                    task_id: 'task-123',
                    status: 'PENDING',
                    progress: 0,
                    message: 'Starting...',
                    measurements_collected: 0,
                });

                expect(useAcquisitionStore.getState().activeTasks.has('task-123')).toBe(true);

                useAcquisitionStore.getState().removeActiveTask('task-123');

                expect(useAcquisitionStore.getState().activeTasks.has('task-123')).toBe(false);
            });

            it('should handle removing non-existent task', () => {
                useAcquisitionStore.getState().removeActiveTask('task-999');
                // Should not throw
                expect(useAcquisitionStore.getState().activeTasks.size).toBe(0);
            });
        });
    });

    describe('clearError Action', () => {
        it('should clear error', () => {
            useAcquisitionStore.setState({ error: 'Some error' });
            
            useAcquisitionStore.getState().clearError();
            
            expect(useAcquisitionStore.getState().error).toBe(null);
        });
    });

    describe('Edge Cases', () => {
        it('should handle concurrent acquisitions', async () => {
            vi.mocked(acquisitionService.triggerAcquisition).mockImplementation(
                async (request) => ({
                    task_id: `task-${request.known_source_id}`,
                    message: 'Started',
                })
            );

            await Promise.all([
                useAcquisitionStore.getState().startAcquisition({
                    known_source_id: 1,
                    frequency_mhz: 145.5,
                    duration_seconds: 60,
                }),
                useAcquisitionStore.getState().startAcquisition({
                    known_source_id: 2,
                    frequency_mhz: 435.0,
                    duration_seconds: 60,
                }),
                useAcquisitionStore.getState().startAcquisition({
                    known_source_id: 3,
                    frequency_mhz: 145.8,
                    duration_seconds: 60,
                }),
            ]);

            const state = useAcquisitionStore.getState();
            expect(state.activeTasks.size).toBe(3);
        });

        it('should maintain state integrity when tasks complete', async () => {
            // Start multiple tasks
            vi.mocked(acquisitionService.triggerAcquisition).mockImplementation(
                async (request) => ({
                    task_id: `task-${request.known_source_id}`,
                    message: 'Started',
                })
            );

            await useAcquisitionStore.getState().startAcquisition({
                known_source_id: 1,
                frequency_mhz: 145.5,
                duration_seconds: 60,
            });

            await useAcquisitionStore.getState().startAcquisition({
                known_source_id: 2,
                frequency_mhz: 435.0,
                duration_seconds: 60,
            });

            expect(useAcquisitionStore.getState().activeTasks.size).toBe(2);

            // Complete one task
            vi.mocked(acquisitionService.pollAcquisitionStatus).mockResolvedValue({
                task_id: 'task-1',
                status: 'COMPLETED',
                progress: 100,
                message: 'Done',
                measurements_collected: 7,
            });

            await useAcquisitionStore.getState().pollTask('task-1');

            const state = useAcquisitionStore.getState();
            expect(state.activeTasks.size).toBe(1);
            expect(state.activeTasks.has('task-2')).toBe(true);
            expect(state.recentAcquisitions).toHaveLength(1);
        });

        it('should handle non-Error exceptions', async () => {
            vi.mocked(acquisitionService.triggerAcquisition).mockRejectedValue('String error');

            await expect(
                useAcquisitionStore.getState().startAcquisition({
                    known_source_id: 1,
                    frequency_mhz: 145.5,
                    duration_seconds: 60,
                })
            ).rejects.toThrow();

            const state = useAcquisitionStore.getState();
            expect(state.error).toBe('Failed to start acquisition');
        });
    });
});
