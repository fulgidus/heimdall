/**
 * Acquisition Store
 * 
 * Manages RF acquisition sessions and tasks
 */

import { create } from 'zustand';
import type { 
    AcquisitionRequest,
    AcquisitionTaskResponse,
    AcquisitionStatusResponse,
} from '@/services/api/types';
import { acquisitionService } from '@/services/api';

interface ActiveTask {
    taskId: string;
    status: AcquisitionStatusResponse;
    startedAt: Date;
}

interface AcquisitionStore {
    activeTasks: Map<string, ActiveTask>;
    recentAcquisitions: AcquisitionStatusResponse[];
    isLoading: boolean;
    error: string | null;
    
    startAcquisition: (request: AcquisitionRequest) => Promise<AcquisitionTaskResponse>;
    getTaskStatus: (taskId: string) => Promise<AcquisitionStatusResponse>;
    pollTask: (
        taskId: string, 
        onProgress?: (status: AcquisitionStatusResponse) => void
    ) => Promise<AcquisitionStatusResponse>;
    
    addActiveTask: (taskId: string, status: AcquisitionStatusResponse) => void;
    updateTaskStatus: (taskId: string, status: AcquisitionStatusResponse) => void;
    removeActiveTask: (taskId: string) => void;
    
    clearError: () => void;
}

export const useAcquisitionStore = create<AcquisitionStore>((set, get) => ({
    activeTasks: new Map(),
    recentAcquisitions: [],
    isLoading: false,
    error: null,

    startAcquisition: async (request: AcquisitionRequest) => {
        set({ isLoading: true, error: null });
        try {
            const response = await acquisitionService.triggerAcquisition(request);
            
            // Add to active tasks
            const status: AcquisitionStatusResponse = {
                task_id: response.task_id,
                status: 'PENDING',
                progress: 0,
                message: response.message,
                measurements_collected: 0,
            };
            
            get().addActiveTask(response.task_id, status);
            
            set({ isLoading: false });
            return response;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to start acquisition';
            set({ error: errorMessage, isLoading: false });
            throw error;
        }
    },

    getTaskStatus: async (taskId: string) => {
        try {
            const status = await acquisitionService.getAcquisitionStatus(taskId);
            get().updateTaskStatus(taskId, status);
            return status;
        } catch (error) {
            console.error('Failed to get task status:', error);
            throw error;
        }
    },

    pollTask: async (
        taskId: string,
        onProgress?: (status: AcquisitionStatusResponse) => void
    ) => {
        const finalStatus = await acquisitionService.pollAcquisitionStatus(
            taskId,
            (status) => {
                get().updateTaskStatus(taskId, status);
                onProgress?.(status);
            }
        );
        
        // Move to recent acquisitions when complete
        set((state) => ({
            recentAcquisitions: [finalStatus, ...state.recentAcquisitions].slice(0, 10),
        }));
        
        get().removeActiveTask(taskId);
        
        return finalStatus;
    },

    addActiveTask: (taskId: string, status: AcquisitionStatusResponse) => {
        set((state) => {
            const newMap = new Map(state.activeTasks);
            newMap.set(taskId, {
                taskId,
                status,
                startedAt: new Date(),
            });
            return { activeTasks: newMap };
        });
    },

    updateTaskStatus: (taskId: string, status: AcquisitionStatusResponse) => {
        set((state) => {
            const task = state.activeTasks.get(taskId);
            if (!task) return state;
            
            const newMap = new Map(state.activeTasks);
            newMap.set(taskId, {
                ...task,
                status,
            });
            return { activeTasks: newMap };
        });
    },

    removeActiveTask: (taskId: string) => {
        set((state) => {
            const newMap = new Map(state.activeTasks);
            newMap.delete(taskId);
            return { activeTasks: newMap };
        });
    },

    clearError: () => set({ error: null }),
}));
