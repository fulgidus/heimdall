/**
 * Tauri Bridge - Non-invasive utility for Tauri desktop integration
 * 
 * This file provides a unified API surface for Tauri IPC commands with graceful
 * fallback for web deployments. No modifications to core frontend code required.
 */

// Type definitions for Tauri API
export interface DataCollectionConfig {
    frequency: number;
    duration_seconds: number;
    websdrs: string[];
}

export interface DataCollectionStatus {
    is_running: boolean;
    progress: number;
    current_websdr: string | null;
    message: string;
}

export interface TrainingConfig {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    model_name: string;
}

export interface TrainingStatus {
    is_running: boolean;
    current_epoch: number;
    total_epochs: number;
    loss: number;
    accuracy: number;
    message: string;
}

export interface GpuInfo {
    available: boolean;
    name: string;
    driver_version: string;
    cuda_version: string;
    memory_total_mb: number;
    memory_free_mb: number;
    memory_used_mb: number;
    utilization_percent: number;
}

export interface AppSettings {
    api_url: string;
    websocket_url: string;
    mapbox_token: string;
    auto_start_backend: boolean;
    backend_port: number;
    enable_gpu: boolean;
    theme: string;
}

/**
 * Check if running in Tauri desktop environment
 */
export const isTauriEnvironment = (): boolean => {
    return typeof window !== 'undefined' && '__TAURI__' in window;
};

/**
 * Get Tauri API instance (only available in desktop mode)
 */
const getTauriAPI = () => {
    if (!isTauriEnvironment()) {
        return null;
    }
    return (window as any).__TAURI__;
};

/**
 * Invoke a Tauri command with error handling
 */
const invokeCommand = async <T>(command: string, args?: any): Promise<T> => {
    const tauri = getTauriAPI();
    
    if (!tauri) {
        throw new Error('Tauri environment not available. This feature only works in desktop mode.');
    }
    
    try {
        const result = await tauri.core.invoke(command, args);
        return result as T;
    } catch (error) {
        console.error(`Tauri command '${command}' failed:`, error);
        throw error;
    }
};

/**
 * Tauri API - Data Collection Commands
 */
export const tauriDataCollection = {
    /**
     * Start RF data collection from configured WebSDRs
     */
    start: async (config: DataCollectionConfig): Promise<string> => {
        return invokeCommand<string>('start_data_collection', { config });
    },

    /**
     * Stop ongoing data collection
     */
    stop: async (): Promise<string> => {
        return invokeCommand<string>('stop_data_collection');
    },

    /**
     * Get current data collection status
     */
    getStatus: async (): Promise<DataCollectionStatus> => {
        return invokeCommand<DataCollectionStatus>('get_collection_status');
    },
};

/**
 * Tauri API - Training Commands
 */
export const tauriTraining = {
    /**
     * Start ML model training
     */
    start: async (config: TrainingConfig): Promise<string> => {
        return invokeCommand<string>('start_training', { config });
    },

    /**
     * Stop ongoing training
     */
    stop: async (): Promise<string> => {
        return invokeCommand<string>('stop_training');
    },

    /**
     * Get current training status
     */
    getStatus: async (): Promise<TrainingStatus> => {
        return invokeCommand<TrainingStatus>('get_training_status');
    },
};

/**
 * Tauri API - GPU Commands
 */
export const tauriGPU = {
    /**
     * Check for available GPU and get information
     */
    check: async (): Promise<GpuInfo> => {
        return invokeCommand<GpuInfo>('check_gpu');
    },

    /**
     * Get current GPU usage statistics
     */
    getUsage: async (): Promise<GpuInfo> => {
        return invokeCommand<GpuInfo>('get_gpu_usage');
    },
};

/**
 * Tauri API - Settings Commands
 */
export const tauriSettings = {
    /**
     * Load application settings from file
     */
    load: async (): Promise<AppSettings> => {
        return invokeCommand<AppSettings>('load_settings');
    },

    /**
     * Save application settings to file
     */
    save: async (settings: AppSettings): Promise<string> => {
        return invokeCommand<string>('save_settings', { settings });
    },

    /**
     * Reset settings to defaults
     */
    reset: async (): Promise<AppSettings> => {
        return invokeCommand<AppSettings>('reset_settings');
    },
};

/**
 * Unified Tauri API object
 */
export const tauriAPI = {
    isTauri: isTauriEnvironment(),
    dataCollection: tauriDataCollection,
    training: tauriTraining,
    gpu: tauriGPU,
    settings: tauriSettings,
};

/**
 * Example usage in React components:
 * 
 * ```tsx
 * import { tauriAPI, isTauriEnvironment } from '@/lib/tauri-bridge';
 * 
 * function MyComponent() {
 *   const handleStartCollection = async () => {
 *     if (!isTauriEnvironment()) {
 *       console.log('Desktop feature not available in web mode');
 *       return;
 *     }
 *     
 *     try {
 *       const result = await tauriAPI.dataCollection.start({
 *         frequency: 145.500,
 *         duration_seconds: 60,
 *         websdrs: ['websdr1', 'websdr2']
 *       });
 *       console.log(result);
 *     } catch (error) {
 *       console.error('Failed to start collection:', error);
 *     }
 *   };
 *   
 *   return (
 *     <button onClick={handleStartCollection}>
 *       Start Collection {isTauriEnvironment() ? '(Desktop)' : '(Web)'}
 *     </button>
 *   );
 * }
 * ```
 */

export default tauriAPI;
