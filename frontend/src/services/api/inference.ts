/**
 * Inference API Service
 * 
 * Handles ML inference operations:
 * - Get model information
 * - Get performance metrics
 * - Make predictions
 */

import api from '@/lib/api';
import type { ModelInfo, ModelPerformanceMetrics } from './types';

/**
 * Get information about the active model
 */
export async function getModelInfo(): Promise<ModelInfo> {
    const response = await api.get<ModelInfo>('/api/v1/inference/model/info');
    return response.data;
}

/**
 * Get model performance metrics
 */
export async function getModelPerformance(): Promise<ModelPerformanceMetrics> {
    const response = await api.get<ModelPerformanceMetrics>('/api/v1/inference/model/performance');
    return response.data;
}

const inferenceService = {
    getModelInfo,
    getModelPerformance,
};

export default inferenceService;
