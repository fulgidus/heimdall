/**
 * Inference API Service
 *
 * Handles ML inference operations:
 * - Get model information
 * - Get performance metrics
 * - Make predictions
 */

import api from '@/lib/api';
import type { ModelInfo, ModelPerformanceMetrics, LocalizationResult } from './types';

/**
 * Get information about the active model
 */
export async function getModelInfo(): Promise<ModelInfo> {
    const response = await api.get<ModelInfo>('/api/v1/analytics/model/info');
    return response.data;
}

/**
 * Get model performance metrics
 */
export async function getModelPerformance(): Promise<ModelPerformanceMetrics> {
    const response = await api.get<ModelPerformanceMetrics>('/api/v1/analytics/model/performance');
    return response.data;
}

/**
 * Prediction request interface
 */
export interface PredictionRequest {
    iq_data: number[][]; // IQ samples as [I, Q] pairs
    session_id?: string;
    cache_enabled?: boolean;
}

/**
 * Prediction response interface
 */
export interface PredictionResponse {
    position: {
        latitude: number;
        longitude: number;
    };
    uncertainty: {
        sigma_x: number;
        sigma_y: number;
        theta: number;
        confidence_interval: number;
    };
    confidence: number;
    model_version: string;
    inference_time_ms: number;
    timestamp: string;
    session_id: string;
    _cache_hit: boolean;
}

/**
 * Make a single localization prediction
 */
export async function predictLocalization(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await api.post<PredictionResponse>('/api/v1/inference/predict', request);
    return response.data;
}

/**
 * Batch prediction request interface
 */
export interface BatchPredictionRequest {
    predictions: PredictionRequest[];
}

/**
 * Batch prediction response interface
 */
export interface BatchPredictionResponse {
    predictions: PredictionResponse[];
    batch_id: string;
    total_time_ms: number;
}

/**
 * Make batch localization predictions
 */
export async function predictLocalizationBatch(request: BatchPredictionRequest): Promise<BatchPredictionResponse> {
    const response = await api.post<BatchPredictionResponse>('/api/v1/inference/predict/batch', request);
    return response.data;
}

/**
 * Get recent localization results
 */
export async function getRecentLocalizations(limit: number = 10): Promise<LocalizationResult[]> {
    const response = await api.get<LocalizationResult[]>('/api/v1/analytics/localizations/recent', {
        params: { limit }
    });
    return response.data;
}

const inferenceService = {
    getModelInfo,
    getModelPerformance,
    predictLocalization,
    predictLocalizationBatch,
    getRecentLocalizations,
};

export default inferenceService;
