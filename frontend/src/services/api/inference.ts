/**
 * Inference API Service
 *
 * Handles ML inference operations:
 * - Get model information
 * - Get performance metrics
 * - Make predictions
 */

import { z } from 'zod';
import api from '@/lib/api';
import {
  ModelInfoSchema,
  ModelPerformanceMetricsSchema,
  LocalizationResultSchema,
  PredictionResponseSchema,
  BatchPredictionResponseSchema,
} from './schemas';
import type { ModelInfo, ModelPerformanceMetrics, LocalizationResult } from './schemas';

/**
 * Get information about the active model
 */
export async function getModelInfo(): Promise<ModelInfo> {
  const response = await api.get('/v1/analytics/model/info');

  // Validate response with Zod
  const validated = ModelInfoSchema.parse(response.data);
  return validated;
}

/**
 * Get model performance metrics
 */
export async function getModelPerformance(): Promise<ModelPerformanceMetrics> {
  const response = await api.get('/v1/analytics/model/performance');

  // Validate response with Zod
  const validated = ModelPerformanceMetricsSchema.parse(response.data);
  return validated;
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
  const response = await api.post('/v1/inference/predict', request);

  // Validate response with Zod
  const validated = PredictionResponseSchema.parse(response.data);
  return validated;
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
export async function predictLocalizationBatch(
  request: BatchPredictionRequest
): Promise<BatchPredictionResponse> {
  const response = await api.post('/v1/inference/predict/batch', request);

  // Validate response with Zod
  const validated = BatchPredictionResponseSchema.parse(response.data);
  return validated;
}

/**
 * Get recent localization results
 */
export async function getRecentLocalizations(limit: number = 10): Promise<LocalizationResult[]> {
  const response = await api.get('/v1/analytics/localizations/recent', {
    params: { limit },
  });

  // Validate response with Zod
  const validated = z.array(LocalizationResultSchema).parse(response.data);
  return validated;
}

const inferenceService = {
  getModelInfo,
  getModelPerformance,
  predictLocalization,
  predictLocalizationBatch,
  getRecentLocalizations,
};

export default inferenceService;
