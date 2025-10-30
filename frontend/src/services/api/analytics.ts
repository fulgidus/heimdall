/**
 * Analytics API Service
 *
 * Handles analytics and metrics operations:
 * - Historical metrics
 * - Prediction trends
 * - Performance analytics
 */

import api from '@/lib/api';

/**
 * Time series data point
 */
export interface TimeSeriesPoint {
    timestamp: string;
    value: number;
}

/**
 * Prediction metrics over time
 */
export interface PredictionMetrics {
    total_predictions: TimeSeriesPoint[];
    successful_predictions: TimeSeriesPoint[];
    failed_predictions: TimeSeriesPoint[];
    average_confidence: TimeSeriesPoint[];
    average_uncertainty: TimeSeriesPoint[];
}

/**
 * WebSDR performance metrics
 */
export interface WebSDRPerformance {
    websdr_id: number;
    name: string;
    uptime_percentage: number;
    average_snr: number;
    total_acquisitions: number;
    successful_acquisitions: number;
}

/**
 * System performance metrics
 */
export interface SystemPerformance {
    cpu_usage: TimeSeriesPoint[];
    memory_usage: TimeSeriesPoint[];
    api_response_times: TimeSeriesPoint[];
    active_tasks: TimeSeriesPoint[];
}

/**
 * Get prediction metrics for the specified time range
 */
export async function getPredictionMetrics(timeRange: string = '7d'): Promise<PredictionMetrics> {
    const response = await api.get<PredictionMetrics>('/v1/analytics/predictions/metrics', {
        params: { time_range: timeRange }
    });
    return response.data;
}

/**
 * Get WebSDR performance metrics
 */
export async function getWebSDRPerformance(timeRange: string = '7d'): Promise<WebSDRPerformance[]> {
    const response = await api.get<WebSDRPerformance[]>('/v1/analytics/websdr/performance', {
        params: { time_range: timeRange }
    });
    return response.data;
}

/**
 * Get system performance metrics
 */
export async function getSystemPerformance(timeRange: string = '7d'): Promise<SystemPerformance> {
    const response = await api.get<SystemPerformance>('/v1/analytics/system/performance', {
        params: { time_range: timeRange }
    });
    return response.data;
}

/**
 * Get localization accuracy distribution
 */
export async function getAccuracyDistribution(timeRange: string = '7d'): Promise<{
    accuracy_ranges: string[];
    counts: number[];
}> {
    const response = await api.get<{
        accuracy_ranges: string[];
        counts: number[];
    }>('/v1/analytics/localizations/accuracy-distribution', {
        params: { time_range: timeRange }
    });
    return response.data;
}

/**
 * Dashboard metrics aggregated from various sources
 */
export interface DashboardMetrics {
    signalDetections: number;
    systemUptime: number;
    modelAccuracy: number;
    predictionsTotal: number;
    predictionsSuccessful: number;
    predictionsFailed: number;
    lastUpdate: string;
}

/**
 * Get aggregated dashboard metrics
 */
export async function getDashboardMetrics(): Promise<DashboardMetrics> {
    const response = await api.get<DashboardMetrics>('/v1/analytics/dashboard/metrics');
    return response.data;
}

const analyticsService = {
    getPredictionMetrics,
    getWebSDRPerformance,
    getSystemPerformance,
    getAccuracyDistribution,
    getDashboardMetrics,
};

export default analyticsService;