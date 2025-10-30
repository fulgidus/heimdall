/**
 * Zod schemas for runtime validation of API responses
 * 
 * These schemas ensure data integrity and catch malformed responses early.
 * All API service functions should validate responses using these schemas.
 */

import { z } from 'zod';

// ============================================================================
// WebSDR Schemas
// ============================================================================

export const WebSDRConfigSchema = z.object({
    id: z.string().uuid(),
    name: z.string(),
    url: z.string().url(),
    location_description: z.string().optional(),
    country: z.string().optional(),
    admin_email: z.string().email().optional(),
    altitude_asl: z.number().optional(),
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    is_active: z.boolean(),
    timeout_seconds: z.number().positive(),
    retry_count: z.number().nonnegative(),
    created_at: z.string().optional(),
    updated_at: z.string().optional(),
});

export const WebSDRHealthStatusSchema = z.object({
    websdr_id: z.string().uuid(),
    name: z.string(),
    status: z.enum(['online', 'offline', 'unknown']),
    response_time_ms: z.number().optional(),
    last_check: z.string(),
    error_message: z.string().optional(),
    uptime: z.number().optional(),
    avg_snr: z.number().nullable().optional(),
});

// ============================================================================
// Acquisition Schemas
// ============================================================================

export const AcquisitionRequestSchema = z.object({
    frequency_mhz: z.number().positive(),
    duration_seconds: z.number().positive(),
    start_time: z.string().optional(),
    websdrs: z.array(z.number()).optional(),
});

export const AcquisitionTaskResponseSchema = z.object({
    task_id: z.string(),
    status: z.string(),
    message: z.string(),
    frequency_mhz: z.number(),
    websdrs_count: z.number().nonnegative(),
});

export const AcquisitionStatusResponseSchema = z.object({
    task_id: z.string(),
    status: z.enum(['PENDING', 'PROGRESS', 'SUCCESS', 'FAILURE', 'REVOKED']),
    progress: z.number().min(0).max(100),
    message: z.string(),
    measurements_collected: z.number().nonnegative(),
    errors: z.array(z.string()).nullable().optional(),
    result: z.record(z.string(), z.unknown()).optional(),
});

// ============================================================================
// Inference/Model Schemas
// ============================================================================

export const ModelInfoSchema = z.object({
    active_version: z.string(),
    stage: z.string(),
    model_name: z.string(),
    accuracy: z.number().min(0).max(1).optional(),
    latency_p95_ms: z.number().optional(),
    cache_hit_rate: z.number().min(0).max(1).optional(),
    loaded_at: z.string(),
    uptime_seconds: z.number().nonnegative(),
    last_prediction_at: z.string().optional(),
    predictions_total: z.number().optional(),
    predictions_successful: z.number().optional(),
    predictions_failed: z.number().optional(),
    is_ready: z.boolean(),
    health_status: z.enum(['healthy', 'degraded', 'unhealthy']),
    error_message: z.string().optional(),
});

export const ModelPerformanceMetricsSchema = z.object({
    inference_latency_ms: z.number(),
    p50_latency_ms: z.number(),
    p95_latency_ms: z.number(),
    p99_latency_ms: z.number(),
    throughput_samples_per_second: z.number(),
    cache_hit_rate: z.number().min(0).max(1),
    success_rate: z.number().min(0).max(1),
    predictions_total: z.number().nonnegative(),
    requests_total: z.number().nonnegative(),
    errors_total: z.number().nonnegative(),
    uptime_seconds: z.number().nonnegative(),
    timestamp: z.string(),
});

export const PredictionResponseSchema = z.object({
    position: z.object({
        latitude: z.number().min(-90).max(90),
        longitude: z.number().min(-180).max(180),
    }),
    uncertainty: z.object({
        sigma_x: z.number(),
        sigma_y: z.number(),
        theta: z.number(),
        confidence_interval: z.number(),
    }),
    confidence: z.number().min(0).max(1),
    model_version: z.string(),
    inference_time_ms: z.number(),
    timestamp: z.string(),
    session_id: z.string(),
    _cache_hit: z.boolean(),
});

export const BatchPredictionResponseSchema = z.object({
    predictions: z.array(PredictionResponseSchema),
    batch_id: z.string(),
    total_time_ms: z.number(),
});

// ============================================================================
// System Schemas
// ============================================================================

export const ServiceHealthSchema = z.object({
    status: z.enum(['healthy', 'unhealthy', 'degraded']),
    service: z.string(),
    version: z.string(),
    timestamp: z.string(),
    details: z.record(z.string(), z.unknown()).optional(),
});

export const SystemMetricsSchema = z.object({
    cpu_percent: z.number().min(0).max(100),
    memory_percent: z.number().min(0).max(100),
    disk_percent: z.number().min(0).max(100),
    active_tasks: z.number().nonnegative(),
    queue_depth: z.number().nonnegative(),
    uptime_seconds: z.number().nonnegative(),
    timestamp: z.string(),
});

// ============================================================================
// Recording Session Schemas
// ============================================================================

export const RecordingSessionSchema = z.object({
    id: z.number(),
    session_name: z.string(),
    frequency_mhz: z.number().positive(),
    duration_seconds: z.number().positive(),
    status: z.enum(['pending', 'in_progress', 'processing', 'completed', 'failed']),
    celery_task_id: z.string().nullable().optional(),
    result_metadata: z.record(z.string(), z.unknown()).nullable().optional(),
    minio_path: z.string().nullable().optional(),
    error_message: z.string().nullable().optional(),
    created_at: z.string(),
    started_at: z.string().nullable().optional(),
    completed_at: z.string().nullable().optional(),
    websdrs_enabled: z.number().optional(),
});

export const RecordingSessionWithDetailsSchema = RecordingSessionSchema.extend({
    source_name: z.string().optional(),
    source_frequency: z.number().optional(),
    source_latitude: z.number().optional(),
    source_longitude: z.number().optional(),
    measurements_count: z.number().optional(),
    approval_status: z.enum(['pending', 'approved', 'rejected']).optional(),
    notes: z.string().optional(),
});

export const SessionListResponseSchema = z.object({
    sessions: z.array(RecordingSessionWithDetailsSchema),
    total: z.number().nonnegative(),
    page: z.number().positive(),
    per_page: z.number().positive(),
});

export const KnownSourceSchema = z.object({
    id: z.string().uuid(),
    name: z.string(),
    description: z.string().optional(),
    frequency_hz: z.number().optional(),
    latitude: z.number().min(-90).max(90).optional(),
    longitude: z.number().min(-180).max(180).optional(),
    power_dbm: z.number().optional(),
    source_type: z.string().optional(),
    is_validated: z.boolean(),
    error_margin_meters: z.number().nonnegative(),
    created_at: z.string(),
    updated_at: z.string(),
});

export const SessionAnalyticsSchema = z.object({
    total_sessions: z.number().nonnegative(),
    completed_sessions: z.number().nonnegative(),
    failed_sessions: z.number().nonnegative(),
    pending_sessions: z.number().nonnegative(),
    success_rate: z.number().min(0).max(100),
    average_duration_seconds: z.number(),
    total_measurements: z.number().nonnegative(),
});

// ============================================================================
// Localization Schemas
// ============================================================================

export const LocalizationResultSchema = z.object({
    id: z.string().uuid(),
    session_id: z.string(),
    timestamp: z.string(),
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    uncertainty_m: z.number().nonnegative(),
    confidence: z.number().min(0).max(1),
    source_frequency_mhz: z.number().positive(),
    snr_avg_db: z.number(),
    websdr_count: z.number().positive(),
});

export const UncertaintyEllipseSchema = z.object({
    center: z.array(z.number()).length(2).refine((arr) => arr[0] >= -90 && arr[0] <= 90 && arr[1] >= -180 && arr[1] <= 180, {
        message: "Center coordinates must be valid [lat, lon]"
    }),
    semi_major_axis_m: z.number().nonnegative(),
    semi_minor_axis_m: z.number().nonnegative(),
    rotation_degrees: z.number().min(0).max(360),
    confidence_level: z.number().min(0).max(1),
});

// ============================================================================
// Analytics Schemas
// ============================================================================

export const TimeSeriesPointSchema = z.object({
    timestamp: z.string(),
    value: z.number(),
});

export const PredictionMetricsSchema = z.object({
    total_predictions: z.array(TimeSeriesPointSchema),
    successful_predictions: z.array(TimeSeriesPointSchema),
    failed_predictions: z.array(TimeSeriesPointSchema),
    average_confidence: z.array(TimeSeriesPointSchema),
    average_uncertainty: z.array(TimeSeriesPointSchema),
});

export const WebSDRPerformanceSchema = z.object({
    websdr_id: z.number(),
    name: z.string(),
    uptime_percentage: z.number().min(0).max(100),
    average_snr: z.number(),
    total_acquisitions: z.number().nonnegative(),
    successful_acquisitions: z.number().nonnegative(),
});

export const SystemPerformanceSchema = z.object({
    cpu_usage: z.array(TimeSeriesPointSchema),
    memory_usage: z.array(TimeSeriesPointSchema),
    api_response_times: z.array(TimeSeriesPointSchema),
    active_tasks: z.array(TimeSeriesPointSchema),
});

export const DashboardMetricsSchema = z.object({
    signalDetections: z.number().nonnegative(),
    systemUptime: z.number().nonnegative(),
    modelAccuracy: z.number().min(0).max(1),
    predictionsTotal: z.number().nonnegative(),
    predictionsSuccessful: z.number().nonnegative(),
    predictionsFailed: z.number().nonnegative(),
    lastUpdate: z.string(),
});

// ============================================================================
// Type exports (inferred from schemas)
// ============================================================================

export type WebSDRConfig = z.infer<typeof WebSDRConfigSchema>;
export type WebSDRHealthStatus = z.infer<typeof WebSDRHealthStatusSchema>;
export type AcquisitionRequest = z.infer<typeof AcquisitionRequestSchema>;
export type AcquisitionTaskResponse = z.infer<typeof AcquisitionTaskResponseSchema>;
export type AcquisitionStatusResponse = z.infer<typeof AcquisitionStatusResponseSchema>;
export type ModelInfo = z.infer<typeof ModelInfoSchema>;
export type ModelPerformanceMetrics = z.infer<typeof ModelPerformanceMetricsSchema>;
export type PredictionResponse = z.infer<typeof PredictionResponseSchema>;
export type ServiceHealth = z.infer<typeof ServiceHealthSchema>;
export type RecordingSession = z.infer<typeof RecordingSessionSchema>;
export type RecordingSessionWithDetails = z.infer<typeof RecordingSessionWithDetailsSchema>;
export type SessionListResponse = z.infer<typeof SessionListResponseSchema>;
export type KnownSource = z.infer<typeof KnownSourceSchema>;
export type LocalizationResult = z.infer<typeof LocalizationResultSchema>;
export type DashboardMetrics = z.infer<typeof DashboardMetricsSchema>;
export type SessionAnalytics = z.infer<typeof SessionAnalyticsSchema>;
