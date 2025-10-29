/**
 * API Types for Heimdall SDR Backend Integration
 * 
 * This file contains TypeScript types matching the FastAPI backend schemas
 */

// ============================================================================
// WebSDR Types
// ============================================================================

export interface WebSDRConfig {
    id: string;  // UUID from backend
    name: string;
    url: string;
    location_description?: string;  // Optional location description from backend
    country?: string;  // Optional country
    admin_email?: string;  // Optional admin email
    altitude_asl?: number;  // Optional altitude ASL
    latitude: number;
    longitude: number;
    is_active: boolean;
    timeout_seconds: number;
    retry_count: number;
    created_at?: string;  // ISO timestamp
    updated_at?: string;  // ISO timestamp
}

export interface WebSDRHealthStatus {
    websdr_id: string;  // UUID from backend
    name: string;
    status: 'online' | 'offline' | 'unknown';
    response_time_ms?: number;
    last_check: string;
    error_message?: string;
    uptime?: number;
    avg_snr?: number | null;
}

// ============================================================================
// Acquisition Types
// ============================================================================

export interface AcquisitionRequest {
    frequency_mhz: number;
    duration_seconds: number;
    start_time?: string; // ISO datetime
    websdrs?: number[]; // WebSDR IDs to use
}

export interface AcquisitionTaskResponse {
    task_id: string;
    status: string;
    message: string;
    frequency_mhz: number;
    websdrs_count: number;
}

export interface AcquisitionStatusResponse {
    task_id: string;
    status: 'PENDING' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED';
    progress: number; // 0-100
    message: string;
    measurements_collected: number;
    errors?: string[] | null;
    result?: Record<string, unknown>;
}

// ============================================================================
// Inference Types
// ============================================================================

export interface ModelInfo {
    active_version: string;
    stage: string;
    model_name: string;
    accuracy?: number;
    latency_p95_ms?: number;
    cache_hit_rate?: number;
    loaded_at: string;
    uptime_seconds: number;
    last_prediction_at?: string;
    predictions_total?: number;
    predictions_successful?: number;
    predictions_failed?: number;
    is_ready: boolean;
    health_status: 'healthy' | 'degraded' | 'unhealthy';
    error_message?: string;
}

export interface ModelPerformanceMetrics {
    inference_latency_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    throughput_samples_per_second: number;
    cache_hit_rate: number;
    success_rate: number;
    predictions_total: number;
    requests_total: number;
    errors_total: number;
    uptime_seconds: number;
    timestamp: string;
}

// ============================================================================
// System Types
// ============================================================================

export interface ServiceHealth {
    status: 'healthy' | 'unhealthy' | 'degraded';
    service: string;
    version: string;
    timestamp: string;
    details?: Record<string, unknown>;
}

export interface SystemMetrics {
    cpu_percent: number;
    memory_percent: number;
    disk_percent: number;
    active_tasks: number;
    queue_depth: number;
    uptime_seconds: number;
    timestamp: string;
}

// ============================================================================
// Recording Session Types
// ============================================================================

export interface RecordingSession {
    id: string;
    name: string;
    frequency_mhz: number;
    duration_seconds: number;
    status: 'pending' | 'recording' | 'processing' | 'completed' | 'failed';
    created_at: string;
    started_at?: string;
    completed_at?: string;
    websdrs_count: number;
    measurements_count: number;
    task_id?: string;
    notes?: string;
}

// ============================================================================
// Localization Types
// ============================================================================

export interface LocalizationResult {
    id: string;
    session_id: string;
    timestamp: string;
    latitude: number;
    longitude: number;
    uncertainty_m: number; // meters
    confidence: number; // 0-1
    source_frequency_mhz: number;
    snr_avg_db: number;
    websdr_count: number;
}

export interface UncertaintyEllipse {
    center: [number, number]; // [lat, lon]
    semi_major_axis_m: number;
    semi_minor_axis_m: number;
    rotation_degrees: number;
    confidence_level: number; // e.g., 0.95
}
