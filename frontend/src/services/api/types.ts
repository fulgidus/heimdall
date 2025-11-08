/**
 * API Types for Heimdall SDR Backend Integration
 *
 * This file re-exports types from schemas.ts for backward compatibility.
 * All new code should import from schemas.ts directly.
 */

// Re-export all types from schemas
export type {
  WebSDRConfig,
  WebSDRHealthStatus,
  AcquisitionRequest,
  AcquisitionTaskResponse,
  AcquisitionStatusResponse,
  ModelInfo,
  ModelPerformanceMetrics,
  PredictionResponse,
  ServiceHealth,
  RecordingSession,
  RecordingSessionWithDetails,
  SessionListResponse,
  KnownSource,
  LocalizationResult,
  DashboardMetrics,
  SessionAnalytics,
} from './schemas';

// Keep legacy exports for other types still used in codebase
export interface UncertaintyEllipse {
  center: [number, number]; // [lat, lon]
  semi_major_axis_m: number;
  semi_minor_axis_m: number;
  rotation_degrees: number;
  confidence_level: number; // e.g., 0.95
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
