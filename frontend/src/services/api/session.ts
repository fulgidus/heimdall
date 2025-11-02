/**
 * Session API Service
 *
 * Handles recording session operations:
 * - List sessions with pagination
 * - Create new sessions
 * - Get session details
 * - Update session status
 * - Get analytics
 * - Manage known sources
 */

import { z } from 'zod';
import api from '@/lib/api';
import {
  RecordingSessionSchema,
  RecordingSessionWithDetailsSchema,
  SessionListResponseSchema,
  KnownSourceSchema,
  SessionAnalyticsSchema,
} from './schemas';

export interface KnownSource {
  id: string;
  name: string;
  description?: string;
  frequency_hz?: number;
  latitude?: number;
  longitude?: number;
  power_dbm?: number;
  source_type?: string;
  is_validated: boolean;
  error_margin_meters: number | null;
  created_at: string;
  updated_at: string;
}

export interface KnownSourceCreate {
  name: string;
  description?: string;
  frequency_hz?: number;
  latitude?: number;
  longitude?: number;
  power_dbm?: number;
  source_type?: string;
  is_validated?: boolean;
  error_margin_meters?: number;
}

export interface KnownSourceUpdate {
  name?: string;
  description?: string;
  frequency_hz?: number;
  latitude?: number;
  longitude?: number;
  power_dbm?: number;
  source_type?: string;
  is_validated?: boolean;
  error_margin_meters?: number;
}

export interface RecordingSession {
  id: string; // UUID as string
  known_source_id: string | null; // UUID as string, null for unknown sources
  session_name: string;
  session_start: string; // ISO datetime
  session_end?: string | null; // ISO datetime, optional
  duration_seconds?: number | null; // Optional
  celery_task_id?: string | null;
  status: 'pending' | 'in_progress' | 'processing' | 'completed' | 'failed';
  approval_status?: 'pending' | 'approved' | 'rejected';
  notes?: string | null;
  created_at: string; // ISO datetime
  updated_at: string; // ISO datetime
}

export interface RecordingSessionWithDetails extends RecordingSession {
  source_name?: string;
  source_frequency?: number;
  source_latitude?: number;
  source_longitude?: number;
  measurements_count?: number;
}

export interface RecordingSessionCreate {
  known_source_id: string | null; // null for unknown sources
  session_name: string;
  frequency_hz: number;
  duration_seconds: number;
  notes?: string;
}

export interface RecordingSessionUpdate {
  session_name?: string;
  notes?: string;
  approval_status?: 'pending' | 'approved' | 'rejected';
}

export interface SessionListResponse {
  sessions: RecordingSessionWithDetails[];
  total: number;
  page: number;
  per_page: number;
}

export interface SessionAnalytics {
  total_sessions: number;
  completed_sessions: number;
  failed_sessions: number;
  pending_sessions: number;
  success_rate: number;
  total_measurements: number;
  average_duration_seconds?: number;
  average_accuracy_meters?: number;
}

/**
 * List recording sessions with pagination and filters
 */
export async function listSessions(params: {
  page?: number;
  per_page?: number;
  status?: string;
  approval_status?: string;
}): Promise<SessionListResponse> {
  const response = await api.get('/v1/sessions', { params });

  // Validate response with Zod
  const validated = SessionListResponseSchema.parse(response.data);
  return validated;
}

/**
 * Get a specific session by ID
 */
export async function getSession(sessionId: string): Promise<RecordingSessionWithDetails> {
  const response = await api.get(`/v1/sessions/${sessionId}`);

  // Validate response with Zod
  const validated = RecordingSessionWithDetailsSchema.parse(response.data);
  return validated;
}

/**
 * Create a new recording session
 */
export async function createSession(session: RecordingSessionCreate): Promise<RecordingSession> {
  const response = await api.post('/v1/sessions', session);

  // Validate response with Zod
  const validated = RecordingSessionSchema.parse(response.data);
  return validated;
}

/**
 * Update session metadata (session_name, notes, approval_status)
 */
export async function updateSession(
  sessionId: string,
  sessionUpdate: RecordingSessionUpdate
): Promise<RecordingSessionWithDetails> {
  const response = await api.patch(`/v1/sessions/${sessionId}`, sessionUpdate);

  // Validate response with Zod
  const validated = RecordingSessionWithDetailsSchema.parse(response.data);
  return validated;
}

/**
 * Update session status
 */
export async function updateSessionStatus(
  sessionId: string,
  status: string,
  celeryTaskId?: string
): Promise<RecordingSession> {
  const response = await api.patch(`/v1/sessions/${sessionId}/status`, null, {
    params: {
      status,
      celery_task_id: celeryTaskId,
    },
  });

  // Validate response with Zod
  const validated = RecordingSessionSchema.parse(response.data);
  return validated;
}

/**
 * Update session approval status
 */
export async function updateSessionApproval(
  sessionId: string,
  approvalStatus: 'pending' | 'approved' | 'rejected'
): Promise<RecordingSession> {
  const response = await api.patch(`/v1/sessions/${sessionId}/approval`, null, {
    params: {
      approval_status: approvalStatus,
    },
  });

  // Validate response with Zod
  const validated = RecordingSessionSchema.parse(response.data);
  return validated;
}

/**
 * Delete a recording session
 */
export async function deleteSession(sessionId: string): Promise<void> {
  await api.delete(`/v1/sessions/${sessionId}`);
}

/**
 * Get session analytics
 */
export async function getSessionAnalytics(): Promise<SessionAnalytics> {
  const response = await api.get('/v1/sessions/analytics');

  // Validate response with Zod
  const validated = SessionAnalyticsSchema.parse(response.data);
  return validated;
}

/**
 * List all known RF sources
 */
export async function listKnownSources(): Promise<KnownSource[]> {
  const response = await api.get('/v1/sessions/known-sources');

  // Validate response with Zod
  const validated = z.array(KnownSourceSchema).parse(response.data);
  return validated;
}

/**
 * Get a specific known RF source
 */
export async function getKnownSource(sourceId: string): Promise<KnownSource> {
  const response = await api.get<KnownSource>(`/v1/sessions/known-sources/${sourceId}`);
  return response.data;
}

/**
 * Create a new known RF source
 */
export async function createKnownSource(source: KnownSourceCreate): Promise<KnownSource> {
  const response = await api.post<KnownSource>('/v1/sessions/known-sources', source);
  return response.data;
}

/**
 * Update a known RF source
 */
export async function updateKnownSource(
  sourceId: string,
  source: KnownSourceUpdate
): Promise<KnownSource> {
  const response = await api.put<KnownSource>(`/v1/sessions/known-sources/${sourceId}`, source);
  return response.data;
}

/**
 * Delete a known RF source
 */
export async function deleteKnownSource(sourceId: string): Promise<void> {
  await api.delete(`/v1/sessions/known-sources/${sourceId}`);
}

const sessionService = {
  listSessions,
  getSession,
  createSession,
  updateSession,
  updateSessionStatus,
  updateSessionApproval,
  deleteSession,
  getSessionAnalytics,
  listKnownSources,
  getKnownSource,
  createKnownSource,
  updateKnownSource,
  deleteKnownSource,
};

export default sessionService;
