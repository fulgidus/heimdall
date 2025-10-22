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

import api from '@/lib/api';

export interface KnownSource {
    id: string;
    name: string;
    description?: string;
    frequency_hz: number;
    latitude: number;
    longitude: number;
    power_dbm?: number;
    source_type?: string;
    is_validated: boolean;
    created_at: string;
    updated_at: string;
}

export interface KnownSourceCreate {
    name: string;
    description?: string;
    frequency_hz: number;
    latitude: number;
    longitude: number;
    power_dbm?: number;
    source_type?: string;
    is_validated?: boolean;
}

export interface RecordingSession {
    id: string;
    known_source_id: string;
    session_name: string;
    session_start: string;
    session_end?: string;
    duration_seconds?: number;
    celery_task_id?: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    approval_status: 'pending' | 'approved' | 'rejected';
    notes?: string;
    created_at: string;
    updated_at: string;
}

export interface RecordingSessionWithDetails extends RecordingSession {
    source_name: string;
    source_frequency: number;
    source_latitude: number;
    source_longitude: number;
    measurements_count: number;
}

export interface RecordingSessionCreate {
    known_source_id: string;
    session_name: string;
    frequency_hz: number;
    duration_seconds: number;
    notes?: string;
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
    const response = await api.get<SessionListResponse>('/api/v1/sessions', { params });
    return response.data;
}

/**
 * Get a specific session by ID
 */
export async function getSession(sessionId: string): Promise<RecordingSessionWithDetails> {
    const response = await api.get<RecordingSessionWithDetails>(`/api/v1/sessions/${sessionId}`);
    return response.data;
}

/**
 * Create a new recording session
 */
export async function createSession(session: RecordingSessionCreate): Promise<RecordingSession> {
    const response = await api.post<RecordingSession>('/api/v1/sessions', session);
    return response.data;
}

/**
 * Update session status
 */
export async function updateSessionStatus(
    sessionId: string,
    status: string,
    celeryTaskId?: string
): Promise<RecordingSession> {
    const response = await api.patch<RecordingSession>(
        `/api/v1/sessions/${sessionId}/status`,
        null,
        {
            params: {
                status,
                celery_task_id: celeryTaskId,
            },
        }
    );
    return response.data;
}

/**
 * Update session approval status
 */
export async function updateSessionApproval(
    sessionId: string,
    approvalStatus: 'pending' | 'approved' | 'rejected'
): Promise<RecordingSession> {
    const response = await api.patch<RecordingSession>(
        `/api/v1/sessions/${sessionId}/approval`,
        null,
        {
            params: {
                approval_status: approvalStatus,
            },
        }
    );
    return response.data;
}

/**
 * Delete a recording session
 */
export async function deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/api/v1/sessions/${sessionId}`);
}

/**
 * Get session analytics
 */
export async function getSessionAnalytics(): Promise<SessionAnalytics> {
    const response = await api.get<SessionAnalytics>('/api/v1/sessions/analytics');
    return response.data;
}

/**
 * List all known RF sources
 */
export async function listKnownSources(): Promise<KnownSource[]> {
    const response = await api.get<KnownSource[]>('/api/v1/sessions/known-sources');
    return response.data;
}

/**
 * Create a new known RF source
 */
export async function createKnownSource(source: KnownSourceCreate): Promise<KnownSource> {
    const response = await api.post<KnownSource>('/api/v1/sessions/known-sources', source);
    return response.data;
}

const sessionService = {
    listSessions,
    getSession,
    createSession,
    updateSessionStatus,
    updateSessionApproval,
    deleteSession,
    getSessionAnalytics,
    listKnownSources,
    createKnownSource,
};

export default sessionService;
